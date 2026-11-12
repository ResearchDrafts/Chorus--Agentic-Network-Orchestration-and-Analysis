"""Tests for run_simulation.py, the launcher.

Nothing covered this module before. It is thin, but two things in it are not
cosmetic: the call counts it prints are what a researcher checks before
spending GPU hours, and `launch` is the only place the gateway is constructed
for a real run -- a setting the schema accepts but `launch` never passes
through is a setting that silently does nothing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import run_simulation
from tests.factories import make_run

BANK_KWARGS = dict(
    regime="bank",
    bank_path="data/bank/test_bank.csv",
    probe_question="Should the fare subsidy continue?",
    neighbor_strategy="graph",
    graph_degree=4,
    M=8,
    N=3,
    K=3,
)


# --- the printed call counts ---------------------------------------------


def test_a_generated_run_counts_one_call_per_agent_turn():
    lines = run_simulation._describe_calls(make_run(M=10, K=4))
    assert lines == ["at most 40 LLM calls (M x K), fewer if memes replace turns"]


def test_a_bank_run_counts_two_calls_per_agent_round_plus_round_zero():
    lines = run_simulation._describe_calls(make_run(**BANK_KWARGS))
    assert "24 reply calls" in lines[0]        # M x K
    assert "32 probe calls" in lines[1]        # M x (K + 1)
    assert "56 LLM calls in total" in lines[2]


def test_the_delivery_off_control_costs_the_same_as_a_treatment_arm():
    """It reads the neutral message and replies to it, so it makes M x K
    reply calls too. It used to make none, and the printed plan said so."""
    run = make_run(
        **BANK_KWARGS, delivery="off", neutral_message="Fares come up a lot."
    )
    lines = run_simulation._describe_calls(run)

    assert "24 reply calls" in lines[0]
    assert "neutral message" in lines[0]
    assert "32 probe calls" in lines[1]
    assert "56 LLM calls in total" in lines[2]
    assert not any("none" in line for line in lines)


def test_the_regime_line_says_what_the_control_reads():
    off = make_run(**BANK_KWARGS, delivery="off", neutral_message="Fares come up.")
    assert "side-free message" in run_simulation._describe_regime(off)
    assert "delivery ON" in run_simulation._describe_regime(make_run(**BANK_KWARGS))


def test_describe_prints_the_neutral_message_for_a_control_run(capsys, tmp_path):
    """So a reviewer can read the one post the whole arm is built on, the same
    way describe() prints the probe question in full."""
    run = make_run(
        **BANK_KWARGS, delivery="off", neutral_message="Fares come up a lot."
    )
    run_simulation.describe(run, tmp_path)
    assert "Fares come up a lot." in capsys.readouterr().out


# --- the probe language warning -------------------------------------------


@pytest.mark.parametrize("condition", ["english", "hinglish", "romanized_hindi"])
def test_the_roman_script_arms_are_never_warned_about(condition):
    """Nothing short of language identification could tell a Hinglish
    question from an English one, and a warning nobody believes is a warning
    nobody reads."""
    run = make_run(**BANK_KWARGS, language_condition=condition)
    assert run_simulation._probe_language_warnings(run) == []


# --- launch wires the gateway settings -----------------------------------


class _RawBoth:
    """One provider response that satisfies both call paths.

    generate() reads choices[0].message.content; score_next_token() reads
    choices[0].logprobs.content[0].top_logprobs. A single object answering
    both lets one patched litellm.acompletion serve a whole run.
    """

    def __init__(self):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content="5"),
                logprobs=SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            top_logprobs=[SimpleNamespace(token="5", logprob=-0.01)]
                        )
                    ]
                ),
            )
        ]
        self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=1)

    def model_dump(self) -> dict:
        return {"fake": True}


LAUNCH_RUN = dict(
    regime="bank",
    bank_path="data/bank/test_bank.csv",
    probe_question="Should the fare subsidy continue?",
    M=4,
    N=1,
    K=1,
)


async def test_launch_sends_the_runs_probe_top_logprobs_to_the_provider(tmp_path):
    """End to end: a non-default probe_top_logprobs reaches
    litellm.acompletion.

    It did not before. The field was validated, recorded in run_config.json
    and never passed, so a run configured to ask for 7 asked for 20 -- and a
    recorded setting the run did not use is worse than no setting at all.
    """
    run = make_run(**LAUNCH_RUN, probe_top_logprobs=7, run_id="launch_probe")
    completion = AsyncMock(side_effect=lambda *a, **kw: _RawBoth())

    with patch("litellm.acompletion", completion):
        await run_simulation.launch(run, tmp_path)

    probe_calls = [
        c for c in completion.await_args_list if "top_logprobs" in c.kwargs
    ]
    assert probe_calls, "no probe call reached the provider"
    assert all(c.kwargs["top_logprobs"] == 7 for c in probe_calls)
    assert all(c.kwargs["logprobs"] is True for c in probe_calls)
    # M x (K + 1) probes, and the reply calls ask for no logprobs at all.
    assert len(probe_calls) == run.M * (run.K + 1)
    assert len(completion.await_args_list) == len(probe_calls) + run.M * run.K


async def test_launch_passes_max_concurrency_to_the_gateway(tmp_path):
    """The bank regime dispatches M calls at once, twice per round, so the
    cap is what keeps a self-hosted server's connection count bounded. It was
    reachable only from the notebook before."""
    captured = {}
    real_gateway = run_simulation.ModelGateway

    def recording(*args, **kwargs):
        captured.update(kwargs)
        return real_gateway(*args, **kwargs)

    run = make_run(
        **LAUNCH_RUN, max_concurrency=3, probe_top_logprobs=11, run_id="launch_cap"
    )
    with patch("litellm.acompletion", AsyncMock(side_effect=lambda *a, **kw: _RawBoth())):
        with patch.object(run_simulation, "ModelGateway", recording):
            await run_simulation.launch(run, tmp_path)

    assert captured["max_concurrency"] == 3
    assert captured["top_logprobs"] == 11


async def test_launch_leaves_both_settings_unbounded_by_default(tmp_path):
    """None means unbounded, which is the right default for a hosted
    provider that pools connections itself."""
    captured = {}
    real_gateway = run_simulation.ModelGateway

    def recording(*args, **kwargs):
        captured.update(kwargs)
        return real_gateway(*args, **kwargs)

    run = make_run(**LAUNCH_RUN, run_id="launch_default")
    assert run.max_concurrency is None
    with patch("litellm.acompletion", AsyncMock(side_effect=lambda *a, **kw: _RawBoth())):
        with patch.object(run_simulation, "ModelGateway", recording):
            await run_simulation.launch(run, tmp_path)

    assert captured["max_concurrency"] is None
    assert captured["top_logprobs"] == 20  # the schema default
