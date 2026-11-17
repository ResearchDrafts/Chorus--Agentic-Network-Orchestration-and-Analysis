"""The orchestrator's end of script compliance, against a gateway that
drifts exactly the way the real one did.

The point of these is the ratchet: in opinion_colab_run4, a slipped reply
was stored, shown back to its author as "Your current opinion:" and
delivered to up to N neighbours, so Roman -> Devanagari cost one unlucky
sample while Devanagari -> Roman had to beat both the model's prior and
the agent's own visible history. Retrying and, failing that, romanizing
before the text is ever stored is what makes a slip stay one slip.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from sandbox.model_gateway import BackendResponse
from sandbox.opinion_models import OpinionExperimentConfig
from sandbox.opinion_orchestrator import run
from sandbox.opinion_script import script_violation

DEVANAGARI_REPLY = (
    "BELIEF: -1.5\nCONFIDENCE: 4\n"
    "मुझे लगता है कि अभिव्यक्ति की स्वतंत्रता पर कुछ सीमाएँ होनी चाहिए।"
)
ROMAN_REPLY = (
    "BELIEF: -1.5\nCONFIDENCE: 4\n"
    "Mujhe lagta hai ki abhivyakti ki swatantrata par kuch seemaayein honi chahiye."
)


class ScriptedGateway:
    """Returns whatever `replies` says, in order, and records every call."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    async def generate(self, prompt_text, **kwargs):
        self.calls.append({"prompt": prompt_text, **kwargs})
        text = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        return BackendResponse(
            text=text, prompt_tokens=10, completion_tokens=10, latency_ms=1,
            raw_provider_response={},
        )


def _config(tmp_path: Path, **overrides) -> OpinionExperimentConfig:
    base = dict(
        run_id="script_test",
        topic="Is there such a thing as a paradox of free speech?",
        M=6, K=1,
        languages={"english": False, "hindi": False,
                   "hinglish": False, "romanized_hindi": True},
        similarity_threshold=2.0,
        N=2, graph_degree=4, graph_rewire_p=0.1,
        allow_short_feed=True,
        seed=1000,
        model_backend_id="hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
        api_base="http://localhost:8000/v1",
        persona_pool_path="data/personas/pool_100.jsonl",
        max_reply_tokens=256,
        max_concurrency=4,
        output_dir=str(tmp_path / "out"),
    )
    base.update(overrides)
    return OpinionExperimentConfig.model_validate(base)


async def _run_with(monkeypatch, tmp_path, replies, **overrides):
    gw = ScriptedGateway(replies)
    monkeypatch.setattr(
        "sandbox.opinion_orchestrator.ModelGateway", lambda *a, **k: gw
    )
    out = await run(_config(tmp_path, **overrides))
    return gw, pd.read_csv(Path(out) / "initial_stance.csv")


async def test_a_devanagari_reply_never_reaches_the_csv_in_a_roman_arm(
    monkeypatch, tmp_path
):
    # Every sample drifts, so retries cannot save it and the repair path is
    # the only thing standing between Devanagari and the stored text.
    gw, df = await _run_with(monkeypatch, tmp_path, [DEVANAGARI_REPLY])

    for col in ["initial_opinion", "round_1_opinion"]:
        for text in df[col]:
            assert script_violation(text, "romanized_hindi") is None, text
    assert set(df["initial_status"]) == {"script_repaired"}
    assert set(df["round_1_status"]) == {"script_repaired"}


async def test_a_slip_is_retried_before_it_is_repaired(monkeypatch, tmp_path):
    # First sample drifts, second complies: the reply that gets stored is
    # the model's own, not a transliteration of its mistake.
    gw, df = await _run_with(
        monkeypatch, tmp_path, [DEVANAGARI_REPLY, ROMAN_REPLY],
    )
    assert df.loc[0, "initial_status"] == "script_retried"
    assert "Mujhe lagta hai" in df.loc[0, "initial_opinion"]


async def test_a_complying_run_costs_exactly_one_call_per_agent_per_round(
    monkeypatch, tmp_path
):
    # The retry must not fire when nothing is wrong, and the first attempt
    # must keep the original seed, so a compliant run's traffic is
    # byte-identical to what it was before script checking existed.
    gw, df = await _run_with(monkeypatch, tmp_path, [ROMAN_REPLY])
    assert len(gw.calls) == 6 * 2            # M agents x (round 0 + round 1)
    assert set(df["initial_status"]) == {"success"}
    assert set(df["round_1_status"]) == {"success"}


async def test_every_retry_is_a_fresh_sample_not_the_same_request(
    monkeypatch, tmp_path
):
    # Re-sending an identical seeded request returns the identical reply --
    # that is what seeding is for -- so a retry that reused the seed would
    # be a guaranteed no-op that still costs a call.
    gw, _ = await _run_with(monkeypatch, tmp_path, [DEVANAGARI_REPLY])
    round0 = [c["seed"] for c in gw.calls[:3]]
    assert len(set(round0)) == 3, round0


async def test_the_feed_warns_that_posts_may_be_in_another_script(
    monkeypatch, tmp_path
):
    gw, _ = await _run_with(monkeypatch, tmp_path, [ROMAN_REPLY])
    update_prompts = [c["prompt"] for c in gw.calls if "you follow" in c["prompt"]]
    assert update_prompts
    for prompt in update_prompts:
        assert "may be written in any script" in prompt
        # Before the feed, not only after it: by the time the directive is
        # read, 2,000 characters of the other script have already been.
        assert prompt.index("may be written in any script") < prompt.index("you follow")


async def test_a_hindi_arm_reply_in_latin_is_flagged_and_kept(monkeypatch, tmp_path):
    # Latin -> Devanagari is not a transliteration this codebase will guess
    # at, so the arm reports the failure instead of inventing text.
    gw, df = await _run_with(
        monkeypatch, tmp_path, [ROMAN_REPLY],
        languages={"english": False, "hindi": True,
                   "hinglish": False, "romanized_hindi": False},
    )
    assert df.loc[0, "initial_status"] == "script_unrepaired"
    assert "Mujhe lagta hai" in df.loc[0, "initial_opinion"]
