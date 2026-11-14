"""Tests for simulation_orchestrator.py, organised around phase4.md Section 2.

The gateway is always mocked. Nothing here makes a network call.
"""

from __future__ import annotations

import json

import pytest

from sandbox.cost_tracker import CostCeilingExceeded
from sandbox.model_gateway import FatalGatewayError
from sandbox.models import ExperimentRun, Interaction, MemeInjectionConfig
from sandbox.simulation_orchestrator import (
    build_orchestrator,
    _current_stance_for_logging,
)
from tests.factories import make_agent, make_backend_response, make_run


def _run(**overrides) -> ExperimentRun:
    base = dict(M=5, N=2, K=2, run_id="orch_run", persona_pool_id="test_pool")
    base.update(overrides)
    return make_run(**base)


def _meme_config(**overrides) -> MemeInjectionConfig:
    base = dict(enabled=True, meme_pool_id="test_pool", injection_rate=1.0)
    base.update(overrides)
    return MemeInjectionConfig(**base)


def _interactions(tmp_path, run_id="orch_run") -> list[Interaction]:
    path = tmp_path / run_id / "interactions.jsonl"
    return [
        Interaction.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _run_config(tmp_path, run_id="orch_run") -> dict:
    return json.loads((tmp_path / run_id / "run_config.json").read_text())


# --- the happy path -------------------------------------------------------


async def test_full_run_writes_one_interaction_per_agent_turn(tmp_path, mock_gateway):
    run = _run()
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    assert len(rows) == run.M * run.K


async def test_every_row_has_exactly_n_neighbours_and_excludes_the_speaker(
    tmp_path, mock_gateway
):
    run = _run()
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    for row in _interactions(tmp_path):
        assert len(row.neighbor_agent_ids) == run.N
        assert row.speaker_agent_id not in row.neighbor_agent_ids


async def test_agents_final_and_checkpoint_are_written(tmp_path, mock_gateway):
    run = _run()
    orchestrator = build_orchestrator(run, mock_gateway, runs_dir=tmp_path)
    await orchestrator.run()

    assert (tmp_path / run.run_id / "agents_final.jsonl").exists()
    assert (tmp_path / run.run_id / "checkpoints" / "latest.json").exists()


async def test_orchestrator_creates_the_run_directory(tmp_path, mock_gateway):
    """LoggingWriter deliberately does not mkdir (phase3.md:166 resolved in
    favour of the caller), so the orchestrator must, or write_run_config
    raises FileNotFoundError on a fresh run."""
    assert not (tmp_path / "orch_run").exists()
    await build_orchestrator(_run(), mock_gateway, runs_dir=tmp_path).run()
    assert (tmp_path / "orch_run" / "run_config.json").exists()


# --- Q4.1 / Fix F: neighbour posts come from the frozen snapshot ---------


async def test_turn_one_sends_no_neighbour_posts(tmp_path, mock_gateway):
    """Nobody has spoken at turn 1, so no prompt may contain a neighbour
    block. Guards against reading turn-t output during turn t."""
    await build_orchestrator(_run(K=1), mock_gateway, runs_dir=tmp_path).run()

    for call in mock_gateway.generate.await_args_list:
        assert "What others in your feed" not in call.args[0]


async def test_later_turns_do_send_neighbour_posts(tmp_path, mock_gateway):
    await build_orchestrator(_run(K=2), mock_gateway, runs_dir=tmp_path).run()

    prompts = [c.args[0] for c in mock_gateway.generate.await_args_list]
    assert any("What others in your feed" in p for p in prompts)


async def test_identical_seeds_produce_identical_runs(tmp_path, mock_gateway):
    """Reproducibility is the whole point of the seeded-RNG discipline, and
    nothing else in the suite exercises SeedManager against its real
    consumers."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    await build_orchestrator(_run(seed=99), mock_gateway, runs_dir=first).run()
    await build_orchestrator(_run(seed=99), mock_gateway, runs_dir=second).run()

    def neighbours(base):
        return [
            (r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids))
            for r in _interactions(base)
        ]

    assert neighbours(first) == neighbours(second)


async def test_different_seeds_produce_different_neighbour_draws(tmp_path, mock_gateway):
    await build_orchestrator(_run(seed=1), mock_gateway, runs_dir=tmp_path / "a").run()
    await build_orchestrator(_run(seed=2), mock_gateway, runs_dir=tmp_path / "b").run()

    def neighbours(base):
        return [tuple(r.neighbor_agent_ids) for r in _interactions(base)]

    assert neighbours(tmp_path / "a") != neighbours(tmp_path / "b")


# --- Q4.8 / 2g: the vision path is actually wired -----------------------


async def test_meme_turns_cost_no_llm_call_and_take_the_dataset_stance(
    tmp_path, mock_gateway
):
    run = _run(K=1, meme_injection=_meme_config())
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    assert all(r.content_type == "meme" for r in rows)
    assert all(r.meme_id is not None for r in rows)
    assert all(r.prompt_token_count == 0 and r.completion_token_count == 0 for r in rows)
    assert mock_gateway.generate.await_count == 0


async def test_a_neighbours_meme_reaches_a_vision_backend_as_an_image(
    tmp_path, mock_gateway
):
    """Everyone posts a meme on turn 1, so every turn-2 prompt carries one."""
    run = _run(K=2, meme_injection=_meme_config(injection_schedule="fixed_turn", fixed_turns={1}))
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    images = [c.kwargs.get("image") for c in mock_gateway.generate.await_args_list]
    assert any(img is not None for img in images)


async def test_text_only_backend_records_used_vision_fallback(
    tmp_path, text_only_gateway
):
    """Without this wiring used_vision_fallback stays False in every run ever
    produced, and vision_fallback.py remains unreachable dead code."""
    run = _run(K=2, meme_injection=_meme_config(injection_schedule="fixed_turn", fixed_turns={1}))
    await build_orchestrator(run, text_only_gateway, runs_dir=tmp_path).run()

    turn_two = [r for r in _interactions(tmp_path) if r.turn == 2]
    assert any(r.used_vision_fallback for r in turn_two)
    assert all(c.kwargs.get("image") is None for c in text_only_gateway.generate.await_args_list)


# --- 2b: the exception taxonomy ------------------------------------------


async def test_unparseable_reply_retries_then_logs_failed_logged_null(
    tmp_path, mock_gateway
):
    mock_gateway.generate.return_value = make_backend_response(text="no stance here")
    await build_orchestrator(_run(K=1), mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    assert all(r.api_call_status == "failed_logged_null" for r in rows)
    assert all(not r.is_valid_for_analysis for r in rows)
    # 3 attempts per agent, per Fix I.
    assert mock_gateway.generate.await_count == 5 * 3


async def test_a_recovered_parse_failure_is_marked_retried_success(
    tmp_path, mock_gateway
):
    mock_gateway.generate.side_effect = [
        make_backend_response(text="junk"),
        make_backend_response(stance=6.0),
    ] + [make_backend_response(stance=5.0)] * 20

    await build_orchestrator(_run(M=2, N=1, K=1), mock_gateway, runs_dir=tmp_path).run()

    statuses = {r.api_call_status for r in _interactions(tmp_path)}
    assert "retried_success" in statuses


async def test_failed_agent_keeps_its_previous_stance(tmp_path, mock_gateway):
    """Fix I: never fabricate a placeholder stance. stance_after repeats
    stance_before, and no StanceRecord is appended."""
    mock_gateway.generate.return_value = make_backend_response(text="no stance here")
    await build_orchestrator(_run(K=1), mock_gateway, runs_dir=tmp_path).run()

    for row in _interactions(tmp_path):
        assert row.stance_after == row.stance_before


async def test_fatal_gateway_error_is_absorbed_per_agent_not_fatal_to_the_run(
    tmp_path, mock_gateway
):
    mock_gateway.generate.side_effect = FatalGatewayError("bad key")
    await build_orchestrator(_run(K=1), mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    assert len(rows) == 5
    assert all(r.api_call_status == "failed_logged_null" for r in rows)


async def test_fatal_gateway_error_is_not_retried(tmp_path, mock_gateway):
    mock_gateway.generate.side_effect = FatalGatewayError("bad key")
    await build_orchestrator(_run(K=1), mock_gateway, runs_dir=tmp_path).run()
    assert mock_gateway.generate.await_count == 5  # one attempt each, not three


async def test_cost_ceiling_aborts_the_whole_run(tmp_path, mock_gateway):
    """The contradiction phase4.md 2b resolves. Sec 3.9's sample `continue`s
    past every gathered exception, which would silently swallow this and let
    the run continue spending."""
    mock_gateway.generate.side_effect = CostCeilingExceeded(10.0, 5.0)

    with pytest.raises(CostCeilingExceeded):
        await build_orchestrator(_run(K=2), mock_gateway, runs_dir=tmp_path).run()


async def test_an_unexpected_exception_aborts_the_whole_run(tmp_path, mock_gateway):
    """Only the two designed failure modes get soft handling; anything else
    is a bug and must surface rather than becoming a failed_logged_null row."""
    mock_gateway.generate.side_effect = RuntimeError("something nobody planned for")

    with pytest.raises(RuntimeError, match="nobody planned for"):
        await build_orchestrator(_run(K=2), mock_gateway, runs_dir=tmp_path).run()


async def test_a_fatal_abort_records_failed_partial_status(tmp_path, mock_gateway):
    """A run that dies must not be left looking like it is still running."""
    mock_gateway.generate.side_effect = [make_backend_response(stance=5.0)] * 5 + [
        RuntimeError("boom")
    ] * 20

    with pytest.raises(RuntimeError):
        await build_orchestrator(_run(K=3), mock_gateway, runs_dir=tmp_path).run()

    assert _run_config(tmp_path)["status"] == "failed_partial"


# --- 2c: the run-lifecycle fields nothing used to read -------------------


async def test_completed_run_persists_its_lifecycle_fields(tmp_path, mock_gateway):
    await build_orchestrator(_run(), mock_gateway, runs_dir=tmp_path).run()

    config = _run_config(tmp_path)
    assert config["status"] == "completed"
    assert config["started_at_utc"]
    assert config["completed_at_utc"]


async def test_checkpoint_every_n_turns_is_honoured(tmp_path, mock_gateway):
    """Sec 3.9's sample checkpoints unconditionally, ignoring the field."""
    from sandbox.checkpoint_manager import CheckpointManager

    run = _run(K=4, checkpoint_every_n_turns=2)
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    assert CheckpointManager(tmp_path).load(run.run_id).last_completed_turn == 4


async def test_final_turn_is_checkpointed_even_when_the_interval_skips_it(
    tmp_path, mock_gateway
):
    """K=3 with checkpoint_every_n_turns=2 checkpoints turn 2 and would
    otherwise never checkpoint turn 3, so a completed run would resume from
    the wrong place."""
    from sandbox.checkpoint_manager import CheckpointManager

    run = _run(K=3, checkpoint_every_n_turns=2)
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    assert CheckpointManager(tmp_path).load(run.run_id).last_completed_turn == 3


# --- resume ---------------------------------------------------------------


async def test_rerunning_a_completed_run_is_a_no_op(tmp_path, mock_gateway):
    run = _run()
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()
    rows_before = len(_interactions(tmp_path))
    calls_before = mock_gateway.generate.await_count

    await build_orchestrator(_run(), mock_gateway, runs_dir=tmp_path).run()

    assert len(_interactions(tmp_path)) == rows_before
    assert mock_gateway.generate.await_count == calls_before


async def test_resume_continues_from_the_checkpoint(tmp_path, mock_gateway):
    """Crash after turn 1, then resume and finish."""
    mock_gateway.generate.side_effect = [make_backend_response(stance=5.0)] * 5 + [
        RuntimeError("boom")
    ] * 50

    with pytest.raises(RuntimeError):
        await build_orchestrator(_run(K=3), mock_gateway, runs_dir=tmp_path).run()
    assert {r.turn for r in _interactions(tmp_path)} == {1}

    mock_gateway.generate.side_effect = None
    mock_gateway.generate.return_value = make_backend_response(stance=5.0)
    await build_orchestrator(_run(K=3), mock_gateway, runs_dir=tmp_path).run()

    assert {r.turn for r in _interactions(tmp_path)} == {1, 2, 3}


async def test_resume_rebuilds_neighbour_posts_including_meme_content(
    tmp_path, mock_gateway
):
    """CheckpointState stores only agent_snapshot, and StanceRecord has no
    content_type, so without rebuilding from interactions.jsonl a resumed run
    would render every past meme as ordinary text and silently change the
    stimulus mid-run.
    """
    memes = _meme_config(injection_schedule="fixed_turn", fixed_turns={1})

    # Turn 1 is all memes, then the run dies before turn 2.
    mock_gateway.generate.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        await build_orchestrator(
            _run(K=2, meme_injection=memes), mock_gateway, runs_dir=tmp_path
        ).run()
    assert all(r.content_type == "meme" for r in _interactions(tmp_path))

    # Resuming must recover those meme posts, not just agent stances.
    resumed = build_orchestrator(
        _run(K=2, meme_injection=memes), mock_gateway, runs_dir=tmp_path
    )
    rebuilt = resumed._rebuild_last_posts(1)  # the checkpoint is at turn 1
    assert rebuilt
    assert all(post.content_type == "meme" for post in rebuilt.values())
    assert all(post.meme_id is not None for post in rebuilt.values())

    # And the resumed turn-2 prompts must actually carry the meme framing.
    mock_gateway.generate.side_effect = None
    mock_gateway.generate.return_value = make_backend_response(stance=5.0)
    await resumed.run()

    turn_two_prompts = [c.args[0] for c in mock_gateway.generate.await_args_list]
    assert any("Caption:" in p for p in turn_two_prompts)


# --- 2f: stance_before must match what neighbour sampling used -----------


def test_current_stance_for_logging_matches_interaction_engine():
    """If these drift, stance_before in the log is not the value sampling
    actually used, and every regression fitted on that column is wrong."""
    from sandbox.interaction_engine import _current_stance
    from sandbox.models import StanceRecord

    history = [
        StanceRecord(turn=1, stance_value=6.0, reason_text="r", interaction_id="i1")
    ]
    for agent in (make_agent(initial_stance=4.0), make_agent(initial_stance=4.0, stance_history=history)):
        for turn in (1, 2, 3):
            assert _current_stance_for_logging(agent, turn) == _current_stance(agent, turn)


async def test_stance_before_tracks_the_previous_turns_stance_after(
    tmp_path, mock_gateway
):
    mock_gateway.generate.return_value = make_backend_response(stance=6.0)
    await build_orchestrator(_run(K=2), mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    by_agent: dict[str, list[Interaction]] = {}
    for row in rows:
        by_agent.setdefault(row.speaker_agent_id, []).append(row)

    for turns in by_agent.values():
        turns.sort(key=lambda r: r.turn)
        assert turns[1].stance_before == turns[0].stance_after


async def test_no_agent_sees_another_agents_same_turn_output(tmp_path, mock_gateway):
    """Fix F, tested against what it actually guarantees.

    Every agent's turn-t prompt must show neighbours as of the end of turn
    t-1. If step 6 were interleaved into dispatch rather than running after
    asyncio.gather, agents dispatched later in a turn would see earlier
    agents' turn-t stances, and the run would stop being reproducible.

    Each call returns a distinct stance, so turn-1 posts carry stances 1-5
    and turn-2 posts carry 6-10. A turn-2 prompt may therefore mention
    positions 1-5 only; a 6 or higher means someone read same-turn output.
    """
    counter = iter(range(1, 200))
    mock_gateway.generate.side_effect = lambda *a, **k: make_backend_response(
        stance=5.0, reason=f"turnmarker-{next(counter)}"
    )

    await build_orchestrator(_run(M=5, N=2, K=3), mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    markers_by_turn = {
        turn: {r.reason_text for r in rows if r.turn == turn and r.reason_text}
        for turn in (1, 2, 3)
    }

    prompts_by_call = [c.args[0] for c in mock_gateway.generate.await_args_list]
    # Turn t prompts are the t-th block of M calls (no retries occur here).
    for turn in (2, 3):
        block = prompts_by_call[(turn - 1) * 5 : turn * 5]
        for prompt in block:
            for later_marker in markers_by_turn[turn]:
                assert later_marker not in prompt, (
                    f"turn {turn} prompt contained same-turn output {later_marker!r}"
                )


# --- audit D9: a failed turn is excluded, not turned into a blank post ---


async def _run_with_one_failure(tmp_path, mock_gateway, **run_kw):
    """turn 1 succeeds for all; agent_0000 fails every attempt on turn 2."""
    mock_gateway.generate.side_effect = (
        [make_backend_response(stance=5.0, reason="TURN1-REAL")] * 3
        + [make_backend_response(text="junk")] * 3
        + [make_backend_response(stance=6.0, reason="TURN2-REAL")] * 2
        + [make_backend_response(stance=6.0, reason="TURN3-REAL")] * 3
    )
    run = _run(M=3, N=2, K=3, **run_kw)
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()
    return _interactions(tmp_path)


async def test_failed_turn_is_still_logged(tmp_path, mock_gateway):
    """Exclusion is from agent state, never from the log. A failed turn is
    data: it is what makes an uneven per-condition failure rate detectable."""
    rows = await _run_with_one_failure(tmp_path, mock_gateway)
    assert len(rows) == 3 * 3
    assert sum(1 for r in rows if r.api_call_status == "failed_logged_null") == 1


async def test_failed_turn_appends_no_stance_history_entry(tmp_path, mock_gateway):
    """full_design_doc.md Sec 3.9: no entry is appended, "so _current_stance()
    falls back to the agent's last known stance automatically"."""
    from sandbox.models import Agent

    rows = await _run_with_one_failure(tmp_path, mock_gateway)
    failed = next(r for r in rows if r.api_call_status == "failed_logged_null")

    agents = {
        a.agent_id: a
        for a in (
            Agent.model_validate_json(line)
            for line in (tmp_path / "orch_run" / "agents_final.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    history = agents[failed.speaker_agent_id].stance_history
    assert [r.turn for r in history] == [1, 3]  # turn 2 absent
    assert all(r.reason_text for r in history)  # no blank entry


async def test_neighbours_see_the_last_real_post_not_a_blank(tmp_path, mock_gateway):
    """The research-impacting half of D9.

    Applying a failed interaction made the blank row the agent's most recent
    post, so neighbours were shown an empty contribution. If one language
    condition fails more often, its agents circulate more blanks and receive
    less context, which is the asymmetry Fix A exists to prevent arriving via
    a path Fix A does not guard.
    """
    await _run_with_one_failure(tmp_path, mock_gateway)

    turn_three = [c.args[0] for c in mock_gateway.generate.await_args_list][-3:]
    saw_failed_agent = [p for p in turn_three if "agent_0000 (position" in p]
    assert saw_failed_agent, "expected some turn-3 prompt to show agent_0000"
    for prompt in saw_failed_agent:
        line = next(l for l in prompt.split("\n") if l.strip().startswith("agent_0000 ("))
        assert line.split(":", 1)[1].strip(), f"blank neighbour post: {line!r}"


async def test_failed_turn_does_not_consume_a_fix_a_memory_slot(tmp_path, mock_gateway):
    await _run_with_one_failure(tmp_path, mock_gateway)
    prompts = [c.args[0] for c in mock_gateway.generate.await_args_list][-3:]
    for prompt in prompts:
        if "What you said recently" in prompt:
            block = prompt.split("What you said recently:")[1].split("\n\n")[0]
            for line in block.strip().split("\n"):
                assert line.split(":", 1)[1].strip(), f"blank memory slot: {line!r}"


# --- audit D12: the prompt is built at most twice, not once per attempt -


async def test_prompt_is_built_at_most_twice_across_the_retry_budget(
    tmp_path, mock_gateway, monkeypatch
):
    """Rebuilding per attempt re-ran vision_fallback._load_image, re-reading
    the meme image from disk on every retry for no benefit."""
    from sandbox.prompt_builder import PromptBuilder

    calls = {"n": 0}
    original = PromptBuilder.build_discussion_prompt

    def counting(self, *a, **kw):
        calls["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(PromptBuilder, "build_discussion_prompt", counting)
    mock_gateway.generate.return_value = make_backend_response(text="no stance here")
    await build_orchestrator(_run(M=2, N=1, K=1), mock_gateway, runs_dir=tmp_path).run()

    # 2 agents x 3 attempts would be 6 builds; caching gives at most 2 each.
    assert calls["n"] <= 4, calls["n"]


# =========================================================================
# RQ1: the bank regime
#
# The generated-regime tests above are untouched. _run_turn branches on the
# regime, so both paths are exercised from the same orchestrator.
# =========================================================================

from tests.factories import make_logprob_response  # noqa: E402

FIXTURE_BANK = "data/bank/test_bank.csv"
FIXTURE_NEUTRAL_BANK = "data/bank/test_neutral.csv"


def _rendered_leans(run) -> set[str]:
    """The seven lean sentences, rendered against this run's anchor labels.

    Matching the whole sentence rather than a shared fragment: the midpoint
    sentence ("undecided between the view that X and the view that Y")
    contains "the view that" twice, so counting that fragment cannot
    distinguish one lean from two.
    """
    from sandbox.prompt_builder import LEAN_SENTENCES

    return {
        sentence.format(low=run.stance_low_label, high=run.stance_high_label)
        for sentence in LEAN_SENTENCES.values()
    }


def _lean_line(prompt: str) -> str:
    """The line directly after the topic line, where the lean always sits."""
    lines = [line for line in prompt.split("\n") if line.strip()]
    topic = next(
        i for i, line in enumerate(lines) if line.startswith("You are taking part")
    )
    return lines[topic + 1]


def _bank_run(**overrides) -> ExperimentRun:
    base = dict(
        M=8, N=3, K=3, run_id="bank_run", persona_pool_id="pool_20",
        neighbor_strategy="graph", graph_degree=4, graph_rewire_p=0.1,
        regime="bank", bank_path=FIXTURE_BANK,
        probe_question="Should the fare subsidy continue?",
        stance_low_label="the subsidy should end",
        stance_high_label="the subsidy should continue",
        # memory_rounds is left at its default of None -- full history --
        # so the tests exercise the production default. Tests that want a
        # window pass memory_rounds=n explicitly.
        # Required only when delivery="off", harmless otherwise, and set
        # unconditionally so a test can flip delivery with one override.
        neutral_bank_path=FIXTURE_NEUTRAL_BANK,
    )
    base.update(overrides)
    return make_run(**base)


def _bank_rows(tmp_path, run_id="bank_run") -> list[Interaction]:
    return _interactions(tmp_path, run_id)


# --- shape ---------------------------------------------------------------


async def test_a_bank_run_logs_one_row_per_agent_per_round_plus_round_zero(
    tmp_path, bank_gateway
):
    run = _bank_run()
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rows = _bank_rows(tmp_path)
    assert len(rows) == run.M * (run.K + 1)
    assert sorted({r.turn for r in rows}) == [0, 1, 2, 3]
    for turn in range(0, run.K + 1):
        assert len([r for r in rows if r.turn == turn]) == run.M


async def test_one_reply_and_one_probe_call_per_agent_round(tmp_path, bank_gateway):
    """Two calls per agent per round, plus one probe each for round 0."""
    run = _bank_run()
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    assert bank_gateway.generate.await_count == run.M * run.K
    assert bank_gateway.score_next_token.await_count == run.M * (run.K + 1)


async def test_every_bank_row_carries_the_bank_and_probe_columns(tmp_path, bank_gateway):
    await build_orchestrator(_bank_run(), bank_gateway, runs_dir=tmp_path).run()

    for row in _bank_rows(tmp_path):
        assert row.content_type == "bank_item"
        assert row.meme_id is None
        assert row.probe_mean is not None
        assert row.probe_dist is not None and len(row.probe_dist) == 7
        assert row.probe_mass is not None and row.side_conf is not None
        if row.turn > 0:
            assert row.posted_item_id is not None
            assert len(row.items_read) == 3
            assert row.reply_text


async def test_stance_before_and_after_are_probe_readings(tmp_path, bank_gateway):
    """A bank agent's stance is what the probe measured, round over round."""
    bank_gateway.score_next_token.side_effect = (
        [make_logprob_response(mean=4.0)] * 8      # round 0
        + [make_logprob_response(mean=6.0)] * 8    # round 1
        + [make_logprob_response(mean=2.0)] * 8    # round 2
        + [make_logprob_response(mean=5.0)] * 8    # round 3
    )
    await build_orchestrator(_bank_run(), bank_gateway, runs_dir=tmp_path).run()

    rows = _bank_rows(tmp_path)
    by_turn = {turn: [r for r in rows if r.turn == turn] for turn in range(4)}
    assert {r.stance_after for r in by_turn[0]} == {4.0}
    assert {r.stance_after for r in by_turn[1]} == {6.0}
    assert {r.stance_before for r in by_turn[1]} == {4.0}  # round 0's reading
    assert {r.stance_before for r in by_turn[2]} == {6.0}
    assert {r.stance_after for r in by_turn[3]} == {5.0}


async def test_round_zero_reads_nothing_and_posts_nothing(tmp_path, bank_gateway):
    """Not a turn: it measures the starting stance before any agent has read
    or written anything, so initial_stance is not the only baseline."""
    await build_orchestrator(_bank_run(), bank_gateway, runs_dir=tmp_path).run()

    for row in (r for r in _bank_rows(tmp_path) if r.turn == 0):
        assert row.items_read == []
        assert row.posted_item_id is None
        assert row.neighbor_agent_ids == []
        assert row.reply_text == ""
        assert row.probe_mean is not None


async def test_round_zero_makes_no_reply_calls(tmp_path, bank_gateway):
    await build_orchestrator(_bank_run(K=1), bank_gateway, runs_dir=tmp_path).run()
    # 8 replies for the single round, none for round 0.
    assert bank_gateway.generate.await_count == 8
    assert bank_gateway.score_next_token.await_count == 16


async def test_round_zero_is_checkpointed_before_round_one(tmp_path, bank_gateway):
    """So a crash in round 1 does not cost the starting measurement, and a
    resume cannot probe round 0 twice."""
    from sandbox.checkpoint_manager import CheckpointManager

    bank_gateway.generate.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        await build_orchestrator(_bank_run(), bank_gateway, runs_dir=tmp_path).run()

    assert CheckpointManager(tmp_path).load("bank_run").last_completed_turn == 0
    assert {r.turn for r in _bank_rows(tmp_path)} == {0}


# --- no numbers reach the model ------------------------------------------


async def test_no_prompt_a_bank_run_sends_contains_a_stance_line(tmp_path, bank_gateway):
    await build_orchestrator(_bank_run(), bank_gateway, runs_dir=tmp_path).run()

    for call in bank_gateway.generate.await_args_list:
        assert "STANCE" not in call.args[0]
    for call in bank_gateway.score_next_token.await_args_list:
        assert "STANCE" not in call.args[0]


async def test_no_prompt_a_bank_run_sends_contains_a_position_or_an_agent_id(
    tmp_path, bank_gateway
):
    await build_orchestrator(_bank_run(), bank_gateway, runs_dir=tmp_path).run()

    prompts = [c.args[0] for c in bank_gateway.generate.await_args_list] + [
        c.args[0] for c in bank_gateway.score_next_token.await_args_list
    ]
    assert prompts
    for prompt in prompts:
        assert "position" not in prompt.lower()
        assert "agent_" not in prompt


# --- sides and items ------------------------------------------------------


async def test_the_side_follows_the_last_probe_reading(tmp_path, bank_gateway):
    """pro above the midpoint, con below. Everyone reads 6.0 at round 0, so
    every round-1 post must be a pro item."""
    bank = __import__("sandbox.bank_manager", fromlist=["BankManager"]).BankManager(FIXTURE_BANK)
    bank_gateway.score_next_token.return_value = make_logprob_response(mean=6.0)
    await build_orchestrator(_bank_run(K=1), bank_gateway, runs_dir=tmp_path).run()

    posted = [r.posted_item_id for r in _bank_rows(tmp_path) if r.turn == 1]
    assert posted
    assert all(bank.side_of(item) == "pro" for item in posted)


async def test_a_low_reading_puts_every_agent_on_the_con_side(tmp_path, bank_gateway):
    bank = __import__("sandbox.bank_manager", fromlist=["BankManager"]).BankManager(FIXTURE_BANK)
    bank_gateway.score_next_token.return_value = make_logprob_response(mean=2.0)
    await build_orchestrator(_bank_run(K=1), bank_gateway, runs_dir=tmp_path).run()

    posted = [r.posted_item_id for r in _bank_rows(tmp_path) if r.turn == 1]
    assert all(bank.side_of(item) == "con" for item in posted)


async def test_a_reading_exactly_at_the_midpoint_is_broken_by_a_seeded_draw(
    tmp_path, bank_gateway
):
    """Never clamped to one side and never left undefined: a seeded draw, so
    the same seed splits the population the same way every time."""
    bank = __import__("sandbox.bank_manager", fromlist=["BankManager"]).BankManager(FIXTURE_BANK)
    bank_gateway.score_next_token.return_value = make_logprob_response(mean=4.0)

    for base in ("a", "b"):
        await build_orchestrator(
            _bank_run(M=20, N=3, K=1, graph_degree=4, seed=5),
            bank_gateway,
            runs_dir=tmp_path / base,
        ).run()

    def sides(where):
        return [
            (r.speaker_agent_id, bank.side_of(r.posted_item_id))
            for r in sorted(_bank_rows(where), key=lambda r: r.speaker_agent_id)
            if r.turn == 1
        ]

    assert sides(tmp_path / "a") == sides(tmp_path / "b")   # reproducible
    assert len({side for _, side in sides(tmp_path / "a")}) == 2  # genuinely split


async def test_an_agent_reads_its_neighbours_posted_items(tmp_path, bank_gateway):
    """items_read is exactly what the neighbours posted that round, in
    delivery order."""
    await build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path).run()

    rows = _bank_rows(tmp_path)
    for turn in (1, 2):
        posted = {r.speaker_agent_id: r.posted_item_id for r in rows if r.turn == turn}
        for row in (r for r in rows if r.turn == turn):
            assert row.items_read == [posted[n] for n in row.neighbor_agent_ids]


async def test_every_row_has_exactly_n_graph_neighbours(tmp_path, bank_gateway):
    run = _bank_run()
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    for row in (r for r in _bank_rows(tmp_path) if r.turn > 0):
        assert len(row.neighbor_agent_ids) == run.N
        assert row.speaker_agent_id not in row.neighbor_agent_ids


async def test_neighbours_come_from_the_graph(tmp_path, bank_gateway):
    run = _bank_run()
    orchestrator = build_orchestrator(run, bank_gateway, runs_dir=tmp_path)
    edges = orchestrator._interaction_engine.edges()
    await orchestrator.run()

    adjacency: dict[str, set[str]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    for row in (r for r in _bank_rows(tmp_path) if r.turn > 0):
        assert set(row.neighbor_agent_ids) <= adjacency[row.speaker_agent_id]


async def test_graph_edges_csv_is_written_once_at_run_start(tmp_path, bank_gateway):
    """Analysis needs the structure the run actually used; rebuilding it from
    the config afterwards would depend on this networkx version producing the
    same graph for the same seed."""
    run = _bank_run()
    orchestrator = build_orchestrator(run, bank_gateway, runs_dir=tmp_path)
    expected = orchestrator._interaction_engine.edges()
    await orchestrator.run()

    path = tmp_path / "bank_run" / "graph_edges.csv"
    lines = path.read_text().splitlines()
    assert lines[0] == "agent_a,agent_b"
    assert [tuple(line.split(",")) for line in lines[1:]] == expected
    assert len(lines) - 1 == run.M * run.graph_degree // 2


async def test_no_graph_edges_file_under_alpha_sampling(tmp_path, bank_gateway):
    await build_orchestrator(
        _bank_run(neighbor_strategy="alpha"), bank_gateway, runs_dir=tmp_path
    ).run()
    assert not (tmp_path / "bank_run" / "graph_edges.csv").exists()


# --- delivery: off, the E4 control ---------------------------------------


async def test_delivery_off_makes_the_same_calls_as_a_treatment_arm(
    tmp_path, bank_gateway
):
    """The control does the same amount of work; only what it reads differs.

    It used to make no reply calls at all, which made it a different activity
    from the treatment rather than the same activity without the manipulated
    factor.
    """
    run = _bank_run(delivery="off")
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    assert bank_gateway.generate.await_count == run.M * run.K
    assert bank_gateway.score_next_token.await_count == run.M * (run.K + 1)


async def test_delivery_off_rows_read_n_prefixed_neutral_ids(tmp_path, bank_gateway):
    """A control row reads N side-free messages, not one and not none.

    N, because the treatment reads N: an arm that read one post a round
    against the treatment's three differs in how much was read as well as in
    whether it argued anything.
    """
    from sandbox.bank_manager import NEUTRAL_PREFIX

    run = _bank_run(delivery="off")
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rows = _bank_rows(tmp_path)
    assert len(rows) == run.M * (run.K + 1)
    for row in rows:
        assert row.api_call_status == "success"
        assert row.probe_mean is not None
        if row.turn == 0:
            assert row.items_read == []
            assert row.shadow_items == []
            assert row.reply_text == ""
        else:
            assert len(row.items_read) == run.N
            assert all(i.startswith(NEUTRAL_PREFIX) for i in row.items_read)
            # One bank item per message, in the same order, so a control row
            # can be lined up against the treatment row it is a baseline for.
            assert len(row.shadow_items) == run.N
            # posted_item_id is still the bank item this agent's side WOULD
            # have been represented by, so the control's draws stay
            # comparable to the treatment's row by row.
            assert row.posted_item_id is not None
            assert row.reply_text


async def test_delivery_off_delivers_neutral_messages_and_no_bank_text(
    tmp_path, bank_gateway
):
    """Every post the control reads comes from the neutral bank, and no
    argument from the bank reaches any prompt in either phase."""
    from sandbox.bank_manager import BankManager, NeutralBank

    run = _bank_run(delivery="off", K=2)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    replies = [c.args[0] for c in bank_gateway.generate.await_args_list]
    probes = [c.args[0] for c in bank_gateway.score_next_token.await_args_list]
    assert replies

    neutral = NeutralBank(FIXTURE_NEUTRAL_BANK)
    neutral_texts = [
        neutral.text(nid, run.language_condition) for nid in neutral.ids()
    ]
    for prompt in replies:
        assert prompt.count("A user posted:") == run.N
        shown = sum(prompt.count(f"A user posted: {t}") for t in neutral_texts)
        assert shown == run.N

    bank = BankManager(FIXTURE_BANK)
    bank_texts = [
        bank.text(item_id, run.language_condition)
        for side in ("pro", "con")
        for item_id in bank.items_for_side(side)
    ]
    for prompt in replies + probes:
        for text in bank_texts:
            assert text not in prompt


async def test_delivery_off_remembers_its_rounds(tmp_path, bank_gateway):
    """The control's history has to change round over round, or the arm
    cannot show drift even in principle."""
    run = _bank_run(delivery="off", K=3)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    replies = [c.args[0] for c in bank_gateway.generate.await_args_list]
    round_one, round_three = replies[: run.M], replies[2 * run.M : 3 * run.M]
    assert all("Earlier in this discussion" not in p for p in round_one)
    assert all(p.count("You replied:") == 2 for p in round_three)
    # One "You read:" line per post, so two remembered rounds of N posts each
    # give 2 x N lines -- the same shape a treatment arm's memory has.
    assert all(p.count("You read:") == 2 * run.N for p in round_three)


async def test_delivery_off_still_resolves_neighbours_and_posts(tmp_path, bank_gateway):
    """Sides, items and neighbours are still resolved, so every seeded stream
    consumes identically to a delivery-on arm. The control then differs from
    the treatment in the persuasive content of what agents READ and in
    nothing else.

    The probe is driven to a constant reading in both arms (bank_gateway's
    default), which is what makes the sides match: in a real run the readings
    diverge as soon as the arms do, and that divergence is the result.
    """
    on, off = tmp_path / "on", tmp_path / "off"
    await build_orchestrator(_bank_run(seed=3), bank_gateway, runs_dir=on).run()
    await build_orchestrator(
        _bank_run(seed=3, delivery="off"), bank_gateway, runs_dir=off
    ).run()

    def signature(where):
        return [
            (r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids), r.posted_item_id)
            for r in _bank_rows(where)
        ]

    assert signature(on) == signature(off)


async def test_delivery_off_probes_still_show_no_feed(tmp_path, bank_gateway):
    """The probe measures; it never delivers, in either mode. Its memory
    block does grow, because the control now has rounds to remember."""
    await build_orchestrator(
        _bank_run(delivery="off"), bank_gateway, runs_dir=tmp_path
    ).run()

    probes = [c.args[0] for c in bank_gateway.score_next_token.await_args_list]
    for prompt in probes:
        assert "Posts in your feed" not in prompt
        assert "A user posted" not in prompt
    assert any("Earlier in this discussion" in p for p in probes)


async def test_delivery_off_probes_all_carry_the_lean(tmp_path, bank_gateway):
    """Every probe is the same instrument, round 0 included.

    This arm is where it matters most: nothing else about a delivery-off
    probe changes between rounds, so a lean shown only at round 0 was the one
    difference between the starting reading and every later one.
    """
    run = _bank_run(delivery="off", K=2)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    prompts = [c.args[0] for c in bank_gateway.score_next_token.await_args_list]
    assert len(prompts) == run.M * (run.K + 1)
    for prompt in prompts:
        assert _lean_line(prompt) in _rendered_leans(run), prompt


# --- memory ---------------------------------------------------------------


async def test_memory_accumulates_reads_and_replies_across_rounds(tmp_path, bank_gateway):
    bank_gateway.generate.side_effect = [
        make_backend_response(text=f"reply number {i}") for i in range(100)
    ]
    await build_orchestrator(_bank_run(K=3), bank_gateway, runs_dir=tmp_path).run()

    prompts = [c.args[0] for c in bank_gateway.generate.await_args_list]
    round_one = prompts[:8]
    round_three = prompts[16:24]

    assert all("Earlier in this discussion" not in p for p in round_one)
    assert all("Earlier in this discussion" in p for p in round_three)
    assert all(p.count("You replied:") == 2 for p in round_three)


async def test_full_history_reaches_round_k_oldest_first(tmp_path, bank_gateway):
    """The default: every earlier round's posts and replies are in round K's
    reply prompt AND in its probe prompt, in the order they happened."""
    # Zero-padded and terminated, so no reply text is a substring of
    # another: "reply 1" would also match "reply 17" and the ordering
    # assertions below would pass on the wrong line.
    bank_gateway.generate.side_effect = [
        make_backend_response(text=f"reply number {i:04d} ends here") for i in range(1000)
    ]
    run = _bank_run(K=5)
    assert run.memory_rounds is None
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rows = {(r.turn, r.speaker_agent_id): r for r in _bank_rows(tmp_path)}
    replies = [c.args[0] for c in bank_gateway.generate.await_args_list]
    probes = [c.args[0] for c in bank_gateway.score_next_token.await_args_list]
    agents = sorted({a for _, a in rows})

    # Round 5's prompts, one per agent, in the order the orchestrator
    # dispatched them (sorted agent id).
    last_replies = replies[(run.K - 1) * run.M : run.K * run.M]
    last_probes = probes[run.K * run.M :]

    for agent_id, reply_prompt, probe_prompt in zip(
        agents, last_replies, last_probes, strict=True
    ):
        earlier = [rows[(t, agent_id)].reply_text for t in range(1, run.K)]
        assert len(earlier) == 4
        for prompt in (reply_prompt, probe_prompt):
            assert prompt.count("You replied:") == len(earlier) + (
                1 if prompt is probe_prompt else 0
            )
            positions = [prompt.index(text) for text in earlier]
            assert positions == sorted(positions), prompt
        # The probe additionally carries this round's own reply, last.
        assert probe_prompt.index(rows[(run.K, agent_id)].reply_text) > max(
            probe_prompt.index(text) for text in earlier
        )


async def test_a_window_still_drops_everything_older(tmp_path, bank_gateway):
    """memory_rounds=2 keeps the option available: only the last two rounds
    reach the prompt."""
    bank_gateway.generate.side_effect = [
        make_backend_response(text=f"reply number {i:04d} ends here") for i in range(1000)
    ]
    run = _bank_run(K=5, memory_rounds=2)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rows = {(r.turn, r.speaker_agent_id): r for r in _bank_rows(tmp_path)}
    replies = [c.args[0] for c in bank_gateway.generate.await_args_list]
    agents = sorted({a for _, a in rows})

    for agent_id, prompt in zip(agents, replies[(run.K - 1) * run.M :], strict=True):
        assert prompt.count("You replied:") == 2
        assert rows[(4, agent_id)].reply_text in prompt
        assert rows[(3, agent_id)].reply_text in prompt
        for turn in (1, 2):
            assert rows[(turn, agent_id)].reply_text not in prompt


# --- the prompt-length ceiling -------------------------------------------


async def test_a_prompt_over_the_ceiling_ends_the_run(tmp_path, bank_gateway):
    """Never truncated silently. The agents that overflow first are the ones
    whose language tokenizes least efficiently, so absorbing this per agent
    would drop one arm's agents and keep another's."""
    from sandbox.simulation_orchestrator import PromptBudgetExceeded

    bank_gateway.generate.return_value = make_backend_response(prompt_tokens=900)
    bank_gateway.score_next_token.return_value = make_logprob_response(
        mean=5.0, prompt_tokens=100
    )
    with pytest.raises(PromptBudgetExceeded) as caught:
        await build_orchestrator(
            _bank_run(K=2, max_prompt_tokens=500), bank_gateway, runs_dir=tmp_path
        ).run()

    error = caught.value
    assert error.call == "reply"
    assert error.turn == 1
    assert error.tokens == 900 and error.ceiling == 500
    assert error.agent_id in str(error)
    assert "round 1" in str(error)
    assert "truncated" in str(error)


async def test_the_ceiling_catches_a_long_probe_prompt_too(tmp_path, bank_gateway):
    from sandbox.simulation_orchestrator import PromptBudgetExceeded

    bank_gateway.score_next_token.return_value = make_logprob_response(
        mean=5.0, prompt_tokens=700
    )
    with pytest.raises(PromptBudgetExceeded) as caught:
        await build_orchestrator(
            _bank_run(K=2, max_prompt_tokens=600), bank_gateway, runs_dir=tmp_path
        ).run()

    # Round 0 is the first probe of the run, so that is where it surfaces.
    assert caught.value.call == "probe"
    assert caught.value.turn == 0


async def test_a_run_under_the_ceiling_completes(tmp_path, bank_gateway):
    """The check is on the provider's reported count, and nothing else about
    the run changes when a ceiling is set."""
    run = _bank_run(K=2, max_prompt_tokens=10_000)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()
    assert len(_bank_rows(tmp_path)) == run.M * (run.K + 1)


async def test_no_ceiling_means_no_ceiling(tmp_path, bank_gateway):
    bank_gateway.generate.return_value = make_backend_response(prompt_tokens=10**6)
    run = _bank_run(K=1)
    assert run.max_prompt_tokens is None
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()
    assert len(_bank_rows(tmp_path)) == run.M * (run.K + 1)


async def test_a_context_window_error_ends_the_run(tmp_path, bank_gateway):
    """A per-agent failure here would quietly drop the agents with the
    longest histories, which correlates with the language arm."""
    from sandbox.model_gateway import ContextLengthExceeded

    bank_gateway.generate.side_effect = ContextLengthExceeded("8192 token limit")
    with pytest.raises(ContextLengthExceeded):
        await build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path).run()


# --- the two prompt lengths ----------------------------------------------


async def test_the_two_prompt_lengths_are_logged_separately(tmp_path, bank_gateway):
    """RQ1 reports prompt length per language, and a reply prompt and a probe
    prompt grow for different reasons, so the sum cannot answer the question
    on its own."""
    bank_gateway.generate.return_value = make_backend_response(
        prompt_tokens=100, completion_tokens=20
    )
    bank_gateway.score_next_token.return_value = make_logprob_response(
        mean=5.0, prompt_tokens=70, completion_tokens=1
    )
    await build_orchestrator(_bank_run(K=1), bank_gateway, runs_dir=tmp_path).run()

    for row in _bank_rows(tmp_path):
        if row.turn == 0:
            assert (row.reply_prompt_tokens, row.probe_prompt_tokens) == (0, 70)
        else:
            assert (row.reply_prompt_tokens, row.probe_prompt_tokens) == (100, 70)
        # The summed column still means what every existing analysis reads.
        assert row.prompt_token_count == (
            row.reply_prompt_tokens + row.probe_prompt_tokens
        )


async def test_memory_is_capped_at_memory_rounds(tmp_path, bank_gateway):
    await build_orchestrator(_bank_run(K=5, memory_rounds=2), bank_gateway, runs_dir=tmp_path).run()

    prompts = [c.args[0] for c in bank_gateway.generate.await_args_list]
    assert all(p.count("You replied:") <= 2 for p in prompts)
    assert any(p.count("You replied:") == 2 for p in prompts)


async def test_the_probe_sees_this_rounds_reply(tmp_path, bank_gateway):
    """Phase B runs after Phase A precisely so the probe reads the reply the
    agent just wrote. That is the agent's own output, read by nobody else, so
    it is not a Fix F violation."""
    bank_gateway.generate.return_value = make_backend_response(text="MARKER-THIS-ROUND")
    await build_orchestrator(_bank_run(K=1), bank_gateway, runs_dir=tmp_path).run()

    round_one_probes = [c.args[0] for c in bank_gateway.score_next_token.await_args_list][8:]
    assert len(round_one_probes) == 8
    assert all("MARKER-THIS-ROUND" in p for p in round_one_probes)


async def test_the_probe_is_never_written_into_memory(tmp_path, bank_gateway):
    """The whole reason the probe is a separate call: measuring an agent must
    not change what it is shown next round."""
    bank_gateway.score_next_token.return_value = make_logprob_response(
        top_logprobs=[("5", -0.01), ("PROBE-ONLY-TOKEN", -9.0)]
    )
    await build_orchestrator(_bank_run(K=3), bank_gateway, runs_dir=tmp_path).run()

    for call in bank_gateway.generate.await_args_list:
        assert "PROBE-ONLY-TOKEN" not in call.args[0]
        assert "Answer with a single digit" not in call.args[0]
        assert "Should the fare subsidy continue?" not in call.args[0]


# --- prompts.jsonl: every call, verbatim ---------------------------------


def _prompt_rows(tmp_path, run_id="bank_run"):
    from sandbox.models import PromptRecord

    path = tmp_path / run_id / "prompts.jsonl"
    return [
        PromptRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def test_every_call_writes_one_prompt_line(tmp_path, bank_gateway):
    """M x K reply lines and M x (K+1) probe lines, with the prompt text the
    model was actually handed.

    The bank regime's central claim is that no number reaches an agent, and
    without this file that claim is checkable only inside the test suite,
    never against the run that produced a dataset.
    """
    run = _bank_run(M=4, N=1, K=2, graph_degree=2)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rows = _prompt_rows(tmp_path)
    replies = [r for r in rows if r.call == "reply"]
    probes = [r for r in rows if r.call == "probe"]
    assert len(replies) == run.M * run.K == 8
    assert len(probes) == run.M * (run.K + 1) == 12

    # Keyed on (turn, agent_id, call), one line each in an uninterrupted run.
    keys = [(r.turn, r.agent_id, r.call) for r in rows]
    assert len(set(keys)) == len(keys)
    assert {r.run_id for r in rows} == {run.run_id}
    assert all(r.prompt_text for r in rows)
    assert all("A user posted:" in r.prompt_text for r in replies)
    assert all(run.probe_question in r.prompt_text for r in probes)
    # Round 0 is probed and never replies.
    assert {r.turn for r in probes} == {0, 1, 2}
    assert {r.turn for r in replies} == {1, 2}


async def test_each_logged_probe_reproduces_its_logged_probe_mean(tmp_path, bank_gateway):
    """The raw distribution, before any scoring: a logged probe_mean can be
    recomputed from the logprobs the line carries."""
    from sandbox.probe_scorer import score as score_probe

    bank_gateway.score_next_token.side_effect = [
        make_logprob_response(probabilities={4: 0.3, 5: 0.7}),
        make_logprob_response(mean=2.0),
        make_logprob_response(probabilities={6: 0.5, 7: 0.5}),
        make_logprob_response(mean=7.0),
    ] + [make_logprob_response(mean=5.0)] * 100

    run = _bank_run(M=4, N=1, K=2, graph_degree=2)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    logged = {
        (r.turn, r.speaker_agent_id): r.probe_mean for r in _bank_rows(tmp_path)
    }
    probes = [r for r in _prompt_rows(tmp_path) if r.call == "probe"]
    assert len(probes) == 12

    for line in probes:
        assert line.top_logprobs, line
        recomputed = score_probe(line.top_logprobs, run.probe_mass_floor)
        assert recomputed.mean == pytest.approx(logged[(line.turn, line.agent_id)])
    # And the four scripted reads really did differ, or the check above would
    # pass against a constant.
    assert len({round(m, 6) for m in logged.values()}) > 1


async def test_a_reply_line_carries_no_logprobs(tmp_path, bank_gateway):
    """None means "this was not a probe", never "the probe came back empty"."""
    await build_orchestrator(
        _bank_run(M=4, N=1, K=1, graph_degree=2), bank_gateway, runs_dir=tmp_path
    ).run()

    for line in _prompt_rows(tmp_path):
        assert (line.top_logprobs is None) == (line.call == "reply")


async def test_the_response_is_logged_next_to_the_prompt(tmp_path, bank_gateway):
    bank_gateway.generate.return_value = make_backend_response(text="MARKER-REPLY")
    await build_orchestrator(
        _bank_run(M=4, N=1, K=1, graph_degree=2), bank_gateway, runs_dir=tmp_path
    ).run()

    replies = [r for r in _prompt_rows(tmp_path) if r.call == "reply"]
    assert all(r.response_text == "MARKER-REPLY" for r in replies)


async def test_a_failed_call_still_logs_its_prompt(tmp_path, bank_gateway):
    """A failed call happened and was paid for, and it is the one most worth
    being able to read. The response is empty rather than absent."""
    bank_gateway.generate.side_effect = FatalGatewayError("bad key")
    run = _bank_run(M=4, N=1, K=1, graph_degree=2)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    replies = [r for r in _prompt_rows(tmp_path) if r.call == "reply"]
    assert len(replies) == run.M
    assert all(r.prompt_text and r.response_text == "" for r in replies)


async def test_an_off_scale_probe_logs_the_distribution_it_answered_with(
    tmp_path, bank_gateway
):
    """ProbeMassTooLow is diagnosable only from the logprobs behind it, and
    the interaction row leaves every probe column null by design (Fix I)."""
    bank_gateway.score_next_token.side_effect = [
        make_logprob_response(mean=5.0)
    ] * 4 + [make_logprob_response(mean=5.0, off_scale=0.9)] * 4

    await build_orchestrator(
        _bank_run(M=4, N=1, K=1, graph_degree=2), bank_gateway, runs_dir=tmp_path
    ).run()

    round_one = [r for r in _prompt_rows(tmp_path) if r.call == "probe" and r.turn == 1]
    assert len(round_one) == 4
    for line in round_one:
        assert line.top_logprobs
        assert any(token.strip() == "I" for token, _ in line.top_logprobs)
    assert all(r.probe_mean is None for r in _bank_rows(tmp_path) if r.turn == 1)


async def test_log_prompts_off_writes_no_file(tmp_path, bank_gateway):
    run = _bank_run(M=4, N=1, K=1, graph_degree=2, log_prompts=False)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    assert not (tmp_path / "bank_run" / "prompts.jsonl").exists()
    assert len(_bank_rows(tmp_path)) == run.M * (run.K + 1)


async def test_the_generated_regime_logs_no_prompts_by_default(tmp_path, mock_gateway):
    """log_prompts defaults by regime, and the generated regime's default is
    off: nothing about it turns on an audit of its prompts, and switching it
    on for existing configs would change what a rerun writes to disk."""
    run = _run(K=1)
    assert run.log_prompts is False
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()
    assert not (tmp_path / run.run_id / "prompts.jsonl").exists()


async def test_the_generated_regime_can_opt_in(tmp_path, mock_gateway):
    """The field means the same thing in both regimes; only its default
    differs."""
    run = _run(M=3, N=1, K=2, log_prompts=True)
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    rows = _prompt_rows(tmp_path, run.run_id)
    assert len(rows) == run.M * run.K
    assert {r.call for r in rows} == {"reply"}
    assert all("STANCE:" in r.prompt_text for r in rows)   # the old contract
    assert all(r.top_logprobs is None for r in rows)


async def test_a_generated_regime_format_retry_logs_both_attempts(tmp_path, mock_gateway):
    """A retry is a second call, and a prompt log that showed one line would
    hide the attempt whose reply could not be parsed."""
    run = _run(M=2, N=1, K=1, log_prompts=True)
    mock_gateway.generate.side_effect = [
        make_backend_response(text="no stance here"),   # agent 0, attempt 1
        make_backend_response(stance=5.0),              # agent 1
        make_backend_response(stance=6.0),              # agent 0, attempt 2
    ]
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    rows = _prompt_rows(tmp_path, run.run_id)
    assert len(rows) == 3
    assert sum(r.response_text == "no stance here" for r in rows) == 1
    # Last write wins on (turn, agent, call): the attempt that was used.
    latest = {}
    for row in rows:
        latest[(row.turn, row.agent_id, row.call)] = row
    assert len(latest) == 2
    assert all("no stance here" != r.response_text for r in latest.values())


async def test_a_resumed_run_may_duplicate_prompt_lines(tmp_path, bank_gateway):
    """Documented, not prevented: the resumed pass re-runs the round its
    checkpoint does not cover, so that round's prompts are appended again.
    Keyed on (turn, agent_id, call) with the last write winning, the later
    copy is the call whose result the run actually used."""
    run = dict(M=4, N=1, K=3, graph_degree=2)
    # Round 3's replies are logged, then its probes die: the one interruption
    # that leaves prompt lines for a round the checkpoint does not cover.
    # (A round that dies in Phase A leaves none, since nothing was called.)
    bank_gateway.generate.return_value = make_backend_response(text="a steady reply")
    bank_gateway.score_next_token.side_effect = [
        make_logprob_response(mean=5.0)
    ] * 12 + [RuntimeError("boom")] * 50
    with pytest.raises(RuntimeError):
        await build_orchestrator(_bank_run(**run), bank_gateway, runs_dir=tmp_path).run()
    assert CheckpointManager(tmp_path).load("bank_run").last_completed_turn == 2

    bank_gateway.score_next_token.side_effect = None
    bank_gateway.score_next_token.return_value = make_logprob_response(mean=5.0)
    bank_gateway.generate.return_value = make_backend_response(text="a resumed reply")
    await build_orchestrator(_bank_run(**run), bank_gateway, runs_dir=tmp_path).run()

    rows = _prompt_rows(tmp_path)
    keys = [(r.turn, r.agent_id, r.call) for r in rows]
    assert len(keys) > len(set(keys))  # duplicates exist

    latest = {}
    for row in rows:  # file order: last write wins
        latest[(row.turn, row.agent_id, row.call)] = row
    replies = [r for r in latest.values() if r.call == "reply"]
    probes = [r for r in latest.values() if r.call == "probe"]
    assert len(replies) == 4 * 3
    assert len(probes) == 4 * 4
    # The surviving copy of the re-run round is the resumed call.
    assert all(
        r.response_text == "a resumed reply" for r in replies if r.turn == 3
    )


# --- Fix F ---------------------------------------------------------------


async def test_no_agent_reads_another_agents_same_round_reply(tmp_path, bank_gateway):
    """Fix F, in the bank regime. Posts are bank items resolved before
    dispatch, so a same-round reply cannot reach another agent's feed -- and
    this checks the replies themselves never appear in anyone else's prompt."""
    counter = iter(range(1, 500))
    bank_gateway.generate.side_effect = lambda *a, **k: make_backend_response(
        text=f"ROUNDMARKER-{next(counter)}"
    )
    await build_orchestrator(_bank_run(K=3), bank_gateway, runs_dir=tmp_path).run()

    rows = _bank_rows(tmp_path)
    for turn in (1, 2, 3):
        this_round = {r.reply_text for r in rows if r.turn == turn}
        block = [c.args[0] for c in bank_gateway.generate.await_args_list][
            (turn - 1) * 8 : turn * 8
        ]
        for prompt in block:
            for marker in this_round:
                assert marker not in prompt, f"round {turn} prompt carried {marker!r}"


async def test_items_are_resolved_before_any_dispatch(tmp_path, bank_gateway):
    """An agent's own posted item must not depend on anyone's turn-t output.
    Driven by making every probe differ: if items were resolved mid-dispatch,
    later agents' sides would reflect earlier agents' new readings."""
    readings = iter([make_logprob_response(mean=m) for m in [4] * 8 + [6, 2] * 50])
    bank_gateway.score_next_token.side_effect = lambda *a, **k: next(readings)

    first, second = tmp_path / "a", tmp_path / "b"
    for where in (first, second):
        readings = iter([make_logprob_response(mean=m) for m in [4] * 8 + [6, 2] * 50])
        bank_gateway.score_next_token.side_effect = lambda *a, **k: next(readings)
        await build_orchestrator(_bank_run(seed=17), bank_gateway, runs_dir=where).run()

    def signature(where):
        return [(r.turn, r.speaker_agent_id, r.posted_item_id) for r in _bank_rows(where)]

    assert signature(first) == signature(second)


# --- the error taxonomy ---------------------------------------------------


async def test_a_reply_failure_is_logged_and_not_applied(tmp_path, bank_gateway):
    """No invented reply, and no stance record: Fix I, unchanged."""
    bank_gateway.generate.side_effect = FatalGatewayError("bad key")
    run = _bank_run(K=1)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rows = _bank_rows(tmp_path)
    round_one = [r for r in rows if r.turn == 1]
    assert len(round_one) == run.M
    for row in round_one:
        assert row.api_call_status == "failed_logged_null"
        assert row.reply_text == ""
        assert row.probe_mean is None
        assert row.stance_after == row.stance_before


async def test_a_failed_reply_spends_no_probe_call(tmp_path, bank_gateway):
    """The agent will not be applied this round whatever the probe says, so
    probing it would spend a call on a reading nothing can consume."""
    bank_gateway.generate.side_effect = FatalGatewayError("bad key")
    run = _bank_run(K=1)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    assert bank_gateway.score_next_token.await_count == run.M  # round 0 only


async def test_a_probe_below_the_mass_floor_is_logged_and_not_applied(
    tmp_path, bank_gateway
):
    """ProbeMassTooLow is a per-agent event, not a reason to end the run: the
    model answered something other than a scale point."""
    bank_gateway.score_next_token.side_effect = (
        [make_logprob_response(mean=5.0)] * 8                    # round 0 succeeds
        + [make_logprob_response(mean=5.0, off_scale=0.9)] * 8   # round 1 is off-scale
    )
    await build_orchestrator(_bank_run(K=1), bank_gateway, runs_dir=tmp_path).run()

    round_one = [r for r in _bank_rows(tmp_path) if r.turn == 1]
    assert len(round_one) == 8
    for row in round_one:
        assert row.api_call_status == "failed_logged_null"
        assert row.probe_mean is None and row.probe_dist is None
        assert row.stance_after == row.stance_before
        assert row.reply_text  # the reply happened; only the measurement failed


async def test_a_failed_probe_leaves_no_stance_record_or_memory_slot(
    tmp_path, bank_gateway
):
    from sandbox.models import Agent

    bank_gateway.score_next_token.side_effect = (
        [make_logprob_response(mean=5.0)] * 8                   # round 0
        + [make_logprob_response(mean=5.0, off_scale=0.9)] * 8  # round 1 fails
        + [make_logprob_response(mean=6.0)] * 8                 # round 2
    )
    await build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path).run()

    agents = [
        Agent.model_validate_json(line)
        for line in (tmp_path / "bank_run" / "agents_final.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for agent in agents:
        assert [r.turn for r in agent.stance_history] == [0, 2]  # round 1 absent

    # Round 2's prompts remember round 2's own feed only, never round 1's.
    round_two_prompts = [c.args[0] for c in bank_gateway.generate.await_args_list][8:]
    assert all(p.count("You replied:") == 0 for p in round_two_prompts)


async def test_a_fatal_probe_error_is_absorbed_per_agent(tmp_path, bank_gateway):
    bank_gateway.score_next_token.side_effect = FatalGatewayError("no logprobs")
    await build_orchestrator(_bank_run(K=1), bank_gateway, runs_dir=tmp_path).run()

    rows = _bank_rows(tmp_path)
    assert len(rows) == 16
    assert all(r.api_call_status == "failed_logged_null" for r in rows)


async def test_an_unexpected_exception_still_aborts_a_bank_run(tmp_path, bank_gateway):
    """Only the designed failure modes get soft handling. gather(...,
    return_exceptions=True) flattens real bugs into return values, so _unwrap
    has to re-raise them."""
    bank_gateway.generate.side_effect = RuntimeError("nobody planned for this")
    with pytest.raises(RuntimeError, match="nobody planned for"):
        await build_orchestrator(_bank_run(), bank_gateway, runs_dir=tmp_path).run()


async def test_a_cost_ceiling_still_aborts_a_bank_run(tmp_path, bank_gateway):
    bank_gateway.score_next_token.side_effect = CostCeilingExceeded(10.0, 5.0)
    with pytest.raises(CostCeilingExceeded):
        await build_orchestrator(_bank_run(), bank_gateway, runs_dir=tmp_path).run()


async def test_a_probe_failure_at_round_zero_does_not_abort_the_run(
    tmp_path, bank_gateway
):
    """An agent with no round-0 reading falls back to initial_stance for its
    first side assignment -- a configured value, not a fabricated one -- and
    is probed again next round like everyone else."""
    bank_gateway.score_next_token.side_effect = [
        make_logprob_response(mean=5.0, off_scale=0.9)
    ] + [make_logprob_response(mean=5.0)] * 100

    await build_orchestrator(_bank_run(K=1), bank_gateway, runs_dir=tmp_path).run()

    rows = _bank_rows(tmp_path)
    failed = [r for r in rows if r.api_call_status == "failed_logged_null"]
    assert len(failed) == 1 and failed[0].turn == 0
    assert len([r for r in rows if r.turn == 1]) == 8


# --- token accounting -----------------------------------------------------


async def test_token_counts_are_summed_over_both_calls(tmp_path, bank_gateway):
    """A round is two calls, and every analysis reads these columns as "what
    this agent-round cost"."""
    bank_gateway.generate.return_value = make_backend_response(
        prompt_tokens=100, completion_tokens=20, latency_ms=50
    )
    bank_gateway.score_next_token.return_value = make_logprob_response(
        mean=5.0, prompt_tokens=70, completion_tokens=1, latency_ms=10
    )
    await build_orchestrator(_bank_run(K=1), bank_gateway, runs_dir=tmp_path).run()

    rows = _bank_rows(tmp_path)
    for row in (r for r in rows if r.turn == 1):
        assert row.prompt_token_count == 170
        assert row.completion_token_count == 21
        assert row.latency_ms == 60
    for row in (r for r in rows if r.turn == 0):
        assert row.prompt_token_count == 70  # probe only
        assert row.completion_token_count == 1


# --- wiring ---------------------------------------------------------------


def test_build_orchestrator_selects_the_graph_strategy_and_the_bank():
    from sandbox.bank_manager import BankManager
    from sandbox.interaction_engine import AlphaSampling, GraphSampling

    graph = build_orchestrator(_bank_run(), mock_gateway_stub())
    assert isinstance(graph._interaction_engine, GraphSampling)
    assert isinstance(graph._bank, BankManager)

    alpha = build_orchestrator(_run(), mock_gateway_stub())
    assert isinstance(alpha._interaction_engine, AlphaSampling)
    assert alpha._bank is None


def mock_gateway_stub():
    from unittest.mock import create_autospec

    from sandbox.model_gateway import ModelGateway

    gateway = create_autospec(ModelGateway, instance=True)
    gateway.model_backend_id = "gpt-4o"
    gateway.supports_vision = True
    return gateway


def test_the_graph_is_built_from_the_graph_seed():
    """So the four language arms of a trial share a structure."""
    english = build_orchestrator(_bank_run(language_condition="english"), mock_gateway_stub())
    hindi = build_orchestrator(_bank_run(language_condition="hindi"), mock_gateway_stub())
    other_seed = build_orchestrator(_bank_run(seed=999), mock_gateway_stub())

    assert english._interaction_engine.edges() == hindi._interaction_engine.edges()
    assert english._interaction_engine.edges() != other_seed._interaction_engine.edges()


def test_a_bank_orchestrator_without_a_bank_raises():
    """Rather than failing on the first pick, with the run directory and the
    config already written."""
    from sandbox.simulation_orchestrator import SimulationOrchestrator

    run = _bank_run()
    with pytest.raises(ValueError, match="requires a BankManager"):
        SimulationOrchestrator(
            run=run,
            agent_manager=None, interaction_engine=None, meme_pool=None,
            gateway=None, logger=None, checkpoint_mgr=None, prompt_builder=None,
            seeds=None, bank=None,
        )


def test_the_graph_nodes_match_the_population_ids():
    """agent_id_for is shared by the wiring and by initialize_population, so a
    drift between them would build a graph whose nodes are not the run's
    agents.

    graph_degree=6 rather than 4: at degree 4 this seed's graph leaves a node
    with 2 neighbours, which GraphSampling now rejects at construction rather
    than on round 1 (see test_interaction_engine). The node ids are what this
    test is about.
    """
    orchestrator = build_orchestrator(_bank_run(M=12, graph_degree=6), mock_gateway_stub())
    nodes = {a for edge in orchestrator._interaction_engine.edges() for a in edge}
    orchestrator._agent_manager.initialize_population()
    population = {a.agent_id for a in orchestrator._agent_manager.all_agents()}
    assert population == nodes


# --- resume ---------------------------------------------------------------


def _crash_in_round(bank_gateway, round_number: int, agents: int = 8):
    """Lets `round_number - 1` rounds of replies succeed, then fails them all.

    Failing every agent in the round matters: the orchestrator writes rows
    only after asyncio.gather returns, and _unwrap re-raises before the write
    loop, so a round in which everyone fails leaves no partial rows behind. A
    round in which only some fail would leave rows with no checkpoint, which
    is the known re-logging gap and a different situation from this one.
    """
    bank_gateway.generate.side_effect = [
        make_backend_response(text=f"reply in round {r}")
        for r in range(1, round_number)
        for _ in range(agents)
    ] + [RuntimeError("boom")] * (agents * 10)


def _crash_in_round_replying(bank_gateway, round_number: int, text: str, agents: int = 8):
    """_crash_in_round, but every pre-crash reply is the same text.

    The row-for-row comparisons below include reply_text, so a crashed run
    whose early rounds said "reply in round 2" cannot be compared against a
    clean run that said something else. Holding the text constant makes the
    comparison about the state the resume rebuilt rather than about the mock.
    """
    bank_gateway.generate.side_effect = [
        make_backend_response(text=text) for _ in range((round_number - 1) * agents)
    ] + [RuntimeError("boom")] * (agents * 10)


async def test_resume_continues_a_bank_run_from_the_checkpoint(tmp_path, bank_gateway):
    _crash_in_round(bank_gateway, 3)
    with pytest.raises(RuntimeError):
        await build_orchestrator(_bank_run(K=4), bank_gateway, runs_dir=tmp_path).run()
    assert {r.turn for r in _bank_rows(tmp_path)} == {0, 1, 2}

    bank_gateway.generate.side_effect = None
    bank_gateway.generate.return_value = make_backend_response(text="a resumed reply")
    await build_orchestrator(_bank_run(K=4), bank_gateway, runs_dir=tmp_path).run()

    assert {r.turn for r in _bank_rows(tmp_path)} == {0, 1, 2, 3, 4}


async def test_a_resume_does_not_probe_round_zero_again(tmp_path, bank_gateway):
    """Two turn-0 rows would mean two turn-0 records in every agent's history
    and a doubled starting measurement in the dataset."""
    _crash_in_round(bank_gateway, 2)
    with pytest.raises(RuntimeError):
        await build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path).run()

    bank_gateway.generate.side_effect = None
    bank_gateway.generate.return_value = make_backend_response(text="a resumed reply")
    await build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path).run()

    round_zero = [r for r in _bank_rows(tmp_path) if r.turn == 0]
    assert len(round_zero) == 8
    assert len({r.speaker_agent_id for r in round_zero}) == 8


async def test_a_resumed_bank_run_posts_the_items_it_would_have_posted(
    tmp_path, bank_gateway
):
    """The bank cycle counts are not in the checkpoint, so without rebuilding
    them from the log a resumed run would restart every sender's cycle and
    post items it had already used."""
    clean = tmp_path / "clean"
    bank_gateway.generate.return_value = make_backend_response(text="a reply")
    await build_orchestrator(_bank_run(K=4), bank_gateway, runs_dir=clean).run()

    crashed = tmp_path / "crashed"
    _crash_in_round(bank_gateway, 3)
    with pytest.raises(RuntimeError):
        await build_orchestrator(_bank_run(K=4), bank_gateway, runs_dir=crashed).run()
    bank_gateway.generate.side_effect = None
    bank_gateway.generate.return_value = make_backend_response(text="a reply")
    await build_orchestrator(_bank_run(K=4), bank_gateway, runs_dir=crashed).run()

    def posted(where):
        return [
            (r.turn, r.speaker_agent_id, r.posted_item_id, tuple(r.items_read))
            for r in _bank_rows(where)
        ]

    assert posted(crashed) == posted(clean)


async def test_restored_bank_counts_match_an_uninterrupted_runs(tmp_path, bank_gateway):
    """Checked directly, not only through the output: the counts are what the
    next pick reads, so an off-by-one here would surface as a subtly different
    item sequence rather than as an error."""
    bank_gateway.generate.return_value = make_backend_response(text="a reply")
    uninterrupted = build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path / "a")
    await uninterrupted.run()

    resumed = build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path / "a")
    resumed._agent_manager.initialize_population()
    resumed._restore_bank_state(2)  # the completed run checkpointed turn 2

    assert resumed._bank.post_counts() == uninterrupted._bank.post_counts()
    assert sum(resumed._bank.post_counts().values()) == 8 * 2


async def test_a_resumed_run_remembers_the_rounds_it_already_ran(tmp_path, bank_gateway):
    """Remembered rounds cannot be checkpointed without putting every post's
    text into agent state, so they are rebuilt from the log. Without that, a
    resumed run would show its agents an empty memory and change the stimulus
    mid-run -- the same defect _rebuild_last_posts exists to prevent in the
    generated regime."""
    _crash_in_round(bank_gateway, 3)
    with pytest.raises(RuntimeError):
        await build_orchestrator(_bank_run(K=4), bank_gateway, runs_dir=tmp_path).run()

    resumed = build_orchestrator(_bank_run(K=4), bank_gateway, runs_dir=tmp_path)
    resumed._agent_manager.restore_from_snapshot(
        __import__("sandbox.checkpoint_manager", fromlist=["CheckpointManager"])
        .CheckpointManager(tmp_path)
        .load("bank_run")
        .agent_snapshot
    )
    resumed._restore_bank_state(2)  # the crash left a checkpoint at round 2

    assert len(resumed._history) == 8
    for remembered in resumed._history.values():
        assert len(remembered) == 2          # rounds 1 and 2
        assert all(r.posts_read for r in remembered)
        assert all(r.reply_text for r in remembered)

    bank_gateway.generate.side_effect = None
    bank_gateway.generate.return_value = make_backend_response(text="a resumed reply")
    await resumed.run()

    round_three_prompts = [c.args[0] for c in bank_gateway.generate.await_args_list][-16:-8]
    assert all("Earlier in this discussion" in p for p in round_three_prompts)
    assert all(p.count("You replied:") == 2 for p in round_three_prompts)


async def test_a_resumed_run_rebuilds_its_full_history(tmp_path, bank_gateway):
    """Over five rounds, crashing after two, with memory_rounds=None.

    Full history makes the resume path's rebuild load-bearing in a way a
    three-round window hid: with a window, a resume that rebuilt only the
    last few rounds was indistinguishable from one that rebuilt everything.
    """
    run = _bank_run(K=5)
    assert run.memory_rounds is None
    bank_gateway.generate.side_effect = [
        make_backend_response(text=f"reply number {i:04d} ends here") for i in range(1000)
    ]
    clean = tmp_path / "clean"
    await build_orchestrator(run, bank_gateway, runs_dir=clean).run()

    crashed = tmp_path / "crashed"
    _crash_in_round(bank_gateway, 3)
    with pytest.raises(RuntimeError):
        await build_orchestrator(run, bank_gateway, runs_dir=crashed).run()

    resumed = build_orchestrator(run, bank_gateway, runs_dir=crashed)
    bank_gateway.generate.side_effect = [
        make_backend_response(text=f"resumed reply {i:04d} ends here") for i in range(1000)
    ]
    bank_gateway.generate.reset_mock()
    await resumed.run()

    # Both rounds 1 and 2 are remembered, and the memory carries them.
    assert all(len(v) == 5 for v in resumed._history.values())
    round_five = [c.args[0] for c in bank_gateway.generate.await_args_list][-run.M :]
    assert all(p.count("You replied:") == 4 for p in round_five)

    def signature(where):
        return [
            (r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids), r.posted_item_id,
             tuple(r.items_read), r.stance_before, r.stance_after)
            for r in sorted(_bank_rows(where), key=lambda r: (r.turn, r.speaker_agent_id))
        ]

    assert signature(crashed) == signature(clean)


async def test_rebuilt_memory_is_capped_at_the_window(tmp_path, bank_gateway):
    bank_gateway.generate.return_value = make_backend_response(text="a reply")
    await build_orchestrator(
        _bank_run(K=5, memory_rounds=2), bank_gateway, runs_dir=tmp_path
    ).run()

    resumed = build_orchestrator(_bank_run(K=5, memory_rounds=2), bank_gateway, runs_dir=tmp_path)
    resumed._restore_bank_state(5)
    assert all(len(v) == 2 for v in resumed._history.values())


async def test_a_failed_round_is_not_rebuilt_into_memory(tmp_path, bank_gateway):
    """Fix I again, on the resume path: a round that was never applied is not
    in the memory an uninterrupted run would be carrying."""
    bank_gateway.score_next_token.side_effect = (
        [make_logprob_response(mean=5.0)] * 8                   # round 0
        + [make_logprob_response(mean=5.0, off_scale=0.9)] * 8  # round 1 fails
        + [make_logprob_response(mean=5.0)] * 80
    )
    bank_gateway.generate.return_value = make_backend_response(text="a reply")
    await build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path).run()

    resumed = build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path)
    resumed._restore_bank_state(2)
    for remembered in resumed._history.values():
        assert len(remembered) == 1  # round 2 only


async def test_re_logged_rows_do_not_double_count_the_bank_cycle(tmp_path, bank_gateway):
    """A run interrupted mid-turn can leave rows with no checkpoint, and the
    resumed pass re-logs that turn. Keyed on (turn, speaker) with the last
    write kept, which is the rule export_csv applies on read -- otherwise the
    rebuilt counts would be too high and the resumed run would skip items."""
    bank_gateway.generate.return_value = make_backend_response(text="a reply")
    await build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path).run()

    path = tmp_path / "bank_run" / "interactions.jsonl"
    original = path.read_text()
    path.write_text(original + original)  # every row logged twice

    resumed = build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path)
    resumed._restore_bank_state(2)
    assert sum(resumed._bank.post_counts().values()) == 8 * 2  # not 32


async def test_a_resumed_delivery_off_run_matches_an_uninterrupted_one(
    tmp_path, bank_gateway
):
    """The control resumes like any other arm.

    It has history to rebuild now -- every round reads N neutral messages
    and writes a reply -- and those ids in items_read have to render through
    the neutral bank rather than the argument bank, which would raise a
    KeyError on them.
    """
    run = dict(delivery="off", K=4)
    bank_gateway.generate.return_value = make_backend_response(text="a reply")
    clean = tmp_path / "clean"
    await build_orchestrator(_bank_run(**run), bank_gateway, runs_dir=clean).run()

    crashed = tmp_path / "crashed"
    _crash_in_round(bank_gateway, 3)
    with pytest.raises(RuntimeError):
        await build_orchestrator(_bank_run(**run), bank_gateway, runs_dir=crashed).run()
    assert {r.turn for r in _bank_rows(crashed)} == {0, 1, 2}

    bank_gateway.generate.side_effect = None
    bank_gateway.generate.return_value = make_backend_response(text="reply in round 3")
    await build_orchestrator(_bank_run(**run), bank_gateway, runs_dir=crashed).run()

    def signature(where):
        return [
            (r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids), r.posted_item_id,
             tuple(r.items_read), r.stance_before, r.stance_after, r.api_call_status)
            for r in sorted(_bank_rows(where), key=lambda r: (r.turn, r.speaker_agent_id))
        ]

    assert signature(crashed) == signature(clean)


async def test_a_resumed_delivery_off_run_remembers_its_neutral_messages(
    tmp_path, bank_gateway
):
    """Rebuilt from the log, where the round is recorded as N neutral ids.

    Those have to render back through the neutral bank: looking them up in
    the argument bank would raise, and a fallback would rebuild the control's
    history out of arguments it never read.
    """
    from sandbox.bank_manager import NeutralBank

    run = _bank_run(delivery="off", K=2)
    bank_gateway.generate.return_value = make_backend_response(text="a reply")
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    live = {
        (r.turn, r.speaker_agent_id): r.items_read for r in _bank_rows(tmp_path)
    }
    resumed = build_orchestrator(run, bank_gateway, runs_dir=tmp_path)
    resumed._restore_bank_state(run.K)

    neutral = NeutralBank(FIXTURE_NEUTRAL_BANK)
    assert len(resumed._history) == run.M
    for agent_id, remembered in resumed._history.items():
        assert len(remembered) == run.K
        for turn, round_memory in enumerate(remembered, start=1):
            assert round_memory.posts_read == [
                neutral.text(nid, run.language_condition)
                for nid in live[(turn, agent_id)]
            ]
            assert len(round_memory.posts_read) == run.N


# --- resume: rows from uncommitted rounds are ignored --------------------

from sandbox.checkpoint_manager import CheckpointManager  # noqa: E402


def _deduped_signature(rows: list[Interaction]) -> list[tuple]:
    """Every row's reproducible content, keeping the last write of each
    (turn, speaker) -- the rule export_csv.py applies on read.

    Excludes interaction_id, timestamp_utc and latency_ms: wall-clock facts
    about one execution rather than properties of the run.
    """
    latest: dict[tuple[int, str], Interaction] = {}
    for row in rows:  # file order, so "last" means the post-resume copy
        latest[(row.turn, row.speaker_agent_id)] = row
    return [
        (
            r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids),
            r.stance_before, r.stance_after, r.reply_text, r.reason_text,
            tuple(r.items_read), r.posted_item_id, r.probe_mean,
            tuple(r.probe_dist or ()), r.probe_mass, r.side_conf,
            r.prompt_token_count, r.completion_token_count, r.api_call_status,
        )
        for _, r in sorted(latest.items())
    ]


async def test_a_resume_ignores_a_partially_written_round(tmp_path, bank_gateway):
    """Half a round on disk with no checkpoint must leave no trace.

    The orchestrator writes rows one at a time and checkpoints only after the
    whole round, so a crash part-way through leaves rows that no checkpoint
    covers. Counting them put those agents' bank cycles and histories one
    step ahead of everyone else's, and the rerun of that round then differed
    from an uninterrupted one -- so whether a resumed run posted the right
    arguments depended on exactly where the crash landed.
    """
    bank_gateway.generate.return_value = make_backend_response(text="a steady reply")
    clean = tmp_path / "clean"
    await build_orchestrator(_bank_run(K=4), bank_gateway, runs_dir=clean).run()

    # Rounds 1 and 2 complete, round 3 fails for everyone, so the checkpoint
    # is at 2 and no round-3 row was written.
    crashed = tmp_path / "crashed"
    _crash_in_round_replying(bank_gateway, 3, "a steady reply")
    with pytest.raises(RuntimeError):
        await build_orchestrator(_bank_run(K=4), bank_gateway, runs_dir=crashed).run()
    assert CheckpointManager(crashed).load("bank_run").last_completed_turn == 2

    # Now stage the partial round: half of round 3's rows, as a crash
    # between two write_interaction calls would have left them.
    log = crashed / "bank_run" / "interactions.jsonl"
    partial = [
        row.model_dump_json()
        for row in sorted(_bank_rows(clean), key=lambda r: (r.turn, r.speaker_agent_id))
        if row.turn == 3
    ][:4]
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(partial) + "\n")
    assert len([r for r in _bank_rows(crashed) if r.turn == 3]) == 4

    bank_gateway.generate.side_effect = None
    bank_gateway.generate.return_value = make_backend_response(text="a steady reply")
    resumed = build_orchestrator(_bank_run(K=4), bank_gateway, runs_dir=crashed)
    await resumed.run()

    assert _deduped_signature(_bank_rows(crashed)) == _deduped_signature(_bank_rows(clean))


async def test_a_resume_ignores_a_full_round_the_checkpoint_does_not_cover(
    tmp_path, bank_gateway
):
    """checkpoint_every_n_turns=2: round 3 completes, round 4 dies, and the
    last checkpoint is at round 2. The whole of round 3 is on disk and must
    be ignored, because the resumed pass is about to run it again."""
    every_two = dict(K=4, checkpoint_every_n_turns=2)
    bank_gateway.generate.return_value = make_backend_response(text="a steady reply")
    clean = tmp_path / "clean"
    await build_orchestrator(_bank_run(**every_two), bank_gateway, runs_dir=clean).run()

    crashed = tmp_path / "crashed"
    _crash_in_round_replying(bank_gateway, 4, "a steady reply")
    with pytest.raises(RuntimeError):
        await build_orchestrator(
            _bank_run(**every_two), bank_gateway, runs_dir=crashed
        ).run()
    assert CheckpointManager(crashed).load("bank_run").last_completed_turn == 2
    assert {r.turn for r in _bank_rows(crashed)} == {0, 1, 2, 3}

    bank_gateway.generate.side_effect = None
    bank_gateway.generate.return_value = make_backend_response(text="a steady reply")
    resumed = build_orchestrator(_bank_run(**every_two), bank_gateway, runs_dir=crashed)
    await resumed.run()

    assert _deduped_signature(_bank_rows(crashed)) == _deduped_signature(_bank_rows(clean))


async def test_rebuilt_state_stops_at_the_checkpoint(tmp_path, bank_gateway):
    """The mechanism, asserted directly: an uncommitted round advances
    neither the bank cycle nor the remembered history."""
    bank_gateway.generate.return_value = make_backend_response(text="a steady reply")
    await build_orchestrator(_bank_run(K=3), bank_gateway, runs_dir=tmp_path).run()

    resumed = build_orchestrator(_bank_run(K=3), bank_gateway, runs_dir=tmp_path)
    # As of round 1, two of the three logged rounds are ahead of us.
    resumed._restore_bank_state(1)

    assert sum(resumed._bank.post_counts().values()) == 8  # round 1 only
    assert all(len(v) == 1 for v in resumed._history.values())
    assert len(resumed._rebuild_last_posts(1)) == 8
    assert all(post.turn == 1 for post in resumed._rebuild_last_posts(1).values())


async def test_rerunning_a_completed_bank_run_is_a_no_op(tmp_path, bank_gateway):
    bank_gateway.generate.return_value = make_backend_response(text="a reply")
    await build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path).run()
    rows_before = len(_bank_rows(tmp_path))
    replies_before = bank_gateway.generate.await_count
    probes_before = bank_gateway.score_next_token.await_count

    await build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path).run()

    assert len(_bank_rows(tmp_path)) == rows_before
    assert bank_gateway.generate.await_count == replies_before
    assert bank_gateway.score_next_token.await_count == probes_before


# --- the reply seed -------------------------------------------------------


def _reply_seeds(bank_gateway) -> list[int | None]:
    """The seed each generate() call carried, in dispatch order."""
    return [c.kwargs.get("seed") for c in bank_gateway.generate.await_args_list]


async def test_every_reply_call_carries_a_seed(tmp_path, bank_gateway):
    """Without one, a rerun of the same config samples different replies, and
    the probe reads a stance out of a history made of them."""
    run = _bank_run(K=2)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    seeds = _reply_seeds(bank_gateway)
    assert len(seeds) == run.M * run.K
    assert all(isinstance(s, int) for s in seeds)
    # One per (turn, agent), so no two calls in a round sample alike.
    assert len(set(seeds)) == len(seeds)


async def test_a_runs_probe_calls_are_never_seeded(tmp_path, bank_gateway):
    """It runs at temperature 0. A seed there would suggest the measurement
    had a draw in it."""
    await build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path).run()

    for call in bank_gateway.score_next_token.await_args_list:
        assert "seed" not in call.kwargs


async def test_two_language_arms_send_identical_reply_seeds(tmp_path, bank_gateway):
    """A matched set differs in the text of the prompt, not in how the reply
    to it was sampled. A seed that varied by arm would leave the four arms
    drawing differently from the one comparison RQ1 reports."""
    logged, dispatched = {}, {}
    for condition in ("english", "hinglish", "romanized_hindi", "hindi"):
        bank_gateway.generate.reset_mock()
        # seed 1000: its WS(8, 4, 0.1) graph starves no node, which the
        # constructor checks before anything is dispatched.
        run = _bank_run(K=2, seed=1000, language_condition=condition,
                        run_id=f"bank_{condition}")
        await build_orchestrator(
            run, bank_gateway, runs_dir=tmp_path / condition
        ).run()
        logged[condition] = {
            (r.turn, r.speaker_agent_id): r.reply_seed
            for r in _bank_rows(tmp_path / condition, f"bank_{condition}")
        }
        # The seeds the calls carried, in dispatch order, not only the ones
        # the rows recorded: the two agree, and it is the first that matters.
        dispatched[condition] = _reply_seeds(bank_gateway)

    assert logged["english"]
    assert len(dispatched["english"]) == 8 * 2
    for condition in logged:
        assert logged[condition] == logged["english"], condition
        assert dispatched[condition] == dispatched["english"], condition

    # Not vacuous: the seeds are not a constant the comparison would pass
    # against regardless. A different base seed moves every one of them.
    other = _bank_run(K=2, seed=1001, run_id="bank_other")
    bank_gateway.generate.reset_mock()
    await build_orchestrator(other, bank_gateway, runs_dir=tmp_path / "other").run()
    assert _reply_seeds(bank_gateway) != dispatched["english"]


async def test_a_resumed_round_sends_the_seed_it_would_have_sent(tmp_path, bank_gateway):
    """The seed is a function of (seed, turn, agent) and of nothing about
    when the call happened, so the rerun of an interrupted round is the same
    request an uninterrupted run would have made."""
    run = dict(K=4)
    bank_gateway.generate.return_value = make_backend_response(text="a steady reply")
    clean = tmp_path / "clean"
    await build_orchestrator(_bank_run(**run), bank_gateway, runs_dir=clean).run()
    clean_seeds = {
        (r.turn, r.speaker_agent_id): r.reply_seed for r in _bank_rows(clean)
    }

    crashed = tmp_path / "crashed"
    _crash_in_round_replying(bank_gateway, 3, "a steady reply")
    with pytest.raises(RuntimeError):
        await build_orchestrator(_bank_run(**run), bank_gateway, runs_dir=crashed).run()

    bank_gateway.generate.side_effect = None
    bank_gateway.generate.return_value = make_backend_response(text="a steady reply")
    bank_gateway.generate.reset_mock()
    await build_orchestrator(_bank_run(**run), bank_gateway, runs_dir=crashed).run()

    resumed_seeds = {
        (r.turn, r.speaker_agent_id): r.reply_seed for r in _bank_rows(crashed)
    }
    assert resumed_seeds == clean_seeds
    # And round 3, the one that was re-run, really was dispatched again with
    # the seed the clean run used.
    rerun = {
        (3, agent): seed for (turn, agent), seed in clean_seeds.items() if turn == 3
    }
    assert set(_reply_seeds(bank_gateway)) >= set(rerun.values())


async def test_round_zero_rows_carry_no_reply_seed(tmp_path, bank_gateway):
    """Nothing replied there, so there is no seed to record. A number would
    claim a call that was never made."""
    run = _bank_run(K=2)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    for row in _bank_rows(tmp_path):
        if row.turn == 0:
            assert row.reply_seed is None
        else:
            assert isinstance(row.reply_seed, int)


async def test_a_failed_row_still_records_its_reply_seed(tmp_path, bank_gateway):
    """The call was made and was seeded; only its result was unusable."""
    bank_gateway.generate.side_effect = FatalGatewayError("bad key")
    run = _bank_run(M=4, N=1, K=1, graph_degree=2)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    failed = [r for r in _bank_rows(tmp_path) if r.turn == 1]
    assert len(failed) == run.M
    assert all(r.api_call_status == "failed_logged_null" for r in failed)
    assert all(isinstance(r.reply_seed, int) for r in failed)


async def test_prompt_lines_carry_the_seed_on_replies_only(tmp_path, bank_gateway):
    run = _bank_run(M=4, N=1, K=2, graph_degree=2)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rows = _prompt_rows(tmp_path)
    replies = [r for r in rows if r.call == "reply"]
    probes = [r for r in rows if r.call == "probe"]
    assert len(replies) == run.M * run.K
    assert all(isinstance(r.seed, int) for r in replies)
    assert all(r.seed is None for r in probes)

    # The logged seed is the one the row records, for the same call.
    by_key = {(r.turn, r.speaker_agent_id): r.reply_seed for r in _bank_rows(tmp_path)}
    for line in replies:
        assert line.seed == by_key[(line.turn, line.agent_id)]


async def test_the_generated_regime_sends_no_seed(tmp_path, mock_gateway):
    """Out of scope by design: its behaviour and its traffic are unchanged."""
    run = _run(M=3, N=1, K=2)
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    assert mock_gateway.generate.await_count == run.M * run.K
    for call in mock_gateway.generate.await_args_list:
        assert "seed" not in call.kwargs


async def test_a_delivery_off_resume_ignores_a_partially_written_round(
    tmp_path, bank_gateway
):
    """The Fix 4 case, for the control arm.

    The control has history to rebuild now -- every round reads N messages
    and writes a reply -- so a half-written round left on disk with no
    checkpoint can put those agents' remembered rounds one step ahead of
    everyone else's, exactly as it could for a treatment arm. The neutral IDs
    in those rows also have to render back through the neutral bank; looking
    them up in the argument bank would raise.
    """
    off = dict(delivery="off", K=4)
    bank_gateway.generate.return_value = make_backend_response(text="a steady reply")
    clean = tmp_path / "clean"
    await build_orchestrator(_bank_run(**off), bank_gateway, runs_dir=clean).run()

    # Rounds 1 and 2 complete, round 3 fails for everyone, so the checkpoint
    # is at 2 and no round-3 row was written.
    crashed = tmp_path / "crashed"
    _crash_in_round_replying(bank_gateway, 3, "a steady reply")
    with pytest.raises(RuntimeError):
        await build_orchestrator(_bank_run(**off), bank_gateway, runs_dir=crashed).run()
    assert CheckpointManager(crashed).load("bank_run").last_completed_turn == 2

    # Stage half of round 3's rows, as a crash between two write_interaction
    # calls would have left them.
    log = crashed / "bank_run" / "interactions.jsonl"
    partial = [
        row.model_dump_json()
        for row in sorted(_bank_rows(clean), key=lambda r: (r.turn, r.speaker_agent_id))
        if row.turn == 3
    ][:4]
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(partial) + "\n")
    assert len([r for r in _bank_rows(crashed) if r.turn == 3]) == 4

    bank_gateway.generate.side_effect = None
    bank_gateway.generate.return_value = make_backend_response(text="a steady reply")
    await build_orchestrator(_bank_run(**off), bank_gateway, runs_dir=crashed).run()

    assert _deduped_signature(_bank_rows(crashed)) == _deduped_signature(_bank_rows(clean))


async def test_a_delivery_off_resume_rebuilds_the_feed_it_read(tmp_path, bank_gateway):
    """Over four rounds, the rebuilt history holds N posts a round, rendered
    from the neutral bank rather than the argument bank."""
    off = dict(delivery="off", K=4)
    bank_gateway.generate.return_value = make_backend_response(text="a steady reply")
    await build_orchestrator(_bank_run(**off), bank_gateway, runs_dir=tmp_path).run()

    run = _bank_run(**off)
    resumed = build_orchestrator(run, bank_gateway, runs_dir=tmp_path)
    resumed._restore_bank_state(run.K)

    assert len(resumed._history) == run.M
    for remembered in resumed._history.values():
        assert len(remembered) == run.K
        assert all(len(r.posts_read) == run.N for r in remembered)


async def test_a_delivery_on_run_cannot_rebuild_a_delivery_off_log(
    tmp_path, bank_gateway
):
    """A neutral ID in the log has no reading in the argument bank. It raises
    rather than falling back, since a fallback would rebuild a history out of
    text the agent never read."""
    off = _bank_run(delivery="off", K=2)
    bank_gateway.generate.return_value = make_backend_response(text="a reply")
    await build_orchestrator(off, bank_gateway, runs_dir=tmp_path).run()

    # Same run directory, delivery on: no neutral bank is wired at all.
    on = build_orchestrator(_bank_run(K=2), bank_gateway, runs_dir=tmp_path)
    with pytest.raises(ValueError, match="neutral bank"):
        on._restore_bank_state(2)


# --- start_mode="posts" in a network run ----------------------------------


async def test_a_posts_run_shows_each_agent_the_same_starters_every_round(
    tmp_path, bank_gateway
):
    """Every prompt an agent gets carries the same starter block, from its
    round-0 probe to its last reply: the section is a trait, and a section
    that changed between rounds would make the probe two instruments."""
    from sandbox.bank_manager import BankManager, starter_items

    run = _bank_run(start_mode="posts", starter_bank_path="data/bank/test_starter.csv")
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    starters = BankManager("data/bank/test_starter.csv")
    agents = {
        json.loads(line)["agent_id"]: json.loads(line)
        for line in (tmp_path / "bank_run" / "agents_final.jsonl").read_text().splitlines()
        if line.strip()
    }
    blocks: dict[str, set[str]] = {}
    for record in _prompt_rows(tmp_path):
        text = record.prompt_text
        assert "Before this discussion, you had written these posts:" in text
        block = text.split("Before this discussion, you had written these posts:\n")[1].split("\n\n")[0]
        blocks.setdefault(record.agent_id, set()).add(block)
    assert set(blocks) == set(agents)
    for agent_id, seen in blocks.items():
        assert len(seen) == 1, agent_id
        expected = starter_items(starters, run.seed, agent_id, agents[agent_id]["initial_stance"])
        for item in expected:
            assert starters.text(item, "english") in next(iter(seen))
