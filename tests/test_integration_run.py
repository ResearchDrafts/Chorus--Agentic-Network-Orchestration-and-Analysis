"""End-to-end integration test (full_design_doc.md Sec 7.4).

The first test in the repo that exercises more than one production module
together. Everything before Phase 4 imported exactly one module plus
sandbox.models, so no seam between components had any coverage at all.

Covers the whole path a real run takes: a YAML config on disk, through
config_loader, into a fully wired orchestrator, out to interactions.jsonl,
agents_final.jsonl, run_config.json and a checkpoint. The only thing faked is
the LLM call itself.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
import yaml

from sandbox.bank_manager import BankManager
from sandbox.checkpoint_manager import CheckpointManager
from sandbox.config_loader import load_run_config
from sandbox.models import (
    LANGUAGE_CONDITIONS,
    Agent,
    ExperimentRun,
    Interaction,
    PromptRecord,
)
from sandbox.simulation_orchestrator import build_orchestrator

EXAMPLE_CONFIG = Path("configs/example_run.yaml")

# Sec 7.4's stated shape: small enough to assert exact counts on.
SMALL_RUN = dict(M=5, N=2, K=2)


@pytest.fixture
def config_path(tmp_path):
    """The shipped example config, shrunk to Sec 7.4's size and pointed at
    the test fixtures. Starting from the real file rather than a handwritten
    dict means this test fails if the example config ever stops loading."""

    def _write(**overrides) -> Path:
        raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
        raw.update(SMALL_RUN)
        raw["run_id"] = "integration_run"
        raw["persona_pool_id"] = "test_pool"
        raw.pop("max_cost_usd", None)
        raw.update(overrides)
        path = tmp_path / "run.yaml"
        path.write_text(yaml.safe_dump(raw))
        return path

    return _write


def _rows(runs_dir, run_id="integration_run") -> list[Interaction]:
    path = runs_dir / run_id / "interactions.jsonl"
    return [
        Interaction.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _prompt_records(runs_dir, run_id="integration_run") -> list[PromptRecord]:
    """Every call a run made, as the run itself recorded it.

    Auditing this rather than the gateway double's call list is the stronger
    check: it is the file a real run leaves behind, so the same assertion can
    be made against a campaign's output and not only inside the suite.
    """
    path = runs_dir / run_id / "prompts.jsonl"
    return [
        PromptRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- the shipped example config ------------------------------------------


def test_the_shipped_example_config_actually_loads():
    """configs/example_run.yaml is the only worked example a researcher has.
    If it drifts out of sync with the schema it is worse than no example."""
    run = load_run_config(EXAMPLE_CONFIG)
    assert run.M == 100 and run.N == 5 and run.K == 10  # Ohagi's baseline
    assert run.git_commit_hash  # injected at load, never read from the YAML


def test_the_example_configs_commented_meme_block_is_valid_when_enabled(tmp_path):
    """The meme fields ship commented out, so nothing validates them. Enabling
    them exactly as documented must produce a loadable config."""
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["meme_injection"] = {
        "enabled": True,
        "meme_pool_id": "test_pool",
        "injection_rate": 0.2,
        "injection_schedule": "random",
    }
    path = tmp_path / "memes.yaml"
    path.write_text(yaml.safe_dump(raw))

    run = load_run_config(path)
    assert run.meme_injection.enabled
    assert run.meme_injection.injection_rate == 0.2


# --- Sec 7.4's integration assertions ------------------------------------


async def test_full_run_from_yaml_to_disk(tmp_path, config_path, mock_gateway):
    run = load_run_config(config_path())
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()
    run_dir = tmp_path / run.run_id

    # Exactly M * K records, meme turns included.
    rows = _rows(tmp_path)
    assert len(rows) == run.M * run.K == 10

    # Every neighbour list is exactly N and never contains the speaker.
    for row in rows:
        assert len(row.neighbor_agent_ids) == run.N
        assert row.speaker_agent_id not in row.neighbor_agent_ids

    # Every record round-trips, and the file is one JSON object per line.
    lines = (run_dir / "interactions.jsonl").read_text().splitlines()
    assert len(lines) == 10
    assert all(json.loads(line) for line in lines)

    # agents_final.jsonl holds one Agent per line.
    finals = [
        Agent.model_validate_json(line)
        for line in (run_dir / "agents_final.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(finals) == run.M

    # run_config.json round-trips and carries terminal lifecycle fields.
    persisted = ExperimentRun.model_validate_json(
        (run_dir / "run_config.json").read_text()
    )
    assert persisted.status == "completed"
    assert persisted.started_at_utc and persisted.completed_at_utc

    # A checkpoint exists at the final turn.
    assert CheckpointManager(tmp_path).load(run.run_id).last_completed_turn == run.K


async def test_meme_enabled_run_produces_valid_meme_rows(
    tmp_path, config_path, mock_gateway
):
    run = load_run_config(
        config_path(
            meme_injection={
                "enabled": True,
                "meme_pool_id": "test_pool",
                "injection_rate": 1.0,
                "injection_schedule": "fixed_turn",
                "fixed_turns": [1],
            }
        )
    )
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    rows = _rows(tmp_path)
    assert len(rows) == 10

    memes = [r for r in rows if r.content_type == "meme"]
    assert len(memes) == 5  # every agent, turn 1 only
    for row in memes:
        assert row.meme_id is not None
        assert row.prompt_token_count == 0
        assert row.completion_token_count == 0
        assert row.latency_ms == 0

    # Turn 2 is ordinary generated text, and those agents were shown the
    # turn-1 memes as neighbour content.
    assert all(r.content_type == "generated_text" for r in rows if r.turn == 2)
    assert any("Caption:" in c.args[0] for c in mock_gateway.generate.await_args_list)


async def test_rerunning_a_completed_run_is_a_no_op(tmp_path, config_path, mock_gateway):
    """Sec 7.4: resume against a finished run_id must not error and must not
    duplicate work, since start_turn > K leaves the loop empty."""
    path = config_path()
    await build_orchestrator(
        load_run_config(path), mock_gateway, runs_dir=tmp_path
    ).run()
    rows_before = len(_rows(tmp_path))
    calls_before = mock_gateway.generate.await_count

    await build_orchestrator(
        load_run_config(path), mock_gateway, runs_dir=tmp_path
    ).run()

    assert len(_rows(tmp_path)) == rows_before
    assert mock_gateway.generate.await_count == calls_before


async def test_the_same_seed_reproduces_the_run_exactly(
    tmp_path, config_path, mock_gateway
):
    """The property every RQ in this project rests on. Fix M also depends on
    it: reusing a seed across language arms is what holds neighbour sampling
    and meme injection constant while language varies."""
    first, second = tmp_path / "a", tmp_path / "b"
    await build_orchestrator(
        load_run_config(config_path(seed=7)), mock_gateway, runs_dir=first
    ).run()
    await build_orchestrator(
        load_run_config(config_path(seed=7)), mock_gateway, runs_dir=second
    ).run()

    def signature(base):
        return [
            (r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids), r.stance_after)
            for r in _rows(base)
        ]

    assert signature(first) == signature(second)


async def test_agent_state_and_the_interaction_log_agree(
    tmp_path, config_path, mock_gateway
):
    """agents_final.jsonl and interactions.jsonl are written by different
    paths from different objects. Post-hoc analysis reads the log; anything
    loading Agent objects reads the other. They must tell the same story."""
    run = load_run_config(config_path())
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    rows = _rows(tmp_path)
    finals = {
        a.agent_id: a
        for a in (
            Agent.model_validate_json(line)
            for line in (tmp_path / run.run_id / "agents_final.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        )
    }

    for agent_id, agent in finals.items():
        logged = sorted(
            (r for r in rows if r.speaker_agent_id == agent_id), key=lambda r: r.turn
        )
        assert [s.turn for s in agent.stance_history] == [r.turn for r in logged]
        assert agent.current_stance() == logged[-1].stance_after
        # Fix A: the window never exceeds 5, whatever K is.
        assert len(agent.memory_window) <= 5


async def test_a_resumed_run_matches_an_uninterrupted_one(
    tmp_path, config_path, mock_gateway
):
    """Checkpoint and resume exist so a failed run is not restarted from
    scratch, which would re-spend the budget. That is only worth anything if
    the resumed output is the output the run would have produced anyway."""
    from tests.factories import make_backend_response

    clean = tmp_path / "clean"
    await build_orchestrator(
        load_run_config(config_path(seed=11)), mock_gateway, runs_dir=clean
    ).run()

    crashed = tmp_path / "crashed"
    mock_gateway.generate.side_effect = [make_backend_response(stance=5.0)] * 5 + [
        RuntimeError("boom")
    ] * 50
    with pytest.raises(RuntimeError):
        await build_orchestrator(
            load_run_config(config_path(seed=11)), mock_gateway, runs_dir=crashed
        ).run()

    mock_gateway.generate.side_effect = None
    mock_gateway.generate.return_value = make_backend_response(stance=5.0)
    await build_orchestrator(
        load_run_config(config_path(seed=11)), mock_gateway, runs_dir=crashed
    ).run()

    def signature(base):
        return [
            (r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids), r.stance_after)
            for r in _rows(base)
        ]

    assert signature(crashed) == signature(clean)


# --- the shape post-hoc analysis will consume ----------------------------


async def test_the_log_is_ready_for_the_phase_5_analysis_modules(
    tmp_path, config_path, mock_gateway
):
    """Sec 7.4 wants the post-hoc path validated end to end, but none of those
    modules exist yet (Phase 5). What can be checked now is that
    interactions.jsonl carries the columns they are specified to read, with
    usable dtypes, since that schema is what unblocks building them.
    """
    pd = pytest.importorskip("pandas")

    run = load_run_config(config_path())
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    df = pd.read_json(tmp_path / run.run_id / "interactions.jsonl", lines=True)

    # stance_regression.py fits stance_after ~ stance_before + neighbour mean.
    for column in ("stance_before", "stance_after", "prompt_token_count", "turn"):
        assert pd.api.types.is_numeric_dtype(df[column]), column

    # coherence_scorer.py and sentiment_proxy.py must filter on this first
    # (Fix N), so it has to be present and exactly two-valued.
    assert set(df["content_type"]) <= {"generated_text", "meme"}

    # Every analysis drops failed rows via this column.
    assert set(df["api_call_status"]) <= {
        "success",
        "retried_success",
        "failed_logged_null",
    }

    # neighbour ids survive as real lists, not stringified ones: the
    # regression joins on them to compute each row's neighbour mean.
    assert all(isinstance(v, list) for v in df["neighbor_agent_ids"])
    assert df["neighbor_agent_ids"].map(len).eq(run.N).all()


async def test_prompt_token_count_is_populated_for_the_fix_a_diagnostic(
    tmp_path, config_path, mock_gateway
):
    """prompt_token_count is the field that turns "memory was capped at 5
    turns" into a measured result. If it were ever left at 0 for generated
    text, the RQ1 truncation-symmetry check could not be reported at all."""
    run = load_run_config(config_path())
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    generated = [r for r in _rows(tmp_path) if r.content_type == "generated_text"]
    assert generated
    assert all(r.prompt_token_count > 0 for r in generated)


# =========================================================================
# RQ1: the bank regime, end to end
#
# Same shape as the tests above -- a real YAML config on disk, through
# config_loader, into a fully wired orchestrator, out to disk -- with only the
# LLM calls faked. The shipped configs/rq1/smoke.yaml is the starting point
# rather than a handwritten dict, so these fail if it ever stops loading.
# =========================================================================

RQ1_CONFIG = Path("configs/rq1/smoke.yaml")

# Small enough to assert exact counts on, and still big enough for a graph:
# M=8 with degree 4 gives every node four neighbours to draw three from.
RQ1_RUN = dict(M=8, N=3, K=3, graph_degree=4)


@pytest.fixture
def rq1_config(tmp_path):
    def _write(filename: str = "rq1.yaml", **overrides) -> Path:
        raw = yaml.safe_load(RQ1_CONFIG.read_text())
        raw.update(RQ1_RUN)
        raw["run_id"] = "rq1_run"
        raw["persona_pool_id"] = "pool_20"
        raw.update(overrides)
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(raw, allow_unicode=True))
        return path

    return _write


def _signature(rows: list[Interaction]) -> list[tuple]:
    """Everything about a row that must be reproducible.

    Excludes interaction_id (a fresh uuid4 per row), timestamp_utc and
    latency_ms, which are wall-clock facts about one execution rather than
    properties of the run. Everything else is included, including the probe
    columns and the posted item: this is the comparison the resume guarantee
    is stated in terms of.
    """
    return [
        (
            r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids),
            r.stance_before, r.stance_after, r.reason_text, r.reply_text,
            r.content_type, tuple(r.items_read), r.posted_item_id,
            r.probe_mean, tuple(r.probe_dist or ()), r.probe_mass, r.side_conf,
            r.prompt_token_count, r.completion_token_count, r.api_call_status,
        )
        for r in sorted(rows, key=lambda r: (r.turn, r.speaker_agent_id))
    ]


# --- 12.1: a full bank run from YAML to disk -----------------------------


async def test_the_shipped_rq1_config_actually_loads():
    """configs/rq1/smoke.yaml is the only worked bank-regime example a
    researcher has. If it drifts out of sync with the schema it is worse than
    no example."""
    run = load_run_config(RQ1_CONFIG)
    assert run.regime == "bank" and run.neighbor_strategy == "graph"
    assert run.delivery == "on"
    assert run.probe_question and run.bank_path
    assert run.git_commit_hash


async def test_a_full_bank_run_writes_every_output_file(tmp_path, rq1_config, bank_gateway):
    run = load_run_config(rq1_config())
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()
    run_dir = tmp_path / run.run_id

    # M * (K + 1): one row per agent per round, plus the round-0 probe.
    rows = _rows(tmp_path, "rq1_run")
    assert len(rows) == run.M * (run.K + 1) == 32
    assert sorted({r.turn for r in rows}) == [0, 1, 2, 3]

    lines = (run_dir / "interactions.jsonl").read_text().splitlines()
    assert len(lines) == 32
    assert all(json.loads(line) for line in lines)

    finals = [
        Agent.model_validate_json(line)
        for line in (run_dir / "agents_final.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(finals) == run.M
    # Round 0 plus K rounds of history, all of it probe readings.
    assert all(len(a.stance_history) == run.K + 1 for a in finals)

    persisted = ExperimentRun.model_validate_json((run_dir / "run_config.json").read_text())
    assert persisted.status == "completed"
    assert persisted.regime == "bank"
    assert persisted.started_at_utc and persisted.completed_at_utc

    assert CheckpointManager(tmp_path).load(run.run_id).last_completed_turn == run.K

    # prompts.jsonl: every call, verbatim. On by default for this regime,
    # because the no-number claim is otherwise checkable only in the test
    # suite and never against the run that produced a dataset.
    from sandbox.models import PromptRecord

    assert run.log_prompts is True
    prompts = [
        PromptRecord.model_validate_json(line)
        for line in (run_dir / "prompts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len([p for p in prompts if p.call == "reply"]) == run.M * run.K == 24
    assert len([p for p in prompts if p.call == "probe"]) == run.M * (run.K + 1) == 32
    assert all(p.run_id == run.run_id for p in prompts)

    # The file the generated regime never writes.
    edges = (run_dir / "graph_edges.csv").read_text().splitlines()
    assert edges[0] == "agent_a,agent_b"
    assert len(edges) - 1 == run.M * run.graph_degree // 2 == 16


async def test_the_bank_run_log_carries_the_columns_rq1_will_analyse(
    tmp_path, rq1_config, bank_gateway
):
    """The bank-regime counterpart of the Phase 5 readiness check: the shape
    the analysis consumes, with usable dtypes, since that schema is what
    unblocks building it."""
    pd = pytest.importorskip("pandas")

    run = load_run_config(rq1_config())
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    df = pd.read_json(tmp_path / run.run_id / "interactions.jsonl", lines=True)

    for column in ("probe_mean", "probe_mass", "side_conf", "stance_before", "stance_after"):
        assert pd.api.types.is_numeric_dtype(df[column]), column
    assert set(df["content_type"]) == {"bank_item"}
    assert set(df["api_call_status"]) <= {"success", "retried_success", "failed_logged_null"}
    assert all(isinstance(v, list) for v in df["probe_dist"])
    assert df["probe_dist"].map(len).eq(7).all()
    assert all(isinstance(v, list) for v in df["items_read"])
    # Round 0 reads nothing; every later round reads exactly N.
    assert df[df.turn == 0]["items_read"].map(len).eq(0).all()
    assert df[df.turn > 0]["items_read"].map(len).eq(run.N).all()


async def test_export_csv_consumes_a_bank_run(tmp_path, rq1_config, bank_gateway):
    """End to end into the analysis CSVs, not only into the JSONL."""
    pd = pytest.importorskip("pandas")

    import export_csv

    run = load_run_config(rq1_config())
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    frame = export_csv.derive(export_csv.mark_duplicates(export_csv.load_run(tmp_path / run.run_id)))

    assert len(frame) == 32
    assert set(frame["regime"]) == {"bank"}
    assert set(frame["delivery"]) == {"on"}
    # x is the stance rescaled to [-1, 1] about the midpoint.
    probed = frame[frame.probe_mean.notna()]
    assert ((probed.x - (probed.probe_mean - 4) / 3).abs() < 1e-9).all()
    assert frame[frame.turn > 0]["n_items_read"].eq(run.N).all()
    assert frame[frame.turn > 0]["items_read"].str.count(";").eq(run.N - 1).all()

    # Round 0 is a measurement, not a turn: its stance_before is the
    # configured initial_stance rather than a reading, so it carries no
    # shift. Left in, every aggregate over the column absorbed M rows of
    # movement that happened before any agent read a word.
    round_zero = frame[frame.turn == 0]
    later = frame[frame.turn > 0]
    assert len(round_zero) == run.M
    assert round_zero["is_round_zero"].all()
    assert not later["is_round_zero"].any()
    for column in ("stance_before", "stance_shift", "abs_stance_shift", "stance_moved"):
        assert round_zero[column].isna().all(), column
        assert later[column].notna().all(), column
    # stance_after survives: it IS the starting measurement.
    assert round_zero["stance_after"].notna().all()
    assert round_zero["probe_mean"].notna().all()

    # The two prompt lengths, separately as well as summed: RQ1 reports
    # prompt length per language, and a reply prompt and a probe prompt grow
    # for different reasons.
    for column in ("reply_prompt_tokens", "probe_prompt_tokens"):
        assert pd.api.types.is_numeric_dtype(frame[column]), column
    assert frame[frame.turn == 0]["reply_prompt_tokens"].eq(0).all()
    assert frame[frame.turn > 0]["reply_prompt_tokens"].gt(0).all()
    assert frame["probe_prompt_tokens"].gt(0).all()
    assert (
        frame.prompt_token_count
        == frame.reply_prompt_tokens + frame.probe_prompt_tokens
    ).all()


async def test_the_agents_table_still_knows_where_a_bank_agent_started(
    tmp_path, rq1_config, bank_gateway
):
    """Nulling round 0's stance_before must not cost the agents table its
    initial_stance.

    It does not: groupby.first() skips the null and lands on round 1's
    stance_before, which is round 0's probe reading -- the measurement, not
    the configured value. That is a better answer than the one it gave
    before, and the column must not quietly become empty.
    """
    pytest.importorskip("pandas")

    import export_csv

    from tests.factories import make_logprob_response

    bank_gateway.score_next_token.side_effect = (
        [make_logprob_response(mean=3.0)] * 8    # round 0
        + [make_logprob_response(mean=6.0)] * 80
    )
    run = load_run_config(rq1_config(K=2))
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    frame = export_csv.derive(
        export_csv.mark_duplicates(export_csv.load_run(tmp_path / run.run_id))
    )
    agents = export_csv.agents_table(frame)

    assert len(agents) == run.M
    assert agents.initial_stance.notna().all()
    assert set(agents.initial_stance) == {3.0}   # the round-0 reading
    assert set(agents.final_stance) == {6.0}
    assert set(agents.net_drift) == {3.0}


async def test_a_generated_run_has_no_round_zero_rows_to_null(
    tmp_path, config_path, mock_gateway
):
    """The generated regime never writes turn 0, so nothing there changes."""
    pytest.importorskip("pandas")

    import export_csv

    run = load_run_config(config_path())
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    frame = export_csv.derive(
        export_csv.mark_duplicates(export_csv.load_run(tmp_path / run.run_id))
    )
    assert not frame["is_round_zero"].any()
    assert frame["stance_before"].notna().all()
    assert frame["stance_shift"].notna().all()


# --- 12.2: the matched set -----------------------------------------------


async def test_a_matched_set_differs_only_in_the_rendering(
    tmp_path, rq1_config, bank_gateway
):
    """The property RQ1 rests on.

    Two arms sharing a seed must draw the same graph, sample the same
    neighbours and post the same bank items whenever sides match -- so that
    the only thing differing between them is the language the text is rendered
    in. The probe is driven to a constant reading in both arms, which is what
    makes the sides match: a real run's sides diverge as soon as its readings
    do, and that divergence is a result, not a defect.
    """
    english = load_run_config(rq1_config("en.yaml", language_condition="english"))
    hindi = load_run_config(
        rq1_config(
            "hi.yaml",
            language_condition="hindi",
            probe_question="क्या शहर की बस किराया सब्सिडी जारी रहनी चाहिए?",
            # The anchor labels are NOT overridden: they stay English in
            # every arm, because they are rendered inside English sentences.
            # Only the bank text, the question and the directive vary.
        )
    )
    assert hindi.stance_low_label == english.stance_low_label
    assert english.seed == hindi.seed

    en_dir, hi_dir = tmp_path / "en", tmp_path / "hi"
    await build_orchestrator(english, bank_gateway, runs_dir=en_dir).run()
    await build_orchestrator(hindi, bank_gateway, runs_dir=hi_dir).run()

    en_rows = sorted(_rows(en_dir, "rq1_run"), key=lambda r: (r.turn, r.speaker_agent_id))
    hi_rows = sorted(_rows(hi_dir, "rq1_run"), key=lambda r: (r.turn, r.speaker_agent_id))

    # Identical neighbour samples.
    assert [tuple(r.neighbor_agent_ids) for r in en_rows] == [
        tuple(r.neighbor_agent_ids) for r in hi_rows
    ]
    # Identical posted items, and identical feeds.
    assert [r.posted_item_id for r in en_rows] == [r.posted_item_id for r in hi_rows]
    assert [tuple(r.items_read) for r in en_rows] == [tuple(r.items_read) for r in hi_rows]
    # Identical graph.
    assert (en_dir / "rq1_run" / "graph_edges.csv").read_text() == (
        hi_dir / "rq1_run" / "graph_edges.csv"
    ).read_text()


async def test_all_four_arms_of_a_matched_set_agree(tmp_path, rq1_config, bank_gateway):
    """Extended to all four conditions, including romanized_hindi, which is
    the condition this work added."""
    signatures = {}
    for condition in ("english", "hinglish", "romanized_hindi", "hindi"):
        run = load_run_config(rq1_config(f"{condition}.yaml", language_condition=condition))
        where = tmp_path / condition
        await build_orchestrator(run, bank_gateway, runs_dir=where).run()
        signatures[condition] = [
            (r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids),
             r.posted_item_id, tuple(r.items_read))
            for r in sorted(_rows(where, "rq1_run"), key=lambda r: (r.turn, r.speaker_agent_id))
        ]

    baseline = signatures["english"]
    for condition, signature in signatures.items():
        assert signature == baseline, f"{condition} diverged from english"


async def test_the_arms_do_differ_in_the_text_the_model_was_shown(
    tmp_path, rq1_config, bank_gateway
):
    """The other half of the matched-set claim: identical structure is only
    interesting if the rendering genuinely differs. A bug that rendered every
    arm in English would pass every assertion above."""
    from sandbox.bank_manager import BankManager

    bank = BankManager("data/bank/test_bank.csv")
    prompts = {}
    for condition in ("english", "hindi"):
        bank_gateway.generate.reset_mock()
        run = load_run_config(rq1_config(f"{condition}.yaml", language_condition=condition))
        await build_orchestrator(run, bank_gateway, runs_dir=tmp_path / condition).run()
        prompts[condition] = [c.args[0] for c in bank_gateway.generate.await_args_list]

    english_text = bank.text("bank_t01", "english")
    hindi_text = bank.text("bank_t01", "hindi")

    assert any(english_text in p for p in prompts["english"])
    assert not any(hindi_text in p for p in prompts["english"])
    assert any(hindi_text in p for p in prompts["hindi"])
    assert not any(english_text in p for p in prompts["hindi"])


# --- 12.3: resume equivalence (the step 9.5 hard requirement) ------------


async def test_a_resumed_bank_run_matches_an_uninterrupted_one(
    tmp_path, rq1_config, bank_gateway
):
    """Killed after round 2 of 4, resumed, and byte-equal on every column
    that is a property of the run rather than of one execution.

    Stronger than the generated regime's version of this test, which compares
    four fields: this compares the whole row except interaction_id, timestamp
    and latency, the probe distribution and the posted item included. It has
    to, because the bank regime carries two pieces of state that are NOT in
    the checkpoint -- each sender's position in its bank cycle, and the rounds
    each agent remembers -- both rebuilt from interactions.jsonl. A resume
    that restarted either would produce a run that completes, looks valid,
    and posts different arguments from the one it was continuing.
    """
    from tests.factories import make_backend_response

    clean = tmp_path / "clean"
    await build_orchestrator(
        load_run_config(rq1_config("clean.yaml", K=4)), bank_gateway, runs_dir=clean
    ).run()

    # Round 3 fails for EVERY agent, so the turn raises before any row is
    # written. A partial round would leave rows with no checkpoint, which is
    # the known re-logging gap and a different scenario from this one.
    crashed = tmp_path / "crashed"
    bank_gateway.generate.side_effect = [
        make_backend_response(text="The cost point is the one that lands for me.")
    ] * 16 + [RuntimeError("boom")] * 100
    with pytest.raises(RuntimeError):
        await build_orchestrator(
            load_run_config(rq1_config("crashed.yaml", K=4)), bank_gateway, runs_dir=crashed
        ).run()
    assert {r.turn for r in _rows(crashed, "rq1_run")} == {0, 1, 2}

    bank_gateway.generate.side_effect = None
    bank_gateway.generate.return_value = make_backend_response(
        text="The cost point is the one that lands for me."
    )
    await build_orchestrator(
        load_run_config(rq1_config("crashed.yaml", K=4)), bank_gateway, runs_dir=crashed
    ).run()

    assert _signature(_rows(crashed, "rq1_run")) == _signature(_rows(clean, "rq1_run"))


async def test_a_resumed_run_keeps_its_graph_edges_file(tmp_path, rq1_config, bank_gateway):
    from tests.factories import make_backend_response

    bank_gateway.generate.side_effect = [RuntimeError("boom")] * 100
    with pytest.raises(RuntimeError):
        await build_orchestrator(
            load_run_config(rq1_config(K=2)), bank_gateway, runs_dir=tmp_path
        ).run()
    first = (tmp_path / "rq1_run" / "graph_edges.csv").read_text()

    bank_gateway.generate.side_effect = None
    bank_gateway.generate.return_value = make_backend_response(text="a reply")
    await build_orchestrator(
        load_run_config(rq1_config(K=2)), bank_gateway, runs_dir=tmp_path
    ).run()

    assert (tmp_path / "rq1_run" / "graph_edges.csv").read_text() == first


# --- 12.4: delivery: off, the E4 control ---------------------------------


NEUTRAL_BANK = "data/bank/test_neutral.csv"


async def test_the_shipped_delivery_off_config_loads(tmp_path):
    """configs/rq1/smoke_off.yaml is the worked example of the E4 control,
    and its neutral bank is validated in full at load like the argument
    bank is."""
    from sandbox.bank_manager import NEUTRAL_PREFIX, NeutralBank

    run = load_run_config(Path("configs/rq1/smoke_off.yaml"))
    assert run.delivery == "off"
    assert run.neutral_bank_path and run.neutral_bank_path.strip()
    ids = NeutralBank(run.neutral_bank_path).ids()
    assert ids and all(i.startswith(NEUTRAL_PREFIX) for i in ids)


async def test_the_delivery_off_control_reads_n_neutral_messages(
    tmp_path, rq1_config, bank_gateway
):
    """It makes the same calls a treatment arm does, reads N side-free
    messages every round, and never sees a bank argument.

    N, not one. The control used to read a single fixed message against the
    treatment's N, so the two arms differed in how much was read and in how
    often something was read twice, as well as in persuasive content.
    """
    from sandbox.bank_manager import NEUTRAL_PREFIX, BankManager

    run = load_run_config(
        rq1_config(delivery="off", neutral_bank_path=NEUTRAL_BANK)
    )
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    assert bank_gateway.generate.await_count == run.M * run.K == 24
    assert bank_gateway.score_next_token.await_count == run.M * (run.K + 1) == 32

    rows = _rows(tmp_path, "rq1_run")
    assert len(rows) == 32
    assert all(r.probe_mean is not None for r in rows)
    for row in rows:
        if row.turn == 0:
            assert row.items_read == [] and row.reply_text == ""
            assert row.shadow_items == []
        else:
            assert len(row.items_read) == run.N == 3
            assert all(i.startswith(NEUTRAL_PREFIX) for i in row.items_read)
            assert len(row.shadow_items) == run.N
            assert row.reply_text

    # The audit, against the file a real run leaves behind rather than only
    # against the mock's call list: no argument from the bank, in any
    # language, reaches any prompt of the control arm.
    bank = BankManager(run.bank_path)
    arguments = [
        bank.text(item_id, language)
        for item_id in bank.all_item_ids()
        for language in LANGUAGE_CONDITIONS
    ]
    logged = _prompt_records(tmp_path, "rq1_run")
    assert len(logged) == run.M * run.K + run.M * (run.K + 1)
    for record in logged:
        assert all(argument not in record.prompt_text for argument in arguments)
    for record in (r for r in logged if r.call == "reply"):
        assert record.prompt_text.count("A user posted:") == run.N


async def test_the_control_feed_mirrors_the_treatment_feed(
    tmp_path, rq1_config, bank_gateway
):
    """Same shape, no arguments.

    With the probe driven to one constant reading in both arms, the sides and
    therefore the bank items match, so the control's shadow_items must equal
    the treatment's items_read row for row -- including the repeats, which is
    the part a single fixed message could not reproduce. The stand-in is a
    pure function of the item, so equal items give equal messages and the
    control reads the same thing twice exactly when the treatment would have.
    """
    on, off = tmp_path / "on", tmp_path / "off"
    await build_orchestrator(
        load_run_config(rq1_config("on.yaml")), bank_gateway, runs_dir=on
    ).run()
    await build_orchestrator(
        load_run_config(
            rq1_config("off.yaml", delivery="off", neutral_bank_path=NEUTRAL_BANK)
        ),
        bank_gateway,
        runs_dir=off,
    ).run()

    treatment = {
        (r.turn, r.speaker_agent_id): r for r in _rows(on, "rq1_run")
    }
    control = {(r.turn, r.speaker_agent_id): r for r in _rows(off, "rq1_run")}
    assert set(treatment) == set(control)

    repeats_seen = 0
    for key, t_row in treatment.items():
        c_row = control[key]
        assert c_row.shadow_items == t_row.items_read, key
        assert c_row.neighbor_agent_ids == t_row.neighbor_agent_ids, key
        assert c_row.posted_item_id == t_row.posted_item_id, key
        # Equal items map to equal messages, so the repeat structure matches.
        for i, j in itertools.combinations(range(len(t_row.items_read)), 2):
            same_item = t_row.items_read[i] == t_row.items_read[j]
            assert (c_row.items_read[i] == c_row.items_read[j]) == same_item
            repeats_seen += same_item

    # Not vacuous: the treatment really did deliver the same argument twice
    # in some round, which is the case a single fixed message got wrong in
    # one direction and N distinct messages would get wrong in the other.
    assert repeats_seen > 0


async def test_the_control_and_the_treatment_share_their_draws(
    tmp_path, rq1_config, bank_gateway
):
    """So the control differs from the treatment in what agents SAW and in
    nothing else. If the two arms drew different neighbours, the control's
    drift would be a baseline for a different run."""
    on = tmp_path / "on"
    off = tmp_path / "off"
    await build_orchestrator(
        load_run_config(rq1_config("on.yaml")), bank_gateway, runs_dir=on
    ).run()
    await build_orchestrator(
        load_run_config(
            rq1_config("off.yaml", delivery="off", neutral_bank_path=NEUTRAL_BANK)
        ),
        bank_gateway,
        runs_dir=off,
    ).run()

    def draws(where):
        return [
            (r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids), r.posted_item_id)
            for r in sorted(_rows(where, "rq1_run"), key=lambda r: (r.turn, r.speaker_agent_id))
        ]

    assert draws(on) == draws(off)
    assert (on / "rq1_run" / "graph_edges.csv").read_text() == (
        off / "rq1_run" / "graph_edges.csv"
    ).read_text()


# --- the instrument is identical across arms -----------------------------

ARM_QUESTIONS = {
    "english": "Should city bus fares stay subsidised?",
    "hinglish": "Kya city bus fares subsidised rehne chahiye?",
    "romanized_hindi": "Kya shahar ki bus kiraya subsidy jaari rehni chahiye?",
    "hindi": "क्या शहर की बस किराया सब्सिडी जारी रहनी चाहिए?",
}


async def test_the_four_arms_differ_only_in_bank_text_question_and_directive(
    tmp_path, rq1_config, bank_gateway
):
    """The validity claim RQ1 rests on, checked against whole runs.

    Exactly three things may vary across the arms: the bank text, the probe
    question, and the one directive line. Everything else the model reads --
    persona, topic, the lean sentence, the anchor labels, every heading, the
    reply instruction, the scale line -- must be byte-identical, or a
    difference between arms is a difference between four instruments as much
    as between four languages.

    The probe is driven to a constant reading, which holds the sides and
    therefore the bank items identical across arms; in a real run they
    diverge as soon as the readings do, and that divergence is the result.
    """
    from sandbox.bank_manager import BankManager
    from sandbox.prompt_builder import LANGUAGE_DIRECTIVES

    bank = BankManager("data/bank/test_bank.csv")
    all_bank_texts = [
        bank.text(item_id, language)
        for side in ("pro", "con")
        for item_id in bank.items_for_side(side)
        for language in LANGUAGE_DIRECTIVES
    ]

    def is_variable(line: str) -> bool:
        """True for a line carrying one of the three things that may vary."""
        if line.strip() in {d.strip() for d in LANGUAGE_DIRECTIVES.values()}:
            return True
        if any(question in line for question in ARM_QUESTIONS.values()):
            return True
        return any(text in line for text in all_bank_texts)

    def skeleton(prompt: str) -> list[str]:
        return [line for line in prompt.split("\n") if not is_variable(line)]

    replies, probes = {}, {}
    for condition, question in ARM_QUESTIONS.items():
        bank_gateway.generate.reset_mock()
        bank_gateway.score_next_token.reset_mock()
        run = load_run_config(
            rq1_config(
                f"{condition}.yaml",
                K=2,
                language_condition=condition,
                probe_question=question,
            )
        )
        # The anchor labels are English in every arm, straight from the
        # shipped config: the decision is in the configs, not in this test.
        assert run.stance_low_label == "fares should not be subsidised"
        assert run.stance_high_label == "fares should stay subsidised"

        await build_orchestrator(run, bank_gateway, runs_dir=tmp_path / condition).run()
        replies[condition] = [c.args[0] for c in bank_gateway.generate.await_args_list]
        probes[condition] = [
            c.args[0] for c in bank_gateway.score_next_token.await_args_list
        ]

    baseline = "english"
    for condition in ARM_QUESTIONS:
        assert len(replies[condition]) == len(replies[baseline])
        assert len(probes[condition]) == len(probes[baseline])
        for prompts in (replies, probes):
            for mine, theirs in zip(prompts[condition], prompts[baseline], strict=True):
                assert skeleton(mine) == skeleton(theirs), (
                    f"{condition} differs from {baseline} outside the bank text, "
                    f"the question and the directive:\n{mine}\n---\n{theirs}"
                )

    # The filter must not be swallowing the whole prompt, or the comparison
    # above would hold for any two strings.
    assert skeleton(replies["hindi"][0]), replies["hindi"][0]
    assert len(skeleton(replies["hindi"][0])) < len(replies["hindi"][0].split("\n"))

    # And the arms genuinely do differ where they are allowed to.
    assert replies["hindi"][0] != replies["english"][0]
    assert ARM_QUESTIONS["hindi"] in probes["hindi"][0]
    assert ARM_QUESTIONS["hindi"] not in probes["english"][0]


async def test_the_lean_and_the_scale_line_are_english_in_every_arm(
    tmp_path, rq1_config, bank_gateway
):
    """The two anchor labels sit inside English sentences, so they stay
    English: a translated clause in an English frame would code-mix the
    instrument and change it between arms at once."""
    lean_lines, scale_lines = set(), set()
    for condition, question in ARM_QUESTIONS.items():
        bank_gateway.score_next_token.reset_mock()
        run = load_run_config(
            rq1_config(
                f"{condition}.yaml",
                K=1,
                language_condition=condition,
                probe_question=question,
            )
        )
        await build_orchestrator(run, bank_gateway, runs_dir=tmp_path / condition).run()

        probe = [c.args[0] for c in bank_gateway.score_next_token.await_args_list][0]
        lines = [line for line in probe.split("\n") if line.strip()]
        lean_lines.add(next(line for line in lines if "the view that" in line))
        scale_lines.add(next(line for line in lines if line.startswith("On this scale")))

    assert len(lean_lines) == 1, lean_lines
    assert len(scale_lines) == 1, scale_lines
    assert not any(
        "\u0900" <= ch <= "\u097f" for ch in "".join(lean_lines | scale_lines)
    )


# --- 12.5: the prompt audit ----------------------------------------------


async def test_the_prompt_audit(tmp_path, rq1_config, bank_gateway):
    """Every prompt the model received in a whole run, checked for leakage.

    Three claims, in increasing order of how badly a violation would damage
    RQ1:

      1. no prompt mentions STANCE -- the bank regime parses nothing, so the
         generated regime's output contract must not have followed it here;
      2. no prompt shows a position number or an agent id in its stimulus --
         an agent that can see one does not have to read the arguments;
      3. no probe text ever appears in a later reply prompt -- the probe is a
         measurement, and a measurement that entered the agent's history
         would have become part of the stimulus.
    """
    run = load_run_config(rq1_config(K=3))
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    replies = [c.args[0] for c in bank_gateway.generate.await_args_list]
    probes = [c.args[0] for c in bank_gateway.score_next_token.await_args_list]
    assert len(replies) == run.M * run.K
    assert len(probes) == run.M * (run.K + 1)

    for prompt in replies + probes:
        # 1. No stance contract anywhere.
        assert "STANCE" not in prompt
        # 2a. No agent ids, and nothing calling itself a position.
        assert "agent_" not in prompt
        assert "position" not in prompt.lower()

    directive = "Write your reply in English."
    for prompt in replies:
        # 2b. The stimulus -- persona, topic, lean, memory, feed -- carries no
        # digit at all. The reply instruction after the directive says "1 to 2
        # sentences", which is a formatting instruction, not a position.
        stimulus = prompt.split(directive)[0]
        assert not any(ch.isdigit() for ch in stimulus), stimulus

    for prompt in probes:
        # 2c. Same for the probe, up to the question. The anchor and
        # instruction lines after it must name 1 and 7: the model cannot
        # answer "a single digit from 1 to 7" without being told the range,
        # and neither digit is anybody's position.
        stimulus = prompt.split(run.probe_question)[0]
        assert not any(ch.isdigit() for ch in stimulus), stimulus
        assert "Answer with a single digit from 1 to 7" in prompt

    # 3. Nothing that is unique to a probe may appear in a reply prompt.
    for prompt in replies:
        assert run.probe_question not in prompt
        assert "Answer with a single digit" not in prompt
        assert "On this scale," not in prompt


async def test_the_lean_sentence_reaches_every_reply_and_every_probe(
    tmp_path, rq1_config, bank_gateway
):
    """It is the one place an agent is told where it started, and it is told
    in words -- in every round, so that round 0 and round k are measured by
    the same instrument. Shown at round 0 and round 1 only, the step between
    the first two readings was partly the removal of a sentence."""
    run = load_run_config(rq1_config(K=3))
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    replies = [c.args[0] for c in bank_gateway.generate.await_args_list]
    probes = [c.args[0] for c in bank_gateway.score_next_token.await_args_list]
    assert len(replies) == run.M * run.K
    assert len(probes) == run.M * (run.K + 1)

    # The lean sits on the line directly after the topic line, in every
    # prompt. Matched as a whole sentence: the midpoint one ("undecided
    # between the view that X and the view that Y") contains the shared
    # fragment twice, so a fragment count cannot tell one lean from two.
    from sandbox.prompt_builder import LEAN_SENTENCES

    expected = {
        sentence.format(low=run.stance_low_label, high=run.stance_high_label)
        for sentence in LEAN_SENTENCES.values()
    }
    for prompt in replies + probes:
        lines = [line for line in prompt.split("\n") if line.strip()]
        topic = next(
            i for i, line in enumerate(lines) if line.startswith("You are taking part")
        )
        assert lines[topic + 1] in expected, prompt


# --- exposure counts ------------------------------------------------------


def _exposure_row(turn, items, *, agent="agent_0000", status="success", run_id="r1",
                  row_index=None):
    """One interactions.jsonl row, as a dict, for a hand-built log."""
    return {
        "run_id": run_id,
        "turn": turn,
        "speaker_agent_id": agent,
        "neighbor_agent_ids": ["agent_0001"],
        "stance_before": 4.0,
        "stance_after": 4.0,
        "reason_text": "",
        "content_type": "bank_item",
        "api_call_status": status,
        "items_read": list(items),
        "regime": "bank",
        "row_index": row_index if row_index is not None else turn,
        "is_duplicate": False,
    }


def _exposure_frame(rows):
    pd = pytest.importorskip("pandas")
    import export_csv

    frame = pd.DataFrame(rows)
    frame["is_valid_for_analysis"] = frame.api_call_status != "failed_logged_null"
    return export_csv._derive_exposure_columns(frame, frame["items_read"])


def test_exposure_counts_over_four_rounds_with_known_repeats():
    """One agent, hand-built, so the expected numbers are read off the log
    rather than recomputed by the same code that produced them.

      round 0  reads nothing
      round 1  a, b        both new
      round 2  b, c        b is a repeat, seen once before
      round 3  b, d        b again, seen twice before
    """
    frame = _exposure_frame([
        _exposure_row(0, []),
        _exposure_row(1, ["a", "b"]),
        _exposure_row(2, ["b", "c"]),
        _exposure_row(3, ["b", "d"]),
    ]).set_index("turn")

    assert list(frame["n_repeat_reads"]) == [0, 0, 1, 1]
    assert list(frame["max_prior_exposures"]) == [0, 0, 1, 2]
    assert list(frame["n_distinct_items_read_so_far"]) == [0, 2, 3, 4]


def test_a_failed_row_in_the_middle_does_not_count_as_an_exposure():
    """A failed round was logged and never applied, so the agent's memory
    does not hold it. Counting it would put the exposure history out of step
    with the prompt the agent actually saw."""
    frame = _exposure_frame([
        _exposure_row(1, ["a"]),
        _exposure_row(2, ["a"], status="failed_logged_null"),
        _exposure_row(3, ["a"]),
    ]).set_index("turn")

    # Round 3 sees "a" as read once, not twice: round 2 did not happen.
    assert frame.loc[3, "n_repeat_reads"] == 1
    assert frame.loc[3, "max_prior_exposures"] == 1
    assert frame.loc[3, "n_distinct_items_read_so_far"] == 1
    # And the failed row itself is null, like every other measured column.
    for column in export_csv_exposure_columns():
        assert pd_isna(frame.loc[2, column]), column


def test_an_item_read_twice_in_one_round_is_not_a_repeat_of_itself():
    """Both counts are taken against the state before the round. Two
    neighbours posting the same argument is one round's stimulus, not a
    second exposure within it -- but it does count twice toward later
    rounds, because the agent read it twice."""
    frame = _exposure_frame([
        _exposure_row(1, ["a", "a"]),
        _exposure_row(2, ["a"]),
    ]).set_index("turn")

    assert frame.loc[1, "n_repeat_reads"] == 0
    assert frame.loc[1, "max_prior_exposures"] == 0
    assert frame.loc[1, "n_distinct_items_read_so_far"] == 1
    assert frame.loc[2, "max_prior_exposures"] == 2


def test_exposure_counts_are_per_agent_and_per_run():
    """Two agents reading the same item have separate histories, and so do
    two runs: an exposure is something that happened to one agent."""
    frame = _exposure_frame([
        _exposure_row(1, ["a"], agent="agent_0000"),
        _exposure_row(1, ["a"], agent="agent_0001"),
        _exposure_row(2, ["a"], agent="agent_0000"),
        _exposure_row(1, ["a"], agent="agent_0000", run_id="r2"),
    ])

    by_key = {
        (r.run_id, r.speaker_agent_id, r.turn): r for r in frame.itertuples()
    }
    assert by_key[("r1", "agent_0000", 2)].n_repeat_reads == 1
    assert by_key[("r1", "agent_0001", 1)].n_repeat_reads == 0
    assert by_key[("r2", "agent_0000", 1)].n_repeat_reads == 0


def test_a_superseded_duplicate_row_is_not_counted():
    """A resumed run logs the re-run round twice and the later write is the
    one the run used. Counting both would show an exposure the agent never
    had."""
    rows = [
        _exposure_row(1, ["a"], row_index=0),
        _exposure_row(2, ["a"], row_index=1),   # superseded below
        _exposure_row(2, ["a"], row_index=2),   # the resumed write
    ]
    rows[1]["is_duplicate"] = True
    frame = _exposure_frame(rows)

    surviving = frame[~frame.is_duplicate].set_index("turn")
    assert surviving.loc[2, "n_repeat_reads"] == 1
    assert surviving.loc[2, "max_prior_exposures"] == 1
    assert pd_isna(frame[frame.is_duplicate].iloc[0]["n_repeat_reads"])


def test_a_generated_run_gets_null_exposure_counts():
    """It has no items_read to count, and a 0 would claim a measurement that
    was never made."""
    pd = pytest.importorskip("pandas")
    import export_csv

    frame = pd.DataFrame([
        {**_exposure_row(1, []), "regime": "generated"},
        {**_exposure_row(2, []), "regime": "generated"},
    ])
    frame["is_valid_for_analysis"] = True
    out = export_csv._derive_exposure_columns(frame, frame["items_read"])

    for column in export_csv_exposure_columns():
        assert out[column].isna().all(), column


def export_csv_exposure_columns():
    import export_csv

    return export_csv.EXPOSURE_COLUMNS


def pd_isna(value) -> bool:
    import pandas as pd

    return bool(pd.isna(value))


async def test_a_real_bank_run_exports_its_exposure_counts(
    tmp_path, rq1_config, bank_gateway
):
    """Through derive(), not the helper directly: the columns have to arrive
    in the exported frame, not merely be computable."""
    pytest.importorskip("pandas")

    import export_csv

    run = load_run_config(rq1_config(K=4))
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    frame = export_csv.derive(
        export_csv.mark_duplicates(export_csv.load_run(tmp_path / run.run_id))
    )

    round_zero = frame[frame.turn == 0]
    later = frame[frame.turn > 0]
    for column in export_csv.EXPOSURE_COLUMNS:
        assert column in frame
        assert (round_zero[column] == 0).all(), column
        assert later[column].notna().all(), column

    # Monotone per agent: a distinct-item count cannot go down, and cannot
    # exceed the number of items the bank has.
    bank_size = len(BankManager(run.bank_path).all_item_ids())
    for _, rows in frame.sort_values("turn").groupby("speaker_agent_id"):
        seen = list(rows["n_distinct_items_read_so_far"])
        assert seen == sorted(seen)
        assert max(seen) <= bank_size

    # Not vacuous: with a 4-item bank over 4 rounds of N=3 reads, somebody
    # read something twice.
    assert (later["n_repeat_reads"] > 0).any()
    assert (later["max_prior_exposures"] > 0).any()


def test_exposure_counts_require_mark_duplicates_to_have_run():
    """Assuming a frame with no is_duplicate column is duplicate-free would
    silently double-count a resumed run's re-logged round, which is the one
    case these columns most need to get right."""
    pd = pytest.importorskip("pandas")

    import export_csv

    frame = pd.DataFrame([_exposure_row(1, ["a"])]).drop(columns=["is_duplicate"])
    frame["is_valid_for_analysis"] = True
    with pytest.raises(ValueError, match="is_duplicate"):
        export_csv._derive_exposure_columns(frame, frame["items_read"])
