"""Fix 9: the similarity-recommender neighbour strategy
(neighbor_strategy="graph_similarity"), end-to-end against a mocked gateway.

GraphSimilaritySampling's own selection logic (threshold filtering,
ranking, ties, the fallback, feed_info) is unit-tested in
tests/test_interaction_engine.py. This file covers what only the
orchestrator can exercise: the empty-feed path (no reply call, a probe
still runs, a distinct api_call_status), and that feed_size /
n_neighbors_over_threshold / recsys_fallback land correctly on the logged
Interaction.

graph_rewire_p=0.0 throughout: at p=0, WS(M, degree, 0) is a plain ring
lattice (node i connects to i-1..i-degree/2 and i+1..i+degree/2, mod M),
which is deterministic regardless of graph_seed -- so which agents are
graph-neighbours of which is known at test-writing time without having to
introspect a randomly rewired graph first.
"""

from __future__ import annotations

import json
from pathlib import Path

from sandbox.models import ExperimentRun, Interaction
from sandbox.simulation_orchestrator import build_orchestrator
from tests.factories import make_logprob_response, make_run

FIXTURE_BANK = "data/bank/test_bank.csv"


def _recsys_run(**overrides) -> ExperimentRun:
    """M=6, degree=2, N=2: a ring where agent_0000's only two graph
    neighbours are agent_0001 and agent_0005 (the wrap-around), with
    graph_rewire_p=0.0 so that stays true regardless of seed."""
    base = dict(
        M=6, N=2, K=1, run_id="recsys_run", persona_pool_id="pool_20",
        neighbor_strategy="graph_similarity", graph_degree=2, graph_rewire_p=0.0,
        recsys_threshold=0.5,
        regime="bank", bank_path=FIXTURE_BANK,
        probe_question="Should the fare subsidy continue?",
        stance_low_label="the subsidy should end",
        stance_high_label="the subsidy should continue",
        checkpoint_every_n_turns=1,
    )
    base.update(overrides)
    return make_run(**base)


def _rows(tmp_path, run_id="recsys_run") -> list[Interaction]:
    path = tmp_path / run_id / "interactions.jsonl"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [Interaction.model_validate_json(line) for line in lines]


async def test_an_agent_whose_neighbours_are_both_over_threshold_gets_no_feed(
    tmp_path, bank_gateway
):
    """Round 0: agent_0000 reads mean=4 (x=0.0); its two ring neighbours,
    agent_0001 and agent_0005, read mean=7 (x=1.0) -- 1.0 away, over the
    0.5 threshold. Round 1's feed for agent_0000 must come up empty: no
    reply call, a probe that still runs, and api_call_status="no_feed"."""
    run = _recsys_run()
    round0_means = [4, 7, 4, 4, 4, 7]  # agent_0000..agent_0005, sorted order
    round1_means = [5, 5, 5, 5, 5, 5]
    bank_gateway.score_next_token.side_effect = [
        make_logprob_response(mean=m) for m in round0_means + round1_means
    ]

    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rows = _rows(tmp_path, run.run_id)
    round1 = {r.speaker_agent_id: r for r in rows if r.turn == 1}
    row = round1["agent_0000"]

    assert row.api_call_status == "no_feed"
    assert row.is_valid_for_analysis  # a real, applicable round -- not a failure
    assert row.neighbor_agent_ids == []
    assert row.items_read == []
    assert row.reply_text == ""
    assert row.reply_seed is None
    assert row.feed_size == 0
    assert row.n_neighbors_over_threshold == 2
    assert row.recsys_fallback is False
    # No reply call happened for this agent, so no reply was ever dispatched
    # to be truncated -- the field means exactly what it means on round 0.
    assert row.reply_truncated is None
    # The probe still ran: a reading was taken, not fabricated (Fix I).
    assert row.probe_mean == 5.0


async def test_a_non_empty_feed_still_carries_the_new_fields(tmp_path, bank_gateway):
    """A reader whose neighbours ARE within threshold gets an ordinary
    "success" row, but still carries feed_size/n_neighbors_over_threshold/
    recsys_fallback -- these are not special to the empty-feed path."""
    run = _recsys_run()
    # agent_0002's ring neighbours are agent_0001 and agent_0003. Keep every
    # round-0 reading equal (mean=4 for everyone) so nothing is filtered.
    round0_means = [4, 4, 4, 4, 4, 4]
    round1_means = [5, 5, 5, 5, 5, 5]
    bank_gateway.score_next_token.side_effect = [
        make_logprob_response(mean=m) for m in round0_means + round1_means
    ]

    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rows = _rows(tmp_path, run.run_id)
    round1 = {r.speaker_agent_id: r for r in rows if r.turn == 1}
    row = round1["agent_0002"]

    assert row.api_call_status == "success"
    assert row.feed_size == 2
    assert row.n_neighbors_over_threshold == 0
    assert row.recsys_fallback is False
    assert len(row.neighbor_agent_ids) == 2


async def test_a_reader_with_no_valid_round_zero_probe_falls_back(tmp_path, bank_gateway):
    """agent_0000's own round-0 probe fails (mass below floor): its
    stance_history stays empty, so round 1 falls back to random graph
    neighbours instead of ranking by similarity -- and says so."""
    run = _recsys_run()
    # agent_0000's round-0 probe answers off-scale (mass floor, default 0.5,
    # fails outright): off_scale=1.0 puts ALL mass off the digits.
    round0 = [
        make_logprob_response(off_scale=1.0, probabilities={4: 1.0}),
        make_logprob_response(mean=4),
        make_logprob_response(mean=4),
        make_logprob_response(mean=4),
        make_logprob_response(mean=4),
        make_logprob_response(mean=4),
    ]
    round1_means = [5, 5, 5, 5, 5, 5]
    bank_gateway.score_next_token.side_effect = round0 + [
        make_logprob_response(mean=m) for m in round1_means
    ]

    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rows = _rows(tmp_path, run.run_id)
    round0_rows = {r.speaker_agent_id: r for r in rows if r.turn == 0}
    round1_rows = {r.speaker_agent_id: r for r in rows if r.turn == 1}

    assert round0_rows["agent_0000"].api_call_status == "failed_logged_null"
    row = round1_rows["agent_0000"]
    assert row.recsys_fallback is True
    assert row.feed_size == 2  # N=2, satisfied by the ring's own 2 neighbours
    assert row.n_neighbors_over_threshold is None


def test_graph_similarity_requires_bank_regime():
    import pytest

    with pytest.raises(ValueError, match="requires regime='bank'"):
        make_run(
            regime="generated", neighbor_strategy="graph_similarity",
            N=2, graph_degree=4, M=10,
        )


def test_graph_similarity_rejects_rewire_rate():
    import pytest

    with pytest.raises(ValueError, match="requires neighbor_strategy='graph'"):
        make_run(
            regime="bank", neighbor_strategy="graph_similarity",
            rewire_rate=0.1, N=2, graph_degree=4, M=10,
            bank_path=FIXTURE_BANK, probe_question="q?",
        )


def test_recsys_threshold_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        make_run(recsys_threshold=0.0)
    with pytest.raises(ValueError):
        make_run(recsys_threshold=-0.5)


def test_a_valid_graph_similarity_config_constructs():
    run = make_run(
        regime="bank", neighbor_strategy="graph_similarity",
        N=2, graph_degree=4, M=10, recsys_threshold=1.5,
        bank_path=FIXTURE_BANK, probe_question="q?",
    )
    assert run.neighbor_strategy == "graph_similarity"
    assert run.recsys_threshold == 1.5
