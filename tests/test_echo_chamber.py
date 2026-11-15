"""E5: echo-chamber formation (rewiring on top of the graph strategy).

Four groups of tests, bottom-up:

    * ExperimentRun validation (rewire_rate ties to regime/neighbor_strategy)
    * GraphSampling's new surface: rewire, degree, neighbors_of, restore_edges
    * AgentManager's balanced lean assignment
    * the orchestrator's rewiring rule end-to-end, against a mocked gateway

Row comparisons across two runs never compare interaction_id or
timestamp_utc, which are random/wall-clock on every run by construction
(see test_simulation_orchestrator.py's own `signature()` helpers) --
rewires.jsonl and graph_edges_by_round.csv carry neither, so those two are
compared as plain text.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from sandbox.agent_manager import AgentManager
from sandbox.interaction_engine import GraphSampling
from sandbox.model_gateway import ModelGateway
from sandbox.models import ExperimentRun, Interaction
from sandbox.seed_manager import SeedManager
from sandbox.simulation_orchestrator import build_orchestrator
from tests.factories import make_agent, make_backend_response, make_logprob_response, make_run

# asyncio_mode = "auto" (pyproject.toml): async def tests run without any
# marker, so this file needs none -- see CLAUDE.md.


# --- ExperimentRun validation ---------------------------------------------


def test_rewire_rate_requires_bank_regime():
    with pytest.raises(ValueError, match="requires regime='bank'"):
        make_run(
            rewire_rate=0.1, regime="generated", neighbor_strategy="graph",
            graph_degree=4, N=2, M=8,
        )


def test_rewire_rate_requires_graph_strategy():
    with pytest.raises(ValueError, match="requires neighbor_strategy='graph'"):
        make_run(
            rewire_rate=0.1, regime="bank", neighbor_strategy="alpha",
            bank_path="data/bank/test_bank.csv", probe_question="Q?",
        )


def test_rewire_rate_zero_needs_neither():
    """0.0 is "off", so it is not held to the same requirements as a
    positive rate -- an alpha-sampling, generated-regime config with
    rewire_rate left at its default must keep validating."""
    run = make_run(regime="generated", neighbor_strategy="alpha")
    assert run.rewire_rate == 0.0
    assert run.rewire_target == "same_side"
    assert run.lean_assignment == "random"


def test_a_valid_echo_chamber_config_constructs():
    run = make_run(
        rewire_rate=0.3, rewire_target="random", lean_assignment="balanced",
        regime="bank", neighbor_strategy="graph", graph_degree=4, N=2, M=8,
        bank_path="data/bank/test_bank.csv", probe_question="Q?",
    )
    assert run.rewire_rate == 0.3
    assert run.rewire_target == "random"


def test_config_loader_surfaces_the_rewire_requirement(tmp_path):
    """The validator runs through config_loader too, not merely through a
    bare ExperimentRun(**kwargs) call -- ConfigLoadError wraps it the same
    way it wraps every other schema error."""
    import yaml

    from sandbox.config_loader import ConfigLoadError, load_run_config

    raw = yaml.safe_load(Path("configs/smoke_test.yaml").read_text())
    raw["rewire_rate"] = 0.2  # smoke_test.yaml is generated/alpha
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigLoadError, match="rewire_rate > 0 requires"):
        load_run_config(path)


def test_rewire_is_a_known_seed_manager_purpose():
    seeds = SeedManager(1)
    # Raises for an unknown purpose, so this is the contract check that
    # "rewire" is actually wired into _TURN_PURPOSES rather than merely
    # planned.
    rng = seeds.turn_rng("rewire", 1)
    assert isinstance(rng, random.Random)


# --- GraphSampling: rewire / degree / neighbors_of / restore_edges --------


def _ids(n: int) -> list[str]:
    return [f"agent_{i:04d}" for i in range(n)]


def _graph(n=10, degree=4, rewire_p=0.1, k=2, graph_seed=42) -> GraphSampling:
    return GraphSampling(_ids(n), degree=degree, rewire_p=rewire_p, k=k, graph_seed=graph_seed)


def test_rewire_moves_one_edge_and_keeps_the_readers_degree():
    strategy = _graph()
    reader = "agent_0000"
    old_neighbor = strategy.neighbors_of(reader)[0]
    non_neighbor = next(
        a for a in _ids(10) if a != reader and a not in strategy.neighbors_of(reader)
    )
    before_degree = strategy.degree(reader)
    before_old_degree = strategy.degree(old_neighbor)
    before_new_degree = strategy.degree(non_neighbor)

    strategy.rewire((reader, old_neighbor), (reader, non_neighbor))

    assert strategy.degree(reader) == before_degree  # Design point 3
    assert old_neighbor not in strategy.neighbors_of(reader)
    assert non_neighbor in strategy.neighbors_of(reader)
    assert reader not in strategy.neighbors_of(old_neighbor)
    assert reader in strategy.neighbors_of(non_neighbor)
    assert strategy.degree(old_neighbor) == before_old_degree - 1
    assert strategy.degree(non_neighbor) == before_new_degree + 1


def test_rewire_rejects_dropping_a_non_neighbour():
    strategy = _graph()
    non_neighbor = next(
        a for a in _ids(10)
        if a != "agent_0000" and a not in strategy.neighbors_of("agent_0000")
    )
    with pytest.raises(ValueError, match="not neighbours"):
        strategy.rewire(("agent_0000", non_neighbor), ("agent_0000", "agent_0005"))


def test_rewire_rejects_adding_an_existing_neighbour():
    strategy = _graph()
    existing = strategy.neighbors_of("agent_0000")[0]
    other = strategy.neighbors_of("agent_0000")[1]
    with pytest.raises(ValueError, match="already neighbours"):
        strategy.rewire(("agent_0000", other), ("agent_0000", existing))


def test_rewire_rejects_a_self_loop():
    strategy = _graph()
    neighbor = strategy.neighbors_of("agent_0000")[0]
    with pytest.raises(ValueError, match="self-loop"):
        strategy.rewire(("agent_0000", neighbor), ("agent_0000", "agent_0000"))


def test_rewire_rejects_an_unknown_agent():
    strategy = _graph()
    neighbor = strategy.neighbors_of("agent_0000")[0]
    with pytest.raises(ValueError, match="not a node"):
        strategy.rewire(("agent_0000", neighbor), ("agent_0000", "ghost"))


def test_no_self_loops_or_duplicate_edges_after_many_rewires():
    strategy = _graph(n=12, degree=4, k=2)
    rng = random.Random(0)
    all_ids = _ids(12)
    for _ in range(30):
        reader = rng.choice(all_ids)
        neighbors = strategy.neighbors_of(reader)
        if not neighbors:
            continue
        drop = rng.choice(neighbors)
        candidates = [
            a for a in all_ids
            if a != reader and a not in strategy.neighbors_of(reader)
        ]
        if not candidates or strategy.degree(drop) <= 2:
            continue
        add = rng.choice(candidates)
        strategy.rewire((reader, drop), (reader, add))

    edges = strategy.edges()
    assert len(edges) == len(set(edges))  # no duplicates
    assert all(a != b for a, b in edges)  # no self-loops
    for agent_id in all_ids:
        assert strategy.degree(agent_id) >= 2  # k


def test_restore_edges_round_trips():
    strategy = _graph()
    neighbor = strategy.neighbors_of("agent_0000")[0]
    other = next(
        a for a in _ids(10)
        if a != "agent_0000" and a not in strategy.neighbors_of("agent_0000")
    )
    strategy.rewire(("agent_0000", neighbor), ("agent_0000", other))
    saved = strategy.edges()

    fresh = _graph()  # same construction params -> same starting graph
    fresh.restore_edges(saved)
    assert fresh.edges() == saved


def test_restore_edges_rejects_an_unknown_agent():
    strategy = _graph()
    with pytest.raises(ValueError, match="not in this run's graph"):
        strategy.restore_edges([("agent_0000", "ghost")])


def test_restore_edges_rejects_a_graph_under_k():
    strategy = _graph(n=10, degree=4, k=2)
    # A graph with no edges at all for agent_0000: everyone else still
    # mutually connected, agent_0000 isolated.
    thin = [e for e in strategy.edges() if "agent_0000" not in e]
    with pytest.raises(ValueError, match="fewer than k=2"):
        strategy.restore_edges(thin)


# --- AgentManager: balanced leans -------------------------------------------


def _population_leans(run, seed=1) -> list[float]:
    manager = AgentManager(run, random.Random(seed))
    agents = manager.initialize_population()
    return [a.initial_stance for a in agents]


def test_balanced_leans_give_every_scale_point_as_evenly_as_possible():
    run = make_run(M=20, persona_pool_id="pool_20", lean_assignment="balanced")
    leans = _population_leans(run, seed=7)
    counts = {point: leans.count(point) for point in run.stance_scale}
    # 20 agents over a 7-point scale: two full blocks (14) plus 6 more drawn
    # without replacement, so six points appear 3 times and one appears
    # twice -- never 4 of one and 1 of another.
    assert sorted(counts.values()) == [2, 3, 3, 3, 3, 3, 3]
    assert sum(counts.values()) == 20


def test_balanced_leans_give_the_same_multiset_on_every_seed():
    run = make_run(M=20, persona_pool_id="pool_20", lean_assignment="balanced")
    first = sorted(_population_leans(run, seed=1))
    second = sorted(_population_leans(run, seed=99))
    assert first == second


def test_balanced_leans_still_shuffle_so_assignment_order_can_differ():
    run = make_run(M=20, persona_pool_id="pool_20", lean_assignment="balanced")
    first = _population_leans(run, seed=1)
    second = _population_leans(run, seed=2)
    assert sorted(first) == sorted(second)
    assert first != second  # different seeds shuffle differently


def test_balanced_two_camps_splits_evenly_and_excludes_the_midpoint():
    run = make_run(M=20, persona_pool_id="pool_20", lean_assignment="balanced_two_camps")
    leans = _population_leans(run, seed=7)
    assert len(leans) == 20
    assert 4.0 not in leans
    con = [lean for lean in leans if lean < 4]
    pro = [lean for lean in leans if lean > 4]
    assert len(con) == 10 and len(pro) == 10
    # Each camp of 10 over 3 points: one block of 3 (9) plus 1 more, so one
    # point appears 4 times and the other two appear 3 times each.
    assert sorted(con.count(p) for p in (1, 2, 3)) == [3, 3, 4]
    assert sorted(pro.count(p) for p in (5, 6, 7)) == [3, 3, 4]


def test_balanced_two_camps_gives_the_same_multiset_on_every_seed_for_even_m():
    """Even M: camp sizes are fixed at M/2 each with no RNG involved, so the
    multiset -- not just the shuffle -- must be identical across seeds."""
    run = make_run(M=20, persona_pool_id="pool_20", lean_assignment="balanced_two_camps")
    first = sorted(_population_leans(run, seed=1))
    second = sorted(_population_leans(run, seed=99))
    assert first == second


def test_balanced_two_camps_still_shuffles_assignment_order():
    run = make_run(M=20, persona_pool_id="pool_20", lean_assignment="balanced_two_camps")
    first = _population_leans(run, seed=1)
    second = _population_leans(run, seed=2)
    assert sorted(first) == sorted(second)
    assert first != second


def test_balanced_two_camps_odd_m_still_splits_within_one_agent():
    """Odd M: the camps cannot be exactly equal, and which camp gets the
    extra agent is drawn from the RNG rather than fixed to either side."""
    run = make_run(M=21, persona_pool_id="pool_20", lean_assignment="balanced_two_camps")
    leans = _population_leans(run, seed=7)
    assert len(leans) == 21
    con = sum(1 for lean in leans if lean < 4)
    pro = sum(1 for lean in leans if lean > 4)
    assert {con, pro} == {10, 11}


def test_balanced_two_camps_odd_m_extra_agents_side_depends_on_the_seed():
    """Not hard-coded to one side: across enough seeds, both camps must
    sometimes be the larger one."""
    run = make_run(M=21, persona_pool_id="pool_20", lean_assignment="balanced_two_camps")
    larger_camps = set()
    for seed in range(1, 30):
        leans = _population_leans(run, seed=seed)
        con = sum(1 for lean in leans if lean < 4)
        pro = sum(1 for lean in leans if lean > 4)
        larger_camps.add("con" if con > pro else "pro")
    assert larger_camps == {"con", "pro"}


def test_balanced_two_camps_rejects_a_scale_with_no_room_on_one_side():
    # sorted([3, 4]) has midpoint 4 (index len // 2 == 1): one point (3)
    # below it and none above, so there is no pro camp to draw from.
    run = make_run(
        M=2, N=1, persona_pool_id="test_pool", lean_assignment="balanced_two_camps",
        stance_scale=[3, 4],
    )
    with pytest.raises(ValueError, match="below and above the midpoint"):
        _population_leans(run, seed=1)


def test_random_lean_assignment_is_unchanged():
    """lean_assignment="random" (the default) must draw exactly the way
    _draw_initial_stance always has: explicitly passing it must equal
    leaving it out."""
    explicit = make_run(M=10, persona_pool_id="test_pool", lean_assignment="random")
    implicit = make_run(M=10, persona_pool_id="test_pool")
    assert _population_leans(explicit, seed=3) == _population_leans(implicit, seed=3)


# --- orchestrator end-to-end ------------------------------------------------

FIXTURE_BANK = "data/bank/test_bank.csv"
FIXTURE_NEUTRAL_BANK = "data/bank/test_neutral.csv"


def _echo_run(**overrides) -> ExperimentRun:
    base = dict(
        M=8, N=2, K=4, run_id="echo_run", persona_pool_id="pool_20",
        neighbor_strategy="graph", graph_degree=4, graph_rewire_p=0.1,
        regime="bank", bank_path=FIXTURE_BANK,
        probe_question="Should the fare subsidy continue?",
        stance_low_label="the subsidy should end",
        stance_high_label="the subsidy should continue",
        neutral_bank_path=FIXTURE_NEUTRAL_BANK,
        checkpoint_every_n_turns=1,
    )
    base.update(overrides)
    return make_run(**base)


def _alternating_probes(n_rounds: int, agents: int = 8) -> list:
    """mean=6 (pro) for even agent indices, mean=2 (con) for odd ones, the
    same pattern every round -- so every agent keeps the same side for the
    whole run and opposite-side reads are reliably present every round.

    `n_rounds` counts round 0 (M calls) plus every turn 1..K (M calls each).
    """
    pattern = [6, 2] * (agents // 2)
    return [make_logprob_response(mean=m) for m in pattern * n_rounds]


def _scripted(bank_gateway, responses: list):
    readings = iter(responses)
    bank_gateway.score_next_token.side_effect = lambda *a, **k: next(readings)
    return bank_gateway


def _rows(tmp_path, run_id="echo_run", name="interactions.jsonl") -> list[str]:
    path = tmp_path / run_id / name
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _interactions(tmp_path, run_id="echo_run") -> list[Interaction]:
    return [Interaction.model_validate_json(line) for line in _rows(tmp_path, run_id)]


def _rewire_statuses(tmp_path, run_id="echo_run") -> list[dict]:
    return [json.loads(line) for line in _rows(tmp_path, run_id, "rewires.jsonl")]


async def test_rewire_rate_zero_writes_no_new_files_and_never_changes_the_graph(
    tmp_path, bank_gateway
):
    run = _echo_run(rewire_rate=0.0)
    _scripted(bank_gateway, _alternating_probes(5))
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    run_dir = tmp_path / run.run_id
    assert not (run_dir / "rewires.jsonl").exists()
    assert not (run_dir / "graph_edges_by_round.csv").exists()

    edges_at_start = (run_dir / "graph_edges.csv").read_text()
    # No per-round file means the graph never moved from the one
    # graph_edges.csv recorded at round 0 -- there is nothing else to check
    # it against, which is itself the point: rewiring never touched it.
    assert edges_at_start  # sanity: the graph strategy did write the file


async def test_degree_invariants_hold_after_rewiring(tmp_path, bank_gateway):
    run = _echo_run(rewire_rate=1.0, rewire_target="same_side")
    _scripted(bank_gateway, _alternating_probes(5))
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    lines = _rows(tmp_path, run.run_id, "graph_edges_by_round.csv")[1:]  # skip header
    final_turn = max(int(line.split(",")[0]) for line in lines)
    final_edges = [
        tuple(line.split(",")[1:]) for line in lines if line.startswith(f"{final_turn},")
    ]
    assert len(final_edges) == len(set(final_edges))  # no duplicate edges
    assert all(a != b for a, b in final_edges)  # no self-loops

    degree: dict[str, int] = {}
    for a, b in final_edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    assert all(d >= run.N for d in degree.values())


async def test_same_side_targets_always_match_the_readers_new_side(tmp_path, bank_gateway):
    run = _echo_run(rewire_rate=1.0, rewire_target="same_side")
    _scripted(bank_gateway, _alternating_probes(5))
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rewired = [r for r in _rewire_statuses(tmp_path, run.run_id) if r["status"] == "rewired"]
    assert rewired  # the scripted probes guarantee opposite-side reads exist

    # Even agent indices read 6 (pro), odd read 2 (con), consistently: the
    # added agent's own index parity must match the reader's side.
    for row in rewired:
        reader_idx = int(row["agent_id"].split("_")[1])
        added_idx = int(row["added"].split("_")[1])
        reader_is_pro = reader_idx % 2 == 0
        added_is_pro = added_idx % 2 == 0
        assert reader_is_pro == added_is_pro, row


async def test_random_target_ignores_side(tmp_path, bank_gateway):
    run = _echo_run(rewire_rate=1.0, rewire_target="random")
    _scripted(bank_gateway, _alternating_probes(5))
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rewired = [r for r in _rewire_statuses(tmp_path, run.run_id) if r["status"] == "rewired"]
    assert rewired

    mismatches = [
        row for row in rewired
        if (int(row["agent_id"].split("_")[1]) % 2) != (int(row["added"].split("_")[1]) % 2)
    ]
    # With a random target and a population split 50/50 by side, at least
    # one rewire landing on the opposite side is overwhelmingly likely; a
    # same_side-only run (tested above) can never produce one.
    assert mismatches


async def test_a_reader_exactly_at_the_midpoint_never_rewires(tmp_path, bank_gateway):
    """Every agent probes to exactly 4 every round: no side, so no mismatch
    can ever be found, and rewires.jsonl must stay entirely empty even
    though every read is still drawn from the rewire stream."""
    run = _echo_run(rewire_rate=1.0, rewire_target="same_side")
    flat = [make_logprob_response(mean=4, probabilities={4: 1.0}) for _ in range(8 * 5)]
    _scripted(bank_gateway, flat)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    run_dir = tmp_path / run.run_id
    assert not (run_dir / "rewires.jsonl").exists()
    # graph_edges_by_round.csv is still written (rewire_rate > 0 unconditionally
    # logs the graph used for reading each round), just with no rewires in it.
    assert (run_dir / "graph_edges_by_round.csv").exists()


async def test_a_same_side_post_never_triggers_a_rewire(tmp_path, bank_gateway):
    """Every agent on the same side: no opposite-side read exists, so no
    rewire can ever be attempted."""
    run = _echo_run(rewire_rate=1.0, rewire_target="same_side")
    all_pro = [make_logprob_response(mean=6) for _ in range(8 * 5)]
    _scripted(bank_gateway, all_pro)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    assert not (tmp_path / run.run_id / "rewires.jsonl").exists()


async def test_identical_probes_give_identical_rewiring_across_languages(
    tmp_path, bank_gateway
):
    en_dir, hi_dir = tmp_path / "en", tmp_path / "hi"
    run_en = _echo_run(rewire_rate=1.0, rewire_target="same_side", language_condition="english")
    run_hi = _echo_run(rewire_rate=1.0, rewire_target="same_side", language_condition="hindi")

    _scripted(bank_gateway, _alternating_probes(5))
    await build_orchestrator(run_en, bank_gateway, runs_dir=en_dir).run()
    _scripted(bank_gateway, _alternating_probes(5))
    await build_orchestrator(run_hi, bank_gateway, runs_dir=hi_dir).run()

    def without_run_id(rows):
        out = []
        for row in rows:
            row = dict(row)
            row.pop("run_id", None)
            out.append(row)
        return out

    assert without_run_id(_rewire_statuses(en_dir, run_en.run_id)) == without_run_id(
        _rewire_statuses(hi_dir, run_hi.run_id)
    )
    assert _rows(en_dir, run_en.run_id, "graph_edges_by_round.csv") == _rows(
        hi_dir, run_hi.run_id, "graph_edges_by_round.csv"
    )


async def test_e4_style_delivery_off_rewires_identically_on_shadow_items(
    tmp_path, bank_gateway
):
    """delivery='off' runs the same rewiring rule on the bank items the
    neutral messages stood in for -- which, since post_map/side_map are
    computed identically regardless of delivery mode, means an off run and
    an on run with the same scripted probes rewire identically."""
    on_dir, off_dir = tmp_path / "on", tmp_path / "off"
    run_on = _echo_run(rewire_rate=1.0, rewire_target="same_side", delivery="on")
    run_off = _echo_run(rewire_rate=1.0, rewire_target="same_side", delivery="off")

    _scripted(bank_gateway, _alternating_probes(5))
    await build_orchestrator(run_on, bank_gateway, runs_dir=on_dir).run()
    _scripted(bank_gateway, _alternating_probes(5))
    await build_orchestrator(run_off, bank_gateway, runs_dir=off_dir).run()

    def without_run_id(rows):
        out = []
        for row in rows:
            row = dict(row)
            row.pop("run_id", None)
            out.append(row)
        return out

    assert without_run_id(_rewire_statuses(on_dir, run_on.run_id)) == without_run_id(
        _rewire_statuses(off_dir, run_off.run_id)
    )


async def test_resume_after_a_mid_round_crash_matches_an_uninterrupted_run(
    tmp_path, bank_gateway
):
    run = _echo_run(rewire_rate=1.0, rewire_target="random")
    clean_dir = tmp_path / "clean"
    crashed_dir = tmp_path / "crashed"

    _scripted(bank_gateway, _alternating_probes(5))
    await build_orchestrator(run, bank_gateway, runs_dir=clean_dir).run()

    # Crash partway through round 3's probes: enough responses for round 0
    # (8) and rounds 1-2 (8 each) = 24, then every later call raises.
    crash_responses = _alternating_probes(3) + [RuntimeError("boom")] * 80
    readings = iter(crash_responses)

    def _side_effect(*a, **k):
        value = next(readings)
        if isinstance(value, Exception):
            raise value
        return value

    bank_gateway.score_next_token.side_effect = _side_effect
    with pytest.raises(RuntimeError):
        await build_orchestrator(run, bank_gateway, runs_dir=crashed_dir).run()
    assert {r.turn for r in _interactions(crashed_dir, run.run_id)} == {0, 1, 2}

    # Resume with fresh responses covering rounds 3-4.
    remaining = _alternating_probes(5)[8 * 3:]
    _scripted(bank_gateway, remaining)
    await build_orchestrator(run, bank_gateway, runs_dir=crashed_dir).run()

    def signature(base):
        return sorted(
            (
                r.turn, r.speaker_agent_id, r.posted_item_id, tuple(r.items_read),
                r.stance_before, r.stance_after, r.api_call_status,
            )
            for r in _interactions(base, run.run_id)
        )

    assert signature(crashed_dir) == signature(clean_dir)
    assert _rows(crashed_dir, run.run_id, "rewires.jsonl") == _rows(
        clean_dir, run.run_id, "rewires.jsonl"
    )
    assert _rows(crashed_dir, run.run_id, "graph_edges_by_round.csv") == _rows(
        clean_dir, run.run_id, "graph_edges_by_round.csv"
    )
