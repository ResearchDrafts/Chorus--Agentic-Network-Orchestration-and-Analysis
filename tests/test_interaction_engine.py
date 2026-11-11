import random
from collections import Counter

import pytest

from sandbox.interaction_engine import AlphaSampling
from sandbox.models import Agent, StanceRecord
from tests.factories import make_agent as _make_agent


def _population(n: int, stance_fn) -> list[Agent]:
    return [_make_agent(f"agent_{i:04d}", stance_fn(i)) for i in range(n)]


def test_construction_rejects_alpha_outside_zero_one():
    with pytest.raises(ValueError):
        AlphaSampling(alpha=1.5, N=3)
    with pytest.raises(ValueError):
        AlphaSampling(alpha=-0.1, N=3)


def test_construction_rejects_n_less_than_one():
    with pytest.raises(ValueError):
        AlphaSampling(alpha=0.5, N=0)


def test_alpha_one_deterministically_returns_closest_stance_candidates():
    # This is a *weighted* sample, not a strict top-k-by-weight pick, so
    # "deterministic" here means: with stances separated enough that the
    # far candidates' weights are negligible next to the close candidates',
    # the closest two win for any rng draw, not just a cherry-picked seed
    # (verified empirically across many seeds during test development).
    strategy = AlphaSampling(alpha=1.0, N=2)
    speaker = _make_agent("speaker", initial_stance=4.0)
    others = [
        _make_agent("far_1", initial_stance=-996.0),
        _make_agent("far_2", initial_stance=1004.0),
        _make_agent("close_1", initial_stance=4.1),
        _make_agent("close_2", initial_stance=3.9),
    ]
    all_agents = [speaker, *others]

    for seed in range(10):
        result = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(seed))
        assert {a.agent_id for a in result} == {"close_1", "close_2"}


def test_alpha_zero_selection_frequency_is_roughly_uniform_across_seeds():
    strategy = AlphaSampling(alpha=0.0, N=1)
    speaker = _make_agent("speaker", initial_stance=4.0)
    candidates = _population(10, lambda i: float(1 + i))  # varied stances, irrelevant at alpha=0
    candidates = [c for c in candidates if c.agent_id != "speaker"]
    all_agents = [speaker, *candidates]

    counts = Counter()
    trials = 2000
    for seed in range(trials):
        chosen = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(seed))
        counts[chosen[0].agent_id] += 1

    expected = trials / len(candidates)
    for c in candidates:
        # wide tolerance: uniform-ish, not exact, over 2000 trials across 10 candidates
        assert 0.5 * expected < counts[c.agent_id] < 1.5 * expected


def test_never_returns_speaker_and_always_returns_n_distinct_agents():
    strategy = AlphaSampling(alpha=0.5, N=3)
    speaker = _make_agent("speaker", initial_stance=4.0)
    others = _population(6, lambda i: float(i))
    all_agents = [speaker, *others]

    result = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(7))

    assert speaker.agent_id not in {a.agent_id for a in result}
    assert len(result) == 3
    assert len({a.agent_id for a in result}) == 3


def test_population_too_small_raises_value_error():
    strategy = AlphaSampling(alpha=0.5, N=5)
    speaker = _make_agent("speaker", initial_stance=4.0)
    others = _population(3, lambda i: float(i))
    all_agents = [speaker, *others]

    with pytest.raises(ValueError):
        strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(0))


def test_identically_seeded_rng_produces_identical_neighbor_sets():
    strategy = AlphaSampling(alpha=0.5, N=3)
    speaker = _make_agent("speaker", initial_stance=4.0)
    others = _population(8, lambda i: float(i))
    all_agents = [speaker, *others]

    result1 = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(42))
    result2 = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(42))

    assert [a.agent_id for a in result1] == [a.agent_id for a in result2]


def test_turn_one_always_reads_initial_stance_even_with_stance_history_present():
    strategy = AlphaSampling(alpha=1.0, N=1)
    # speaker's initial_stance (4.0) is close to "matches_initial"; its
    # stance_history (if wrongly read at turn=1) would instead be close to
    # "matches_history" -- this distinguishes which one select_neighbors used.
    speaker = _make_agent(
        "speaker",
        initial_stance=4.0,
        stance_history=[StanceRecord(turn=1, stance_value=1.0, reason_text="r", interaction_id="i1")],
    )
    matches_initial = _make_agent("matches_initial", initial_stance=4.0)
    matches_history = _make_agent("matches_history", initial_stance=1.0)
    all_agents = [speaker, matches_initial, matches_history]

    result = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(0))

    assert result[0].agent_id == "matches_initial"


# --- phase4.md D6: sampling weights must be finite and strictly positive --


def test_weighted_sample_rejects_zero_weight():
    """1.0 / w raised a bare ZeroDivisionError before the guard."""
    from sandbox.interaction_engine import _weighted_sample_without_replacement

    items = _population(3, lambda i: float(i + 1))
    with pytest.raises(ValueError, match="finite and > 0"):
        _weighted_sample_without_replacement(items, [1.0, 0.0, 1.0], 2, random.Random(1))


def test_weighted_sample_rejects_negative_weight():
    """The dangerous case: random() ** negative > 1, and keys sort
    descending, so a negative-weight item was GUARANTEED to be picked first.
    A silent wrong answer rather than an error."""
    from sandbox.interaction_engine import _weighted_sample_without_replacement

    items = _population(3, lambda i: float(i + 1))
    with pytest.raises(ValueError, match="finite and > 0"):
        _weighted_sample_without_replacement(items, [1.0, -1.0, 1.0], 2, random.Random(1))


def test_weighted_sample_rejects_nan_weight():
    """A NaN stance propagates through distance -> weight and makes the sort
    order meaningless rather than raising."""
    from sandbox.interaction_engine import _weighted_sample_without_replacement

    items = _population(3, lambda i: float(i + 1))
    with pytest.raises(ValueError, match="finite and > 0"):
        _weighted_sample_without_replacement(
            items, [1.0, float("nan"), 1.0], 2, random.Random(1)
        )


def test_weighted_sample_rejects_length_mismatch():
    """zip() silently truncated on a mismatch before strict=True."""
    from sandbox.interaction_engine import _weighted_sample_without_replacement

    items = _population(3, lambda i: float(i + 1))
    with pytest.raises(ValueError, match="length mismatch"):
        _weighted_sample_without_replacement(items, [1.0, 1.0], 2, random.Random(1))


# --- RQ1: GraphSampling ---------------------------------------------------


def _graph_population(n: int) -> list[Agent]:
    return [_make_agent(f"agent_{i:04d}", 4.0) for i in range(n)]


def _ids(n: int) -> list[str]:
    return [f"agent_{i:04d}" for i in range(n)]


def _graph(n=20, degree=6, rewire_p=0.1, k=3, graph_seed=42) -> "GraphSampling":
    from sandbox.interaction_engine import GraphSampling

    return GraphSampling(_ids(n), degree=degree, rewire_p=rewire_p, k=k, graph_seed=graph_seed)


def test_graph_sampling_returns_only_graph_neighbours():
    """The point of the strategy: an agent's candidate set is structural, not
    a function of where it currently stands."""
    strategy = _graph()
    population = _graph_population(20)

    for speaker in population:
        chosen = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(5))
        adjacency = {
            b for a, b in strategy.edges() if a == speaker.agent_id
        } | {a for a, b in strategy.edges() if b == speaker.agent_id}
        assert {a.agent_id for a in chosen} <= adjacency
        assert speaker.agent_id not in {a.agent_id for a in chosen}
        assert len(chosen) == len({a.agent_id for a in chosen}) == 3


def test_graph_sampling_ignores_stance_entirely():
    """AlphaSampling at alpha=1 would always pick the closest stance. The
    graph must not, or the selection effect it exists to remove is still
    there."""
    strategy = _graph(n=20, degree=6, k=3)
    population = [_make_agent(f"agent_{i:04d}", float(1 + (i % 7))) for i in range(20)]
    speaker = population[0]

    from_varied = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(5))
    flat = [_make_agent(f"agent_{i:04d}", 4.0) for i in range(20)]
    from_flat = strategy.select_neighbors(flat[0], flat, turn=1, rng=random.Random(5))

    assert [a.agent_id for a in from_varied] == [a.agent_id for a in from_flat]


def test_same_seed_gives_the_same_graph_and_the_same_samples():
    """Two runs of a matched set share a seed, so they must share a graph and
    read the same neighbours; only the rendering may differ."""
    a, b = _graph(graph_seed=42), _graph(graph_seed=42)
    assert a.edges() == b.edges()

    population = _graph_population(20)
    for speaker in population:
        assert [
            n.agent_id
            for n in a.select_neighbors(speaker, population, turn=1, rng=random.Random(7))
        ] == [
            n.agent_id
            for n in b.select_neighbors(speaker, population, turn=1, rng=random.Random(7))
        ]


def test_a_different_seed_gives_a_different_graph():
    assert _graph(graph_seed=42).edges() != _graph(graph_seed=43).edges()


def test_a_node_with_too_few_neighbours_raises_rather_than_returning_fewer():
    """A short feed is a different stimulus. An agent that quietly read two
    posts where everyone else read three would be a confound nothing in the
    output records."""
    from sandbox.interaction_engine import GraphSampling

    # Degree 2 is a ring: every node has exactly 2 neighbours, so k=2 works
    # and k=3 cannot be constructed at all.
    with pytest.raises(ValueError, match="cannot exceed degree"):
        GraphSampling(_ids(10), degree=2, rewire_p=0.0, k=3, graph_seed=1)

    # The call-time path: a node whose neighbourhood is smaller than k. Built
    # legally (k <= degree), then the graph is edited to strand one node.
    strategy = GraphSampling(_ids(10), degree=4, rewire_p=0.0, k=3, graph_seed=1)
    strategy._neighbors["agent_0000"] = ["agent_0001", "agent_0002"]
    with pytest.raises(ValueError, match="need 3"):
        strategy.select_neighbors(
            _make_agent("agent_0000", 4.0), _graph_population(10), turn=1, rng=random.Random(0)
        )


def test_an_agent_outside_the_graph_raises():
    """The graph is built once from the initial population, so an agent that
    is not a node is a wiring error, not a run-time condition."""
    strategy = _graph(n=10)
    with pytest.raises(ValueError, match="not a node"):
        strategy.select_neighbors(
            _make_agent("agent_9999", 4.0), _graph_population(10), turn=1, rng=random.Random(0)
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(degree=5), "even"),
        (dict(degree=1), ">= 2"),
        (dict(degree=20), "less than the population"),
        (dict(rewire_p=1.5), r"\[0,1\]"),
        (dict(rewire_p=-0.1), r"\[0,1\]"),
        (dict(k=0), "k must be >= 1"),
        (dict(k=8), "cannot exceed degree"),
    ],
)
def test_graph_construction_rejects_invalid_parameters(kwargs, message):
    """Never silently clamped: the same contract AlphaSampling holds."""
    from sandbox.interaction_engine import GraphSampling

    base = dict(agent_ids=_ids(20), degree=6, rewire_p=0.1, k=3, graph_seed=1)
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        GraphSampling(**base)


def test_duplicate_agent_ids_are_rejected():
    """Node indices map to agent IDs positionally, so a duplicate would make
    two nodes the same agent and silently halve the population."""
    from sandbox.interaction_engine import GraphSampling

    with pytest.raises(ValueError, match="unique"):
        GraphSampling(["a", "b", "b", "c", "d"], degree=2, rewire_p=0.1, k=1, graph_seed=1)


def test_edges_are_sorted_pairs_covering_the_population():
    strategy = _graph(n=20, degree=6)
    edges = strategy.edges()

    assert edges == sorted(edges)
    assert all(a < b for a, b in edges)
    assert len(edges) == 20 * 6 // 2  # rewiring moves edges, never adds or drops them
    assert {a for a, _ in edges} | {b for _, b in edges} == set(_ids(20))


def test_the_graph_is_connected():
    """A disconnected population is two smaller experiments sharing an output
    file, which is why the connected WS variant is used."""
    import networkx as nx

    strategy = _graph(n=30, degree=4, rewire_p=0.3, graph_seed=99)
    graph = nx.Graph()
    graph.add_nodes_from(_ids(30))
    graph.add_edges_from(strategy.edges())
    assert nx.is_connected(graph)


def test_sampling_consumes_the_rng_exactly_once_per_agent():
    """A second rng.sample call per agent would both advance the stream twice
    and return a set other than the one just validated.

    Checked by state rather than by output: one call's result alone cannot
    distinguish "sampled once" from "sampled twice and returned the second",
    whereas how far the stream advanced can.
    """
    strategy = _graph(n=20, degree=6, k=3)
    population = _graph_population(20)
    speaker = population[0]
    candidates = sorted(
        {b for a, b in strategy.edges() if a == speaker.agent_id}
        | {a for a, b in strategy.edges() if b == speaker.agent_id}
    )

    used = random.Random(3)
    chosen = strategy.select_neighbors(speaker, population, turn=1, rng=used)

    reference = random.Random(3)
    expected = reference.sample(candidates, 3)

    assert [a.agent_id for a in chosen] == expected
    assert used.random() == reference.random()  # the streams are still in step


def test_turn_does_not_change_the_draw_for_a_fixed_rng():
    """The graph does not change over the run; which neighbours are read comes
    from rng, which the orchestrator derives per turn."""
    strategy = _graph()
    population = _graph_population(20)
    speaker = population[4]

    first = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(11))
    second = strategy.select_neighbors(speaker, population, turn=9, rng=random.Random(11))
    assert [a.agent_id for a in first] == [a.agent_id for a in second]
