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

    # graph_seed=2 rather than 99: at this rewiring rate seed 99 leaves a
    # node with 2 neighbours, which construction now rejects outright (see
    # the degree tests below). Connectivity is what this test is about.
    strategy = _graph(n=30, degree=4, rewire_p=0.3, graph_seed=2)
    graph = nx.Graph()
    graph.add_nodes_from(_ids(30))
    graph.add_edges_from(strategy.edges())
    assert nx.is_connected(graph)


# --- every node must have k neighbours, checked at construction ----------


def test_a_node_short_of_k_neighbours_is_rejected_at_construction():
    """WS(8, 4, 0.1) with this seed leaves one node with 2 neighbours.

    `k <= degree` is necessary but not sufficient: every node starts the ring
    with exactly `degree` neighbours and rewiring moves edges rather than
    adding them. Caught here rather than in select_neighbors on round 1,
    which is after the population is built, the config is written and the
    round-0 probe has been paid for -- M calls for a graph that was never
    going to work.
    """
    import networkx as nx

    # The seed is pinned, so this states a fact about this graph rather than
    # hoping: node degrees are [2, 3, 4, 4, 4, 4, 5, 6].
    degrees = sorted(dict(nx.connected_watts_strogatz_graph(8, 4, 0.1, seed=30).degree()).values())
    assert degrees[0] == 2

    with pytest.raises(ValueError) as caught:
        _graph(n=8, degree=4, rewire_p=0.1, k=3, graph_seed=30)

    message = str(caught.value)
    assert "graph_seed=30" in message
    assert "k=3" in message
    assert "has 2" in message       # the offending node's degree
    assert "agent_" in message      # named, so it can be looked up


def test_the_error_names_every_short_node():
    """All of them at once, like every other validation in this codebase: a
    graph with two bad nodes should take one run to diagnose, not two."""
    with pytest.raises(ValueError) as caught:
        _graph(n=8, degree=4, rewire_p=0.1, k=4, graph_seed=45)

    # Seed 45 gives degrees [2, 2, 4, 4, 5, 5, 5, 5], so k=4 leaves two short.
    message = str(caught.value)
    assert "2 node(s)" in message
    assert message.count("has 2") == 2


def test_a_graph_whose_every_node_meets_k_is_accepted():
    """Seed 0 gives every node at least 3 neighbours at this degree."""
    strategy = _graph(n=8, degree=4, rewire_p=0.1, k=3, graph_seed=0)
    assert all(len(strategy._neighbors[a]) >= 3 for a in _ids(8))


def test_the_check_is_per_node_not_on_the_average():
    """A graph can average well above k and still starve one node, which is
    the whole reason the schema's k <= degree check is not enough."""
    import networkx as nx

    graph = nx.connected_watts_strogatz_graph(8, 4, 0.1, seed=30)
    degrees = list(dict(graph.degree()).values())
    assert sum(degrees) / len(degrees) > 3      # the average is fine
    with pytest.raises(ValueError):             # one node is not
        _graph(n=8, degree=4, rewire_p=0.1, k=3, graph_seed=30)


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


# --- Fix 9: GraphSimilaritySampling ---------------------------------------


def _similarity(n=10, degree=4, rewire_p=0.1, k=2, graph_seed=7, threshold=1.0):
    from sandbox.interaction_engine import GraphSimilaritySampling

    return GraphSimilaritySampling(
        _ids(n), degree=degree, rewire_p=rewire_p, k=k, graph_seed=graph_seed,
        threshold=threshold,
    )


def _probed(agent_id: str, raw_stance: float) -> Agent:
    """An agent with a valid probe reading -- current_stance() then returns
    this round's measurement rather than falling back to initial_stance."""
    return _make_agent(
        agent_id, raw_stance,
        stance_history=[
            StanceRecord(turn=0, stance_value=raw_stance, reason_text="", interaction_id="x")
        ],
    )


def test_construction_rejects_non_positive_threshold():
    from sandbox.interaction_engine import GraphSimilaritySampling

    with pytest.raises(ValueError, match="threshold"):
        GraphSimilaritySampling(_ids(10), degree=4, rewire_p=0.1, k=2, graph_seed=1, threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        GraphSimilaritySampling(_ids(10), degree=4, rewire_p=0.1, k=2, graph_seed=1, threshold=-1.0)


def test_shares_byte_identical_graph_construction_with_graph_sampling():
    """A seed must produce the same graph under both strategies -- e8 and e9
    are only a meaningful comparison if they read from the same structure."""
    from sandbox.interaction_engine import GraphSampling

    plain = GraphSampling(_ids(20), degree=6, rewire_p=0.1, k=3, graph_seed=99)
    similarity = _similarity(n=20, degree=6, rewire_p=0.1, k=3, graph_seed=99)
    assert plain.edges() == similarity.edges()


def test_never_delivers_a_non_neighbour():
    """A huge threshold filters nobody, so the only agents that can appear
    at all are the reader's own graph neighbours."""
    strategy = _similarity(n=10, degree=4, k=2, threshold=10.0, graph_seed=7)
    population = [_probed(agent_id, 4.0) for agent_id in _ids(10)]
    by_id = {a.agent_id: a for a in population}
    speaker = by_id["agent_0000"]

    chosen = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(5))
    candidates = set(strategy.neighbors_of(speaker.agent_id))
    assert {a.agent_id for a in chosen} <= candidates
    assert speaker.agent_id not in {a.agent_id for a in chosen}


def test_threshold_of_two_in_x_filters_nobody():
    """x is in [-1, 1], so no two readings can differ by more than 2.0 --
    a threshold that wide must filter out none of the reader's graph
    neighbours, however far apart their readings are (n_neighbors_over_
    threshold stays 0), even though only k of them are actually delivered."""
    strategy = _similarity(n=10, degree=4, k=2, threshold=2.0, graph_seed=7)
    candidates = strategy.neighbors_of("agent_0000")
    raws = {candidates[0]: 1.0, candidates[1]: 7.0, candidates[2]: 1.0, candidates[3]: 7.0}
    population = [_probed(agent_id, raws.get(agent_id, 4.0)) for agent_id in _ids(10)]
    by_id = {a.agent_id: a for a in population}
    speaker = by_id["agent_0000"]

    chosen = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(5))
    assert len(chosen) == 2
    assert strategy.feed_info(speaker.agent_id).n_neighbors_over_threshold == 0


def test_a_neighbour_over_the_threshold_is_never_delivered():
    """threshold=0.5 in x units is a raw-scale gap of 1.5 (x = (raw-4)/3):
    a neighbour 3 full scale points away must never appear in the feed, and
    must be counted in n_neighbors_over_threshold."""
    strategy = _similarity(n=10, degree=4, k=2, threshold=0.5, graph_seed=7)
    candidates = strategy.neighbors_of("agent_0000")
    population = [_probed(agent_id, 4.0) for agent_id in _ids(10)]
    by_id = {a.agent_id: a for a in population}
    far_id = candidates[0]
    by_id[far_id] = _probed(far_id, 7.0)  # x = 1.0, 1.0 away from the speaker's 0.0
    population = [by_id[agent_id] for agent_id in _ids(10)]
    speaker = by_id["agent_0000"]

    chosen = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(2))
    assert far_id not in {a.agent_id for a in chosen}
    assert strategy.feed_info("agent_0000").n_neighbors_over_threshold == 1


def test_the_closest_neighbours_are_chosen():
    """Among graph neighbours within the threshold, the k closest by |x_i -
    x_j| are delivered -- not merely "some that qualify"."""
    strategy = _similarity(n=10, degree=4, k=2, threshold=2.0, graph_seed=7)
    candidates = strategy.neighbors_of("agent_0000")
    assert len(candidates) == 4
    # Speaker at x=0 (raw 4.0); candidates at x = 0.1, 0.5, 0.9, -0.2 via raw
    # stances 4.3, 5.5, 6.7, 3.4 -- the two closest are the first and last.
    raws = {candidates[0]: 4.3, candidates[1]: 5.5, candidates[2]: 6.7, candidates[3]: 3.4}
    population = [
        _probed(agent_id, raws.get(agent_id, 4.0)) for agent_id in _ids(10)
    ]
    by_id = {a.agent_id: a for a in population}
    speaker = by_id["agent_0000"]

    chosen = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(3))
    assert {a.agent_id for a in chosen} == {candidates[0], candidates[3]}


def test_ties_are_reproducible_from_the_seed():
    """Candidates equidistant from the reader break ties off `rng`, not off
    agent_id order -- so the same seed must always produce the same pick,
    and (empirically, across many seeds) not always the same one."""
    strategy = _similarity(n=10, degree=4, k=1, threshold=2.0, graph_seed=7)
    candidates = strategy.neighbors_of("agent_0000")
    # All four candidates equidistant from the speaker (x = 0.5 for every one).
    population = [
        _probed(agent_id, 5.5 if agent_id in candidates else 4.0) for agent_id in _ids(10)
    ]
    by_id = {a.agent_id: a for a in population}
    speaker = by_id["agent_0000"]

    picks = set()
    for seed in range(20):
        chosen = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(seed))
        picks.add(chosen[0].agent_id)
        # Reproducible: the same seed, run again, picks the same one.
        again = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(seed))
        assert chosen[0].agent_id == again[0].agent_id
    assert len(picks) > 1


def test_short_feed_delivers_fewer_than_k_without_backfilling():
    """Fewer qualifying candidates than k: deliver what there is, never pad
    back up with the next-closest -- that is what lets a tighter threshold
    show up as a shorter average feed."""
    strategy = _similarity(n=10, degree=4, k=2, threshold=0.5, graph_seed=7)
    candidates = strategy.neighbors_of("agent_0000")
    # Only one candidate within threshold; the rest are far away.
    raws = {candidates[0]: 4.2}
    population = [
        _probed(agent_id, raws.get(agent_id, 7.0) if agent_id in candidates else 4.0)
        for agent_id in _ids(10)
    ]
    by_id = {a.agent_id: a for a in population}
    speaker = by_id["agent_0000"]

    chosen = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(4))
    assert [a.agent_id for a in chosen] == [candidates[0]]
    assert strategy.feed_info("agent_0000").feed_size == 1


def test_empty_feed_when_every_candidate_is_filtered_out():
    strategy = _similarity(n=10, degree=4, k=2, threshold=0.1, graph_seed=7)
    candidates = strategy.neighbors_of("agent_0000")
    population = [
        _probed(agent_id, 7.0) if agent_id in candidates else _probed(agent_id, 4.0)
        for agent_id in _ids(10)
    ]
    by_id = {a.agent_id: a for a in population}
    speaker = by_id["agent_0000"]

    chosen = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(4))
    assert chosen == []
    assert strategy.feed_info("agent_0000").feed_size == 0
    assert strategy.feed_info("agent_0000").fallback is False


def test_a_candidate_with_no_valid_probe_is_dropped_not_ranked():
    """A graph neighbour that has never been successfully probed has no x
    to rank against, so it is excluded from the candidate set entirely
    rather than treated as "infinitely far"."""
    strategy = _similarity(n=10, degree=4, k=2, threshold=2.0, graph_seed=7)
    candidates = strategy.neighbors_of("agent_0000")
    population = [
        _make_agent(agent_id, 4.0)  # no stance_history: never probed
        if agent_id == candidates[0]
        else _probed(agent_id, 4.0)
        for agent_id in _ids(10)
    ]
    by_id = {a.agent_id: a for a in population}
    speaker = by_id["agent_0000"]

    chosen = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(4))
    assert candidates[0] not in {a.agent_id for a in chosen}
    # Only 3 of the speaker's 4 neighbours had a valid probe to compare; the
    # unprobed one is not counted as "over threshold" either.
    assert strategy.feed_info("agent_0000").n_neighbors_over_threshold == 0


def test_a_reader_with_no_valid_probe_falls_back_to_random_neighbours():
    """stance_history empty: there is nothing to rank candidates against, so
    the rule falls back to k random graph neighbours, same as GraphSampling,
    and says so in feed_info."""
    strategy = _similarity(n=10, degree=4, k=2, threshold=0.1, graph_seed=7)
    candidates = set(strategy.neighbors_of("agent_0000"))
    population = [
        _make_agent(agent_id, 4.0) if agent_id == "agent_0000" else _probed(agent_id, 7.0)
        for agent_id in _ids(10)
    ]
    by_id = {a.agent_id: a for a in population}
    speaker = by_id["agent_0000"]
    assert not speaker.stance_history

    chosen = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(6))
    assert len(chosen) == 2
    assert {a.agent_id for a in chosen} <= candidates
    info = strategy.feed_info("agent_0000")
    assert info.fallback is True
    assert info.n_neighbors_over_threshold is None


def test_feed_order_is_shuffled_after_ranking():
    """Delivery order is not itself a clue to similarity: the two chosen
    candidates (one clearly closest, one clearly second) are reshuffled
    before delivery, so the closest is not always first in the feed."""
    strategy = _similarity(n=10, degree=4, k=2, threshold=2.0, graph_seed=7)
    candidates = strategy.neighbors_of("agent_0000")
    raws = {candidates[0]: 4.1, candidates[1]: 4.5, candidates[2]: 6.0, candidates[3]: 7.0}
    population = [_probed(agent_id, raws.get(agent_id, 4.0)) for agent_id in _ids(10)]
    by_id = {a.agent_id: a for a in population}
    speaker = by_id["agent_0000"]

    orders = set()
    for seed in range(30):
        chosen = strategy.select_neighbors(speaker, population, turn=1, rng=random.Random(seed))
        assert {a.agent_id for a in chosen} == {candidates[0], candidates[1]}
        orders.add(tuple(a.agent_id for a in chosen))
    assert len(orders) > 1


def test_nothing_in_the_class_reads_language():
    """Selection must be identical regardless of each agent's
    language_condition -- the rule is specified over opinions alone."""
    strategy = _similarity(n=10, degree=4, k=2, threshold=2.0, graph_seed=7)
    candidates = strategy.neighbors_of("agent_0000")
    raws = {candidates[0]: 4.1, candidates[1]: 4.9, candidates[2]: 6.0, candidates[3]: 3.0}

    def population_in(language: str) -> list[Agent]:
        return [
            _probed(agent_id, raws.get(agent_id, 4.0) if agent_id != "agent_0000" else 4.0)
            for agent_id in _ids(10)
        ]

    english = [a.model_copy(update={"language_condition": "english"}) for a in population_in("english")]
    hindi = [a.model_copy(update={"language_condition": "hindi"}) for a in population_in("hindi")]
    speaker_en = next(a for a in english if a.agent_id == "agent_0000")
    speaker_hi = next(a for a in hindi if a.agent_id == "agent_0000")

    chosen_en = strategy.select_neighbors(speaker_en, english, turn=1, rng=random.Random(9))
    chosen_hi = strategy.select_neighbors(speaker_hi, hindi, turn=1, rng=random.Random(9))
    assert [a.agent_id for a in chosen_en] == [a.agent_id for a in chosen_hi]
