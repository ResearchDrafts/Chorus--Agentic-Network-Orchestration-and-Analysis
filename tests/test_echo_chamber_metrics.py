"""E5's echo-chamber metrics in analysis/metrics.py: t_echo, the permutation
null behind assort_null_mean/assort_null_hi, cross_side_edge_share and
minority_share.

Every expected value here is worked out by hand in the test itself, the
same discipline test_analysis_metrics.py already uses for sign_of/classify:
these are the definitions t_echo and auc_assort_excess are made of, and a
definition that drifted would change a reported round without changing a
line of the analysis that reads it.
"""

from __future__ import annotations

import networkx as nx
import pytest

from analysis import metrics as m

SORTED_GRAPH = nx.Graph([("a", "b"), ("c", "d")])  # two same-side pairs
MIXED_GRAPH = nx.Graph([("a", "c"), ("b", "d")])  # two opposite-side pairs


def _round(turn: int, **overrides) -> dict:
    base = {
        "turn": turn, "assortativity": None, "assort_null_hi": None,
        "minority_share": None,
    }
    base.update(overrides)
    return base


# --- t_echo ----------------------------------------------------------------


def test_t_echo_at_the_first_of_two_consecutive_qualifying_rounds():
    """Rounds 2 and 3 are the first pair where assortativity clears its own
    null for two rounds running; t_echo reports the LATER of the pair, the
    same convention stationary_round uses for its own window."""
    rounds = [
        _round(0, assortativity=0.0, assort_null_hi=0.1, minority_share=0.5),
        _round(1, assortativity=0.05, assort_null_hi=0.1, minority_share=0.5),  # below null
        _round(2, assortativity=0.30, assort_null_hi=0.1, minority_share=0.5),  # clears
        _round(3, assortativity=0.35, assort_null_hi=0.1, minority_share=0.5),  # clears again
        _round(4, assortativity=0.40, assort_null_hi=0.1, minority_share=0.5),
    ]
    assert m.echo_round(rounds, k=4) == (3, False)


def test_t_echo_is_censored_when_it_never_forms():
    rounds = [
        _round(0, assortativity=0.0, assort_null_hi=0.1, minority_share=0.5),
        _round(1, assortativity=0.05, assort_null_hi=0.1, minority_share=0.5),
        _round(2, assortativity=0.08, assort_null_hi=0.1, minority_share=0.5),
    ]
    assert m.echo_round(rounds, k=2) == (3, True)  # k + 1, censored


def test_t_echo_is_censored_when_the_population_has_collapsed_to_one_side():
    """Assortativity clears the null at every round, but minority_share
    never reaches the floor: consensus, not an echo chamber, so t_echo must
    not report it as one."""
    rounds = [
        _round(0, assortativity=0.30, assort_null_hi=0.1, minority_share=0.1),
        _round(1, assortativity=0.35, assort_null_hi=0.1, minority_share=0.1),
        _round(2, assortativity=0.40, assort_null_hi=0.1, minority_share=0.1),
    ]
    assert m.echo_round(rounds, k=2) == (3, True)


def test_t_echo_requires_actually_consecutive_turn_numbers():
    """A missing round (every probe failed that turn, so round_rows never
    produced a row for it) must not bridge a gap: two qualifying rounds with
    a hole between them are not "2 consecutive rounds"."""
    rounds = [
        _round(0, assortativity=0.30, assort_null_hi=0.1, minority_share=0.5),
        # turn 1 missing entirely
        _round(2, assortativity=0.30, assort_null_hi=0.1, minority_share=0.5),
        _round(3, assortativity=0.30, assort_null_hi=0.1, minority_share=0.5),
    ]
    assert m.echo_round(rounds, k=3) == (3, False)  # turns 2,3 are the real pair


def test_t_echo_treats_an_undefined_assortativity_as_not_qualifying():
    rounds = [
        _round(0, assortativity=None, assort_null_hi=None, minority_share=0.5),
        _round(1, assortativity=0.30, assort_null_hi=0.1, minority_share=0.5),
        _round(2, assortativity=0.30, assort_null_hi=0.1, minority_share=0.5),
    ]
    assert m.echo_round(rounds, k=2) == (2, False)


# --- auc_assort_excess -------------------------------------------------------


def test_auc_assort_excess_averages_rounds_one_through_k_only():
    rounds = [
        _round(0, assort_excess=10.0),  # excluded: round 0 is the baseline
        _round(1, assort_excess=0.2),
        _round(2, assort_excess=0.4),
    ]
    assert m.auc_assort_excess(rounds) == pytest.approx(0.3)


def test_auc_assort_excess_drops_undefined_rounds_rather_than_zeroing_them():
    rounds = [_round(0, assort_excess=0.0), _round(1, assort_excess=None), _round(2, assort_excess=0.6)]
    assert m.auc_assort_excess(rounds) == pytest.approx(0.6)


def test_auc_assort_excess_is_none_with_nothing_to_average():
    rounds = [_round(0, assort_excess=0.0)]
    assert m.auc_assort_excess(rounds) is None


# --- the permutation null ---------------------------------------------------


def test_assortativity_null_is_reproducible_from_the_same_seed_key():
    x = {"a": 1.0, "b": 1.0, "c": -1.0, "d": -1.0}
    first = m.assortativity_null(MIXED_GRAPH, x, "1001|3|assort_null")
    second = m.assortativity_null(MIXED_GRAPH, x, "1001|3|assort_null")
    assert first == second


def test_assortativity_null_differs_across_seed_keys():
    x = {"a": 1.0, "b": 1.0, "c": -1.0, "d": -1.0}
    a = m.assortativity_null(MIXED_GRAPH, x, "1001|3|assort_null")
    b = m.assortativity_null(MIXED_GRAPH, x, "1001|4|assort_null")
    assert a != b


def test_assortativity_null_is_none_when_assortativity_itself_is_none():
    assert m.assortativity_null(None, {"a": 1.0, "b": -1.0}, "k") == (None, None)
    assert m.assortativity_null(FOUR := SORTED_GRAPH, {n: 0.3 for n in "abcd"}, "k") == (None, None)


def test_a_perfectly_sorted_graph_clears_its_own_null():
    """A real echo chamber -- every edge joins two agents on the same side
    -- must have assortativity above the null most permutations produce,
    since shuffling destroys the very sorting assortativity=1 reports.

    Ten nodes in two cliques rather than two disjoint pairs: with only two
    edges, several of the handful of distinct permutations still happen to
    pair like signs by chance, which pushes the 97.5th percentile itself up
    to 1.0 and makes "clears its own null" vacuous. Two 5-cliques make a
    perfect-sort permutation rare enough (2 of C(10,5) = 252 label
    assignments) that the null's upper tail sits below 1.0 reliably.
    """
    pro = [f"p{i}" for i in range(5)]
    con = [f"c{i}" for i in range(5)]
    # Two fully-connected 5-cliques and no cross edges: assortativity is
    # exactly 1.0 only when a permutation's five +1s land on EXACTLY one of
    # the two cliques, which is 2 out of C(10, 5) = 252 ways -- under 1%, so
    # the top 2.5% of 1,000 resamples essentially never lands on it.
    wide = nx.Graph()
    for group in (pro, con):
        for i, u in enumerate(group):
            for v in group[i + 1:]:
                wide.add_edge(u, v)
    x = {**{n: 1.0 for n in pro}, **{n: -1.0 for n in con}}
    assort = m.assortativity(wide, x)
    null_mean, null_hi = m.assortativity_null(wide, x, "seed|1")
    assert assort == pytest.approx(1.0)
    assert null_mean < assort
    assert null_hi < assort


# --- cross_side_edge_share and minority_share -------------------------------


def test_cross_side_edge_share_on_a_sorted_graph_is_zero():
    x = {"a": 1.0, "b": 1.0, "c": -1.0, "d": -1.0}
    assert m.cross_side_edge_share(SORTED_GRAPH, x) == pytest.approx(0.0)


def test_cross_side_edge_share_on_a_mixed_graph_is_one():
    x = {"a": 1.0, "b": 1.0, "c": -1.0, "d": -1.0}
    assert m.cross_side_edge_share(MIXED_GRAPH, x) == pytest.approx(1.0)


def test_cross_side_edge_share_excludes_agents_exactly_at_the_midpoint():
    """x == 0 has no side, per sign_of's own rule: an edge touching such an
    agent is neither same-side nor cross-side, and must not be counted as
    either."""
    x = {"a": 1.0, "b": 0.0, "c": -1.0, "d": -1.0}
    # Only edge c-d survives the exclusion (a-b touches the undecided node).
    assert m.cross_side_edge_share(SORTED_GRAPH, x) == pytest.approx(0.0)


def test_cross_side_edge_share_is_none_with_no_surviving_edge():
    assert m.cross_side_edge_share(None, {"a": 1.0}) is None
    assert m.cross_side_edge_share(SORTED_GRAPH, {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}) is None


def test_minority_share_is_the_smaller_sides_fraction():
    x = {"a": 1.0, "b": 1.0, "c": 1.0, "d": -1.0}
    assert m.minority_share(x) == pytest.approx(0.25)


def test_minority_share_excludes_undecided_agents():
    x = {"a": 1.0, "b": -1.0, "c": 0.0}
    assert m.minority_share(x) == pytest.approx(0.5)


def test_minority_share_is_none_when_nobody_has_a_side():
    assert m.minority_share({"a": 0.0, "b": 0.0}) is None


# --- Fix 4: global_disagreement, (pairwise/local) disagreement, nci, -------
# --- bimodality, on hand-built six-node graphs ------------------------------
#
# Two separated camps (two disjoint triangles) and a mixed ring (alternating
# sides around a 6-cycle) use the SAME multiset of x values -- three +1s and
# three -1s -- so pairwise_disagreement, which ignores structure entirely,
# is identical on both; every structural metric differs.

TWO_CAMPS = nx.Graph([
    ("a", "b"), ("b", "c"), ("a", "c"),  # triangle 1
    ("d", "e"), ("e", "f"), ("d", "f"),  # triangle 2, no edge to triangle 1
])
MIXED_RING = nx.Graph([
    ("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "a"),
])
CAMPS_X = {"a": 1.0, "b": 1.0, "c": 1.0, "d": -1.0, "e": -1.0, "f": -1.0}
RING_X = {"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0, "e": 1.0, "f": -1.0}
ALL_EQUAL_X = {n: 0.5 for n in "abcdef"}


def test_global_disagreement_is_zero_within_two_separated_camps():
    """Every edge joins two agents who agree exactly, so every DG_i is 0."""
    assert m.global_disagreement(TWO_CAMPS, CAMPS_X, population=6) == pytest.approx(0.0)


def test_global_disagreement_on_a_mixed_ring():
    """Each node's two ring neighbours are both on the opposite side (diff
    2, squared 4), so DG_i = 4 for every node: DG = (6 * 4) / (2 * 6) = 2."""
    assert m.global_disagreement(MIXED_RING, RING_X, population=6) == pytest.approx(2.0)


def test_global_disagreement_on_a_graph_with_an_isolated_node():
    """The isolated node has no neighbour and contributes no DG_i, but the
    denominator is still the full population -- not the count of agents
    that qualified."""
    graph = nx.Graph([("a", "b"), ("c", "d")])
    graph.add_node("iso")  # no edges at all
    x = {"a": 1.0, "b": 1.0, "c": -1.0, "d": -1.0, "iso": 5.0}
    # a-b and c-d each agree exactly: both DG_i = 0. Population is 5.
    assert m.global_disagreement(graph, x, population=5) == pytest.approx(0.0)


def test_global_disagreement_is_none_when_nobody_has_a_known_neighbour():
    graph = nx.Graph()
    graph.add_node("iso")
    assert m.global_disagreement(graph, {"iso": 1.0}, population=1) is None
    assert m.global_disagreement(None, {"a": 1.0}, population=1) is None


def test_pairwise_disagreement_ignores_structure():
    """Same multiset of values, no graph at all: 9 cross-camp pairs (of 15)
    differ by 2, so the mean is 18 / 15 = 1.2 -- identical for the camps'
    values and the ring's, since both are three +1s and three -1s."""
    assert m.pairwise_disagreement(CAMPS_X) == pytest.approx(1.2)
    assert m.pairwise_disagreement(RING_X) == pytest.approx(1.2)


def test_pairwise_disagreement_is_none_below_two_agents():
    assert m.pairwise_disagreement({"a": 1.0}) is None


def test_local_disagreement_on_two_separated_camps_is_zero():
    assert m.local_disagreement(TWO_CAMPS, CAMPS_X) == pytest.approx(0.0)


def test_local_disagreement_on_a_mixed_ring():
    """Every edge differs by exactly 2."""
    assert m.local_disagreement(MIXED_RING, RING_X) == pytest.approx(2.0)


def test_local_disagreement_is_none_with_no_surviving_edge():
    assert m.local_disagreement(None, {"a": 1.0}) is None


def test_disagreement_ratio_below_one_means_neighbours_agree_more_than_random():
    """Two camps: local 0.0 / pairwise 1.2 = 0.0 -- neighbours agree
    completely, far below two random agents."""
    ratio = m.disagreement_ratio(
        m.local_disagreement(TWO_CAMPS, CAMPS_X),
        m.pairwise_disagreement(CAMPS_X),
    )
    assert ratio == pytest.approx(0.0)


def test_disagreement_ratio_above_one_means_neighbours_disagree_more_than_random():
    """Mixed ring: local 2.0 / pairwise 1.2 = 5/3 -- every edge joins
    opposite sides, worse than two random agents."""
    ratio = m.disagreement_ratio(
        m.local_disagreement(MIXED_RING, RING_X),
        m.pairwise_disagreement(RING_X),
    )
    assert ratio == pytest.approx(5 / 3)


def test_disagreement_ratio_is_none_when_everyone_agrees():
    """Both local and pairwise disagreement are exactly 0: the ratio is
    0/0, undefined -- not 0 ("neighbours agree less than random"), since
    there is no disagreement anywhere to compare."""
    local = m.local_disagreement(TWO_CAMPS, ALL_EQUAL_X)
    pairwise = m.pairwise_disagreement(ALL_EQUAL_X)
    assert local == pytest.approx(0.0) and pairwise == pytest.approx(0.0)
    assert m.disagreement_ratio(local, pairwise) is None


# --- nci and its permutation null -------------------------------------------


def test_nci_is_perfect_within_two_separated_camps():
    """Every node's neighbours all share its own exact value, so each
    node's neighbour-mean equals its own x: a perfect correlation."""
    assert m.nci(TWO_CAMPS, CAMPS_X) == pytest.approx(1.0)


def test_nci_is_perfectly_negative_on_a_mixed_ring():
    """Every node's two ring neighbours are both on the opposite side, so
    the neighbour-mean is the exact negation of the node's own x."""
    assert m.nci(MIXED_RING, RING_X) == pytest.approx(-1.0)


def test_nci_is_none_when_everyone_agrees():
    """Zero variance on both sides of the correlation: None, not 0 ("no
    relationship"), since there is no spread to correlate."""
    assert m.nci(TWO_CAMPS, ALL_EQUAL_X) is None


def test_nci_null_is_reproducible_from_the_same_seed_key():
    first = m.nci_null(MIXED_RING, RING_X, "1001|3|nci_null")
    second = m.nci_null(MIXED_RING, RING_X, "1001|3|nci_null")
    assert first == second


def test_nci_null_differs_across_seed_keys():
    a = m.nci_null(MIXED_RING, RING_X, "1001|3|nci_null")
    b = m.nci_null(MIXED_RING, RING_X, "1001|4|nci_null")
    assert a != b


def test_nci_null_is_none_when_nci_itself_is_none():
    assert m.nci_null(TWO_CAMPS, ALL_EQUAL_X, "k") == (None, None)
    assert m.nci_null(None, CAMPS_X, "k") == (None, None)


def test_a_perfectly_polarized_network_clears_its_own_nci_null():
    """Two 3-node camps only has 2/C(6,3) = 10% of permutations landing a
    perfect split by chance, which the 97.5th-percentile null still
    reaches; two 5-cliques (as assortativity's own analogous test uses)
    makes a perfect permutation rare enough (2/C(10,5) = 252, under 1%)
    that the null's upper tail sits reliably below the real value."""
    pro = [f"p{i}" for i in range(5)]
    con = [f"c{i}" for i in range(5)]
    wide = nx.Graph()
    for group in (pro, con):
        for i, u in enumerate(group):
            for v in group[i + 1:]:
                wide.add_edge(u, v)
    x = {**{n: 1.0 for n in pro}, **{n: -1.0 for n in con}}
    nci_value = m.nci(wide, x)
    null_mean, null_hi = m.nci_null(wide, x, "seed|1")
    assert nci_value == pytest.approx(1.0)
    assert null_mean < nci_value and null_hi < nci_value


# --- bimodality --------------------------------------------------------------


def test_bimodality_of_a_perfect_two_point_split():
    """Three +1s and three -1s: mean 0, m2 = 1, m3 = 0 (symmetric), m4 = 1.
    g = 0, k = 1/1 - 3 = -2. denom = -2 + 3*25/(4*3) = 4.25 = 17/4.
    bimodality = 1 / (17/4) = 4/17."""
    values = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    assert m.bimodality(values) == pytest.approx(4 / 17)


def test_bimodality_is_none_below_four_values():
    assert m.bimodality([1.0, 2.0, 3.0]) is None


def test_bimodality_is_none_when_variance_is_zero():
    assert m.bimodality([1.0, 1.0, 1.0, 1.0]) is None
