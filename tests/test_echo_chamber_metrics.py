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
