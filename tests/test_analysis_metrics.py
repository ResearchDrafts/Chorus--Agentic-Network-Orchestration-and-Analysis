"""analysis/metrics.py: the classification rules, by hand.

Every test here states the expected answer as arithmetic rather than as
"whatever the function returns". These are the definitions every reported
flip, harden and t_stationary is made of, and a definition that drifted would
change a result without changing a line of the analysis that reads it.
"""

from __future__ import annotations

import json

import networkx as nx
import pandas as pd
import pytest

from analysis import metrics as m

TAU = 0.1
TAU_CONF = 0.1


# --- sign, and the x == 0 rule -------------------------------------------


def test_zero_is_its_own_sign():
    """The midpoint is the one reading for which "which side is it on" has no
    answer. Folding it in with a pole would make the midpoint arbitrarily
    belong to one side and turn half of the undecided population's movements
    into hardening."""
    assert m.sign_of(0.0) == 0
    assert m.sign_of(1e-9) == 1
    assert m.sign_of(-1e-9) == -1
    assert m.sign_of(0.5) == 1 and m.sign_of(-0.5) == -1


def test_moving_off_exactly_zero_is_a_flip():
    """Because 0 is its own sign, a move off it is a change of sign -- and if
    it clears τ, a flip."""
    assert m.classify(0.5, 0.0, 0.5, 0.0, TAU, TAU_CONF) == "flip"
    assert m.classify(-0.5, 0.0, -0.5, 0.0, TAU, TAU_CONF) == "flip"


def test_moving_onto_exactly_zero_is_a_flip():
    assert m.classify(-0.5, 0.5, 0.0, 0.0, TAU, TAU_CONF) == "flip"


def test_moving_off_zero_by_less_than_tau_is_not_a_flip():
    """An agent oscillating either side of the midpoint by less than the
    instrument's own noise has not changed its mind, and counting it would
    make the flip rate a function of how many agents sit near 0."""
    assert m.classify(0.05, 0.0, 0.05, 0.0, TAU, TAU_CONF) == "none"


# --- the six classes ------------------------------------------------------


@pytest.mark.parametrize(
    ("dx", "x_before", "x_after", "dconf", "expected"),
    [
        # Crossed the midpoint and cleared τ.
        (-0.6, 0.3, -0.3, 0.0, "flip"),
        # Crossed, but by less than τ: not a flip.
        (-0.08, 0.04, -0.04, 0.0, "none"),
        # Same side, further out by more than τ.
        (0.3, 0.3, 0.6, 0.0, "harden"),
        # Same side on the con pole, further out.
        (-0.3, -0.3, -0.6, 0.0, "harden"),
        # Same side, back towards the middle by more than τ.
        (-0.3, 0.6, 0.3, 0.0, "soften"),
        # Flipped AND became much more decided: movement wins, since that is
        # what is being measured.
        (0.15, -0.075, 0.075, 0.9, "flip"),
        # Did not move, became more decided.
        (0.0, 0.5, 0.5, 0.2, "conf_up"),
        # Did not move, became less decided.
        (0.0, 0.5, 0.5, -0.2, "conf_down"),
        # Did not move, confidence change under τ_conf.
        (0.0, 0.5, 0.5, 0.05, "none"),
        # Moved a lot AND confidence changed: movement wins, since that is
        # what is being measured.
        (0.4, 0.2, 0.6, 0.9, "harden"),
        # Exactly at the thresholds: "more than" means strictly more.
        (TAU, 0.0, TAU, 0.0, "none"),
        (0.0, 0.5, 0.5, TAU_CONF, "none"),
    ],
)
def test_each_outcome_class(dx, x_before, x_after, dconf, expected):
    assert m.classify(dx, x_before, x_after, dconf, TAU, TAU_CONF) == expected


def test_every_class_is_reachable_and_named_once():
    assert set(m.OUTCOMES) == {
        "flip", "harden", "soften", "conf_up", "conf_down", "none"
    }


# --- the population metrics ----------------------------------------------


def test_population_variance_has_no_bessel_correction():
    """These agents ARE the population; the inferential step happens later,
    across runs. ddof=1 would also be undefined at n=1, which is a real case
    when every probe but one failed in a round."""
    assert m.population_variance([0.0, 1.0]) == pytest.approx(0.25)
    assert m.population_variance([0.5]) == pytest.approx(0.0)
    assert m.population_variance([-1.0, 1.0]) == pytest.approx(1.0)


def test_variance_of_nothing_raises():
    with pytest.raises(ValueError, match="undefined"):
        m.population_variance([])


def test_extremity_counts_strictly_beyond_two_thirds():
    """|x| > 2/3 is a probe mean beyond 6 or below 2 on the 7-point scale."""
    assert m.EXTREMITY_THRESHOLD == pytest.approx(2 / 3)
    xs = [0.0, 0.5, 2 / 3, 0.667, -0.9, 1.0]
    share = sum(1 for x in xs if abs(x) > m.EXTREMITY_THRESHOLD) / len(xs)
    # 0.667, -0.9 and 1.0 -- exactly 2/3 does not count.
    assert share == pytest.approx(3 / 6)


# --- assortativity --------------------------------------------------------


FOUR_NODES = nx.Graph([("a", "b"), ("c", "d")])


def test_assortativity_is_one_when_each_edge_joins_equals():
    """Two edges, each joining two agents at the same position: perfectly
    sorted."""
    assert m.assortativity(
        FOUR_NODES, {"a": 1.0, "b": 1.0, "c": -1.0, "d": -1.0}
    ) == pytest.approx(1.0)


def test_assortativity_is_minus_one_when_each_edge_joins_opposites():
    assert m.assortativity(
        FOUR_NODES, {"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0}
    ) == pytest.approx(-1.0)


def test_assortativity_is_null_when_everyone_agrees():
    """The coefficient is a correlation, so zero variance divides by zero.
    None, not 0 ("not sorted") or 1 ("perfectly sorted"), both of which would
    be read as findings."""
    assert m.assortativity(FOUR_NODES, {n: 0.3 for n in "abcd"}) is None


def test_assortativity_is_null_without_a_graph():
    """An alpha-sampling run has no fixed structure: "agents are not adjacent
    to their own side" and "there is no adjacency" are different facts."""
    assert m.assortativity(None, {"a": 1.0, "b": -1.0}) is None


def test_assortativity_ignores_agents_with_no_reading():
    """An agent that never produced one is absent from the subgraph rather
    than present with a fabricated position."""
    graph = nx.Graph([("a", "b"), ("c", "d"), ("a", "z")])
    value = m.assortativity(graph, {"a": 1.0, "b": 1.0, "c": -1.0, "d": -1.0})
    assert value == pytest.approx(1.0)


def test_assortativity_is_null_when_no_edge_survives():
    graph = nx.Graph([("a", "b")])
    assert m.assortativity(graph, {"a": 1.0, "z": -1.0}) is None


# --- t_stationary ---------------------------------------------------------


EPSILON = {"var_x": 0.01, "extremity": 0.01, "assortativity": 0.01}


def _series(values: list[tuple[float, float, float]]) -> list[dict]:
    """One round per entry: (var_x, extremity, assortativity)."""
    return [
        {"turn": turn, "var_x": v, "extremity": e, "assortativity": a}
        for turn, (v, e, a) in enumerate(values)
    ]


def test_t_stationary_is_the_first_round_ending_three_quiet_ones():
    """Rounds 1 and 2 move; 3, 4 and 5 do not. The first round ending a run
    of three consecutive small changes is 5."""
    rounds = _series(
        [
            (0.10, 0.10, 0.10),
            (0.30, 0.30, 0.30),   # round 1: moved
            (0.50, 0.50, 0.50),   # round 2: moved
            (0.502, 0.502, 0.502),  # 3: quiet
            (0.503, 0.503, 0.503),  # 4: quiet
            (0.504, 0.504, 0.504),  # 5: quiet -> three in a row ending here
            (0.505, 0.505, 0.505),
        ]
    )
    assert m.stationary_round(rounds, EPSILON, k=6) == (5, True)


def test_a_series_that_never_settles_reports_k_and_says_so():
    """K with stationary False, so a run that never settled is not reported
    as having settled on its last round."""
    rounds = _series([(0.1 * i, 0.1 * i, 0.1 * i) for i in range(7)])
    assert m.stationary_round(rounds, EPSILON, k=6) == (6, False)


def test_three_quiet_rounds_are_needed_not_two():
    rounds = _series(
        [
            (0.10, 0.10, 0.10),
            (0.50, 0.50, 0.50),
            (0.501, 0.501, 0.501),
            (0.502, 0.502, 0.502),
        ]
    )
    # Only rounds 2 and 3 are quiet.
    assert m.stationary_round(rounds, EPSILON, k=3) == (3, False)


def test_one_metric_still_moving_blocks_stationarity():
    """All three have to be quiet. A population whose variance has settled
    while its agents are still re-sorting on the graph has not stopped
    changing shape."""
    rounds = _series(
        [
            (0.50, 0.50, 0.10),
            (0.501, 0.501, 0.40),
            (0.502, 0.502, 0.70),
            (0.503, 0.503, 1.00),
        ]
    )
    assert m.stationary_round(rounds, EPSILON, k=3) == (3, False)


def test_an_undefined_assortativity_is_not_within_epsilon():
    """"We cannot tell whether this metric moved" is not evidence that it
    stopped moving -- and the alternative would make a run stationary
    precisely because its population had collapsed to one position."""
    # Six rounds, every metric flat. Nulling round 1's assortativity makes
    # the changes at rounds 1 and 2 uncomputable, leaving 3, 4 and 5 as the
    # first run of three.
    rounds = _series([(0.5, 0.5, 0.5)] * 6)
    rounds[1]["assortativity"] = None
    assert m.stationary_round(rounds, EPSILON, k=5) == (5, True)
    # Null round 3 as well and no window of three survives, even though
    # nothing in the series moved.
    rounds[3]["assortativity"] = None
    assert m.stationary_round(rounds, EPSILON, k=5) == (5, False)


def test_each_metric_gets_its_own_epsilon():
    """A variance, a proportion and a correlation are on different scales, so
    one number would be far too loose for one and far too tight for
    another."""
    rounds = _series([(0.50, 0.50, 0.50), (0.55, 0.501, 0.501),
                      (0.60, 0.502, 0.502), (0.65, 0.503, 0.503)])
    loose = {"var_x": 0.1, "extremity": 0.01, "assortativity": 0.01}
    assert m.stationary_round(rounds, loose, k=3) == (3, True)
    assert m.stationary_round(rounds, EPSILON, k=3) == (3, False)


def test_a_missing_epsilon_raises():
    with pytest.raises(ValueError, match="no epsilon"):
        m.stationary_round(_series([(0.5, 0.5, 0.5)] * 4), {"var_x": 0.1}, k=3)


def test_parse_epsilon_requires_all_three():
    assert m.parse_epsilon(
        ["var_x=0.01", "extremity=0.02", "assortativity=0.05"]
    ) == {"var_x": 0.01, "extremity": 0.02, "assortativity": 0.05}
    with pytest.raises(ValueError, match="missing"):
        m.parse_epsilon(["var_x=0.01"])
    with pytest.raises(ValueError, match="unknown metric"):
        m.parse_epsilon(["mean_x=0.01"])
    with pytest.raises(ValueError, match="metric=value"):
        m.parse_epsilon(["var_x"])


def test_epsilon_from_e4_is_the_95th_percentile_of_the_controls_wobble():
    """Hand-computed: the control's own round-to-round change, per metric."""
    rounds = []
    # One E4 run whose var_x steps by 0.00, 0.10, 0.20, 0.30.
    for turn, var in enumerate([0.0, 0.0, 0.1, 0.3, 0.6]):
        rounds.append(
            {
                "run_id": "e4_english_ws_s01", "experiment": "e4", "turn": turn,
                "var_x": var, "extremity": 0.0, "assortativity": 0.0,
            }
        )
    # And one E3 run with wild swings, which must not contribute.
    for turn in range(5):
        rounds.append(
            {
                "run_id": "e3_english_ws_s01", "experiment": "e3", "turn": turn,
                "var_x": 10.0 * turn, "extremity": 0.0, "assortativity": 0.0,
            }
        )

    epsilon = m.epsilon_from_e4(rounds)
    # |changes| = [0.0, 0.1, 0.2, 0.3]; the 95th percentile by linear
    # interpolation is at position 3 * 0.95 = 2.85, so 0.2*0.15 + 0.3*0.85.
    assert epsilon["var_x"] == pytest.approx(0.2 * 0.15 + 0.3 * 0.85)
    assert epsilon["extremity"] == pytest.approx(0.0)


def test_epsilon_from_e4_without_a_control_raises():
    """An ε invented here would set the threshold for every reported
    t_stationary, and a wrong one is invisible in the output."""
    rounds = [
        {"run_id": "e3_x", "experiment": "e3", "turn": t,
         "var_x": 0.1, "extremity": 0.1, "assortativity": 0.1}
        for t in range(3)
    ]
    with pytest.raises(ValueError, match="needs E4 runs"):
        m.epsilon_from_e4(rounds)


# --- τ --------------------------------------------------------------------


def test_a_language_with_no_tau_is_refused(tmp_path):
    """Classifying an uncalibrated arm's movements against another arm's
    noise would look exactly like a calibrated result."""
    path = tmp_path / "tau.json"
    path.write_text(json.dumps({"languages": {"english": {"tau": 0.1, "tau_conf": 0.1}}}))
    tau = m.load_tau(path)
    assert m.tau_for(tau, "english", path) == (0.1, 0.1)
    with pytest.raises(m.MissingTau, match="hindi"):
        m.tau_for(tau, "hindi", path)


def test_a_null_tau_is_refused(tmp_path):
    path = tmp_path / "tau.json"
    path.write_text(json.dumps({"languages": {"hindi": {"tau": None, "tau_conf": None}}}))
    with pytest.raises(m.MissingTau):
        m.tau_for(m.load_tau(path), "hindi", path)


def test_a_missing_tau_file_names_the_script_that_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate_tau"):
        m.load_tau(tmp_path / "nope.json")


def test_a_smoke_tau_is_flagged(tmp_path, capsys):
    path = tmp_path / "tau.json"
    path.write_text(json.dumps({"smoke": True, "languages": {}}))
    m.load_tau(path)
    assert "smoke run" in capsys.readouterr().out


# --- the experiment label -------------------------------------------------


def test_the_experiment_comes_off_the_run_id_prefix():
    assert m.experiment_of("e3_english_ws_s01") == "e3"
    assert m.experiment_of("e4_romanized_hindi_ws_s12") == "e4"
    # Not a raise: a hand-named run still appears in the tables instead of
    # stopping the analysis.
    assert m.experiment_of("my_pilot_run") == "other"


# --- probe rows against a tiny frame --------------------------------------


def _frame(rows: list[dict]) -> pd.DataFrame:
    """A frame with the columns probe_rows reads, defaulted."""
    base = {
        "is_valid_for_analysis": True, "side_conf": 1.0, "probe_mean": 4.0,
        "probe_mass": 1.0, "reply_prompt_tokens": 10.0, "probe_prompt_tokens": 8.0,
        "n_repeat_reads": 0.0, "max_prior_exposures": 0.0,
        "n_distinct_items_read_so_far": 0.0,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


CONFIG = {
    "run_id": "e3_english_ws_s01", "language_condition": "english", "seed": 1001,
    "delivery": "on", "M": 2, "N": 1, "K": 3, "topic": "t",
    "model_backend_id": "gpt-4o", "neighbor_strategy": "graph",
}


def test_round_zero_carries_no_outcome():
    rows = m.probe_rows(
        _frame([
            {"speaker_agent_id": "a", "turn": 0, "x": 0.0},
            {"speaker_agent_id": "a", "turn": 1, "x": 0.5},
        ]),
        CONFIG, TAU, TAU_CONF,
    )
    assert rows[0]["turn"] == 0
    assert rows[0]["outcome"] is None and rows[0]["dx"] is None
    assert rows[1]["outcome"] == "flip" and rows[1]["dx"] == pytest.approx(0.5)


def test_dx_is_measured_against_the_previous_valid_round():
    """A round whose probe failed produced no reading. Comparing to the last
    thing that WAS measured beats inventing one, and prev_turn records that
    a gap was crossed."""
    rows = m.probe_rows(
        _frame([
            {"speaker_agent_id": "a", "turn": 0, "x": 0.0},
            {"speaker_agent_id": "a", "turn": 1, "x": None,
             "is_valid_for_analysis": False},
            {"speaker_agent_id": "a", "turn": 2, "x": 0.5},
        ]),
        CONFIG, TAU, TAU_CONF,
    )
    assert [r["turn"] for r in rows] == [0, 2]
    assert rows[1]["prev_turn"] == 0
    assert rows[1]["dx"] == pytest.approx(0.5)


def test_round_metrics_count_a_missing_agent_as_failed():
    """n_failed is M minus the agents with a valid reading: an agent whose
    reply failed has a row and one that never appeared does not, and both are
    the same kind of missing here."""
    frame = _frame([
        {"speaker_agent_id": "a", "turn": 0, "x": 0.0},
        {"speaker_agent_id": "b", "turn": 0, "x": 1.0},
        {"speaker_agent_id": "a", "turn": 1, "x": 0.0},
    ])
    probes = m.probe_rows(frame, CONFIG, TAU, TAU_CONF)
    rounds = m.round_rows(frame, probes, CONFIG, None)
    assert [(r["turn"], r["n_valid"], r["n_failed"]) for r in rounds] == [
        (0, 2, 0), (1, 1, 1)
    ]
    assert rounds[0]["var_x"] == pytest.approx(0.25)
    assert rounds[1]["var_x"] == pytest.approx(0.0)


def test_the_graph_metric_carries_a_missing_agent_forward():
    """And nothing else does: a mean over carried-forward values would count
    one reading twice, while an assortativity over a shrinking node set would
    move as agents dropped out and read as structural change."""
    frame = _frame([
        {"speaker_agent_id": "a", "turn": 0, "x": 1.0},
        {"speaker_agent_id": "b", "turn": 0, "x": 1.0},
        {"speaker_agent_id": "c", "turn": 0, "x": -1.0},
        {"speaker_agent_id": "d", "turn": 0, "x": -1.0},
        {"speaker_agent_id": "a", "turn": 1, "x": 1.0},
        {"speaker_agent_id": "b", "turn": 1, "x": 1.0},
        {"speaker_agent_id": "c", "turn": 1, "x": -1.0},
    ])
    config = {**CONFIG, "M": 4}
    probes = m.probe_rows(frame, config, TAU, TAU_CONF)
    rounds = m.round_rows(frame, probes, config, FOUR_NODES)

    # d is missing at round 1, carried forward at -1.0, so the graph still
    # reads perfectly sorted...
    assert rounds[1]["assortativity"] == pytest.approx(1.0)
    # ...while the mean is over the three agents actually measured.
    assert rounds[1]["mean_x"] == pytest.approx(1 / 3)
    assert rounds[1]["n_valid"] == 3 and rounds[1]["n_failed"] == 1


def test_outcome_shares_sum_to_one_and_are_null_at_round_zero():
    frame = _frame([
        {"speaker_agent_id": "a", "turn": 0, "x": 0.5},
        {"speaker_agent_id": "b", "turn": 0, "x": 0.5},
        {"speaker_agent_id": "a", "turn": 1, "x": 0.9},   # harden
        {"speaker_agent_id": "b", "turn": 1, "x": 0.5},   # none
    ])
    probes = m.probe_rows(frame, CONFIG, TAU, TAU_CONF)
    rounds = m.round_rows(frame, probes, CONFIG, None)
    assert all(rounds[0][f"share_{name}"] is None for name in m.OUTCOMES)
    assert rounds[1]["share_harden"] == pytest.approx(0.5)
    assert rounds[1]["share_none"] == pytest.approx(0.5)
    assert sum(rounds[1][f"share_{n}"] for n in m.OUTCOMES) == pytest.approx(1.0)


def test_run_row_reports_the_change_from_round_zero():
    frame = _frame([
        {"speaker_agent_id": "a", "turn": 0, "x": 0.0},
        {"speaker_agent_id": "b", "turn": 0, "x": 0.0},
        {"speaker_agent_id": "a", "turn": 1, "x": 1.0},
        {"speaker_agent_id": "b", "turn": 1, "x": -1.0},
    ])
    probes = m.probe_rows(frame, CONFIG, TAU, TAU_CONF)
    rounds = m.round_rows(frame, probes, CONFIG, None)
    row = m.run_row(rounds, probes, CONFIG, EPSILON)

    assert row["final_var_x"] == pytest.approx(1.0)
    assert row["d_var_x"] == pytest.approx(1.0)
    assert row["final_extremity"] == pytest.approx(1.0)
    assert row["d_extremity"] == pytest.approx(1.0)
    assert row["rate_flip"] == pytest.approx(1.0)
    assert row["experiment"] == "e3" and row["language"] == "english"
    # Two rounds of changes is fewer than the three-round window, so it
    # cannot have been shown stationary.
    assert row["stationary"] is False and row["t_stationary"] == CONFIG["K"]
