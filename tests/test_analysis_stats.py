"""analysis/stats.py: the arithmetic, against hand calculations.

The difference-in-differences and the Holm correction are checked against
numbers worked out in the docstrings rather than against the functions'
own output. Both are small enough to get wrong invisibly: a DiD with the
sign flipped and a Holm correction applied to the wrong family both produce
a perfectly plausible table.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
import pytest

from analysis import stats as st


def _run_metrics(rows: list[dict]) -> pd.DataFrame:
    base = {
        "final_var_x": 0.0, "final_extremity": 0.0,
        "final_assortativity": 0.0, "t_stationary": 0,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


# --- paired differences and the DiD --------------------------------------


def test_paired_differences_are_within_a_seed():
    """Four runs sharing a seed share a graph, the same neighbour draws and
    the same bank items, so an English run and a Hindi run at seed 1003 are a
    matched pair."""
    frame = _run_metrics([
        {"experiment": "e3", "language": "english", "seed": 1001, "final_var_x": 0.10},
        {"experiment": "e3", "language": "hindi", "seed": 1001, "final_var_x": 0.30},
        {"experiment": "e3", "language": "english", "seed": 1002, "final_var_x": 0.20},
        {"experiment": "e3", "language": "hindi", "seed": 1002, "final_var_x": 0.25},
    ])
    diffs = st.paired_differences(frame, "final_var_x", "e3", "hindi")
    assert diffs == {1001: pytest.approx(0.20), 1002: pytest.approx(0.05)}


def test_an_unpaired_seed_contributes_nothing():
    """Silently dropping the pair is right where silently filling in the
    missing half would not be."""
    frame = _run_metrics([
        {"experiment": "e3", "language": "english", "seed": 1001, "final_var_x": 0.1},
        {"experiment": "e3", "language": "hindi", "seed": 1001, "final_var_x": 0.3},
        {"experiment": "e3", "language": "hindi", "seed": 1002, "final_var_x": 0.9},
    ])
    assert list(st.paired_differences(frame, "final_var_x", "e3", "hindi")) == [1001]


def test_a_null_metric_drops_its_pair():
    frame = _run_metrics([
        {"experiment": "e3", "language": "english", "seed": 1001,
         "final_assortativity": None},
        {"experiment": "e3", "language": "hindi", "seed": 1001,
         "final_assortativity": 0.3},
    ])
    assert st.paired_differences(frame, "final_assortativity", "e3", "hindi") == {}


def test_the_difference_in_differences_arithmetic():
    """Hand-computed.

        seed 1001   E3: hindi 0.30 - english 0.10 = +0.20
                    E4: hindi 0.12 - english 0.10 = +0.02
                    DiD                            = +0.18
        seed 1002   E3: hindi 0.25 - english 0.20 = +0.05
                    E4: hindi 0.22 - english 0.20 = +0.02
                    DiD                            = +0.03

        mean DiD = (0.18 + 0.03) / 2 = 0.105

    Which is the whole point of running the control: the raw E3 difference
    averages 0.125, and 0.02 of that is present in an arm whose agents read
    nothing that argues anything.
    """
    frame = _run_metrics([
        {"experiment": "e3", "language": "english", "seed": 1001, "final_var_x": 0.10},
        {"experiment": "e3", "language": "hindi", "seed": 1001, "final_var_x": 0.30},
        {"experiment": "e4", "language": "english", "seed": 1001, "final_var_x": 0.10},
        {"experiment": "e4", "language": "hindi", "seed": 1001, "final_var_x": 0.12},
        {"experiment": "e3", "language": "english", "seed": 1002, "final_var_x": 0.20},
        {"experiment": "e3", "language": "hindi", "seed": 1002, "final_var_x": 0.25},
        {"experiment": "e4", "language": "english", "seed": 1002, "final_var_x": 0.20},
        {"experiment": "e4", "language": "hindi", "seed": 1002, "final_var_x": 0.22},
    ])
    rows = st.network_contrasts(frame)
    by_scope = {
        row["scope"]: row
        for row in rows
        if row["metric"] == "final_var_x" and row["language"] == "hindi"
    }
    assert by_scope["within_e3"]["mean_diff"] == pytest.approx(0.125)
    assert by_scope["within_e4"]["mean_diff"] == pytest.approx(0.02)
    assert by_scope["did"]["mean_diff"] == pytest.approx(0.105)
    assert by_scope["did"]["n"] == 2
    assert by_scope["did"]["seeds"] == "1001;1002"


def test_the_did_only_uses_seeds_present_in_both_experiments():
    frame = _run_metrics([
        {"experiment": "e3", "language": "english", "seed": 1001, "final_var_x": 0.1},
        {"experiment": "e3", "language": "hindi", "seed": 1001, "final_var_x": 0.3},
        {"experiment": "e3", "language": "english", "seed": 1002, "final_var_x": 0.1},
        {"experiment": "e3", "language": "hindi", "seed": 1002, "final_var_x": 0.9},
        {"experiment": "e4", "language": "english", "seed": 1001, "final_var_x": 0.1},
        {"experiment": "e4", "language": "hindi", "seed": 1001, "final_var_x": 0.1},
    ])
    did = next(
        r for r in st.network_contrasts(frame)
        if r["metric"] == "final_var_x" and r["scope"] == "did"
    )
    assert did["n"] == 1 and did["mean_diff"] == pytest.approx(0.2)


def test_every_metric_and_language_gets_all_three_scopes():
    frame = _run_metrics([
        {"experiment": e, "language": lang, "seed": 1001}
        for e in ("e3", "e4")
        for lang in ("english", "hinglish", "romanized_hindi", "hindi")
    ])
    rows = st.network_contrasts(frame)
    assert len(rows) == len(st.NETWORK_METRICS) * 3 * 3
    assert {r["scope"] for r in rows} == {"within_e3", "within_e4", "did"}


# --- the bootstrap --------------------------------------------------------


def test_the_bootstrap_is_reproducible():
    """An interval that moved between two runs of the same script on the same
    data could not be reported."""
    values = [0.1, 0.3, -0.2, 0.4, 0.0]
    assert st.bootstrap_ci(values) == st.bootstrap_ci(values)


def test_the_bootstrap_brackets_the_mean():
    values = [0.1, 0.3, -0.2, 0.4, 0.0]
    low, high = st.bootstrap_ci(values)
    assert low <= np.mean(values) <= high


def test_one_seed_gives_a_point_not_a_zero_width_interval():
    """One seed is one observation: a bootstrap over a single value returns
    the value and would look like certainty either way, so it is reported as
    the point it is."""
    assert st.bootstrap_ci([0.42]) == (0.42, 0.42)


def test_no_seeds_gives_nan():
    low, high = st.bootstrap_ci([])
    assert np.isnan(low) and np.isnan(high)


# --- the Wilcoxon test ----------------------------------------------------


def test_wilcoxon_matches_scipy_on_a_known_sample():
    from scipy import stats as scipy_stats

    values = [0.2, 0.3, -0.1, 0.4, 0.5, 0.25]
    assert st.wilcoxon_p(values) == pytest.approx(
        scipy_stats.wilcoxon(values, zero_method="zsplit").pvalue
    )


def test_all_zero_differences_give_p_one():
    """scipy refuses to compute it, and the answer is obvious: no evidence of
    any difference at all."""
    assert st.wilcoxon_p([0.0, 0.0, 0.0]) == 1.0


def test_fewer_than_two_observations_has_no_test():
    assert st.wilcoxon_p([0.3]) is None
    assert st.wilcoxon_p([]) is None


def test_the_underpowered_warning_states_the_attainable_p(capsys):
    """With n paired observations the smallest attainable two-sided p is
    2/2**n, which is 0.0625 at n = 5 -- so a non-significant row below six
    seeds says nothing about the effect, only about the design."""
    st.warn_if_underpowered([{"n": 5}, {"n": 4}])
    printed = capsys.readouterr().out
    assert "at most 5 paired seed(s)" in printed
    assert "0.0625" in printed


def test_no_warning_at_six_seeds(capsys):
    st.warn_if_underpowered([{"n": 6}])
    assert capsys.readouterr().out == ""


# --- Holm -----------------------------------------------------------------


def test_holm_against_a_hand_calculation():
    """Holm-Bonferroni, step-down, with the running maximum enforced.

        sorted p   0.010  0.020  0.030
        multiplier     3      2      1
        product    0.030  0.040  0.030
        running max 0.030  0.040  0.040   <- the third cannot fall below the
                                             second; monotonicity is the half
                                             of Holm that is easy to omit

    So the adjusted values are 0.030, 0.040 and 0.040, and at α = 0.05 all
    three are rejected.
    """
    rows = [
        {"family": "f", "p_raw": 0.030},
        {"family": "f", "p_raw": 0.010},
        {"family": "f", "p_raw": 0.020},
    ]
    adjusted = {row["p_raw"]: row for row in st.holm_within_families(rows)}
    assert adjusted[0.010]["p_holm"] == pytest.approx(0.030)
    assert adjusted[0.020]["p_holm"] == pytest.approx(0.040)
    assert adjusted[0.030]["p_holm"] == pytest.approx(0.040)
    assert all(row["reject_at_05"] for row in rows)
    assert all(row["family_size"] == 3 for row in rows)


def test_holm_does_not_reject_what_bonferroni_would_not():
    rows = [{"family": "f", "p_raw": p} for p in (0.02, 0.9, 0.9)]
    out = st.holm_within_families(rows)
    # 0.02 * 3 = 0.06, above 0.05.
    assert out[0]["p_holm"] == pytest.approx(0.06)
    assert out[0]["reject_at_05"] is False


def test_holm_is_applied_within_a_family_not_across_all_of_them():
    """Correcting the language comparisons for the strength checks run
    alongside them would penalise each question for the other being asked."""
    rows = [
        {"family": "a", "p_raw": 0.01},
        {"family": "a", "p_raw": 0.02},
        {"family": "b", "p_raw": 0.01},
    ]
    out = st.holm_within_families(rows)
    assert out[0]["p_holm"] == pytest.approx(0.02)   # family of 2
    assert out[2]["p_holm"] == pytest.approx(0.01)   # family of 1
    assert out[2]["family_size"] == 1


def test_the_raw_p_survives_the_correction():
    """Kept in every row so a reader who wants a different family can
    recompute."""
    rows = [{"family": "f", "p_raw": 0.01}, {"family": "f", "p_raw": 0.4}]
    out = st.holm_within_families(rows)
    assert [r["p_raw"] for r in out] == [0.01, 0.4]


def test_the_network_holm_family_is_one_metric_and_one_scope():
    frame = _run_metrics([
        {"experiment": e, "language": lang, "seed": seed,
         "final_var_x": 0.1 * (lang != "english") + 0.01 * seed}
        for e in ("e3", "e4")
        for lang in ("english", "hinglish", "romanized_hindi", "hindi")
        for seed in range(1001, 1009)
    ])
    rows = st.network_contrasts(frame)
    var_did = [r for r in rows if r["metric"] == "final_var_x" and r["scope"] == "did"]
    assert len(var_did) == 3
    assert {r["family_size"] for r in var_did} == {3}


# --- E1 -------------------------------------------------------------------


def _e1_rows(
    language_effect: float = 0.0,
    strength_effect: float = 0.15,
    n_items: int = 20,
    n_personas: int = 6,
) -> list[dict]:
    """A sweep with a known strong-minus-weak difference and real nuisance.

    Items and personas carry their own offsets, and trials carry noise,
    because the model this feeds is specified against exactly that structure:
    with no item-to-item variation at all the random intercept has nothing to
    estimate and MixedLM does not converge, so a noiseless fixture would
    silently test the OLS fallback in every test below.

    Twenty items rather than the four-item test bank, and strength alternates
    across them rather than splitting them in half, so the strength effect is
    identified between items that otherwise differ.
    """
    rng = random.Random(7)
    items = {f"bank_t{index:02d}": rng.gauss(0, 0.10) for index in range(n_items)}
    personas = {index: rng.gauss(0, 0.08) for index in range(n_personas)}
    rows = []
    for index, (item, item_offset) in enumerate(items.items()):
        strength = "strong" if index % 2 == 0 else "weak"
        side = "pro" if index % 4 < 2 else "con"
        for persona, persona_offset in personas.items():
            for language in ("english", "hinglish", "romanized_hindi", "hindi"):
                for lean in (2, 4, 6):
                    toward = (
                        (strength_effect if strength == "strong" else 0.0)
                        + (language_effect if language != "english" else 0.0)
                        + item_offset
                        + persona_offset
                        + rng.gauss(0, 0.12)
                    )
                    rows.append(
                        {
                            "key": f"e1|{language}|{item}|p{persona:03d}|lean{lean}",
                            "language": language, "item_id": item,
                            "arg_id": f"arg_{index}", "strength": strength,
                            "side": side, "persona_index": persona, "lean": lean,
                            "toward": toward, "status": "ok",
                        }
                    )
    return rows


@pytest.fixture
def e1_path(tmp_path):
    def _write(rows: list[dict]) -> "object":
        path = tmp_path / "trials.jsonl"
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        return path

    return _write


def test_load_e1_keeps_only_ok_trials(e1_path):
    rows = _e1_rows()
    rows.append({**rows[0], "key": "e1|english|bank_t00|p999|lean2",
                 "status": "failed", "toward": None})
    frame = st.load_e1(e1_path(rows))
    assert len(frame) == len(rows) - 1
    assert frame.toward.notna().all()


def test_load_e1_refuses_a_sweep_with_no_ok_trials(e1_path):
    """Treating a failed trial as zero movement would bias every estimate
    towards no effect."""
    rows = [{**r, "status": "failed", "toward": None} for r in _e1_rows()[:4]]
    with pytest.raises(ValueError, match="no ok trials"):
        st.load_e1(e1_path(rows))


def test_the_manipulation_check_recovers_the_strength_effect(e1_path):
    """Reported first, because if a strong argument does not move an English
    reader further than a weak one, the bank's own strength labels do not
    describe the stimuli and no language comparison over them means
    anything."""
    frame = st.load_e1(e1_path(_e1_rows(strength_effect=0.15)))
    fit = st.fit_e1(frame)
    rows = st.e1_contrasts(fit, frame)

    assert fit.kind == "MixedLM" and fit.converged
    assert rows[0]["is_manipulation_check"] is True
    assert rows[0]["contrast"] == "strong - weak in english"
    # The interval covers the effect that was built in, which is the claim
    # the contrast vector has to get right; the point estimate carries the
    # fixture's own sampling noise.
    assert rows[0]["ci_low"] <= 0.15 <= rows[0]["ci_high"]
    assert rows[0]["estimate"] > 0


def test_the_language_contrast_is_averaged_over_strength(e1_path):
    """With the interaction in the model, the bare coefficient is the
    difference among strong items only. The reported main effect averages
    the two strengths."""
    frame = st.load_e1(e1_path(_e1_rows(language_effect=-0.04)))
    fit = st.fit_e1(frame)
    rows = st.e1_contrasts(fit, frame)

    main = [r for r in rows if r["family"] == "language_main"]
    assert len(main) == 3
    for row in main:
        assert row["estimate"] == pytest.approx(-0.04, abs=0.015)
        assert row["ci_low"] <= -0.04 <= row["ci_high"]
        assert "averaged over strong and weak" in row["contrast"]


def test_the_two_contrast_families_are_corrected_separately(e1_path):
    frame = st.load_e1(e1_path(_e1_rows(language_effect=-0.04)))
    rows = st.e1_contrasts(st.fit_e1(frame), frame)
    sizes = {row["family"]: row["family_size"] for row in rows}
    assert sizes == {"strength_diff": 4, "language_main": 3}


def test_every_contrast_records_which_model_it_came_from(e1_path):
    """A cluster-robust OLS estimate and a mixed-model estimate are not the
    same quantity."""
    frame = st.load_e1(e1_path(_e1_rows()))
    fit = st.fit_e1(frame)
    rows = st.e1_contrasts(fit, frame)
    assert {row["model"] for row in rows} == {fit.kind}
    assert all(row["n_trials"] == len(frame) for row in rows)


def test_a_non_converged_fit_falls_back_and_says_so(e1_path, monkeypatch):
    """Never reported as a mixed model: its point estimates would be whatever
    the optimizer stopped on."""
    import statsmodels.formula.api as smf

    real_mixedlm = smf.mixedlm

    class NonConverging:
        def __init__(self, inner):
            self._inner = inner

        def fit(self, *a, **k):
            result = self._inner.fit(*a, **k)
            result.converged = False
            return result

    monkeypatch.setattr(
        st.smf, "mixedlm", lambda *a, **k: NonConverging(real_mixedlm(*a, **k))
    )

    frame = st.load_e1(e1_path(_e1_rows(strength_effect=0.15)))
    fit = st.fit_e1(frame)

    assert fit.converged is False
    assert fit.kind.startswith("OLS (cluster-robust")
    assert "DID NOT CONVERGE" in fit.note
    # The fallback still estimates the manipulation check, from the same
    # contrast vector.
    rows = st.e1_contrasts(fit, frame)
    assert rows[0]["ci_low"] <= 0.15 <= rows[0]["ci_high"]


def test_the_model_file_leads_with_the_convergence_status(tmp_path, e1_path):
    """It changes what every number below it means, so a reader who stops
    after the first paragraph should already know which model they have."""
    frame = st.load_e1(e1_path(_e1_rows()))
    fit = st.fit_e1(frame)
    rows = st.e1_contrasts(fit, frame)
    path = tmp_path / "e1_model.txt"
    st.write_e1_model(path, fit, frame, rows)

    text = path.read_text(encoding="utf-8")
    assert text.index(f"model     {fit.kind}") < text.index("MANIPULATION CHECK")
    assert text.index("MANIPULATION CHECK") < text.index("=" * 72)
    assert text.index("=" * 72) < text.index("Dependent Variable")
    assert st.E1_FORMULA in text


def test_a_sweep_with_one_strength_cannot_report_a_manipulation_check(e1_path):
    rows = [r for r in _e1_rows() if r["strength"] == "strong"]
    frame = st.load_e1(e1_path(rows))
    with pytest.raises(ValueError, match="manipulation check"):
        st.e1_contrasts(st.fit_e1(frame), frame)


# --- the entry point ------------------------------------------------------


def test_main_writes_all_three_files(tmp_path, e1_path):
    results = tmp_path / "results"
    results.mkdir()
    _run_metrics([
        {"experiment": e, "language": lang, "seed": seed,
         "final_var_x": 0.1 + 0.01 * seed}
        for e in ("e3", "e4")
        for lang in ("english", "hinglish", "romanized_hindi", "hindi")
        for seed in range(1001, 1007)
    ]).to_csv(results / "run_metrics.csv", index=False)

    assert st.main([
        "--e1", str(e1_path(_e1_rows())),
        "--results", str(results),
        "--out", str(results),
    ]) == 0

    assert (results / "e1_model.txt").exists()
    contrasts = pd.read_csv(results / "e1_contrasts.csv")
    assert list(contrasts.columns) == list(st.E1_CONTRAST_COLUMNS)
    network = pd.read_csv(results / "network_contrasts.csv")
    assert list(network.columns) == list(st.NETWORK_COLUMNS)
    assert (network.n == 6).all()


def test_skip_e1_reports_only_the_network_contrasts(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _run_metrics([
        {"experiment": "e3", "language": lang, "seed": 1001}
        for lang in ("english", "hindi")
    ]).to_csv(results / "run_metrics.csv", index=False)

    assert st.main([
        "--e1", str(tmp_path / "absent.jsonl"),
        "--results", str(results), "--out", str(results), "--skip-e1",
    ]) == 0
    assert (results / "network_contrasts.csv").exists()
    assert not (results / "e1_contrasts.csv").exists()


def test_a_missing_run_metrics_names_the_stage_that_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="analysis.metrics"):
        st.load_run_metrics(tmp_path)


# --- too few item clusters ------------------------------------------------


def _force_fallback(monkeypatch):
    import statsmodels.formula.api as smf

    real_mixedlm = smf.mixedlm

    class NonConverging:
        def __init__(self, inner):
            self._inner = inner

        def fit(self, *a, **k):
            result = self._inner.fit(*a, **k)
            result.converged = False
            return result

    monkeypatch.setattr(
        st.smf, "mixedlm", lambda *a, **k: NonConverging(real_mixedlm(*a, **k))
    )


def test_the_fallback_on_two_items_reports_no_inference(e1_path, monkeypatch):
    """A live smoke run had two items. The cluster-robust standard error came
    out as exactly 0, so every contrast was p = 0 and "reject". Point
    estimates only: nothing here may read as a result."""
    _force_fallback(monkeypatch)
    frame = st.load_e1(e1_path(_e1_rows(n_items=2)))

    fit = st.fit_e1(frame)
    rows = st.e1_contrasts(fit, frame)

    assert fit.inference_ok is False
    assert "point estimates only" in fit.kind
    assert "2 item" in fit.note
    for row in rows:
        assert np.isfinite(row["estimate"])
        for column in ("se", "statistic", "p_raw", "p_holm", "ci_low", "ci_high"):
            assert np.isnan(row[column]), column
        assert row["reject_at_05"] is False
        assert row["model"] == fit.kind


def test_the_fallback_at_the_cluster_floor_still_reports_inference(
    e1_path, monkeypatch
):
    _force_fallback(monkeypatch)
    frame = st.load_e1(
        e1_path(_e1_rows(n_items=st.MIN_CLUSTERS_FOR_INFERENCE))
    )

    fit = st.fit_e1(frame)
    rows = st.e1_contrasts(fit, frame)

    assert fit.inference_ok is True
    assert fit.kind.startswith("OLS (cluster-robust")
    assert all(np.isfinite(row["p_raw"]) for row in rows)


def test_main_survives_a_sweep_with_too_few_items(tmp_path, e1_path, monkeypatch, capsys):
    """The model file and the console line both format p and the interval;
    NaN must print rather than raise."""
    _force_fallback(monkeypatch)
    results = tmp_path / "results"
    results.mkdir()
    _run_metrics(
        [
            {"experiment": e, "language": lang, "seed": seed, "final_var_x": 0.1 * seed}
            for e in ("e3", "e4")
            for lang in ("english", "hindi")
            for seed in (1, 2)
        ]
    ).to_csv(results / "run_metrics.csv", index=False)

    code = st.main(
        ["--e1", str(e1_path(_e1_rows(n_items=2))), "--results", str(results),
         "--out", str(results)]
    )

    assert code == 0
    assert "too few items" in capsys.readouterr().out
    contrasts = pd.read_csv(results / "e1_contrasts.csv")
    assert contrasts.p_raw.isna().all()
    assert not contrasts.reject_at_05.any()
