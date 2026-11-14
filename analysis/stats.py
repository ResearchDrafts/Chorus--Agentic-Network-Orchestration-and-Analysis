"""The models and the planned contrasts.

    python -m analysis.stats --e1 e1/trials.jsonl --results results/ --out results/

Two analyses, with two different units.

**E1 is a trial-level analysis.** Its unit is one reader reading one item, and
the two nuisance structures are the item (some arguments are simply more
persuasive than others) and the persona (some readers move more than others).
Both are absorbed as random effects rather than estimated, because neither is
the question and both are shared across the four arms -- so a model that
treated trials as independent would attribute an item's own persuasiveness to
whichever language happened to be sampled alongside it.

**The network analysis is a run-level analysis, paired by seed.** Its unit is
one whole run, because that is the level the manipulation is applied at and
the level the metrics are defined on: a hundred agents' readings within one
run are not a hundred independent observations of what language does to a
population. Four runs sharing a seed share a graph, the same neighbour draws
and the same bank items, so an English run and a Hindi run at seed 1003 are a
matched pair and the difference between them is taken within the pair. Paired
differences across five seeds is five observations, which is what the n
column in every row says, and what the fewer-than-six-seeds warning is about.

**A non-converged fit is never reported as a result.** `MixedLM` can fail to
converge on a design this small, and its point estimates are then whatever
the optimizer stopped on. The fallback is OLS with cluster-robust standard
errors clustered on item, which makes a weaker assumption and says so in the
output, rather than a fit nobody can interpret dressed up as a mixed model.

Writes `e1_model.txt`, `e1_contrasts.csv` and `network_contrasts.csv`.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sandbox.single_exposure import STATUS_OK, load_trials

REFERENCE_LANGUAGE = "english"

# The RQ1 plan's own formula. `toward` is already signed by the item's side
# (see scripts/run_e1.py), so the two sides pool; `lean` is a covariate
# rather than an interaction because a reader's starting position changes how
# far it CAN move, which is a ceiling to adjust for and not an effect to
# report.
E1_FORMULA = (
    f"toward ~ C(language, Treatment('{REFERENCE_LANGUAGE}')) * C(strength) + C(lean)"
)

# The run-level metrics the network contrasts are computed over.
NETWORK_METRICS = (
    "final_var_x",
    "final_extremity",
    "final_assortativity",
    "t_stationary",
)

# Bootstrap settings. Seeded, so re-running the analysis reproduces the
# interval: a confidence interval that moved between two runs of the same
# script on the same data would be impossible to report.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_101
BOOTSTRAP_ALPHA = 0.05

# Below this many seeds, the smallest attainable two-sided Wilcoxon p value
# is above 0.05 -- with n paired observations it is 2/2**n, which is 0.0625
# at n=5. So a non-significant result at n<6 says nothing about the effect,
# only about the design, and the output has to say so.
MIN_SEEDS_FOR_SIGNIFICANCE = 6

HOLM = "holm"


# --- E1 -------------------------------------------------------------------


@dataclass
class Fit:
    """A fitted model, and whether it is the one that was asked for.

    `converged` is carried rather than checked at the call site because every
    reported contrast has to say which model it came from: a cluster-robust
    OLS estimate and a mixed-model estimate are not the same quantity, and a
    reader comparing two campaigns' numbers needs to know which they have.
    """

    result: object
    kind: str
    converged: bool
    design_info: patsy.DesignInfo
    note: str
    # False when the model's standard errors cannot be trusted at all (the
    # cluster-robust fallback on too few item clusters). The point estimates
    # are still reported; se, CI and p are written as NaN and nothing rejects.
    inference_ok: bool = True


def load_e1(path: Path) -> pd.DataFrame:
    """The ok trials of an E1 sweep, as a frame.

    Deduplicated on trial key by `load_trials`, so a resumed sweep's retried
    trials are counted once -- and only `status == "ok"` rows, since a failed
    trial has no `toward` to model and silently coercing one to 0 would pull
    every estimate towards no effect.
    """
    rows = [r for r in load_trials(path) if r["status"] == STATUS_OK]
    if not rows:
        raise ValueError(
            f"{path} has no ok trials. Nothing can be fitted to failures, and "
            "treating a failed trial as zero movement would bias every "
            "estimate towards no effect."
        )
    frame = pd.DataFrame(rows)
    for column in ("toward", "language", "strength", "lean", "item_id", "persona_index"):
        if column not in frame:
            raise ValueError(f"{path} has no {column!r} column")
    return frame


def fit_e1(frame: pd.DataFrame) -> Fit:
    """The mixed model, or the cluster-robust fallback if it does not converge.

    Random intercept for item (`groups=item_id`) and a variance component for
    persona. The two are not interchangeable: an item is a property of the
    stimulus and is crossed with every language, while a persona is a
    property of the reader; modelling the second as a variance component
    rather than a second grouping factor is what `MixedLM` supports for a
    crossed design.

    The convergence warnings are captured rather than suppressed, and land in
    e1_model.txt: a fit that converged only after a retry on the boundary of
    the parameter space is a fact about the result.
    """
    # Built here as well as inside the fit so the contrast vectors can be
    # constructed from the same design. patsy's column order is the model's
    # exog order, which is asserted below rather than assumed.
    _, design = patsy.dmatrices(E1_FORMULA, frame, return_type="dataframe")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.mixedlm(
            E1_FORMULA,
            frame,
            groups=frame["item_id"],
            re_formula="1",
            vc_formula={"persona": "0 + C(persona_index)"},
        )
        result = model.fit()
    messages = sorted({str(w.message) for w in caught})

    if list(design.design_info.column_names) != list(result.model.exog_names):
        raise AssertionError(
            "the design used for the contrasts is not the design the model "
            "was fitted on"
        )

    if getattr(result, "converged", False):
        return Fit(
            result=result,
            kind="MixedLM",
            converged=True,
            design_info=design.design_info,
            note=(
                "Mixed model converged."
                + ("  Warnings during fitting: " + " | ".join(messages) if messages else "")
            ),
        )

    fallback = smf.ols(E1_FORMULA, frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame["item_id"]}
    )
    n_clusters = int(frame["item_id"].nunique())
    if n_clusters < MIN_CLUSTERS_FOR_INFERENCE:
        return Fit(
            result=fallback,
            kind=f"OLS (point estimates only: {n_clusters} item clusters)",
            converged=False,
            design_info=design.design_info,
            inference_ok=False,
            note=(
                "THE MIXED MODEL DID NOT CONVERGE and is not reported. The "
                f"fallback clusters on item, and this sweep has {n_clusters} "
                f"item(s), fewer than the {MIN_CLUSTERS_FOR_INFERENCE} needed "
                "for a cluster-robust standard error to mean anything. ONLY "
                "POINT ESTIMATES ARE REPORTED: se, CI and p are NaN and no "
                "contrast rejects. Run the sweep over more items."
                + ("  Warnings during fitting: " + " | ".join(messages) if messages else "")
            ),
        )
    return Fit(
        result=fallback,
        kind="OLS (cluster-robust, clustered on item_id)",
        converged=False,
        design_info=design.design_info,
        note=(
            "THE MIXED MODEL DID NOT CONVERGE and is not reported. Its point "
            "estimates would be whatever the optimizer stopped on. These "
            "numbers come from OLS with standard errors clustered on item, "
            "which makes a weaker assumption about the persona structure "
            "(it is not modelled at all) and does not pool across items."
            + ("  Warnings during fitting: " + " | ".join(messages) if messages else "")
        ),
    )


def _design_row(fit: Fit, template: pd.DataFrame, **overrides) -> np.ndarray:
    row = template.copy()
    for column, value in overrides.items():
        row[column] = value
    return np.asarray(patsy.build_design_matrices([fit.design_info], row)[0])[0]


def _average_over_strength(
    fit: Fit, template: pd.DataFrame, language: str, strengths: list[str]
) -> np.ndarray:
    """The design row for `language`, averaged over the strength levels.

    This is what makes the language contrast a MAIN effect rather than the
    effect at whichever strength happens to be the reference level. With the
    interaction in the model, the bare `C(language)[T.hindi]` coefficient is
    the Hindi-minus-English difference among strong items only; averaging the
    design over both strengths with equal weights gives the difference
    averaged over the two, which is the quantity the plan names.

    Equal weights, not the observed proportions: the bank is designed with
    both strengths in it deliberately, so the estimand is the average over
    the design rather than over however many of each the sweep happened to
    finish.
    """
    rows = [
        _design_row(fit, template, language=language, strength=strength)
        for strength in strengths
    ]
    return np.mean(rows, axis=0)


# Below this many item clusters the cluster-robust fallback reports no
# inference at all. Its variance estimate is a sum over clusters; with two it
# collapsed to a standard error of exactly 0 on a live smoke run and every
# contrast came out as p = 0, "reject". Ten is a floor for refusing outright,
# not a claim that ten is enough: cluster-robust errors are known to be too
# small with few clusters, so read anything under about thirty with care.
MIN_CLUSTERS_FOR_INFERENCE = 10


def _contrast(fit: Fit, vector: np.ndarray) -> dict:
    test = fit.result.t_test(np.asarray(vector, dtype=float).reshape(1, -1))
    low, high = np.ravel(test.conf_int())
    if not fit.inference_ok:
        nan = float("nan")
        return {
            "estimate": float(np.ravel(test.effect)[0]),
            "se": nan, "statistic": nan, "p_raw": nan, "ci_low": nan, "ci_high": nan,
        }
    return {
        "estimate": float(np.ravel(test.effect)[0]),
        "se": float(np.ravel(test.sd)[0]),
        "statistic": float(np.ravel(test.statistic)[0]),
        "p_raw": float(np.ravel(test.pvalue)[0]),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def e1_contrasts(fit: Fit, frame: pd.DataFrame) -> list[dict]:
    """The planned contrasts, manipulation check first.

    Two families, Holm-corrected separately because they answer different
    questions: the three language main effects (does the rendering change how
    far a reader moves) and the four within-language strength differences
    (does a strong argument move a reader further than a weak one, in each
    arm). Correcting the seven together would penalise each question for the
    other being asked.

    The manipulation check -- strong minus weak in English -- is reported
    first and is a member of the strength family. It is the one contrast whose
    failure invalidates everything else: if a strong argument does not move an
    English reader further than a weak one, the bank's own strength labels do
    not describe the stimuli, and no language comparison over them means
    anything.
    """
    languages = [
        language
        for language in sorted(frame.language.unique())
        if language != REFERENCE_LANGUAGE
    ]
    strengths = sorted(frame.strength.unique())
    if len(strengths) < 2:
        raise ValueError(
            f"the sweep has only {strengths} items, so no strength difference "
            "can be estimated and the manipulation check cannot be reported"
        )
    strong, weak = strengths[0], strengths[-1]
    template = frame.iloc[[0]].copy()

    rows: list[dict] = []

    def strength_difference(language: str) -> np.ndarray:
        return _design_row(
            fit, template, language=language, strength=strong
        ) - _design_row(fit, template, language=language, strength=weak)

    # Reported first: see the docstring.
    rows.append(
        {
            "family": "strength_diff",
            "contrast": f"{strong} - {weak} in {REFERENCE_LANGUAGE}",
            "language": REFERENCE_LANGUAGE,
            "is_manipulation_check": True,
            **_contrast(fit, strength_difference(REFERENCE_LANGUAGE)),
        }
    )
    for language in languages:
        rows.append(
            {
                "family": "strength_diff",
                "contrast": f"{strong} - {weak} in {language}",
                "language": language,
                "is_manipulation_check": False,
                **_contrast(fit, strength_difference(language)),
            }
        )
    for language in languages:
        vector = _average_over_strength(
            fit, template, language, strengths
        ) - _average_over_strength(fit, template, REFERENCE_LANGUAGE, strengths)
        rows.append(
            {
                "family": "language_main",
                "contrast": f"{language} - {REFERENCE_LANGUAGE} "
                            f"(averaged over {strong} and {weak})",
                "language": language,
                "is_manipulation_check": False,
                **_contrast(fit, vector),
            }
        )

    rows = holm_within_families(rows)
    for row in rows:
        row["model"] = fit.kind
        row["n_trials"] = int(len(frame))
        row["n_items"] = int(frame.item_id.nunique())
        row["n_personas"] = int(frame.persona_index.nunique())
    return rows


def holm_within_families(rows: list[dict], key: str = "family") -> list[dict]:
    """Holm-Bonferroni within each family, preserving row order.

    Applied per family rather than across everything, because a family is a
    set of tests that answer one question: correcting the language
    comparisons for the strength checks being run alongside them would
    penalise each question for the other being asked. `p_raw` is kept in
    every row, so a reader who wants a different family can recompute.
    """
    out = list(rows)
    for family in sorted({row[key] for row in out}):
        members = [row for row in out if row[key] == family]
        if any(np.isnan(row["p_raw"]) for row in members):
            # No inference was possible for this family (see Fit.inference_ok).
            # Holm over NaN would return NaN flags; say "no rejection" outright.
            for row in members:
                row["p_holm"] = float("nan")
                row["reject_at_05"] = False
                row["family_size"] = len(members)
            continue
        reject, adjusted, _, _ = multipletests(
            [row["p_raw"] for row in members], method=HOLM
        )
        for row, flag, value in zip(members, reject, adjusted, strict=True):
            row["p_holm"] = float(value)
            row["reject_at_05"] = bool(flag)
            row["family_size"] = len(members)
    return out


E1_CONTRAST_COLUMNS = (
    "family", "contrast", "language", "is_manipulation_check", "model",
    "estimate", "se", "ci_low", "ci_high", "statistic",
    "p_raw", "p_holm", "reject_at_05", "family_size",
    "n_trials", "n_items", "n_personas",
)


def write_e1_model(path: Path, fit: Fit, frame: pd.DataFrame, rows: list[dict]) -> None:
    """e1_model.txt: the fit, in full, with its convergence status at the top.

    The convergence note is first because it changes what every number below
    it means, and a reader who stops after the first paragraph should already
    know whether they are looking at a mixed model.
    """
    check = next(r for r in rows if r["is_manipulation_check"])
    lines = [
        "E1: does one argument move one reader, and does that depend on language?",
        "",
        f"model     {fit.kind}",
        f"formula   {E1_FORMULA}",
        "          random intercept for item_id, variance component for persona",
        f"trials    {len(frame)} ok  "
        f"({frame.item_id.nunique()} items, {frame.persona_index.nunique()} personas, "
        f"{frame.language.nunique()} languages, leans {sorted(frame.lean.unique())})",
        "",
        fit.note,
        "",
        "MANIPULATION CHECK (reported first: if this fails, the bank's own",
        "strength labels do not describe the stimuli and no language",
        "comparison over them means anything)",
        f"  {check['contrast']}",
        f"    estimate {check['estimate']:+.4f}  "
        f"95% CI [{check['ci_low']:+.4f}, {check['ci_high']:+.4f}]  "
        f"p = {check['p_raw']:.4g}",
        "",
        "=" * 72,
        "",
        str(fit.result.summary()),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --- the network contrasts ------------------------------------------------


def load_run_metrics(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "run_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m analysis.metrics` first: the "
            "network contrasts are differences between its rows."
        )
    frame = pd.read_csv(path)
    missing = [m for m in NETWORK_METRICS if m not in frame.columns]
    if missing:
        raise ValueError(f"{path} has no {missing} column(s)")
    return frame


def paired_differences(
    frame: pd.DataFrame, metric: str, experiment: str, language: str
) -> dict[int, float]:
    """`metric` for `language` minus the same for English, per shared seed.

    Within one experiment, so an E3-minus-E3 difference is a treatment
    comparison and an E4-minus-E4 one is a comparison of the controls. Keyed
    by seed, and only seeds where BOTH arms produced a value: an unpaired
    seed contributes nothing to a paired test, and silently dropping the pair
    is right where silently filling in the missing half would not be.
    """
    arm = frame[frame.experiment == experiment]
    reference = arm[arm.language == REFERENCE_LANGUAGE].set_index("seed")[metric]
    other = arm[arm.language == language].set_index("seed")[metric]
    shared = sorted(set(reference.index) & set(other.index))
    return {
        int(seed): float(other[seed]) - float(reference[seed])
        for seed in shared
        if pd.notna(reference[seed]) and pd.notna(other[seed])
    }


def bootstrap_ci(values: list[float], seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """A percentile bootstrap interval over the seeds, resampling seeds.

    The resampling unit is the seed, which is the unit of analysis: resampling
    agents or rounds would treat one run's internal correlation as independent
    information and produce an interval far too narrow.

    Seeded from a constant, so the interval is reproducible. An interval that
    moved between two runs of the same script on the same data could not be
    reported.
    """
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        # One seed is one observation: the interval is the point, and saying
        # so beats a bootstrap over a single value, which would return a
        # zero-width interval and look like certainty.
        return (values[0], values[0])
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    means = array[draws].mean(axis=1)
    low, high = np.percentile(
        means, [100 * BOOTSTRAP_ALPHA / 2, 100 * (1 - BOOTSTRAP_ALPHA / 2)]
    )
    return (float(low), float(high))


def wilcoxon_p(values: list[float]) -> float | None:
    """The two-sided signed-rank p value, or None when there is no test.

    None below two observations, where the test is undefined. 1.0 when every
    difference is exactly zero, which scipy refuses to compute and which has
    an obvious answer: no evidence of any difference at all. `zsplit` handles
    the mixed case, where some differences are zero and some are not, by
    splitting the zero ranks rather than discarding them -- discarding would
    shrink n and make a run of exact ties look like stronger evidence.
    """
    if len(values) < 2:
        return None
    array = np.asarray(values, dtype=float)
    if np.all(array == 0):
        return 1.0
    return float(stats.wilcoxon(array, zero_method="zsplit").pvalue)


NETWORK_COLUMNS = (
    "metric", "scope", "language", "contrast", "n",
    "mean_diff", "ci_low", "ci_high", "p_raw", "p_holm", "reject_at_05",
    "family_size", "underpowered", "seeds",
)


def network_contrasts(frame: pd.DataFrame) -> list[dict]:
    """Three scopes per metric per language: within E3, within E4, and DiD.

    The difference-in-differences is the primary one. A raw E3 difference
    between English and Hindi mixes two things: what the language did to how
    agents persuade each other, and what it did to the measurement itself --
    a Hindi probe prompt is longer, tokenizes differently, and the reading
    drifts differently under it. The E4 control is exposed to the second and
    not the first, because its agents read nothing that argues anything. So
    E3-minus-E4 is the part attributable to the manipulation, and the two
    within-experiment rows are reported alongside it because a DiD near zero
    means something quite different depending on whether both halves were
    near zero or both were large.
    """
    languages = [
        language
        for language in sorted(frame.language.unique())
        if language != REFERENCE_LANGUAGE
    ]
    rows: list[dict] = []
    for metric in NETWORK_METRICS:
        for language in languages:
            e3 = paired_differences(frame, metric, "e3", language)
            e4 = paired_differences(frame, metric, "e4", language)
            did = {
                seed: e3[seed] - e4[seed]
                for seed in sorted(set(e3) & set(e4))
            }
            for scope, values, label in (
                ("within_e3", e3, f"{language} - {REFERENCE_LANGUAGE} (E3)"),
                ("within_e4", e4, f"{language} - {REFERENCE_LANGUAGE} (E4)"),
                ("did", did, f"({language} - {REFERENCE_LANGUAGE}), E3 - E4"),
            ):
                rows.append(
                    _network_row(metric, scope, language, label, values)
                )
    return holm_within_families(rows, key="_family")


def _network_row(
    metric: str, scope: str, language: str, label: str, values: dict[int, float]
) -> dict:
    ordered = [values[seed] for seed in sorted(values)]
    low, high = bootstrap_ci(ordered)
    p = wilcoxon_p(ordered)
    return {
        # Holm family: one metric, one scope, the three non-English
        # languages. That is the set of tests answering one question, which
        # is what a correction family is.
        "_family": f"{metric}|{scope}",
        "metric": metric,
        "scope": scope,
        "language": language,
        "contrast": label,
        # Stated in every row: a mean difference over three seeds and one
        # over twenty are not the same claim, and the column is the only
        # place that shows it.
        "n": len(ordered),
        "mean_diff": float(np.mean(ordered)) if ordered else None,
        "ci_low": low if ordered else None,
        "ci_high": high if ordered else None,
        # p_raw of 1.0 rather than None when the test cannot run, so Holm has
        # a value to work with; `n` and `underpowered` say why.
        "p_raw": 1.0 if p is None else p,
        "underpowered": len(ordered) < MIN_SEEDS_FOR_SIGNIFICANCE,
        "seeds": ";".join(str(seed) for seed in sorted(values)),
    }


def warn_if_underpowered(rows: list[dict]) -> None:
    """Says so when no result could have been significant.

    With n paired observations the smallest attainable two-sided signed-rank
    p is 2 / 2**n, which is 0.0625 at n = 5. So below six seeds a
    non-significant row says nothing about the effect, only about the design,
    and a reader who does not know that will read it as a null result.
    """
    counts = sorted({row["n"] for row in rows})
    worst = max(counts) if counts else 0
    if worst >= MIN_SEEDS_FOR_SIGNIFICANCE:
        return
    smallest = 2 / 2**worst if worst else float("inf")
    print(
        f"\n  WARNING: at most {worst} paired seed(s) per contrast. The "
        f"smallest attainable two-sided Wilcoxon p value is {smallest:.4f}, "
        "which is above 0.05, so NO row here could have been significant "
        f"however large the effect. Run at least "
        f"{MIN_SEEDS_FOR_SIGNIFICANCE} seeds before reading the p values."
    )


# --- entry point ----------------------------------------------------------


def write_table(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    frame = pd.DataFrame(rows, columns=list(columns))
    frame.to_csv(path, index=False)
    print(f"  {path}  {len(frame):>4} rows")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--e1", type=Path, required=True, help="e1/trials.jsonl")
    parser.add_argument("--results", type=Path, required=True,
                        help="the directory analysis.metrics wrote")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--skip-e1", action="store_true",
        help="report only the network contrasts, for a campaign analysed "
             "before its E1 sweep has finished",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.skip_e1:
        frame = load_e1(args.e1)
        fit = fit_e1(frame)
        rows = e1_contrasts(fit, frame)
        write_e1_model(args.out / "e1_model.txt", fit, frame, rows)
        print(f"\n  {args.out / 'e1_model.txt'}   {fit.kind}")
        if not fit.inference_ok:
            print("  WARNING: too few items for any inference. Point estimates "
                  "only; se, CI and p are NaN. Run E1 over more items.")
        elif not fit.converged:
            print("  WARNING: the mixed model did not converge; "
                  "cluster-robust OLS reported instead")
        write_table(args.out / "e1_contrasts.csv", rows, E1_CONTRAST_COLUMNS)
        check = next(r for r in rows if r["is_manipulation_check"])
        print(
            f"  manipulation check  {check['contrast']}: "
            f"{check['estimate']:+.4f} "
            f"[{check['ci_low']:+.4f}, {check['ci_high']:+.4f}]  "
            f"p = {check['p_raw']:.4g}"
        )

    runs = load_run_metrics(args.results)
    network = network_contrasts(runs)
    if not network:
        print("no non-English arms in run_metrics.csv to contrast", file=sys.stderr)
        return 1
    write_table(args.out / "network_contrasts.csv", network, NETWORK_COLUMNS)
    warn_if_underpowered(network)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
