"""Runs in, three tidy tables out.

    python -m analysis.metrics --runs-dir runs --tau calibration/tau.json \
            --out results/ --epsilon-from-e4

Reads each bank-regime run folder through `export_csv.load_run`,
`mark_duplicates` and `derive` -- the same three functions the CSV export
uses, so an analysis and an export cannot disagree about which rows are
duplicates or how `x` is computed -- plus `graph_edges.csv` for the
structure and `run_config.json` for the config.

Writes:

    probes.csv          one row per agent per round: x, side_conf, dx, dconf,
                        and the outcome class the change falls in
    round_metrics.csv   one row per run per round: mean_x, var_x, extremity,
                        assortativity, the outcome shares, and the counts
    run_metrics.csv     one row per run: the config, the final round's
                        metrics, the change from round 0, the outcome rates
                        over the whole run, and t_stationary

**Every classification is against τ.** A probe reading moves on its own --
the history grows, the sampler draws -- and `scripts/calibrate_tau.py`
measures how much, per language, when nothing persuasive happened. A "flip"
reported without that threshold is a claim that a number changed, not that
an agent did, and the measurement noise is not the same size in all four
arms, so one τ for the campaign would be wrong in three of them.

**The sign of x, and x == 0.** `sign(x)` is -1 below the midpoint, +1 above
it, and **0 exactly at it** -- zero is its own sign, not grouped with either
pole. So an agent at exactly 0 that moves to +0.5 has changed sign, and (if
the move clears τ) flipped. The alternative -- folding 0 in with one side --
would make the midpoint arbitrarily belong to one pole and turn half of the
undecided population's movements into hardening. An agent sitting exactly at
0 is the one case where "which side is it on" has no answer, and the
classification says so rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from export_csv import derive, load_run, mark_duplicates
from sandbox.single_exposure import percentile

# |x| above this is "extreme": two thirds of the way from the midpoint to a
# pole, which on the 7-point scale is a probe mean beyond 6 or below 2. A
# round number in x's own units rather than a scale point, because every RQ1
# model is specified in those units.
EXTREMITY_THRESHOLD = 2.0 / 3.0

# The outcome classes, in the order they are checked and reported.
OUTCOMES = ("flip", "harden", "soften", "conf_up", "conf_down", "none")

# The three metrics t_stationary is defined over.
STATIONARY_METRICS = ("var_x", "extremity", "assortativity")

# How many consecutive rounds of small change make a series stationary.
STATIONARY_WINDOW = 3

# Which percentile of E4's own round-to-round change becomes ε under
# --epsilon-from-e4. The same percentile τ uses, and for the same reason: the
# control is what "nothing persuasive happened" looks like at the level of a
# whole population, so its own round-to-round wobble is the right scale for
# "this metric has stopped moving".
EPSILON_PERCENTILE = 95.0


class MissingTau(Exception):
    """A run's language has no τ. Refused rather than defaulted.

    A default τ would classify the movements of an uncalibrated arm against
    another arm's noise, which is worse than reporting nothing: the output
    would look exactly like a calibrated result.
    """


# --- loading --------------------------------------------------------------


def load_tau(path: Path) -> dict:
    """tau.json, checked for the shape the rest of this module assumes."""
    if not path.exists():
        raise FileNotFoundError(
            f"τ file not found: {path}. Run scripts/calibrate_tau.py first -- "
            "every outcome class is a comparison against it."
        )
        # (no fallback: see MissingTau)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "languages" not in raw:
        raise ValueError(f"{path} has no 'languages' block: not a tau.json")
    if raw.get("smoke"):
        print(
            f"  WARNING: {path} was written by a smoke run (--limit, or a "
            "lowered --min-ok-trials). Its thresholds are not a calibration."
        )
    return raw


def tau_for(tau: dict, language: str, path: Path) -> tuple[float, float]:
    arm = tau["languages"].get(language)
    if arm is None or arm.get("tau") is None or arm.get("tau_conf") is None:
        raise MissingTau(
            f"{path} has no τ for {language!r}. Calibrate that arm before "
            "analysing it: classifying its movements against another arm's "
            "noise would look exactly like a calibrated result."
        )
    return float(arm["tau"]), float(arm["tau_conf"])


def load_graph(run_dir: Path) -> nx.Graph | None:
    """The run's own graph, as it was written at run start.

    Read from `graph_edges.csv` rather than rebuilt from the config, for the
    reason the orchestrator writes it: rebuilding would depend on this
    networkx version producing the same graph from the same seed, which is
    not something to rely on across a reinstall.

    None for an alpha-sampling run, which has no fixed structure -- then
    assortativity is null rather than zero, since "agents are not adjacent to
    their own side" and "there is no adjacency" are different facts.
    """
    path = run_dir / "graph_edges.csv"
    if not path.exists():
        return None
    graph = nx.Graph()
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if "agent_a" not in header:
            raise ValueError(f"{path} has no agent_a,agent_b header")
        for line in handle:
            if not line.strip():
                continue
            a, b = line.strip().split(",")
            graph.add_edge(a, b)
    return graph


def discover_runs(runs_dir: Path) -> list[Path]:
    """Every bank-regime run directory under `runs_dir`, sorted.

    Bank only: a generated-regime run has no probe, so it has no x, and every
    metric here is defined on x. Skipped with a note rather than silently, so
    a campaign directory holding both does not look like it produced fewer
    runs than it did.
    """
    if not runs_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {runs_dir}")
    out = []
    for candidate in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        config = candidate / "run_config.json"
        if not config.exists():
            continue
        if json.loads(config.read_text(encoding="utf-8")).get("regime") != "bank":
            print(f"  skipped {candidate.name}: not a bank-regime run")
            continue
        out.append(candidate)
    return out


def load_run_frame(run_dir: Path) -> pd.DataFrame | None:
    """One run's rows, through the export's own pipeline.

    `mark_duplicates` before `derive`, which `derive` itself requires: the
    exposure counts cannot be computed without knowing which rows a resumed
    run superseded.
    """
    frame = load_run(run_dir)
    if frame is None:
        return None
    frame = derive(mark_duplicates(frame))
    return frame[~frame.is_duplicate.astype(bool)]


# --- probes.csv -----------------------------------------------------------


def sign_of(x: float) -> int:
    """-1, 0 or +1, with 0 its own sign.

    Zero is the midpoint: the one reading for which "which side is it on" has
    no answer. Folding it in with a pole would make the midpoint arbitrarily
    belong to one side and turn half of the undecided population's movements
    into hardening, so it gets its own value and a move off it counts as a
    change of sign.
    """
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def classify(dx: float, x_before: float, x_after: float, dconf: float,
             tau: float, tau_conf: float) -> str:
    """Which outcome class one round-to-round change falls in.

    Checked in this order, and the order is the definition:

        flip        the sign of x changed AND |dx| > tau
        harden      same sign, and |x| grew by more than tau
        soften      same sign, and |x| shrank by more than tau
        conf_up     |dx| <= tau and dconf >  tau_conf
        conf_down   |dx| <= tau and dconf < -tau_conf
        none        anything else

    The three position classes come first because a change in where an agent
    stands is the thing being measured; the two confidence classes describe a
    round in which it did not move but became more or less decided, which is
    only interesting once movement has been ruled out -- hence the `|dx| <=
    tau` guard on them rather than an `else`.

    A sign change that does not clear tau is NOT a flip. An agent oscillating
    either side of the midpoint by less than the instrument's own noise has
    not changed its mind, and counting it would make the flip rate a function
    of how many agents sit near 0.
    """
    moved = abs(dx) > tau
    if sign_of(x_before) != sign_of(x_after) and moved:
        return "flip"
    if sign_of(x_before) == sign_of(x_after):
        if abs(x_after) - abs(x_before) > tau:
            return "harden"
        if abs(x_before) - abs(x_after) > tau:
            return "soften"
    if not moved:
        if dconf > tau_conf:
            return "conf_up"
        if dconf < -tau_conf:
            return "conf_down"
    return "none"


PROBE_COLUMNS = (
    "run_id", "experiment", "language", "seed", "delivery", "agent_id", "turn",
    "prev_turn", "x", "side_conf", "probe_mean", "probe_mass", "dx", "dconf",
    "outcome", "tau", "tau_conf", "reply_prompt_tokens", "probe_prompt_tokens",
    "n_repeat_reads", "max_prior_exposures", "n_distinct_items_read_so_far",
)


def probe_rows(frame: pd.DataFrame, config: dict, tau: float, tau_conf: float) -> list[dict]:
    """One row per agent per VALID round, with dx against its previous one.

    dx is measured against the agent's previous valid reading, not against
    round r-1 unconditionally. A round whose probe failed produced no
    reading, and inventing one -- or skipping the comparison -- would both be
    worse than comparing to the last thing that was actually measured;
    `prev_turn` records which round that was, so a reader can see when a gap
    was crossed.

    Round 0 has no previous reading, so it carries no dx, no dconf and no
    outcome: it is the measurement taken before anything was read.
    """
    valid = frame[frame.is_valid_for_analysis & frame.x.notna()]
    rows: list[dict] = []
    experiment = experiment_of(config["run_id"])
    for agent_id, group in valid.groupby("speaker_agent_id", sort=True):
        previous: dict | None = None
        for _, row in group.sort_values("turn").iterrows():
            record = {
                "run_id": config["run_id"],
                "experiment": experiment,
                "language": config["language_condition"],
                "seed": config["seed"],
                "delivery": config.get("delivery"),
                "agent_id": agent_id,
                "turn": int(row.turn),
                "prev_turn": None if previous is None else previous["turn"],
                "x": float(row.x),
                "side_conf": _float_or_none(row.side_conf),
                "probe_mean": _float_or_none(row.probe_mean),
                "probe_mass": _float_or_none(row.probe_mass),
                "dx": None,
                "dconf": None,
                "outcome": None,
                "tau": tau,
                "tau_conf": tau_conf,
                "reply_prompt_tokens": _float_or_none(row.reply_prompt_tokens),
                "probe_prompt_tokens": _float_or_none(row.probe_prompt_tokens),
                "n_repeat_reads": _float_or_none(row.n_repeat_reads),
                "max_prior_exposures": _float_or_none(row.max_prior_exposures),
                "n_distinct_items_read_so_far": _float_or_none(
                    row.n_distinct_items_read_so_far
                ),
            }
            if previous is not None:
                record["dx"] = record["x"] - previous["x"]
                if record["side_conf"] is not None and previous["side_conf"] is not None:
                    record["dconf"] = record["side_conf"] - previous["side_conf"]
                record["outcome"] = classify(
                    record["dx"],
                    previous["x"],
                    record["x"],
                    # A missing side_conf cannot become a confidence class, so
                    # 0 here is correct rather than a fabricated reading: it
                    # means "no confidence change was observed", and the two
                    # classes it could produce both require clearing
                    # tau_conf > 0.
                    record["dconf"] if record["dconf"] is not None else 0.0,
                    tau,
                    tau_conf,
                )
            rows.append(record)
            previous = record
    return rows


def _float_or_none(value) -> float | None:
    return None if pd.isna(value) else float(value)


def experiment_of(run_id: str) -> str:
    """The experiment, from the run_id prefix -- "e3", "e4", or "other".

    Read off the id rather than out of a column, because there is no column:
    `delivery` says on or off, which is the mechanism, while the experiment
    label is what the campaign and every contrast are written in terms of.
    "other" rather than a raise, so a hand-named run still appears in the
    tables instead of stopping the analysis.
    """
    prefix = run_id.split("_", 1)[0]
    return prefix if prefix in ("e3", "e4") else "other"


# --- round_metrics.csv ----------------------------------------------------


ROUND_COLUMNS = (
    "run_id", "experiment", "language", "seed", "delivery", "turn",
    "mean_x", "var_x", "extremity", "assortativity",
    *(f"share_{name}" for name in OUTCOMES),
    "n_valid", "n_failed",
    "mean_reply_prompt_tokens", "mean_probe_prompt_tokens",
)


def round_rows(
    frame: pd.DataFrame,
    probes: list[dict],
    config: dict,
    graph: nx.Graph | None,
) -> list[dict]:
    """One row per round: the population's shape, and what moved in it.

    `n_failed` is M minus the agents with a valid reading this round, read off
    the config's own M rather than counted from the rows -- an agent whose
    reply failed has a row, and an agent that never appeared at all does not,
    and both are the same kind of missing for this purpose.

    The graph metric alone carries a missing agent's last valid x forward. An
    assortativity computed over a shrinking node set would move as agents
    dropped out, which would read as structural change; carrying the last
    reading forward keeps the node set fixed and says, correctly, that
    nothing new is known about that agent. Nothing else does this: a mean or
    a variance over carried-forward values would count one reading twice.
    """
    by_turn: dict[int, list[dict]] = {}
    for row in probes:
        by_turn.setdefault(row["turn"], []).append(row)

    population = int(config["M"])
    experiment = experiment_of(config["run_id"])
    last_known: dict[str, float] = {}
    out: list[dict] = []

    for turn in sorted(by_turn):
        rows = by_turn[turn]
        xs = [r["x"] for r in rows]
        for row in rows:
            last_known[row["agent_id"]] = row["x"]

        turn_frame = frame[frame.turn == turn]
        outcomes = [r["outcome"] for r in rows if r["outcome"] is not None]
        record = {
            "run_id": config["run_id"],
            "experiment": experiment,
            "language": config["language_condition"],
            "seed": config["seed"],
            "delivery": config.get("delivery"),
            "turn": turn,
            "mean_x": sum(xs) / len(xs),
            "var_x": population_variance(xs),
            "extremity": sum(1 for x in xs if abs(x) > EXTREMITY_THRESHOLD) / len(xs),
            "assortativity": assortativity(graph, dict(last_known)),
            "n_valid": len(rows),
            "n_failed": population - len(rows),
            "mean_reply_prompt_tokens": _mean_or_none(
                turn_frame.reply_prompt_tokens
            ),
            "mean_probe_prompt_tokens": _mean_or_none(
                turn_frame.probe_prompt_tokens
            ),
        }
        for name in OUTCOMES:
            record[f"share_{name}"] = (
                None
                if not outcomes
                else sum(1 for o in outcomes if o == name) / len(outcomes)
            )
        out.append(record)
    return out


def population_variance(values: list[float]) -> float:
    """Variance with no Bessel correction.

    These agents are the population, not a sample from one: var_x is a
    descriptive statistic about how spread out this run's agents ended up,
    and the inferential step happens later, across runs, in
    `analysis/stats.py`. ddof=1 would also be undefined at n=1, which is a
    real case when every probe but one failed in a round.
    """
    if not values:
        raise ValueError("variance of nothing is undefined")
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def assortativity(graph: nx.Graph | None, x_by_agent: dict[str, float]) -> float | None:
    """`numeric_assortativity_coefficient` of x over the run's graph.

    None when there is no graph, when fewer than one edge survives, or when
    the variance of x is zero. The last is the important one: the coefficient
    is a correlation, so a population where everyone holds the same position
    divides by zero -- and the honest answer is that assortativity is not
    defined there, rather than 0 ("agents are not sorted") or 1 ("perfectly
    sorted"), both of which would be read as findings.

    Computed over the subgraph induced on the agents that have a reading, so
    an agent that never produced one is absent rather than present with a
    fabricated position.
    """
    if graph is None:
        return None
    known = {a: x for a, x in x_by_agent.items() if a in graph}
    if len(set(known.values())) < 2:
        return None
    subgraph = graph.subgraph(known)
    if subgraph.number_of_edges() == 0:
        return None
    nx.set_node_attributes(subgraph := nx.Graph(subgraph), known, "x")
    return float(nx.numeric_assortativity_coefficient(subgraph, "x"))


def _mean_or_none(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    return None if cleaned.empty else float(cleaned.mean())


# --- run_metrics.csv ------------------------------------------------------


RUN_COLUMNS = (
    "run_id", "experiment", "language", "seed", "delivery", "K", "M", "N",
    "topic", "model_backend_id", "neighbor_strategy",
    "final_turn", "final_mean_x", "final_var_x", "final_extremity",
    "final_assortativity",
    "d_var_x", "d_extremity",
    *(f"rate_{name}" for name in OUTCOMES),
    "mean_n_repeat_reads",
    "t_stationary", "stationary",
    "n_valid_total", "n_failed_total",
    "mean_reply_prompt_tokens", "mean_probe_prompt_tokens",
)


def run_row(
    rounds: list[dict],
    probes: list[dict],
    config: dict,
    epsilon: dict[str, float],
) -> dict:
    """One row per run: the final state, the change, and the rates.

    `d_var_x` and `d_extremity` are final minus round 0, so they answer "did
    this population spread out over the run" directly rather than leaving two
    columns to be subtracted. Round 0 is the measurement before anything was
    read, which is exactly the right baseline for that question.

    The three rates are over every classified round of the run, not over the
    final one: a flip rate is a property of the trajectory.
    """
    ordered = sorted(rounds, key=lambda r: r["turn"])
    first, final = ordered[0], ordered[-1]
    classified = [r for r in probes if r["outcome"] is not None]
    repeats = [
        r["n_repeat_reads"] for r in probes if r["n_repeat_reads"] is not None
    ]
    t_stationary, stationary = stationary_round(ordered, epsilon, int(config["K"]))

    row = {
        "run_id": config["run_id"],
        "experiment": experiment_of(config["run_id"]),
        "language": config["language_condition"],
        "seed": config["seed"],
        "delivery": config.get("delivery"),
        "K": config["K"],
        "M": config["M"],
        "N": config["N"],
        "topic": config["topic"],
        "model_backend_id": config["model_backend_id"],
        "neighbor_strategy": config.get("neighbor_strategy"),
        "final_turn": final["turn"],
        "final_mean_x": final["mean_x"],
        "final_var_x": final["var_x"],
        "final_extremity": final["extremity"],
        "final_assortativity": final["assortativity"],
        "d_var_x": final["var_x"] - first["var_x"],
        "d_extremity": final["extremity"] - first["extremity"],
        "mean_n_repeat_reads": (
            sum(repeats) / len(repeats) if repeats else None
        ),
        "t_stationary": t_stationary,
        "stationary": stationary,
        "n_valid_total": sum(r["n_valid"] for r in ordered),
        "n_failed_total": sum(r["n_failed"] for r in ordered),
        "mean_reply_prompt_tokens": _mean_of(
            [r["mean_reply_prompt_tokens"] for r in ordered]
        ),
        "mean_probe_prompt_tokens": _mean_of(
            [r["mean_probe_prompt_tokens"] for r in ordered]
        ),
    }
    for name in OUTCOMES:
        row[f"rate_{name}"] = (
            None
            if not classified
            else sum(1 for r in classified if r["outcome"] == name) / len(classified)
        )
    return row


def _mean_of(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def stationary_round(
    rounds: list[dict], epsilon: dict[str, float], k: int
) -> tuple[int, bool]:
    """The first round after which the population stopped changing shape.

    The first round r such that `var_x`, `extremity` and `assortativity` each
    changed by less than their own ε for `STATIONARY_WINDOW` consecutive
    rounds ending at r. When no such r exists, K -- with `stationary` False,
    so a run that never settled is not reported as having settled on its last
    round.

    A round whose change cannot be computed -- an undefined assortativity at
    either end -- is not within ε. "We cannot tell whether this metric
    moved" is not evidence that it stopped moving, and the alternative would
    make a run stationary precisely because its population had collapsed to
    one position.

    ε per metric, because the three are on different scales: var_x is a
    variance of a quantity bounded in [-1, 1], extremity is a proportion, and
    assortativity is a correlation. One number for all three would be far too
    loose for one and far too tight for another.
    """
    missing = [m for m in STATIONARY_METRICS if m not in epsilon]
    if missing:
        raise ValueError(f"no epsilon for {missing}")

    changes: dict[int, bool] = {}
    for previous, current in zip(rounds, rounds[1:], strict=False):
        changes[current["turn"]] = all(
            _within(previous[metric], current[metric], epsilon[metric])
            for metric in STATIONARY_METRICS
        )

    turns = sorted(changes)
    for index in range(STATIONARY_WINDOW - 1, len(turns)):
        window = turns[index - STATIONARY_WINDOW + 1 : index + 1]
        # Consecutive rounds, so a gap in the turn numbers cannot be read as
        # a run of small changes.
        if window[-1] - window[0] != STATIONARY_WINDOW - 1:
            continue
        if all(changes[turn] for turn in window):
            return turns[index], True
    return k, False


def _within(before, after, epsilon: float) -> bool:
    if before is None or after is None:
        return False
    return abs(after - before) < epsilon


def epsilon_from_e4(rounds: list[dict]) -> dict[str, float]:
    """ε per metric, from the control arm's own round-to-round wobble.

    The 95th percentile of |change| across every E4 run in the folder, per
    metric. The control is what "nothing persuasive happened" looks like at
    the population level -- its agents read side-free messages and reply to
    them -- so its own movement is the right scale for "this metric has
    stopped moving", in the same way τ is the right scale for one agent's.

    Raises when there are no E4 runs, rather than falling back to a constant:
    an ε invented here would set the stationarity threshold for every
    reported t_stationary, and a wrong one is invisible in the output.
    """
    control = [r for r in rounds if r["experiment"] == "e4"]
    if not control:
        raise ValueError(
            "--epsilon-from-e4 needs E4 runs in the folder, and there are "
            "none. Pass --epsilon explicitly, or run the control arm: an ε "
            "invented here would set the threshold for every t_stationary."
        )
    by_run: dict[str, list[dict]] = {}
    for row in control:
        by_run.setdefault(row["run_id"], []).append(row)

    out: dict[str, float] = {}
    for metric in STATIONARY_METRICS:
        deltas: list[float] = []
        for run_rounds in by_run.values():
            ordered = sorted(run_rounds, key=lambda r: r["turn"])
            for previous, current in zip(ordered, ordered[1:], strict=False):
                if previous[metric] is None or current[metric] is None:
                    continue
                deltas.append(abs(current[metric] - previous[metric]))
        if not deltas:
            raise ValueError(
                f"the E4 runs produced no round-to-round change in {metric!r} "
                "to take a percentile of"
            )
        out[metric] = percentile(deltas, EPSILON_PERCENTILE)
    return out


def parse_epsilon(pairs: list[str] | None) -> dict[str, float]:
    """`--epsilon var_x=.. extremity=.. assortativity=..` into a dict."""
    if not pairs:
        return {}
    out: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--epsilon takes metric=value, got {pair!r}")
        metric, _, value = pair.partition("=")
        metric = metric.strip()
        if metric not in STATIONARY_METRICS:
            raise ValueError(
                f"unknown metric {metric!r} in --epsilon; expected one of "
                f"{list(STATIONARY_METRICS)}"
            )
        out[metric] = float(value)
    missing = [m for m in STATIONARY_METRICS if m not in out]
    if missing:
        raise ValueError(
            f"--epsilon needs a value for every metric; missing {missing}. "
            "The three are on different scales -- a variance, a proportion "
            "and a correlation -- so one number cannot serve all three."
        )
    return out


# --- the whole thing ------------------------------------------------------


def compute(
    runs_dir: Path, tau_path: Path, epsilon: dict[str, float], from_e4: bool
) -> tuple[list[dict], list[dict], list[dict]]:
    """Every table, from every bank run under `runs_dir`."""
    tau = load_tau(tau_path)
    probes: list[dict] = []
    rounds: list[dict] = []
    per_run: list[tuple[list[dict], list[dict], dict]] = []

    for run_dir in discover_runs(runs_dir):
        frame = load_run_frame(run_dir)
        if frame is None or frame.empty:
            print(f"  skipped {run_dir.name}: no usable rows")
            continue
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        arm_tau, arm_tau_conf = tau_for(tau, config["language_condition"], tau_path)

        run_probes = probe_rows(frame, config, arm_tau, arm_tau_conf)
        if not run_probes:
            print(f"  skipped {run_dir.name}: no valid probe readings")
            continue
        run_rounds = round_rows(frame, run_probes, config, load_graph(run_dir))

        probes.extend(run_probes)
        rounds.extend(run_rounds)
        per_run.append((run_rounds, run_probes, config))

    if from_e4:
        epsilon = epsilon_from_e4(rounds)
        print("  ε from the E4 control arm: " + ", ".join(
            f"{k}={v:.5f}" for k, v in epsilon.items()
        ))

    runs = [run_row(r, p, c, epsilon) for r, p, c in per_run]
    return probes, rounds, runs


def write_table(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    """A CSV with exactly `columns`, in that order, even when rows is empty.

    Fixed columns so a downstream reader can rely on the schema: a table
    whose columns depend on which runs happened to be present is one a
    figure script would have to guess at.
    """
    frame = pd.DataFrame(rows, columns=list(columns))
    frame.to_csv(path, index=False)
    print(f"  {path}  {len(frame):>6} rows x {len(frame.columns):>2} cols")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--tau", type=Path, required=True,
                        help="calibration/tau.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--epsilon", nargs="+", default=None, metavar="METRIC=VALUE",
        help="stationarity tolerance per metric, e.g. "
             "--epsilon var_x=0.01 extremity=0.02 assortativity=0.05",
    )
    parser.add_argument(
        "--epsilon-from-e4", action="store_true",
        help=f"take each ε as the {EPSILON_PERCENTILE:g}th percentile of that "
             "metric's absolute round-to-round change across the E4 runs in "
             "the folder",
    )
    args = parser.parse_args(argv)

    if bool(args.epsilon) == bool(args.epsilon_from_e4):
        parser.error(
            "pass exactly one of --epsilon and --epsilon-from-e4. "
            "t_stationary is meaningless without a tolerance, and a default "
            "one would set the threshold for every reported value."
        )

    epsilon = parse_epsilon(args.epsilon)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\n{args.runs_dir} -> {args.out}/")

    probes, rounds, runs = compute(
        args.runs_dir, args.tau, epsilon, args.epsilon_from_e4
    )
    if not runs:
        print("no bank-regime runs with usable rows", file=sys.stderr)
        return 1

    write_table(args.out / "probes.csv", probes, PROBE_COLUMNS)
    write_table(args.out / "round_metrics.csv", rounds, ROUND_COLUMNS)
    write_table(args.out / "run_metrics.csv", runs, RUN_COLUMNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
