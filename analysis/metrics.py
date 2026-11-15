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
import random
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

# --- E5: echo-chamber formation -------------------------------------------
#
# A population at the minority_share floor or below has collapsed onto one
# side -- consensus, not an echo chamber, which needs two camps that have
# each pulled away from the other.
MINORITY_SHARE_FLOOR = 0.2

# "2 consecutive rounds" in t_echo's definition.
ECHO_WINDOW = 2

# How many permutations assort_null_mean/assort_null_hi are built from, and
# which percentile of that distribution assort_null_hi reports. 1,000 is
# cheap enough to run per round per run and stable enough at the 97.5th
# percentile (a two-sided 95% reference region's upper arm) to use as a
# per-round significance line for one correlation coefficient.
ASSORT_NULL_RESAMPLES = 1_000
ASSORT_NULL_PERCENTILE = 97.5


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


def load_graphs_by_round(run_dir: Path) -> dict[int, nx.Graph] | None:
    """turn -> the graph used for that round's reading, from
    graph_edges_by_round.csv.

    None when the file does not exist -- every run before E5, and every E5
    run with rewire_rate=0.0 (SimulationOrchestrator skips writing it in
    that case). The caller falls back to the static graph_edges.csv graph
    for every round in that case, which is correct: a graph that never
    rewires is the same graph every round.

    Round 0 is never a key here: the round-0 probe reads nothing, so there
    is no "graph used for reading" to record for it, and the caller's
    fallback to the static graph covers it too.
    """
    path = run_dir / "graph_edges_by_round.csv"
    if not path.exists():
        return None
    by_turn: dict[int, nx.Graph] = {}
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if "turn" not in header:
            raise ValueError(f"{path} has no turn,agent_a,agent_b header")
        for line in handle:
            if not line.strip():
                continue
            turn_text, a, b = line.strip().split(",")
            by_turn.setdefault(int(turn_text), nx.Graph()).add_edge(a, b)
    return by_turn


def _graph_for_round(
    turn: int,
    graphs_by_round: dict[int, nx.Graph] | None,
    static_graph: nx.Graph | None,
) -> nx.Graph | None:
    """The graph to score round `turn` against.

    The per-round file's own entry when there is one, else the static
    graph_edges.csv graph -- which is also what a run with no per-round file
    at all uses for every round, and what round 0 always uses.
    """
    if graphs_by_round and turn in graphs_by_round:
        return graphs_by_round[turn]
    return static_graph


def load_rewires(run_dir: Path) -> dict[int, int]:
    """turn -> how many rewires were actually applied that round, from
    rewires.jsonl.

    Empty for a run with no file -- rewiring was off, so zero happened,
    which is a known fact rather than a missing one. A round with no entry
    at all (every run's round 0, and every round of a rewire_rate=0.0 run)
    reads back as 0 for the same reason.
    """
    path = run_dir / "rewires.jsonl"
    counts: dict[int, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "rewired":
            counts[row["turn"]] = counts.get(row["turn"], 0) + 1
    return counts


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


# Every experiment experiment_of recognizes. e5/e6/e7 are E5's echo-chamber
# arms (treatment, the shadow-item control, and the rewiring-without-
# selection control); recognizing them here is what lets t_echo and
# assort_excess find them by run_id the same way e3/e4 already are, and lets
# stats.py's e5-vs-e6 contrast find its two arms. e8/e9 are Fix 9's
# similarity-recommender arm and its same-feed-size random control.
EXPERIMENT_PREFIXES = ("e3", "e4", "e5", "e6", "e7", "e8", "e9")


def experiment_of(run_id: str) -> str:
    """The experiment, from the run_id prefix -- one of EXPERIMENT_PREFIXES,
    or "other".

    Read off the id rather than out of a column, because there is no column:
    `delivery` says on or off, which is the mechanism, while the experiment
    label is what the campaign and every contrast are written in terms of.
    "other" rather than a raise, so a hand-named run still appears in the
    tables instead of stopping the analysis.
    """
    prefix = run_id.split("_", 1)[0]
    return prefix if prefix in EXPERIMENT_PREFIXES else "other"


# --- round_metrics.csv ----------------------------------------------------


ROUND_COLUMNS = (
    "run_id", "experiment", "language", "seed", "delivery", "turn",
    "mean_x", "var_x", "extremity", "assortativity",
    "assort_null_mean", "assort_null_hi", "assort_excess",
    "cross_side_edge_share", "minority_share",
    "polarization_index", "global_disagreement",
    "pairwise_disagreement", "local_disagreement", "disagreement_ratio",
    "nci", "nci_null_mean", "nci_null_hi", "nci_excess", "bimodality",
    "mean_belief_shift", "mean_signed_shift",
    "n_rewires", "n_side_switches",
    # Fix 9: the similarity recommender's own round shape. All None for a
    # round with no feed_size column at all (every pre-Fix-9 run, and every
    # non-graph_similarity strategy) -- "no such measurement", the same
    # convention every other optional metric here follows.
    "mean_feed_size", "share_empty_feed", "share_neighbors_filtered",
    "mean_feed_gap", "mean_neighbor_gap",
    *(f"share_{name}" for name in OUTCOMES),
    "n_valid", "n_failed",
    "mean_reply_prompt_tokens", "mean_probe_prompt_tokens",
)


def round_rows(
    frame: pd.DataFrame,
    probes: list[dict],
    config: dict,
    graph: nx.Graph | None,
    graphs_by_round: dict[int, nx.Graph] | None = None,
    rewires: dict[int, int] | None = None,
) -> list[dict]:
    """One row per round: the population's shape, and what moved in it.

    `n_failed` is M minus the agents with a valid reading this round, read off
    the config's own M rather than counted from the rows -- an agent whose
    reply failed has a row, and an agent that never appeared at all does not,
    and both are the same kind of missing for this purpose.

    The graph metrics alone carry a missing agent's last valid x forward. An
    assortativity computed over a shrinking node set would move as agents
    dropped out, which would read as structural change; carrying the last
    reading forward keeps the node set fixed and says, correctly, that
    nothing new is known about that agent. Nothing else does this: a mean or
    a variance over carried-forward values would count one reading twice.

    `graph` is the static graph_edges.csv graph, used for every round when
    `graphs_by_round` is None or has no entry for a round -- which is every
    round of a run with no per-round file, and always round 0, which never
    rewires. `rewires` is turn -> count of rewires applied that round, from
    rewires.jsonl; a round with no entry reads back as 0 rewires, which is
    correct whether that is because rewiring never ran or because nothing
    happened to rewire.
    """
    by_turn: dict[int, list[dict]] = {}
    for row in probes:
        by_turn.setdefault(row["turn"], []).append(row)

    population = int(config["M"])
    experiment = experiment_of(config["run_id"])
    last_known: dict[str, float] = {}
    out: list[dict] = []
    # Round 0's own readings, never carried forward: mean_belief_shift and
    # mean_signed_shift compare a round's own valid reading against the
    # agent's ACTUAL round-0 probe, not against whatever was last known --
    # an agent whose round 0 failed contributes no shift at any later round.
    round0_x = {r["agent_id"]: r["x"] for r in by_turn.get(0, [])}

    for turn in sorted(by_turn):
        rows = by_turn[turn]
        xs = [r["x"] for r in rows]
        # Captured before this round's readings are folded in, so
        # n_side_switches compares against the side as of the END of the
        # previous round -- which may itself be carried forward from an
        # earlier round if this agent's immediately preceding probe failed,
        # the same carry-forward convention the graph metrics use.
        previously_known = dict(last_known)
        for row in rows:
            last_known[row["agent_id"]] = row["x"]
        known = dict(last_known)

        round_graph = _graph_for_round(turn, graphs_by_round, graph)
        assort = assortativity(round_graph, known)
        null_mean, null_hi = assortativity_null(
            round_graph, known, f"{config['seed']}|{turn}|assort_null"
        )
        assort_excess = (
            None if assort is None or null_mean is None else assort - null_mean
        )
        nci_value = nci(round_graph, known)
        nci_null_mean, nci_null_hi = nci_null(
            round_graph, known, f"{config['seed']}|{turn}|nci_null"
        )
        nci_excess = (
            None if nci_value is None or nci_null_mean is None
            else nci_value - nci_null_mean
        )
        local_dis = local_disagreement(round_graph, known)
        pairwise_dis = pairwise_disagreement(known)

        shifts = [
            row["x"] - round0_x[row["agent_id"]]
            for row in rows
            if row["agent_id"] in round0_x
        ]

        if turn == 0:
            n_side_switches = None
        else:
            n_side_switches = sum(
                1
                for row in rows
                if row["agent_id"] in previously_known
                and sign_of(row["x"]) != 0
                and sign_of(previously_known[row["agent_id"]]) != 0
                and sign_of(row["x"]) != sign_of(previously_known[row["agent_id"]])
            )

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
            "assortativity": assort,
            "assort_null_mean": null_mean,
            "assort_null_hi": null_hi,
            "assort_excess": assort_excess,
            "cross_side_edge_share": cross_side_edge_share(round_graph, known),
            "minority_share": minority_share(known),
            # Equal to var_x: both are (1/n) * sum((x - mean)^2) over this
            # round's valid readings. Kept as two columns because Wang et
            # al. name it Pz and report it on their own [-2, 2] scale,
            # where it reads 4x ours for the same underlying state.
            "polarization_index": population_variance(xs),
            "global_disagreement": global_disagreement(round_graph, known, population),
            "pairwise_disagreement": pairwise_dis,
            "local_disagreement": local_dis,
            "disagreement_ratio": disagreement_ratio(local_dis, pairwise_dis),
            "nci": nci_value,
            "nci_null_mean": nci_null_mean,
            "nci_null_hi": nci_null_hi,
            "nci_excess": nci_excess,
            "bimodality": bimodality(xs),
            "mean_belief_shift": (
                sum(abs(s) for s in shifts) / len(shifts) if shifts else None
            ),
            "mean_signed_shift": sum(shifts) / len(shifts) if shifts else None,
            "n_rewires": (rewires or {}).get(turn, 0),
            "n_side_switches": n_side_switches,
            "mean_feed_size": _mean_or_none(turn_frame.get("feed_size", pd.Series(dtype=float))),
            "share_empty_feed": _share_empty_feed(turn_frame),
            "share_neighbors_filtered": share_neighbors_filtered(
                round_graph, known, config.get("recsys_threshold")
            ),
            "mean_feed_gap": mean_feed_gap(turn_frame, known),
            "mean_neighbor_gap": local_dis,
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


def assortativity_null(
    graph: nx.Graph | None, x_by_agent: dict[str, float], seed_key: str
) -> tuple[float | None, float | None]:
    """The permutation null for this round's assortativity: mean and the
    ASSORT_NULL_PERCENTILE-th percentile over ASSORT_NULL_RESAMPLES shuffles
    of x across the graph's nodes.

    None, None under exactly the conditions `assortativity` itself returns
    None under (no graph, fewer than two distinct values, no surviving
    edge): a null distribution for a statistic that is not defined is not
    defined either, and every permutation would otherwise raise the same
    way `assortativity` declines to.

    Seeded by `seed_key` (the run's seed and the turn, per the design) so
    the null is reproducible from the run alone -- re-running this module
    over the same runs_dir must not change assort_null_hi out from under a
    reported t_echo.
    """
    if graph is None:
        return None, None
    known = {a: x for a, x in x_by_agent.items() if a in graph}
    if len(set(known.values())) < 2:
        return None, None
    subgraph = graph.subgraph(known)
    if subgraph.number_of_edges() == 0:
        return None, None
    work = nx.Graph(subgraph)
    nodes = sorted(known)
    values = [known[node] for node in nodes]

    rng = random.Random(seed_key)
    samples: list[float] = []
    for _ in range(ASSORT_NULL_RESAMPLES):
        shuffled = list(values)
        rng.shuffle(shuffled)
        nx.set_node_attributes(work, dict(zip(nodes, shuffled, strict=True)), "x")
        samples.append(float(nx.numeric_assortativity_coefficient(work, "x")))
    samples.sort()
    mean = sum(samples) / len(samples)
    return mean, percentile(samples, ASSORT_NULL_PERCENTILE)


def cross_side_edge_share(graph: nx.Graph | None, x_by_agent: dict[str, float]) -> float | None:
    """Share of this round's edges whose two ends sit on opposite sides of
    the midpoint.

    Sign of x, with 0 excluded on either end exactly as sign_of excludes it
    everywhere else: an edge touching an undecided agent is not "same side"
    or "different side", it is neither, so it is dropped from both the
    numerator and the denominator rather than counted as agreement.

    None when no edge survives with both ends known and decided -- the same
    "not defined" answer assortativity gives, for the same reason: a share
    of nothing is not zero agreement, it is no measurement.
    """
    if graph is None:
        return None
    sides = {
        agent: sign_of(x)
        for agent, x in x_by_agent.items()
        if agent in graph and sign_of(x) != 0
    }
    edges = [(a, b) for a, b in graph.edges if a in sides and b in sides]
    if not edges:
        return None
    cross = sum(1 for a, b in edges if sides[a] != sides[b])
    return cross / len(edges)


def minority_share(x_by_agent: dict[str, float]) -> float | None:
    """Share of agents on the smaller of the two sides this round.

    A population-level quantity, not a graph one -- every agent with a
    known, decided (non-zero-sign) reading counts, whether or not it is in
    the run's graph. None when nobody has a decided side (every reading is
    exactly 0, or there are no readings at all), where "the smaller side" has
    no answer.

    t_echo's floor on this is what tells consensus apart from an echo
    chamber: a population that collapsed onto one side has minority_share
    near 0, and that is not two camps that pulled apart from each other.
    """
    sides = [sign_of(x) for x in x_by_agent.values() if sign_of(x) != 0]
    if not sides:
        return None
    pro = sum(1 for s in sides if s > 0)
    con = len(sides) - pro
    return min(pro, con) / len(sides)


# --- E5 extra echo-chamber metrics (Wang et al.) --------------------------
#
# Wang et al., "Decoding Echo Chambers: LLM-Powered Simulations Revealing
# Polarization in Social Networks" (COLING 2025), use opinions in [-2, 2];
# this codebase's x is in [-1, 1]. Their Pz (polarization_index) and DG
# (global_disagreement) are therefore 4x ours for the same underlying state
# -- a scale factor, not a different quantity.


def global_disagreement(
    graph: nx.Graph | None, x_by_agent: dict[str, float], population: int
) -> float | None:
    """DG in Wang et al.: how much an agent disagrees with its own
    neighbours, averaged over the population.

        DG_i = mean over i's neighbours j of (x_i - x_j)^2
        DG   = (1 / (2 * population)) * sum_i DG_i

    Summed only over agents with at least one known neighbour -- an
    isolated node (or one with no neighbour that has a reading) has no
    DG_i to contribute, and is not the same as contributing 0 -- but the
    denominator is still `population`, not the count of agents that
    qualified, which is what makes this a population-level rate rather
    than a per-qualifying-agent average.

    None when nobody in the graph has a known neighbour to compare
    against at all, the same "no measurement" answer the other graph
    metrics give rather than 0 ("agents agree with their neighbours").
    """
    if graph is None:
        return None
    total = 0.0
    any_qualified = False
    for agent in graph.nodes:
        if agent not in x_by_agent:
            continue
        neighbors = [j for j in graph.neighbors(agent) if j in x_by_agent]
        if not neighbors:
            continue
        any_qualified = True
        xi = x_by_agent[agent]
        total += sum((xi - x_by_agent[j]) ** 2 for j in neighbors) / len(neighbors)
    if not any_qualified:
        return None
    return total / (2 * population)


def pairwise_disagreement(x_by_agent: dict[str, float]) -> float | None:
    """Mean |x_i - x_j| over all unordered pairs of agents with a known
    reading -- not in Wang et al., but the population-wide baseline
    `disagreement_ratio` needs: a graph structure's own edges can disagree
    less, or more, than two agents picked at random would.

    None below two known agents, where no pair exists.
    """
    values = list(x_by_agent.values())
    n = len(values)
    if n < 2:
        return None
    total = sum(
        abs(values[i] - values[j]) for i in range(n) for j in range(i + 1, n)
    )
    pairs = n * (n - 1) // 2
    return total / pairs


def local_disagreement(graph: nx.Graph | None, x_by_agent: dict[str, float]) -> float | None:
    """Mean |x_i - x_j| over this round's edges whose both ends have a
    known reading. Unlike cross_side_edge_share, an agent at x == 0 is not
    excluded -- this is a magnitude of disagreement, not a side comparison.

    None when no edge survives with both ends known.
    """
    if graph is None:
        return None
    known = {a: x for a, x in x_by_agent.items() if a in graph}
    edges = [(a, b) for a, b in graph.edges if a in known and b in known]
    if not edges:
        return None
    return sum(abs(known[a] - known[b]) for a, b in edges) / len(edges)


def disagreement_ratio(local: float | None, pairwise: float | None) -> float | None:
    """local_disagreement / pairwise_disagreement. Below 1 means neighbours
    disagree less than two agents picked at random would; None when the
    denominator is undefined or exactly 0 (every agent at the same x)."""
    if local is None or pairwise is None or pairwise == 0:
        return None
    return local / pairwise


# --- Fix 9: the similarity recommender's own round shape -------------------
#
# These three ask whether the recommender did anything, independent of
# whether it moved any opinion: mean_feed_size/share_empty_feed are a
# property of the FEED (what was actually delivered, under any strategy --
# GraphSampling and AlphaSampling never deliver fewer than N, so these two
# are a no-op everywhere but neighbor_strategy="graph_similarity");
# share_neighbors_filtered and mean_feed_gap are a property of the graph the
# feed was drawn from and compare it against mean_neighbor_gap, which is
# local_disagreement under a second name -- the round's own full
# neighbourhood, filtered by nothing -- so the two numbers side by side show
# whether the recommender's feed disagreed less than the raw graph would
# have.


def _share_empty_feed(turn_frame: pd.DataFrame) -> float | None:
    """Share of this round's rows whose feed_size is exactly 0.

    None when the column is absent or every value is null -- a run with no
    feed_size column never ran the similarity recommender, and "0 agents had
    an empty feed" would claim a measurement that was never made.
    """
    if "feed_size" not in turn_frame.columns:
        return None
    col = turn_frame["feed_size"].dropna()
    if col.empty:
        return None
    return float((col == 0).mean())


def mean_feed_gap(turn_frame: pd.DataFrame, x_by_agent: dict[str, float]) -> float | None:
    """Mean |x_i - x_j| over every (reader, delivered post's sender) pair
    actually assembled this round -- whatever the strategy, since every
    strategy logs neighbor_agent_ids as the feed it delivered.

    Under GraphSampling/AlphaSampling this is the same N posts every reader
    always gets; under graph_similarity it is the (up to N) posts the
    recommender actually chose, which is the number mean_neighbor_gap (the
    round's full candidate neighbourhood, filtered by nothing) is the
    baseline for.

    None when no pair has both ends known -- round 0, where nothing was
    delivered, and any round with no usable readings at all.
    """
    gaps: list[float] = []
    for row in turn_frame.itertuples():
        reader = row.speaker_agent_id
        if reader not in x_by_agent:
            continue
        raw = getattr(row, "neighbor_agent_ids", "") or ""
        for sender in str(raw).split("|"):
            if sender and sender in x_by_agent:
                gaps.append(abs(x_by_agent[reader] - x_by_agent[sender]))
    if not gaps:
        return None
    return sum(gaps) / len(gaps)


def share_neighbors_filtered(
    graph: nx.Graph | None, x_by_agent: dict[str, float], threshold: float | None
) -> float | None:
    """Share of this round's graph edges -- the full candidate
    neighbourhood, not the delivered feed -- whose two known ends disagree
    by more than `threshold`.

    None when there is no graph, no threshold (a run logged before
    recsys_threshold existed), or no edge survives with both ends known --
    the same "not defined" answer every other graph metric here gives.
    """
    if graph is None or threshold is None:
        return None
    known = {a: x for a, x in x_by_agent.items() if a in graph}
    edges = [(a, b) for a, b in graph.edges if a in known and b in known]
    if not edges:
        return None
    over = sum(1 for a, b in edges if abs(known[a] - known[b]) > threshold)
    return over / len(edges)


def _pearson(a: list[float], b: list[float]) -> float | None:
    """The Pearson correlation of `a` and `b`, or None below two
    observations or when either has zero variance -- a correlation needs
    spread on both sides to mean anything, and 0 there would be read as "no
    relationship" rather than "undefined"."""
    n = len(a)
    if n < 2:
        return None
    mean_a, mean_b = sum(a) / n, sum(b) / n
    var_a = sum((v - mean_a) ** 2 for v in a)
    var_b = sum((v - mean_b) ** 2 for v in b)
    if var_a == 0 or var_b == 0:
        return None
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    return cov / (var_a * var_b) ** 0.5


def _neighbor_mean_pairs(
    graph: nx.Graph, x_by_agent: dict[str, float]
) -> tuple[list[float], list[float]]:
    """(x_i, mean of i's known neighbours' x) for every agent with at
    least one known neighbour. Shared by `nci` and `nci_null`'s permutation
    loop, so a reshuffled assignment is scored the same way the real one
    is."""
    xs: list[float] = []
    means: list[float] = []
    for agent in graph.nodes:
        if agent not in x_by_agent:
            continue
        neighbors = [j for j in graph.neighbors(agent) if j in x_by_agent]
        if not neighbors:
            continue
        xs.append(x_by_agent[agent])
        means.append(sum(x_by_agent[j] for j in neighbors) / len(neighbors))
    return xs, means


def nci(graph: nx.Graph | None, x_by_agent: dict[str, float]) -> float | None:
    """NCI in Wang et al.: the correlation between an agent's own opinion
    and the mean opinion of its neighbours, over agents with at least one
    known neighbour.

    Follows their text description ("the correlation between a node's
    opinion and its neighbours' average opinion"), not their printed
    formula -- the paper's own formula does not match that description,
    and the text is what every other implementation of NCI follows.

    None when fewer than two agents qualify, or when either side of the
    correlation has zero variance (e.g. every qualifying agent's own x is
    identical).
    """
    if graph is None:
        return None
    xs, means = _neighbor_mean_pairs(graph, x_by_agent)
    return _pearson(xs, means)


def nci_null(
    graph: nx.Graph | None, x_by_agent: dict[str, float], seed_key: str
) -> tuple[float | None, float | None]:
    """The permutation null for `nci`: mean and the ASSORT_NULL_PERCENTILE-th
    percentile of `nci` over ASSORT_NULL_RESAMPLES shuffles of x across the
    graph's nodes -- built exactly like `assortativity_null`, including its
    seeding scheme, so the two nulls are reproducible the same way and from
    the same draw count.

    None, None when the real `nci` would itself be None for lack of
    variance to permute (fewer than two distinct values among the known
    agents) -- every permutation would read back exactly the same
    non-answer `nci` already gives.
    """
    if graph is None:
        return None, None
    known = {a: x for a, x in x_by_agent.items() if a in graph}
    if len(set(known.values())) < 2:
        return None, None
    nodes = sorted(known)
    values = [known[node] for node in nodes]

    rng = random.Random(seed_key)
    samples: list[float] = []
    for _ in range(ASSORT_NULL_RESAMPLES):
        shuffled = list(values)
        rng.shuffle(shuffled)
        sample = nci(graph, dict(zip(nodes, shuffled, strict=True)))
        if sample is not None:
            samples.append(sample)
    if not samples:
        return None, None
    samples.sort()
    mean = sum(samples) / len(samples)
    return mean, percentile(samples, ASSORT_NULL_PERCENTILE)


def bimodality(values: list[float]) -> float | None:
    """Sarle's bimodality coefficient: (g^2 + 1) / (k + 3*(n-1)^2 / ((n-2)*(n-3))),
    g the sample skewness and k the sample excess kurtosis (population
    moments, consistent with var_x's own ddof=0).

    Variance alone cannot tell a wide single group from two separated ones
    -- both can have the same spread -- and a coefficient above 5/9
    suggests two humps rather than one. None below n = 4, where the
    denominator's (n-2)(n-3) term is undefined at n=3 and the statistic is
    not meaningful below that regardless, or when the variance is 0 (every
    value identical, where skewness and kurtosis are themselves undefined).
    """
    n = len(values)
    if n < 4:
        return None
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if m2 == 0:
        return None
    m3 = sum((v - mean) ** 3 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    g = m3 / m2**1.5
    k = m4 / m2**2 - 3
    denom = k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g**2 + 1) / denom


def _mean_or_none(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    return None if cleaned.empty else float(cleaned.mean())


def reply_truncated_rate(frame: pd.DataFrame) -> float | None:
    """Share of reply calls in this run that hit max_reply_tokens.

    Over every row with a reply call, not just rows whose probe later
    succeeded -- a reply can be truncated regardless of what the probe that
    followed it read. None when the column is absent (a pre-Fix-1 dataset)
    or every row's flag is null (no reply call ever happened, e.g. a run
    with only a round-0 probe).
    """
    if "reply_truncated" not in frame.columns:
        return None
    col = frame["reply_truncated"]
    valid = col[col.notna()]
    if valid.empty:
        return None
    return float(valid.astype(bool).mean())


# --- run_metrics.csv ------------------------------------------------------


RUN_COLUMNS = (
    "run_id", "experiment", "language", "seed", "delivery", "K", "M", "N",
    "topic", "model_backend_id", "neighbor_strategy",
    "rewire_rate", "rewire_target", "lean_assignment", "recsys_threshold",
    "final_turn", "final_mean_x", "final_var_x", "final_extremity",
    "final_assortativity",
    "final_assort_excess", "final_cross_side_edge_share", "final_minority_share",
    "auc_assort_excess", "t_echo", "echo_censored",
    "d_var_x", "d_extremity",
    "final_polarization_index", "final_global_disagreement",
    "final_pairwise_disagreement", "final_local_disagreement",
    "final_disagreement_ratio", "final_nci", "final_nci_null_mean",
    "final_nci_null_hi", "final_nci_excess", "final_bimodality",
    "final_mean_belief_shift", "final_mean_signed_shift",
    "delta_polarization_index", "delta_global_disagreement", "delta_nci",
    "auc_nci_excess", "total_side_switches", "total_rewires",
    *(f"rate_{name}" for name in OUTCOMES),
    "rate_reply_truncated",
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
    rate_reply_truncated: float | None = None,
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
    t_echo, echo_censored = echo_round(ordered, int(config["K"]))

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
        "rewire_rate": config.get("rewire_rate"),
        "rewire_target": config.get("rewire_target"),
        "lean_assignment": config.get("lean_assignment"),
        "recsys_threshold": config.get("recsys_threshold"),
        "final_turn": final["turn"],
        "final_mean_x": final["mean_x"],
        "final_var_x": final["var_x"],
        "final_extremity": final["extremity"],
        "final_assortativity": final["assortativity"],
        "final_assort_excess": final["assort_excess"],
        "final_cross_side_edge_share": final["cross_side_edge_share"],
        "final_minority_share": final["minority_share"],
        "auc_assort_excess": auc_assort_excess(ordered),
        "t_echo": t_echo,
        "echo_censored": echo_censored,
        "d_var_x": final["var_x"] - first["var_x"],
        "d_extremity": final["extremity"] - first["extremity"],
        "final_polarization_index": final["polarization_index"],
        "final_global_disagreement": final["global_disagreement"],
        "final_pairwise_disagreement": final["pairwise_disagreement"],
        "final_local_disagreement": final["local_disagreement"],
        "final_disagreement_ratio": final["disagreement_ratio"],
        "final_nci": final["nci"],
        "final_nci_null_mean": final["nci_null_mean"],
        "final_nci_null_hi": final["nci_null_hi"],
        "final_nci_excess": final["nci_excess"],
        "final_bimodality": final["bimodality"],
        "final_mean_belief_shift": final["mean_belief_shift"],
        "final_mean_signed_shift": final["mean_signed_shift"],
        "delta_polarization_index": _delta(
            final["polarization_index"], first["polarization_index"]
        ),
        "delta_global_disagreement": _delta(
            final["global_disagreement"], first["global_disagreement"]
        ),
        "delta_nci": _delta(final["nci"], first["nci"]),
        "auc_nci_excess": auc_nci_excess(ordered),
        "total_side_switches": _sum_or_none(
            [r["n_side_switches"] for r in ordered]
        ),
        "total_rewires": sum(r["n_rewires"] for r in ordered),
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
        "rate_reply_truncated": rate_reply_truncated,
    }
    for name in OUTCOMES:
        row[f"rate_{name}"] = (
            None
            if not classified
            else sum(1 for r in classified if r["outcome"] == name) / len(classified)
        )
    return row


def _delta(final: float | None, first: float | None) -> float | None:
    """final minus first, or None when either side is undefined -- unlike
    d_var_x/d_extremity, which are always defined, polarization_index's
    own siblings (global_disagreement, nci) can legitimately be None at
    either end."""
    if final is None or first is None:
        return None
    return final - first


def _sum_or_none(values: list[int | None]) -> int | None:
    """Sum, dropping Nones (round 0's n_side_switches) -- None only when
    every value is None, which cannot happen for a run with K >= 1."""
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def auc_nci_excess(rounds: list[dict]) -> float | None:
    """Mean nci_excess over rounds 1..K -- the primary echo-chamber speed
    measure (see Fix 5): unlike t_echo, it needs no threshold and is
    defined for every run, censored or not.

    Round 0 excluded for auc_assort_excess's own reason: it precedes any
    reading or rewiring, so it is not a round the manipulation could have
    moved. None rounds are dropped rather than zeroed.
    """
    values = [
        r["nci_excess"]
        for r in rounds
        if r["turn"] >= 1 and r["nci_excess"] is not None
    ]
    return sum(values) / len(values) if values else None


def auc_assort_excess(rounds: list[dict]) -> float | None:
    """Mean assort_excess over rounds 1..K: a speed measure that needs no
    threshold, unlike t_echo.

    Round 0 is excluded -- it is the measurement before anything was read,
    with no rewiring and no reading behind it yet, so it is not a round the
    manipulation could have moved. None rounds (an undefined assort_excess,
    e.g. a round where the population already agrees on everything) are
    dropped rather than treated as 0: "we cannot tell" is not "no excess".
    """
    values = [
        r["assort_excess"]
        for r in rounds
        if r["turn"] >= 1 and r["assort_excess"] is not None
    ]
    return sum(values) / len(values) if values else None


def echo_round(rounds: list[dict], k: int) -> tuple[int, bool]:
    """t_echo: the first round at which the population has formed an echo
    chamber, per the design's definition.

    A round qualifies when its assortativity clears its own permutation
    null (assortativity > assort_null_hi) AND its minority_share is at
    least MINORITY_SHARE_FLOOR -- the second clause is what keeps a
    population that collapsed onto one side (consensus, not an echo
    chamber) from reporting one. t_echo is the LATER of the first
    ECHO_WINDOW consecutive qualifying rounds, mirroring
    stationary_round's own convention of reporting the end of its window
    rather than the start.

    Consecutive means adjacent round numbers, not merely two qualifying
    rows in a row of a possibly-gappy sequence (a round with no valid
    probes at all has no row here and would otherwise bridge a gap that
    is not evidence of anything).

    K + 1 with echo_censored=True when no such window exists -- never K,
    which is a real round number and would read as "formed on the last
    round" rather than "never formed".
    """
    ordered = sorted(rounds, key=lambda r: r["turn"])
    qualifies = [
        r["assortativity"] is not None
        and r["assort_null_hi"] is not None
        and r["assortativity"] > r["assort_null_hi"]
        and r["minority_share"] is not None
        and r["minority_share"] >= MINORITY_SHARE_FLOOR
        for r in ordered
    ]
    turns = [r["turn"] for r in ordered]
    for index in range(ECHO_WINDOW - 1, len(turns)):
        window = turns[index - ECHO_WINDOW + 1 : index + 1]
        if window[-1] - window[0] != ECHO_WINDOW - 1:
            continue  # a gap in turn numbers, not ECHO_WINDOW consecutive rounds
        if all(qualifies[index - ECHO_WINDOW + 1 : index + 1]):
            return turns[index], False
    return k + 1, True


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


# --- Fix 7: belief_shift_by_trait.csv --------------------------------------
#
# Reads the structured persona attributes (off by default via
# ExperimentRun.persona_attributes) back out of agents_final.jsonl, the
# same file export_csv.py's agents.csv joins them from -- this module
# never reads export_csv.py's output, so it is read independently here.

TRAIT_COLUMNS = (
    "gender", "education",
    "trait_openness", "trait_conscientiousness", "trait_extraversion",
    "trait_agreeableness", "trait_neuroticism",
)

BELIEF_SHIFT_BY_TRAIT_COLUMNS = (
    "trait", "level", "language", "n", "mean_abs_shift", "var_abs_shift",
)


def load_agent_traits(run_dir: Path) -> dict[str, dict] | None:
    """agent_id -> {trait_column: level}, from agents_final.jsonl.

    None when the file does not exist yet (an interrupted run) or when
    every row's TRAIT_COLUMNS are null (persona_attributes was "none",
    which is every pre-Fix-7 run) -- the same "nothing to report" signal
    the rest of this module gives rather than an empty-but-present table.
    age is deliberately not in TRAIT_COLUMNS: it is continuous, not a
    level, and "mean shift by age" is a different question (a regression,
    not a group-by) than this table answers.
    """
    path = run_dir / "agents_final.jsonl"
    if not path.exists():
        return None
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return None
    if not any(any(row.get(c) is not None for c in TRAIT_COLUMNS) for row in rows):
        return None
    return {row["agent_id"]: {c: row.get(c) for c in TRAIT_COLUMNS} for row in rows}


def agent_belief_shifts(probes: list[dict], k: int) -> dict[str, float]:
    """agent_id -> |x(k) - x(0)|, for agents with a valid probe at both
    round 0 and round k of THIS run's own configured K -- not "the last
    round that agent had a valid probe", which belief_shift_by_trait
    would otherwise be comparing across agents at different rounds.
    """
    by_agent: dict[str, dict[int, float]] = {}
    for row in probes:
        by_agent.setdefault(row["agent_id"], {})[row["turn"]] = row["x"]
    return {
        agent: abs(turns[k] - turns[0])
        for agent, turns in by_agent.items()
        if 0 in turns and k in turns
    }


def belief_shift_by_trait(rows: list[dict]) -> list[dict]:
    """Groups belief-shift rows (one per agent per run, from `compute`)
    by (trait, level, language): the mean and population variance
    (ddof=0, consistent with population_variance elsewhere in this
    module) of |x(K) - x(0)|.
    """
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (row["trait"], row["level"], row["language"])
        groups.setdefault(key, []).append(row["shift"])
    out = []
    for (trait, level, language), shifts in sorted(groups.items()):
        n = len(shifts)
        mean = sum(shifts) / n
        variance = sum((s - mean) ** 2 for s in shifts) / n
        out.append({
            "trait": trait, "level": level, "language": language,
            "n": n, "mean_abs_shift": mean, "var_abs_shift": variance,
        })
    return out


def compute(
    runs_dir: Path, tau_path: Path, epsilon: dict[str, float], from_e4: bool
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Every table, from every bank run under `runs_dir`.

    The fourth element is belief_shift_by_trait's own rows -- empty when
    no run under `runs_dir` carries any TRAIT_COLUMNS, which main() reads
    to decide whether to write the file at all.
    """
    tau = load_tau(tau_path)
    probes: list[dict] = []
    rounds: list[dict] = []
    per_run: list[tuple[list[dict], list[dict], dict, float | None]] = []
    belief_shift_rows: list[dict] = []

    for run_dir in discover_runs(runs_dir):
        frame = load_run_frame(run_dir)
        if frame is None or frame.empty:
            print(f"  skipped {run_dir.name}: no usable rows")
            continue
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        arm_tau, arm_tau_conf = tau_for(tau, config["language_condition"], tau_path)
        run_mode = config.get("start_mode", "sentence")
        tau_mode = tau.get("start_mode", "sentence")
        if run_mode != tau_mode:
            print(
                f"  WARNING: {run_dir.name} ran with start_mode={run_mode!r} but "
                f"{tau_path} was calibrated with {tau_mode!r}. τ is the noise of "
                "one instrument; its flips and hardenings here are judged "
                "against another's."
            )

        run_probes = probe_rows(frame, config, arm_tau, arm_tau_conf)
        if not run_probes:
            print(f"  skipped {run_dir.name}: no valid probe readings")
            continue
        run_rounds = round_rows(
            frame, run_probes, config, load_graph(run_dir),
            graphs_by_round=load_graphs_by_round(run_dir),
            rewires=load_rewires(run_dir),
        )

        probes.extend(run_probes)
        rounds.extend(run_rounds)
        per_run.append((run_rounds, run_probes, config, reply_truncated_rate(frame)))

        traits = load_agent_traits(run_dir)
        if traits is not None:
            shifts = agent_belief_shifts(run_probes, int(config["K"]))
            for agent_id, shift in shifts.items():
                for trait_name, level in traits.get(agent_id, {}).items():
                    if level is None:
                        continue
                    belief_shift_rows.append({
                        "trait": trait_name, "level": str(level),
                        "language": config["language_condition"], "shift": shift,
                    })

    if from_e4:
        epsilon = epsilon_from_e4(rounds)
        print("  ε from the E4 control arm: " + ", ".join(
            f"{k}={v:.5f}" for k, v in epsilon.items()
        ))

    runs = [run_row(r, p, c, epsilon, rt) for r, p, c, rt in per_run]
    return probes, rounds, runs, belief_shift_rows


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

    probes, rounds, runs, belief_shift_rows = compute(
        args.runs_dir, args.tau, epsilon, args.epsilon_from_e4
    )
    if not runs:
        print("no bank-regime runs with usable rows", file=sys.stderr)
        return 1

    write_table(args.out / "probes.csv", probes, PROBE_COLUMNS)
    write_table(args.out / "round_metrics.csv", rounds, ROUND_COLUMNS)
    write_table(args.out / "run_metrics.csv", runs, RUN_COLUMNS)
    # Only when at least one run actually carried a structured persona
    # attribute (Fix 7, off by default): a campaign that never set
    # persona_attributes must not grow an all-empty belief_shift_by_trait.csv.
    if belief_shift_rows:
        write_table(
            args.out / "belief_shift_by_trait.csv",
            belief_shift_by_trait(belief_shift_rows),
            BELIEF_SHIFT_BY_TRAIT_COLUMNS,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
