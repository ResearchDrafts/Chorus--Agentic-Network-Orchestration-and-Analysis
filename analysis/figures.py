"""The four figures, as PDF and PNG.

    python -m analysis.figures --results results/ --e1 e1/trials.jsonl \
            --calibration calibration/ --out results/figures/

    1  trajectories.{pdf,png}   mean x and var_x over rounds, per language,
                                E3 solid and E4 dashed, with a band over seeds
    2  e1_toward.{pdf,png}      mean `toward` by language x strength
    3  tau.{pdf,png}            τ and τ_conf per language
    4  instrument.{pdf,png}     probe mass per language, and mean reply
                                prompt length per language by round

Matplotlib only, and no styling beyond what is set here, so a figure looks the
same in a notebook, in CI and on a laptop with somebody's matplotlibrc.

**One colour per language, used in all four figures.** Language is the
identity being compared, so it is the only thing colour encodes anywhere
here; E3 and E4 are distinguished by line style and strength by position,
never by a second set of hues. A reader who learns the four colours once
reads every panel.

The four hues are the first three slots of the reference categorical palette
plus violet, which is the four-colour set that clears the all-pairs
colourblind-separation and normal-vision floors in light mode -- the default
fourth slot (yellow) does not, because yellow beside orange falls below the
normal-vision floor. Aqua sits under 3:1 against the surface, so every series
is also directly labelled and every number is in the CSVs these figures are
drawn from; colour is never the only encoding.

Light surface only, deliberately: these are figures for a paper, and a PDF has
one rendering.

**Nothing is computed here that `metrics.py` or `stats.py` already computes.**
A figure that recomputed a metric would be a second implementation of it, and
the one that disagreed would be the one on the page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display in CI, in Colab, or over ssh

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from analysis import metrics
from sandbox.models import LANGUAGE_CONDITIONS
from sandbox.single_exposure import STATUS_OK, load_trials

# Language -> colour. Fixed order, never cycled: the hue follows the arm, so
# dropping an arm from a figure must not repaint the others.
LANGUAGE_COLORS = {
    "english": "#2a78d6",          # blue
    "hinglish": "#eb6834",         # orange
    "romanized_hindi": "#1baf7a",  # aqua
    "hindi": "#4a3aa7",            # violet
}

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2de"

# Experiment -> line style. E3 solid because it is the treatment and the thing
# being read; E4 dashed because it is the baseline underneath it.
EXPERIMENT_STYLES = {"e3": "-", "e4": "--"}
EXPERIMENT_LABELS = {"e3": "E3 (delivery on)", "e4": "E4 (control)"}

# E5's echo-chamber arms, in the same solid-treatment / dashed-control
# convention: e5 is rewiring on, e6 is the shadow-item control that runs the
# same rewiring rule without delivering anything persuasive. e7 is dotted,
# rather than a third dash style sharing a lane with e4's convention
# elsewhere: it rewires without selecting a same-side neighbour, a control
# for whether rewiring ALONE sorts anything, and drawing it the same as a
# delivery-off control would misread it as one.
ECHO_EXPERIMENT_STYLES = {"e5": "-", "e6": "--", "e7": ":"}
ECHO_EXPERIMENT_LABELS = {
    "e5": "E5 (rewiring on)", "e6": "E6 (control)", "e7": "E7 (random rewiring)",
}

# Fix 9's similarity-recommender arm (e8, solid) and its same-feed-size
# random control (e9, dashed) -- a separate dict from ECHO_EXPERIMENT_STYLES
# rather than added into it, since e8/e9 never rewire and folding them into
# the E5-family dict would silently add their lines to figures captioned
# "solid: E5 · dashed: E6 · dotted: E7".
RECSYS_EXPERIMENT_STYLES = {"e8": "-", "e9": "--"}
RECSYS_EXPERIMENT_LABELS = {
    "e8": "E8 (similarity recommender)", "e9": "E9 (random, same feed size)",
}

# Side categories for the network-snapshot figure, by sign of x -- the same
# three-way split cross_side_edge_share and minority_share use upstream in
# analysis/metrics.py, computed here rather than imported, since that
# module does not expose a standalone sign-of-x classifier (sign_of returns
# -1/0/+1, which this just names for the legend).
SIDE_COLORS = {"pro": "#1baf7a", "undecided": "#9c9a94", "con": "#eb6834"}
UNDECIDED_BAND = 1.0 / 3.0  # |x| below this is "undecided" in the snapshot

LINE_WIDTH = 2.0
MARKER_SIZE = 5.0
BAND_ALPHA = 0.15

# How many item-level resamples back figure 2's intervals. Fewer than the
# inference bootstrap in stats.py, because an interval on a figure is read to
# one or two significant figures and 2,000 is already stable there.
FIGURE_BOOTSTRAP = 2_000
FIGURE_BOOTSTRAP_SEED = 20_260_102


def apply_style() -> None:
    """Recessive grid, recessive axes, thin marks, readable text.

    Set explicitly rather than through a style sheet name so the figures do
    not depend on which matplotlib version ships which style.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            # Behind the marks. A grid line showing through a bar reads as a
            # segment boundary that is not there.
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "text.color": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def languages_in(frame: pd.DataFrame) -> list[str]:
    """The arms present, in LANGUAGE_CONDITIONS order with English first.

    English first everywhere, because every other arm is reported as a
    difference from it.
    """
    present = set(frame.language.dropna().unique())
    return [language for language in LANGUAGE_CONDITIONS if language in present]


def color_for(language: str) -> str:
    """Raises for an unknown arm rather than falling back to a default hue.

    A fifth language silently sharing a colour with one of the four would
    make two arms indistinguishable in a figure while looking perfectly
    normal.
    """
    if language not in LANGUAGE_COLORS:
        raise KeyError(
            f"no colour assigned to {language!r}. Add one to LANGUAGE_COLORS "
            "and re-run the palette validator: a new arm sharing a hue would "
            "make two arms indistinguishable while looking normal."
        )
    return LANGUAGE_COLORS[language]


def save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    """Both formats, same stem. PDF for the paper, PNG for a notebook."""
    paths = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


# --- 1: trajectories ------------------------------------------------------


def figure_trajectories(rounds: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Mean x and var_x over rounds, per language, with a band over seeds.

    Two stacked panels, one measure each. Not one panel with two y-scales: a
    dual axis lets the reader's eye infer a relationship between two
    arbitrarily scaled series, and mean x and var_x genuinely move
    independently -- a population can polarize (var up) with its mean
    unchanged, which is the central thing these panels have to be able to
    show separately.

    The band is the full range across seeds, not a confidence interval. With
    five seeds a parametric interval would be a stronger claim than the data
    supports; min-to-max says exactly what it is, and `stats.py` is where the
    inference happens.
    """
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.8), sharex=True)
    panels = (
        (axes[0], "mean_x", "mean x\n(−1 = con pole, +1 = pro pole)"),
        (axes[1], "var_x", "var(x)\n(x² units; 0 = all agents together)"),
    )

    for axis, metric, ylabel in panels:
        for language in languages_in(rounds):
            for experiment, style in EXPERIMENT_STYLES.items():
                arm = rounds[
                    (rounds.language == language) & (rounds.experiment == experiment)
                ]
                if arm.empty:
                    continue
                grouped = arm.groupby("turn")[metric]
                turns = sorted(arm.turn.unique())
                mean = grouped.mean().reindex(turns)
                axis.plot(
                    turns,
                    mean.values,
                    style,
                    color=color_for(language),
                    marker="o" if experiment == "e3" else None,
                    label=f"{language} · {EXPERIMENT_LABELS[experiment]}",
                )
                low = grouped.min().reindex(turns)
                high = grouped.max().reindex(turns)
                if arm.seed.nunique() > 1:
                    axis.fill_between(
                        turns, low.values, high.values,
                        color=color_for(language), alpha=BAND_ALPHA, linewidth=0,
                    )
        axis.set_ylabel(ylabel)

    axes[0].set_title(
        "Stance trajectories by language\n"
        "band: range over seeds · solid: E3 treatment · dashed: E4 control",
        loc="left",
    )
    axes[1].set_xlabel("round  (0 = probed before reading anything)")
    axes[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    # One legend for the figure, below both panels. Under the upper panel it
    # would sit inside the lower one's y-label and collide with it.
    # Laid out first, then hung below the figure box, for the reason
    # figure_tau's header is placed that way: reserving a fraction with
    # tight_layout's rect means guessing how tall the legend will be, and the
    # guess leaves a band of empty surface. savefig(bbox_inches="tight")
    # crops to the legend either way.
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, ncol=min(len(handles), 2),
            loc="upper center", bbox_to_anchor=(0.5, -0.01),
        )
    return save(fig, out_dir, "trajectories")


# --- E5: echo-chamber formation --------------------------------------------
#
# Four figures, read together: the two line-by-round panels below show
# WHETHER and HOW FAST a population sorted itself into camps; the snapshot
# grid shows WHAT that looks like structurally, for one seed; and the dot
# plot shows HOW MUCH the sorting speed varies across seeds and languages.
# Nothing here recomputes assort_excess, cross_side_edge_share, minority_share
# or t_echo -- all four are read straight from round_metrics.csv/
# run_metrics.csv, which analysis/metrics.py already computed them into.


def _line_by_round_figure(
    rounds: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    stem: str,
    out_dir: Path,
    styles: dict[str, str],
    labels: dict[str, str],
) -> list[Path]:
    """One panel, one metric, one line per (language, experiment) pair.

    Shared by figure_assort_excess and figure_cross_side_edge_share, which
    differ only in which column they read and how they describe themselves
    -- the same band-over-seeds, solid-treatment/dashed-control drawing
    figure_trajectories already established for E3/E4.
    """
    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    for language in languages_in(rounds):
        for experiment, style in styles.items():
            arm = rounds[(rounds.language == language) & (rounds.experiment == experiment)]
            arm = arm[arm[metric].notna()]
            if arm.empty:
                continue
            grouped = arm.groupby("turn")[metric]
            turns = sorted(arm.turn.unique())
            mean = grouped.mean().reindex(turns)
            axis.plot(
                turns, mean.values, style, color=color_for(language),
                marker="o" if experiment == list(styles)[0] else None,
                label=f"{language} · {labels[experiment]}",
            )
            low = grouped.min().reindex(turns)
            high = grouped.max().reindex(turns)
            if arm.seed.nunique() > 1:
                axis.fill_between(
                    turns, low.values, high.values,
                    color=color_for(language), alpha=BAND_ALPHA, linewidth=0,
                )
    axis.set_ylabel(ylabel)
    axis.set_xlabel("round  (0 = probed before reading anything)")
    axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axis.set_title(title, loc="left")
    fig.tight_layout()
    handles, labels_ = axis.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels_, ncol=min(len(handles), 2),
            loc="upper center", bbox_to_anchor=(0.5, -0.02),
        )
    return save(fig, out_dir, stem)


def figure_assort_excess(rounds: pd.DataFrame, out_dir: Path) -> list[Path]:
    """assort_excess by round: assortativity above its own permutation null,
    one line per language, E5 solid and E6 (the shadow-item control) dashed.

    A population that has not sorted itself at all has assort_excess near
    0; a rising line is camps forming faster than chance adjacency alone
    would produce.
    """
    return _line_by_round_figure(
        rounds, "assort_excess",
        "assortativity − null mean\n(excess over random mixing)",
        "Echo-chamber formation by language\n"
        "band: range over seeds · solid: E5 (rewiring on) · dashed: E6 (control)",
        "assort_excess", out_dir,
        ECHO_EXPERIMENT_STYLES, ECHO_EXPERIMENT_LABELS,
    )


def figure_cross_side_edge_share(rounds: pd.DataFrame, out_dir: Path) -> list[Path]:
    """cross_side_edge_share by round: the share of edges still linking
    opposite sides, one line per language, E5 solid and E6 dashed.

    The complementary read to assort_excess: a falling line is neighbours
    increasingly agreeing with each other, which is what rewiring toward
    one's own side looks like from the edge list alone.
    """
    return _line_by_round_figure(
        rounds, "cross_side_edge_share",
        "share of edges crossing sides\n(0 = fully sorted, higher = more mixed)",
        "Cross-side edges by language\n"
        "band: range over seeds · solid: E5 (rewiring on) · dashed: E6 (control)",
        "cross_side_edge_share", out_dir,
        ECHO_EXPERIMENT_STYLES, ECHO_EXPERIMENT_LABELS,
    )


def figure_nci_excess(rounds: pd.DataFrame, out_dir: Path) -> list[Path]:
    """nci_excess by round: the neighbour-correlation index above its own
    permutation null, one line per language, E5/E6/E7 in the usual styles.

    auc_nci_excess (its area under this curve) is the primary echo-chamber
    outcome (see analysis/stats.py); this figure is what that number is
    the area under.
    """
    return _line_by_round_figure(
        rounds, "nci_excess",
        "neighbour-correlation index − null mean\n(excess over random mixing)",
        "Neighbour-correlation excess by language\n"
        "band: range over seeds · solid: E5 · dashed: E6 · dotted: E7",
        "nci_excess", out_dir,
        ECHO_EXPERIMENT_STYLES, ECHO_EXPERIMENT_LABELS,
    )


def figure_disagreement_ratio(rounds: pd.DataFrame, out_dir: Path) -> list[Path]:
    """disagreement_ratio by round: local (edge) disagreement over
    pairwise (population-wide) disagreement, one line per language.

    Below 1 means neighbours disagree less than two agents picked at
    random would -- a falling line is the same sorting assort_excess and
    cross_side_edge_share show, read in units of opinion distance instead
    of a correlation or an edge share.
    """
    return _line_by_round_figure(
        rounds, "disagreement_ratio",
        "local / pairwise disagreement\n(below 1 = neighbours agree more than random)",
        "Disagreement ratio by language\n"
        "band: range over seeds · solid: E5 · dashed: E6 · dotted: E7",
        "disagreement_ratio", out_dir,
        ECHO_EXPERIMENT_STYLES, ECHO_EXPERIMENT_LABELS,
    )


def figure_minority_share(rounds: pd.DataFrame, out_dir: Path) -> list[Path]:
    """minority_share by round: the smaller side's population share, one
    line per language.

    t_echo's own floor on this is what keeps a population that collapsed
    onto one side (consensus) from being read as an echo chamber; this
    figure shows the quantity that floor is checked against directly.
    """
    return _line_by_round_figure(
        rounds, "minority_share",
        "minority share\n(share of agents on the smaller side)",
        "Minority share by language\n"
        "band: range over seeds · solid: E5 · dashed: E6 · dotted: E7",
        "minority_share", out_dir,
        ECHO_EXPERIMENT_STYLES, ECHO_EXPERIMENT_LABELS,
    )


def figure_n_side_switches(rounds: pd.DataFrame, out_dir: Path) -> list[Path]:
    """n_side_switches by round: how many agents changed which side of
    the midpoint they are on, one line per language.

    Read against n_rewires (round_metrics.csv's own column, not drawn a
    second time here): a population can sort by converting agents to the
    majority's side (switches) or by agents dropping an opposite-side
    neighbour for a same-side one (rewires), and this is the conversion
    half of that race.
    """
    return _line_by_round_figure(
        rounds, "n_side_switches",
        "side switches\n(agents whose side differs from the previous round)",
        "Side switches by language -- read against n_rewires\n"
        "band: range over seeds · solid: E5 · dashed: E6 · dotted: E7",
        "n_side_switches", out_dir,
        ECHO_EXPERIMENT_STYLES, ECHO_EXPERIMENT_LABELS,
    )


def figure_t_echo(runs: pd.DataFrame, out_dir: Path) -> list[Path]:
    """t_echo per language: one dot per seed, for every experiment present
    that has a t_echo column with values (e5, e6, e7, if present).

    A censored run (echo_censored=True, t_echo == K+1) is drawn as an open
    marker rather than dropped, so a language that never formed an echo
    chamber within K rounds is visibly different from one that is simply
    missing from the plot.
    """
    arms = [e for e in ("e5", "e6", "e7") if (runs.experiment == e).any()]
    fig, axis = plt.subplots(figsize=(7.4, 0.9 + 0.9 * max(len(arms), 1)))
    y_labels: list[str] = []
    y = 0
    for experiment in arms:
        for language in languages_in(runs[runs.experiment == experiment]):
            arm = runs[(runs.experiment == experiment) & (runs.language == language)]
            censored = arm.get("echo_censored")
            for _, row in arm.iterrows():
                is_censored = bool(row.get("echo_censored", False))
                axis.scatter(
                    row["t_echo"], y, color=color_for(language),
                    marker="o" if not is_censored else "x",
                    s=40, zorder=3,
                )
            y_labels.append(f"{language} · {experiment.upper()}")
            y += 1
    axis.set_yticks(range(len(y_labels)))
    axis.set_yticklabels(y_labels)
    axis.set_xlabel("t_echo  (round an echo chamber first formed; × = never within K)")
    axis.set_title(
        "Speed of echo-chamber formation\none dot per seed · × = censored at K+1",
        loc="left",
    )
    axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return save(fig, out_dir, "t_echo")


class RunGraphs:
    """One run's graph history, read from its run directory.

    Backs figure_network_snapshots: edges_at(turn) is the graph used for
    reading in that round (graph_edges_by_round.csv, falling back to
    graph_edges.csv exactly as analysis.metrics reads the same two files
    for round_metrics.csv); x_at(turn, probes) looks up each node's x in the
    already-aggregated probes.csv rather than re-deriving it from the run's
    own interactions.jsonl, so the snapshot's node colours agree with every
    other figure's x values by construction.
    """

    def __init__(self, run_dir: Path, run_id: str, k: int):
        self.run_id = run_id
        self.k = k
        self._static = metrics.load_graph(run_dir)
        self._by_round = metrics.load_graphs_by_round(run_dir)

    def edges_at(self, turn: int) -> list[tuple[str, str]]:
        graph = metrics._graph_for_round(turn, self._by_round, self._static)
        return list(graph.edges) if graph is not None else []

    def x_at(self, turn: int, probes: pd.DataFrame) -> dict[str, float]:
        rows = probes[(probes.run_id == self.run_id) & (probes.turn == turn)]
        return dict(zip(rows.agent_id, rows.x, strict=True))


def load_run_graphs(
    runs_dir: Path, runs: pd.DataFrame, experiment: str = "e5"
) -> dict[str, RunGraphs]:
    """One RunGraphs per language: the lowest-seed `experiment` run present
    in run_metrics.csv for that language.

    The lowest seed, rather than any particular one, so the figure is
    reproducible from the same inputs every time it is drawn -- "one seed
    per language" (the design's own phrase) needs a rule, not a free choice.
    Skips a language whose chosen run directory is not actually present
    under runs_dir (e.g. a campaign analysed from a copy of just the three
    results CSVs), rather than raising: this is the one figure that needs
    raw run directories, and main() already treats it as optional.
    """
    out: dict[str, RunGraphs] = {}
    arm = runs[runs.experiment == experiment]
    for language in languages_in(arm):
        candidates = arm[arm.language == language].sort_values("seed")
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        run_dir = runs_dir / str(row["run_id"])
        if not run_dir.exists():
            continue
        out[language] = RunGraphs(run_dir, str(row["run_id"]), int(row["K"]))
    return out


def figure_network_snapshots(
    probes: pd.DataFrame,
    run_graphs: dict[str, RunGraphs],
    out_dir: Path,
    stem: str = "network_snapshots",
    experiment_label: str | None = None,
) -> list[Path]:
    """One row per language, one column per snapshot round (0, K/2, K): the
    social graph laid out by spring_layout, nodes coloured by side.

    `run_graphs` is keyed by language and supplies, for one chosen seed's
    run, the edges at each of the three rounds (RunGraphs.edges_at) -- read
    from graph_edges_by_round.csv with the graph_edges.csv fallback, exactly
    as analysis/metrics.py reads them, which main() builds via
    `load_run_graphs` before calling this.

    spring_layout is seeded (FIGURE_LAYOUT_SEED) and recomputed per round:
    an echo chamber is a cluster that forms over rounds, and a layout fixed
    at round 0 would not show the separation that only exists in later
    rounds' edges.

    `stem`/`experiment_label` let main() call this once per experiment
    (e5, e6, e7) that has runs, each to its own file, rather than always
    overwriting one "network_snapshots" figure with whichever experiment
    ran last.
    """
    languages = [lang for lang in LANGUAGE_CONDITIONS if lang in run_graphs]
    if not languages:
        return []
    fig, axes = plt.subplots(
        len(languages), len(SNAPSHOT_LABELS),
        figsize=(3.2 * len(SNAPSHOT_LABELS), 3.0 * len(languages)),
        squeeze=False,
    )
    for row, language in enumerate(languages):
        graphs = run_graphs[language]
        for col, (turn, label) in enumerate(SNAPSHOT_LABELS_FOR(graphs.k)):
            axis = axes[row][col]
            edges = graphs.edges_at(turn)
            x_by_agent = graphs.x_at(turn, probes)
            _draw_one_snapshot(axis, edges, x_by_agent)
            if row == 0:
                axis.set_title(label)
            if col == 0:
                axis.set_ylabel(language, rotation=0, ha="right", va="center")
    title = "Network structure over rounds, one seed per language"
    if experiment_label:
        title += f" ({experiment_label})"
    fig.suptitle(
        f"{title}\nnode colour: pro / undecided / against the midpoint",
        x=0.02, ha="left", fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return save(fig, out_dir, stem)


def _draw_one_snapshot(axis, edges: list[tuple[str, str]], x_by_agent: dict[str, float]) -> None:
    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph.add_nodes_from(x_by_agent)  # an isolate (no edges this round) still appears
    # k = 1/sqrt(n) and 300 iterations: the default k (1/sqrt(n) is itself
    # networkx's default, made explicit here) under-separates two camps
    # that share only a few cross edges, since the spring layout settles
    # into a single blob before the repulsive force has pulled same-side
    # clusters apart; 300 iterations (up from the default 50) gives it
    # room to actually reach that separated layout rather than stopping
    # partway through converging to it.
    k = 1.0 / (len(graph) ** 0.5) if len(graph) else None
    layout = nx.spring_layout(graph, k=k, iterations=300, seed=FIGURE_LAYOUT_SEED)
    colors = [
        SIDE_COLORS["pro"] if x_by_agent.get(node, 0.0) > UNDECIDED_BAND
        else SIDE_COLORS["con"] if x_by_agent.get(node, 0.0) < -UNDECIDED_BAND
        else SIDE_COLORS["undecided"]
        for node in graph.nodes
    ]
    nx.draw_networkx_edges(graph, layout, ax=axis, edge_color=GRID, width=0.8)
    nx.draw_networkx_nodes(graph, layout, ax=axis, node_color=colors, node_size=60)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


FIGURE_LAYOUT_SEED = 20_260_103


def SNAPSHOT_LABELS_FOR(k: int) -> list[tuple[int, str]]:
    """The three snapshot rounds for a run of length K: 0, K//2, K -- with
    the middle round dropped if it collides with either end (K < 2), so a
    very short smoke run draws two panels rather than three identical ones.
    """
    rounds = sorted({0, k // 2, k})
    return [(r, f"round {r}") for r in rounds]


SNAPSHOT_LABELS = ("round 0", "round K/2", "round K")  # the usual (K >= 2) case


# --- 2: E1 ----------------------------------------------------------------


def _bootstrap_interval_over_items(
    frame: pd.DataFrame, seed: int = FIGURE_BOOTSTRAP_SEED
) -> tuple[float, float]:
    """A 95% interval on mean `toward`, resampling ITEMS.

    Items, not trials. Trials sharing an item are not independent -- some
    arguments are simply more persuasive -- so resampling trials would give an
    interval narrowed by however many personas read each item, which is a
    fact about the sweep's size rather than about the uncertainty. This is the
    same clustering the model in `stats.py` handles with a random intercept
    per item.
    """
    items = sorted(frame.item_id.unique())
    if len(items) < 2:
        return (float("nan"), float("nan"))
    by_item = {item: frame[frame.item_id == item].toward.to_numpy() for item in items}
    rng = np.random.default_rng(seed)
    means = np.empty(FIGURE_BOOTSTRAP)
    for draw in range(FIGURE_BOOTSTRAP):
        picked = rng.integers(0, len(items), size=len(items))
        values = np.concatenate([by_item[items[i]] for i in picked])
        means[draw] = values.mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(low), float(high))


def figure_e1(e1: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Mean `toward` by language and strength, with intervals.

    Grouped bars: strength on the x axis, language by colour and position.
    Position as well as colour, so the four arms are separable without
    relying on hue -- which the aqua slot's contrast against the surface
    requires anyway.

    A zero line, because `toward` is signed: a bar below it is a reader moving
    AWAY from what it read, which is a different result from a small effect
    and would be invisible without the reference.
    """
    languages = languages_in(e1)
    strengths = sorted(e1.strength.unique())
    fig, axis = plt.subplots(figsize=(7.6, 4.6))

    width = 0.8 / max(len(languages), 1)
    for index, language in enumerate(languages):
        centers, means, lows, highs = [], [], [], []
        for position, strength in enumerate(strengths):
            cell = e1[(e1.language == language) & (e1.strength == strength)]
            if cell.empty:
                continue
            low, high = _bootstrap_interval_over_items(cell)
            mean = float(cell.toward.mean())
            centers.append(position + (index - (len(languages) - 1) / 2) * width)
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)
        if not centers:
            continue
        axis.bar(
            centers, means, width * 0.88,
            color=color_for(language), label=language,
            # A 2px surface gap between adjacent bars, drawn as an edge in the
            # surface colour.
            edgecolor=SURFACE, linewidth=2.0,
        )
        axis.errorbar(
            centers, means,
            yerr=[np.abs(lows), np.abs(highs)],
            fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1.2, capsize=3,
        )
        # Labelled above the whisker, not above the bar: a label at the bar's
        # own height sits inside the interval and hides the end of it.
        for center, mean, high in zip(centers, means, highs, strict=True):
            top = mean + (abs(high) if np.isfinite(high) else 0.0)
            axis.annotate(
                f"{mean:+.3f}",
                (center, top if mean >= 0 else mean - abs(high)),
                textcoords="offset points",
                xytext=(0, 5 if mean >= 0 else -13),
                ha="center", fontsize=7.5, color=TEXT_SECONDARY,
            )

    axis.axhline(0, color=TEXT_SECONDARY, linewidth=1.0, zorder=1)
    axis.set_xticks(range(len(strengths)))
    axis.set_xticklabels(strengths)
    axis.set_xlabel("argument strength (as labelled in the bank)")
    axis.set_ylabel("mean toward\n(x units moved the way the item argued)")
    axis.set_title(
        "E1: how far one argument moves one reader\n"
        "bars: mean over trials · whiskers: 95% bootstrap interval, resampling items",
        loc="left",
    )
    fig.tight_layout()
    axis.legend(
        ncol=min(len(languages), 4), loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
    )
    return save(fig, out_dir, "e1_toward")


# --- 3: tau ---------------------------------------------------------------


def figure_tau(tau: dict, out_dir: Path) -> list[Path]:
    """τ and τ_conf per language, side by side.

    Two panels, not one grouped chart: τ is in x units and τ_conf is a
    probability, so putting them on one axis would invite a comparison of
    their heights that means nothing.

    Why this is a figure at all: τ is the threshold every outcome in the
    campaign is classified against, and if it differs markedly between arms
    then an equal amount of real movement is called a flip in one arm and
    nothing in another. That is a property of the instrument worth seeing
    before reading any result.
    """
    languages = [
        language
        for language in LANGUAGE_CONDITIONS
        if language in tau["languages"] and tau["languages"][language].get("tau") is not None
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    panels = (
        (axes[0], "tau", "τ  (x units)", "Movement threshold"),
        (axes[1], "tau_conf", "τ_conf  (probability)", "Confidence threshold"),
    )
    for axis, key, ylabel, title in panels:
        values = [float(tau["languages"][language][key]) for language in languages]
        colors = [color_for(language) for language in languages]
        axis.bar(range(len(languages)), values, 0.62, color=colors,
                 edgecolor=SURFACE, linewidth=2.0)
        for position, value in enumerate(values):
            axis.annotate(
                f"{value:.3f}", (position, value),
                textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=8, color=TEXT_SECONDARY,
            )
        axis.set_xticks(range(len(languages)))
        axis.set_xticklabels(
            [language.replace("_", "\n") for language in languages]
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")

    # Reported when the file carries the counts, omitted when it does not: a
    # τ summarized over an unstated number of trials is worse than one with no
    # number beside it, because the reader will assume there were enough.
    counts = [tau["languages"][language].get("n_ok") for language in languages]
    provenance = (
        f"{tau['percentile']:g}th percentile over {sum(counts)} trials"
        if all(c is not None for c in counts)
        else f"{tau['percentile']:g}th percentile (trial counts not recorded)"
    )
    # Laid out first, then the header placed just above the figure box with
    # va="bottom". Reserving space for it with tight_layout's rect instead
    # left a band of empty surface, because the reserved fraction has to be
    # guessed and a two-line title at this size needs less than looks safe --
    # and savefig(bbox_inches="tight") crops to the text either way.
    fig.tight_layout()
    fig.suptitle(
        "Calibration: how far a reading moves when nothing persuasive happened\n"
        + provenance
        + ("   (SMOKE RUN -- not a calibration)" if tau.get("smoke") else ""),
        x=0.01, y=1.01, ha="left", va="bottom", fontsize=11, fontweight="bold",
    )
    return save(fig, out_dir, "tau")


# --- 4: the instrument ----------------------------------------------------


def figure_instrument(probes: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Probe mass per language, and reply prompt length per language by round.

    Both panels are about the instrument rather than the result, and both are
    things RQ1 has to report rather than diagnose away.

    Probe mass is how much of the next-token distribution landed on the seven
    digits. If one arm answers off-scale more than another, its readings are
    not as trustworthy and its failures are not missing at random.

    Prompt length is the known confound the bank regime reduces and does not
    remove: the same argument in Hinglish tokenizes to more than in English,
    and with full history that gap compounds over a run rather than sitting at
    a fixed window. Reporting the gap is part of the result.
    """
    languages = languages_in(probes)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))

    mass = axes[0]
    # Filtered as pairs, so an arm with no readings drops its box AND its tick
    # label together -- filtering only one would shift every label by one and
    # silently attribute each arm's distribution to its neighbour.
    measured = [
        (language, probes[probes.language == language].probe_mass.dropna().to_numpy())
        for language in languages
    ]
    measured = [(language, values) for language, values in measured if len(values)]
    languages = [language for language, _ in measured]
    parts = mass.boxplot(
        [values for _, values in measured], patch_artist=True, widths=0.55,
        medianprops={"color": TEXT_PRIMARY, "linewidth": 1.4},
        flierprops={"marker": ".", "markersize": 3, "markerfacecolor": TEXT_SECONDARY,
                    "markeredgecolor": "none"},
        whiskerprops={"color": TEXT_SECONDARY}, capprops={"color": TEXT_SECONDARY},
    )
    for patch, language in zip(parts["boxes"], languages, strict=True):
        patch.set_facecolor(color_for(language))
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(2.0)
        patch.set_alpha(0.85)
    mass.set_xticklabels([language.replace("_", "\n") for language in languages])
    mass.set_ylabel("probe mass\n(share of next-token probability on digits 1–7)")
    mass.set_title("Probe quality by language", loc="left")

    tokens = axes[1]
    for language in languages:
        arm = probes[(probes.language == language) & probes.reply_prompt_tokens.notna()]
        # Round 0 makes no reply call, so its logged length is 0 rather than a
        # measurement; plotting it would put a false floor on every line.
        arm = arm[arm.turn > 0]
        if arm.empty:
            continue
        by_turn = arm.groupby("turn").reply_prompt_tokens.mean()
        tokens.plot(
            by_turn.index.to_numpy(), by_turn.to_numpy(), "-o",
            color=color_for(language), label=language,
        )
        last_turn = by_turn.index.max()
        tokens.annotate(
            language.replace("_", " "),
            (last_turn, by_turn.loc[last_turn]),
            textcoords="offset points", xytext=(5, 0),
            fontsize=7.5, color=TEXT_SECONDARY, va="center",
        )
    tokens.set_xlabel("round")
    tokens.set_ylabel("mean reply prompt length\n(tokens, as the provider reported)")
    tokens.set_title("Prompt length by language", loc="left")
    if languages:
        # Room on the right for the direct labels. Without it they are drawn
        # outside the axes and clipped, which is worse than no label at all.
        # The ticks are fixed first, so the padding cannot pull a tick out to
        # a round number past the last round anybody ran.
        turns = sorted(probes[probes.turn > 0].turn.unique())
        tokens.set_xticks(turns)
        tokens.set_xlim(min(turns) - 0.3, max(turns) + 0.18 * max(len(turns), 1))
        # No legend box on this panel: all four series are directly labelled
        # at their right-hand end, which identifies them without the reader
        # matching a swatch, and a legend as well would sit on top of the
        # topmost label.

    fig.tight_layout()
    return save(fig, out_dir, "instrument")


# --- entry point ----------------------------------------------------------


def load_e1_frame(path: Path) -> pd.DataFrame:
    rows = [r for r in load_trials(path) if r["status"] == STATUS_OK]
    if not rows:
        raise ValueError(f"{path} has no ok trials to plot")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, required=True,
                        help="the directory analysis.metrics wrote")
    parser.add_argument("--e1", type=Path, required=True, help="e1/trials.jsonl")
    parser.add_argument("--calibration", type=Path, required=True,
                        help="the directory holding tau.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--runs-dir", type=Path, default=None,
        help="the run directories themselves (not just --results' CSVs); "
             "only needed for the network_snapshots figure, which reads "
             "each run's own graph files. Omit to skip that one figure.",
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()

    rounds = pd.read_csv(args.results / "round_metrics.csv")
    probes = pd.read_csv(args.results / "probes.csv")
    tau = json.loads((args.calibration / "tau.json").read_text(encoding="utf-8"))
    e1 = load_e1_frame(args.e1)

    written: list[Path] = []
    written += figure_trajectories(rounds, args.out)
    written += figure_e1(e1, args.out)
    written += figure_tau(tau, args.out)
    written += figure_instrument(probes, args.out)

    # E5's four figures. Guarded on column presence rather than unconditional:
    # round_metrics.csv/run_metrics.csv written before this feature existed
    # (or by a campaign that never ran E5) have none of these columns, and
    # that is not an error here -- it just means there is nothing yet to draw.
    if "assort_excess" in rounds.columns:
        written += figure_assort_excess(rounds, args.out)
    if "cross_side_edge_share" in rounds.columns:
        written += figure_cross_side_edge_share(rounds, args.out)
    if "nci_excess" in rounds.columns:
        written += figure_nci_excess(rounds, args.out)
    if "disagreement_ratio" in rounds.columns:
        written += figure_disagreement_ratio(rounds, args.out)
    if "minority_share" in rounds.columns:
        written += figure_minority_share(rounds, args.out)
    if "n_side_switches" in rounds.columns:
        written += figure_n_side_switches(rounds, args.out)

    run_metrics_path = args.results / "run_metrics.csv"
    runs = pd.read_csv(run_metrics_path) if run_metrics_path.exists() else None
    if runs is not None and "t_echo" in runs.columns:
        written += figure_t_echo(runs, args.out)

    if runs is not None and args.runs_dir is not None:
        # One snapshot figure per echo-chamber experiment actually present,
        # rather than always overwriting a single "network_snapshots" file
        # with whichever of e5/e6/e7 happened to run last.
        for experiment in ("e5", "e6", "e7"):
            if not (runs.experiment == experiment).any():
                continue
            run_graphs = load_run_graphs(args.runs_dir, runs, experiment=experiment)
            if not run_graphs:
                continue
            stem = (
                "network_snapshots" if experiment == "e5"
                else f"network_snapshots_{experiment}"
            )
            written += figure_network_snapshots(
                probes, run_graphs, args.out, stem=stem,
                experiment_label=ECHO_EXPERIMENT_LABELS.get(experiment, experiment.upper()),
            )

    print(f"\n{args.out}/")
    for path in written:
        print(f"  {path.name}  {path.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
