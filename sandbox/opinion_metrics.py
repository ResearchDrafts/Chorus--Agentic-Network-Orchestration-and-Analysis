"""Per-round Pz/DG/NCI/mean-belief-shift for the opinion-CSV experiment.

Reuses analysis/metrics.py's global_disagreement/nci/population_variance
rather than reimplementing Jo et al.'s formulas. Those three all take `x` on
the codebase's [-1, 1] scale and convert to v = 2x internally; this
pipeline's belief is already on the v scale ([-2, 2]), so the adapter is
x = belief / 2.0. mean_belief_shift needs no adapter at all: algebraically
shift_v = 2*(x - x0) = belief - belief0, so it is computed directly from the
raw belief values.
"""

from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx

from analysis.metrics import global_disagreement, nci, population_variance

ROUND_METRICS_COLUMNS = [
    "round",
    "polarization_index",
    "global_disagreement",
    "nci",
    "mean_belief_shift",
    "mean_abs_individual_shift",
]


def build_graph(agent_ids: list[str], edges: list[tuple[str, str]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(agent_ids)
    graph.add_edges_from(edges)
    return graph


def compute_round_metrics(
    graph: nx.Graph,
    beliefs: dict[str, float],
    beliefs_round0: dict[str, float],
) -> dict:
    x_by_agent = {aid: belief / 2.0 for aid, belief in beliefs.items()}

    polarization_index = 4.0 * population_variance(list(x_by_agent.values()))
    dg = global_disagreement(graph, x_by_agent)
    nci_value = nci(graph, x_by_agent)

    shifts = [
        beliefs[aid] - beliefs_round0[aid]
        for aid in beliefs
        if aid in beliefs_round0
    ]
    mean_belief_shift = sum(shifts) / len(shifts) if shifts else None
    mean_abs_individual_shift = (
        sum(abs(s) for s in shifts) / len(shifts) if shifts else None
    )

    return {
        "polarization_index": polarization_index,
        "global_disagreement": dg,
        "nci": nci_value,
        "mean_belief_shift": mean_belief_shift,
        "mean_abs_individual_shift": mean_abs_individual_shift,
    }


def write_round_metrics_row(path: str | Path, round_num: int, metrics: dict) -> None:
    """Appends one row. Writes the header only if the file does not exist
    yet -- the same pattern logging_writer.write_graph_edges_round uses for
    its own per-round CSV."""
    path = Path(path)
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROUND_METRICS_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({"round": round_num, **metrics})
