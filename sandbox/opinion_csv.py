"""Writes initial_stance.csv for the opinion-CSV experiment.

A widening-columns CSV (3-4 new columns appended per round) can't be done
as a pure JSONL-style append the way logging_writer.py's other files are --
each round rewrites the whole file via pandas, the same library
export_csv.py already uses elsewhere in this repo. Fine at this pipeline's
scale (one pool, ~100 agents, a handful of rounds).

The orchestrator keeps its own in-memory OpinionAgentState per agent as the
live state it reasons from each round; this module is the durable on-disk
record, written after the fact, not read back as state.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INITIAL_COLUMNS = [
    "agent_id",
    "language",
    "persona",
    "initial_belief",
    "initial_opinion",
    "initial_confidence",
]


def write_initial_rows(path: str | Path, rows: list[dict]) -> None:
    """rows: one dict per agent, keys matching INITIAL_COLUMNS exactly."""
    path = Path(path)
    for row in rows:
        missing = set(INITIAL_COLUMNS) - set(row)
        if missing:
            raise ValueError(f"row for {row.get('agent_id')!r} missing columns: {missing}")
    df = pd.DataFrame(rows, columns=INITIAL_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def append_round_columns(
    path: str | Path,
    round_num: int,
    updates: dict[str, tuple[str, float, int, str]],
) -> None:
    """updates: agent_id -> (opinion_text, belief, confidence, status).

    status is "success" or "no_feed" (Fix-9-style: no eligible neighbour this
    round, so the agent's previous opinion/belief/confidence was copied
    forward rather than re-generated).

    Every agent_id already present in the CSV must appear in updates, and
    vice versa -- a partial round would leave some agents' columns as NaN
    with no record of why, which this raises on rather than writing.
    """
    path = Path(path)
    df = pd.read_csv(path)

    existing_ids = set(df["agent_id"])
    update_ids = set(updates)
    if existing_ids != update_ids:
        raise ValueError(
            f"round {round_num}: updates do not match the CSV's agents -- "
            f"missing={sorted(existing_ids - update_ids)} "
            f"extra={sorted(update_ids - existing_ids)}"
        )

    opinion_col = f"round_{round_num}_opinion"
    belief_col = f"round_{round_num}_belief"
    confidence_col = f"round_{round_num}_confidence"
    status_col = f"round_{round_num}_status"

    df[opinion_col] = df["agent_id"].map(lambda aid: updates[aid][0])
    df[belief_col] = df["agent_id"].map(lambda aid: updates[aid][1])
    df[confidence_col] = df["agent_id"].map(lambda aid: updates[aid][2])
    df[status_col] = df["agent_id"].map(lambda aid: updates[aid][3])

    df.to_csv(path, index=False)
