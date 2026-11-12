#!/usr/bin/env python3
"""Flatten a campaign's runs into analysis-ready CSVs.

    python export_csv.py outputs/chorus_run1
    python export_csv.py outputs/chorus_run1 --out-dir analysis --keep-duplicates

Produces three files:

  interactions.csv  one row per agent-turn, with run-level config joined on.
                    The main dataset. Bank-regime runs additionally carry
                    probe_mean, x (the stance rescaled to [-1, 1] about the
                    midpoint), probe_mass, side_conf, items_read,
                    n_items_read, posted_item_id, shadow_items,
                    is_round_zero, reply_seed, the two per-call prompt
                    lengths and the three exposure counts
                    (n_repeat_reads, max_prior_exposures,
                    n_distinct_items_read_so_far); these are present but
                    empty for generated-regime rows, so a campaign mixing
                    both regimes still exports as one table.
  agents.csv        one row per agent per run: final stance, total movement,
                    and how far it drifted from where it started. For a bank
                    run, initial_stance is the round-0 probe reading: the
                    round-0 row's own stance_before is null (see
                    is_round_zero below) and pandas' groupby.first() skips
                    it, landing on round 1's stance_before, which IS that
                    reading.
  runs.csv          one row per run: the config, plus summary statistics you
                    would otherwise recompute every time.

**Round 0.** A bank run probes every agent once before anything is read. That
row's stance_after is a measurement, but its stance_before is the configured
initial_stance rather than a reading, so stance_before and every shift column
derived from it are null there and `is_round_zero` marks the row. Without it,
a population seeded at 4 that reads 5 at round 0 contributes M rows of "+1
movement" before any agent has read a word.

**Deduplication.** A run interrupted mid-turn can log the same
(turn, speaker_agent_id) twice: the orchestrator writes interactions one at a
time and checkpoints only after the whole turn, so a crash part-way through
leaves rows on disk with no checkpoint, and the resumed pass re-runs that turn
and appends again. The later row is the authoritative one, verified against
agents_final.jsonl, whose stance_history references the post-resume
interaction_id in every observed case. Earlier copies are orphans: their
effect on agent state was lost with the un-checkpointed memory.

Duplicates are dropped by default and reported. Pass --keep-duplicates to
inspect them, with an `is_duplicate` column marking the ones that would be
removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Joined onto every interaction row so the CSV is groupable without a second file.
# Set by _derive_exposure_columns, and listed here so the three are
# declared in one place: a bank-free campaign still exports all three, empty.
EXPOSURE_COLUMNS = (
    "n_repeat_reads",
    "max_prior_exposures",
    "n_distinct_items_read_so_far",
)

RUN_FIELDS = (
    "run_id", "topic", "language_condition", "trial_number", "seed", "alpha",
    "M", "N", "K", "rq_target", "model_backend_id", "stance_low_label",
    "stance_high_label", "status", "git_commit_hash",
    # RQ1's design dimensions. Every one of these is something an analysis
    # groups or filters by, and joining them on here is what lets a whole
    # campaign be read from interactions.csv alone: without `delivery`,
    # an E4 control row reads like a treatment row whose single post
    # happened not to move anybody.
    "regime", "delivery", "neighbor_strategy", "graph_degree", "graph_rewire_p",
    "bank_path", "probe_mass_floor", "memory_rounds",
)


def load_run(run_dir: Path) -> pd.DataFrame | None:
    """One run's interactions with its config joined on, or None if unusable."""
    cfg_path, rows_path = run_dir / "run_config.json", run_dir / "interactions.jsonl"
    if not cfg_path.exists() or not rows_path.exists():
        return None

    cfg = json.loads(cfg_path.read_text())
    df = pd.read_json(rows_path, lines=True)
    if df.empty:
        return None

    for field in RUN_FIELDS:
        df[field] = cfg.get(field)

    # topic_key is the grouping dimension: run_ids look like politics_english_t1.
    df["topic_key"] = run_dir.name.split("_")[0]
    df["meme_injection_enabled"] = cfg.get("meme_injection", {}).get("enabled", False)

    # Row order within a run is write order, which is what makes "keep last"
    # mean "keep the post-resume copy".
    df["row_index"] = range(len(df))
    return df


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flags all but the final write of each (run_id, turn, speaker_agent_id)."""
    keys = ["run_id", "turn", "speaker_agent_id"]
    df = df.sort_values(["run_id", "row_index"])
    df["is_duplicate"] = df.duplicated(subset=keys, keep="last")
    return df


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Columns every analysis recomputes, added once here."""
    df["stance_shift"] = df.stance_after - df.stance_before
    df["abs_stance_shift"] = df.stance_shift.abs()
    df["stance_moved"] = df.stance_shift != 0
    df = _null_out_round_zero_shifts(df)
    df["is_valid_for_analysis"] = df.api_call_status != "failed_logged_null"
    df["n_neighbors"] = df.neighbor_agent_ids.map(len)
    # Lists do not survive CSV round-trips as lists; make it explicit and
    # re-splittable rather than letting str() of a Python list leak through.
    df["neighbor_agent_ids"] = df.neighbor_agent_ids.map(lambda v: "|".join(v))
    df["reason_char_count"] = df.reason_text.fillna("").str.len()
    df["reason_word_count"] = df.reason_text.fillna("").str.split().map(len)
    return _derive_bank_columns(df)


def _null_out_round_zero_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """Round 0 has no "before", so it gets no shift.

    A bank run's round-0 row is the starting measurement: the probe is asked
    before the agent has read anything. Its stance_after is that reading, but
    its stance_before is the agent's configured initial_stance, which is a
    value the config chose and not something the model reported -- the whole
    premise of the probe being that what an agent answers is not what it was
    told it believes. stance_shift on that row is therefore the gap between a
    configuration and a measurement, in the same column where every other row
    carries the movement between two measurements.

    Left in, it biased every aggregate that touches the column: a population
    seeded at 4 and reading 5 at round 0 shows M rows of +1 "movement" before
    anybody has read a word, and mean_abs_shift, turns_moved and
    polarization_change all absorb it. So stance_before and the three columns
    derived from it are null on those rows, and is_round_zero marks them --
    explicitly, because a null is also what a pre-bank dataset's missing
    column looks like, and an analysis filtering round 0 out should be able to
    say so rather than infer it from a turn number.

    Generated-regime runs never write turn 0, so nothing there is touched.
    """
    is_round_zero = (_column(df, "regime", object) == "bank") & (df.turn == 0)
    df["is_round_zero"] = is_round_zero
    if not is_round_zero.any():
        return df
    for column in ("stance_before", "stance_shift", "abs_stance_shift"):
        df[column] = df[column].mask(is_round_zero)
    # astype("boolean") first: masking a numpy bool column would upcast it to
    # object and put a float NaN in a column of True/False.
    df["stance_moved"] = df.stance_moved.astype("boolean").mask(is_round_zero)
    return df


def _derive_bank_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The RQ1 bank-regime columns.

    Added for every run, not only bank ones, so one campaign directory mixing
    both regimes still produces a single rectangular CSV -- a generated run
    simply has them empty. The columns are read with df.get() because runs
    produced before the bank regime existed have no such keys in their JSONL
    at all, and those datasets (outputs/chorus_run1) must keep exporting.
    """
    probe_mean = _column(df, "probe_mean", float)

    # x is the stance on a signed [-1, 1] scale centred on the 7-point
    # midpoint: -1 is the con pole, 0 undecided, +1 the pro pole. Every RQ1
    # model is specified in these units, and computing it in each analysis
    # instead would mean four chances to get the sign or the divisor wrong.
    df["x"] = (probe_mean - 4) / 3

    # Read once, before the column is overwritten below: counting from the
    # joined string would count zero for every row, since the strings are no
    # longer lists by then.
    items = _column(df, "items_read", object)
    df["n_items_read"] = items.map(lambda v: len(v) if isinstance(v, list) else 0)
    # Semicolon-joined, not pipe-joined like neighbor_agent_ids: that column
    # is already pipe-joined and a reader splitting the wrong column on the
    # wrong separator gets a silently wrong answer rather than an error.
    df["items_read"] = items.map(_join_items)
    # What a control row's neutral messages stood in for, in the same order,
    # semicolon-joined like items_read. Empty on a delivery-on row and at
    # round 0. This is what lines a control row up against the treatment row
    # it is the baseline for.
    df["shadow_items"] = _column(df, "shadow_items", object).map(_join_items)
    df["posted_item_id"] = _column(df, "posted_item_id", object)
    # How much of the next-token distribution actually landed on the scale.
    # A quality measure, not a stance: a mean of 5.0 read off 0.95 mass and
    # the same mean off 0.51 are not equally trustworthy, and filtering on
    # this is how an analysis excludes the latter.
    df["probe_mass"] = _column(df, "probe_mass", float)
    # How decided the agent is, as distinct from where it stands.
    df["side_conf"] = _column(df, "side_conf", float)
    df["probe_mean"] = probe_mean
    # The two calls' prompt lengths, as well as their sum in
    # prompt_token_count. RQ1 reports prompt length per language, and a reply
    # prompt and a probe prompt grow for different reasons, so the sum alone
    # cannot answer the question. Read as float rather than int because a
    # pre-bank dataset has no such column and lands here as NaN, which an
    # int column cannot hold.
    df["reply_prompt_tokens"] = _column(df, "reply_prompt_tokens", float)
    df["probe_prompt_tokens"] = _column(df, "probe_prompt_tokens", float)
    # The seed the reply call carried. Float for the same reason the two
    # above are: a pre-seed dataset has no such column and lands here as NaN,
    # which an int column cannot hold. Empty at round 0 and on a generated
    # row, neither of which sends a seed.
    df["reply_seed"] = _column(df, "reply_seed", float)
    return _derive_exposure_columns(df, items)


def _derive_exposure_columns(df: pd.DataFrame, items: pd.Series) -> pd.DataFrame:
    """How often an agent had already seen what it is reading now.

    With full history an agent sees some bank items more than once, and how
    often is not something an analysis can recover from the exported strings
    without replaying the whole log per agent. Three columns, per row:

      n_repeat_reads               of this row's items, how many this agent
                                   had read in an EARLIER round
      max_prior_exposures          over this row's items, the largest number
                                   of earlier reads of that item
      n_distinct_items_read_so_far distinct items this agent has read up to
                                   and including this round

    Both of the first two are counted against the state before this round, so
    an item delivered twice within one round is not a repeat of itself. It
    does count twice toward later rounds' totals: the agent read it twice,
    and the column says how many times an argument has been in front of it.

    Four rules about which rows participate:

    * Only bank-regime rows. A generated row has no items_read to count, and
      a 0 there would claim a measurement that was never made, so it is null.
    * Only rows valid for analysis. A failed round was logged and never
      applied -- the agent's memory does not hold it -- so counting it would
      put the exposure history out of step with the prompt the agent actually
      saw. Failed rows themselves are null for the same reason the stance
      columns are: nothing about them was measured.
    * Only the surviving write of a re-logged round, since a resumed run logs
      that round twice and the second copy is the one the run used. Superseded
      rows are null; they are dropped from the export by default anyway. This
      is why derive() has to run after mark_duplicates(), and why a frame
      without is_duplicate raises here rather than being assumed clean.
    * Round 0 reads nothing, so it is 0, 0, 0 rather than null. It genuinely
      read nothing, which is a different fact from not being measured.

    Under `delivery: off` these count neutral messages, since those are what
    the agent read; shadow_items carries the bank items they stood in for.
    """
    is_bank = _column(df, "regime", object) == "bank"
    if not is_bank.any():
        for name in EXPOSURE_COLUMNS:
            df[name] = _column(df, name, float)
        return df

    if "is_duplicate" not in df:
        # Raised rather than assumed. Treating a frame with no such column as
        # duplicate-free would silently double-count a resumed run's re-logged
        # round, which is the one case these columns most need to get right.
        raise ValueError(
            "exposure counts need is_duplicate: call mark_duplicates() before "
            "derive(), as main() does"
        )
    eligible = is_bank & df.is_valid_for_analysis & ~df.is_duplicate.astype(bool)

    counts = {name: pd.Series(float("nan"), index=df.index) for name in EXPOSURE_COLUMNS}
    ordered = df[eligible].sort_values(["run_id", "speaker_agent_id", "turn"])
    for _, group in ordered.groupby(["run_id", "speaker_agent_id"], sort=False):
        seen: dict[str, int] = {}
        for index in group.index:
            row_items = items.loc[index]
            row_items = row_items if isinstance(row_items, list) else []
            counts["n_repeat_reads"][index] = sum(
                1 for item in row_items if seen.get(item, 0) > 0
            )
            counts["max_prior_exposures"][index] = max(
                (seen.get(item, 0) for item in row_items), default=0
            )
            for item in row_items:
                seen[item] = seen.get(item, 0) + 1
            counts["n_distinct_items_read_so_far"][index] = len(seen)

    for name, series in counts.items():
        df[name] = series
    return df


def _column(df: pd.DataFrame, name: str, dtype) -> pd.Series:
    """The column, or an all-null one of the right dtype if it is absent."""
    if name in df.columns:
        return df[name]
    if dtype is object:
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    return pd.Series([float("nan")] * len(df), index=df.index, dtype=float)


def _join_items(value) -> str:
    """Semicolon-joined item IDs, empty for a row that read nothing.

    An empty string is the round-0 probe, the only row where an agent
    genuinely read nothing -- as distinct from a missing column, which is a
    pre-bank dataset and also lands here as empty. A `delivery: off` row
    carries N neutral-bank IDs, all prefixed "neutral_", since the control
    reads one side-free message for each bank item it would have been
    delivered; shadow_items says which items those were. n_items_read carries
    the count so an analysis does not have to split the string to know.
    """
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return ""


def agents_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per agent per run: trajectory summary."""
    valid = df[df.is_valid_for_analysis]
    first = valid.sort_values("turn").groupby(["run_id", "speaker_agent_id"]).first()
    last = valid.sort_values("turn").groupby(["run_id", "speaker_agent_id"]).last()
    agg = valid.groupby(["run_id", "speaker_agent_id"]).agg(
        topic_key=("topic_key", "first"),
        language_condition=("language_condition", "first"),
        turns_completed=("turn", "nunique"),
        total_abs_movement=("abs_stance_shift", "sum"),
        mean_abs_shift=("abs_stance_shift", "mean"),
        turns_moved=("stance_moved", "sum"),
        mean_prompt_tokens=("prompt_token_count", "mean"),
        mean_completion_tokens=("completion_token_count", "mean"),
    )
    agg["initial_stance"] = first.stance_before
    agg["final_stance"] = last.stance_after
    # Net drift, as distinct from total movement: an agent can move a lot and
    # end where it began.
    agg["net_drift"] = agg.final_stance - agg.initial_stance
    return agg.reset_index()


def runs_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run: config plus the summaries worth precomputing."""
    valid = df[df.is_valid_for_analysis]
    final_turn = valid[valid.turn == valid.groupby("run_id").turn.transform("max")]

    runs = df.groupby("run_id").agg(
        topic_key=("topic_key", "first"),
        topic=("topic", "first"),
        language_condition=("language_condition", "first"),
        trial_number=("trial_number", "first"),
        seed=("seed", "first"),
        alpha=("alpha", "first"),
        M=("M", "first"), N=("N", "first"), K=("K", "first"),
        model_backend_id=("model_backend_id", "first"),
        status=("status", "first"),
        n_interactions=("interaction_id", "count"),
        n_failed=("is_valid_for_analysis", lambda s: (~s).sum()),
        mean_latency_ms=("latency_ms", "mean"),
        mean_prompt_tokens=("prompt_token_count", "mean"),
    )
    runs["parse_failure_rate"] = runs.n_failed / runs.n_interactions
    runs["mean_abs_shift"] = valid.groupby("run_id").abs_stance_shift.mean()
    # Spread of final stances: the headline polarization measure. Higher means
    # agents ended further apart.
    runs["final_stance_std"] = final_turn.groupby("run_id").stance_after.std()
    runs["final_stance_mean"] = final_turn.groupby("run_id").stance_after.mean()
    runs["initial_stance_std"] = (
        valid[valid.turn == 1].groupby("run_id").stance_before.std()
    )
    runs["polarization_change"] = runs.final_stance_std - runs.initial_stance_std
    return runs.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("campaign_dir", type=Path, help="directory holding one subdir per run")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the CSVs (default: alongside the runs)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep re-logged rows, marked with is_duplicate")
    ap.add_argument("--exclude", nargs="*", default=["colab_smoke"],
                    help="run_ids to skip (default: colab_smoke)")
    args = ap.parse_args()

    if not args.campaign_dir.is_dir():
        print(f"not a directory: {args.campaign_dir}", file=sys.stderr)
        return 1

    frames, skipped = [], []
    for d in sorted(p for p in args.campaign_dir.iterdir() if p.is_dir()):
        if d.name in args.exclude:
            skipped.append(f"{d.name} (excluded)")
            continue
        got = load_run(d)
        if got is None:
            skipped.append(f"{d.name} (no usable output)")
        else:
            frames.append(got)

    if not frames:
        print("no runs loaded", file=sys.stderr)
        return 1

    df = derive(mark_duplicates(pd.concat(frames, ignore_index=True)))

    n_dupes = int(df.is_duplicate.sum())
    if n_dupes and not args.keep_duplicates:
        affected = sorted(df[df.is_duplicate].run_id.unique())
        df = df[~df.is_duplicate].drop(columns=["is_duplicate"])
        print(f"dropped {n_dupes} re-logged rows from interrupted runs: {', '.join(affected)}")
        print("  (a run that crashed mid-turn re-logs that turn on resume; the later")
        print("   write is authoritative, per agents_final.jsonl's stance_history)")
    elif n_dupes:
        print(f"kept {n_dupes} re-logged rows, marked in is_duplicate")

    out = args.out_dir or args.campaign_dir
    out.mkdir(parents=True, exist_ok=True)
    df = df.drop(columns=["row_index"]).sort_values(["run_id", "turn", "speaker_agent_id"])

    for name, table in (("interactions", df),
                        ("agents", agents_table(df)),
                        ("runs", runs_table(df))):
        path = out / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"  {path}  {len(table):>5} rows x {len(table.columns):>2} cols "
              f"({path.stat().st_size/1e6:.1f} MB)")

    for s in skipped:
        print(f"  skipped {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
