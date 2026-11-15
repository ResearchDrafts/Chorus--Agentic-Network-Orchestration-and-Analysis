#!/usr/bin/env python3
"""calibrate_starts.py: what probe reading a starter-post MIX produces.

    python scripts/calibrate_starts.py --runs-dir runs --sample 200
    python scripts/calibrate_starts.py --runs-dir runs --suggest 1.5,2.5,3.5,4.5,5.5,6.5

Fix 3's motivating problem: under lean_assignment="balanced", lean 4 (the
scale's own midpoint) is read back by the probe at about 4.1 and so posts
"pro", which on a 20-agent, 7-point population tips the count to 9-vs-11
before round 0. A live pilot found the deeper problem: bank_manager's
STARTER_COUNTS gives leans {1,2,3} a weaker starter composition than
{5,6,7}'s mirror, so the con camp starts more persuadable and gets absorbed
before an echo chamber can be measured at all.

This script does not test a *lean*. It tests a *starter-post mix*: for each
candidate (n_con, n_pro) from (0, 0) to (4, 4), a sample of personas in
every language is given exactly that many con/pro starter posts (via
ExperimentRun.starter_counts) and probed once, cold -- no item delivered, no
reply, nothing else in its history (SingleReaderHarness.probe at
TURN_BASELINE, the same call calibrate_tau.py uses for P0). The reading a
mix produces is a property of the mix and the model, independent of which
nominal lean a run's starter_counts would otherwise map it from.

Writes, under `--runs-dir`:

    calibration/starts_trials.jsonl   one row per (mix, language, persona)
    calibration/starts.csv            one row per mix: pooled mean/sd and
                                       each language's own mean

`--suggest TARGETS` (default "1.5,2.5,3.5,4.5,5.5,6.5", one per lean in
{1,2,3,5,6,7} -- lean 4 keeps STARTER_COUNTS[4] = (1, 1), unchanged) prints,
for each lean, the mix whose pooled mean is closest to its target and the
largest gap between any two languages' means for that mix, and writes a
ready-to-paste `starter_counts` YAML block to
`calibration/starter_counts.yaml`.

The (0, 0) row is also the topic check Fix 3 asks for: a reader given no
starter posts at all who still answers far from 4 means the model carries a
prior on this topic that no starter-post mix can correct.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python scripts/calibrate_starts.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sandbox.bank_manager import BankManager
from sandbox.models import ExperimentRun
from sandbox.single_exposure import (
    DEFAULT_CONFIG_DIR,
    LANGUAGES,
    STATUS_FAILED,
    STATUS_OK,
    TURN_BASELINE,
    Reading,
    SingleReaderHarness,
    TrialStore,
    apply_sample,
    chunked,
    gather_all,
    load_base_configs,
    load_trials,
    personas_for,
)

OUTPUT_SUBDIR = "calibration"

# (0, 0) .. (4, 4): every combination, con first. 4 is one more than
# STARTER_COUNTS's own maximum (3), so the sweep can tell whether a stronger
# mix than the module default moves the reading further, not only whether
# the module default is well-calibrated.
MAX_PER_SIDE = 4
MIXES: list[tuple[int, int]] = [
    (n_con, n_pro) for n_con in range(MAX_PER_SIDE + 1) for n_pro in range(MAX_PER_SIDE + 1)
]

# The six leans a default --suggest covers, in the order its six default
# targets name them. Lean 4 is deliberately absent: it is the scale's own
# midpoint, STARTER_COUNTS already gives it (1, 1) -- one post each side --
# and no target for it would even be well-posed (closest to its OWN starting
# point 4.0 is a question a mix search cannot usefully answer).
SUGGEST_LEANS: tuple[int, ...] = (1, 2, 3, 5, 6, 7)
DEFAULT_TARGETS: tuple[float, ...] = (1.5, 2.5, 3.5, 4.5, 5.5, 6.5)

# What a run that never calibrated starts anything still needs for lean 4,
# so a suggestion block is a COMPLETE starter_counts mapping -- pasteable as
# is, per ExperimentRun's own requirement that every assignable lean has an
# entry.
MIDPOINT_LEAN = 4
MIDPOINT_COUNTS: tuple[int, int] = (1, 1)

# How far the (0, 0) row's pooled mean may sit from the scale's own midpoint
# before the model is judged to be carrying a prior on the topic itself.
TOPIC_PRIOR_WARNING_THRESHOLD = 0.5


def mix_key(language: str, n_con: int, n_pro: int, persona_index: int) -> str:
    return f"start|{language}|{n_con}-{n_pro}|p{persona_index:03d}"


def mix_start_key(n_con: int, n_pro: int, persona_index: int) -> str:
    """Language-free: the same mix and persona draw the same starter items
    in every arm, so the four languages differ only in rendering."""
    return f"start|{n_con}-{n_pro}|p{persona_index:03d}"


def with_starter_counts(run: ExperimentRun, n_con: int, n_pro: int) -> ExperimentRun:
    """`run`, with start_mode="posts" and every assignable lean mapped to
    the same (n_con, n_pro) mix.

    Every point gets the SAME mix rather than just the one lean this script
    actually builds readers at: only MIDPOINT_LEAN is ever used as a
    reader's `initial_stance` here (the digit itself never reaches the
    agent, so which constant names it does not matter), but
    ExperimentRun.check_starter_counts requires an entry for every
    stance_scale point regardless of which ones a particular caller reads.
    Re-validated through the model, the same pattern with_start_mode uses,
    so a mix the starter bank cannot support is caught by the pre-flight
    check below rather than by a confusing mid-sweep crash.
    """
    data = run.model_dump()
    data["start_mode"] = "posts"
    points = {int(point) for point in run.stance_scale if point == int(point)}
    data["starter_counts"] = {point: (n_con, n_pro) for point in points}
    return ExperimentRun.model_validate(data)


def check_starter_bank_covers_the_sweep(run: ExperimentRun) -> None:
    """Fails once, before any call, naming the shortfall -- rather than
    failing on whichever trial happens to hit the largest mix first."""
    starters = BankManager(run.starter_bank_path)
    for side in ("con", "pro"):
        available = len(starters.items_for_side(side))
        if available < MAX_PER_SIDE:
            raise ValueError(
                f"starter bank {run.starter_bank_path} has {available} {side} "
                f"item(s); this sweep needs at least {MAX_PER_SIDE} to try "
                f"every mix up to ({MAX_PER_SIDE}, {MAX_PER_SIDE})"
            )


async def measure_mix(
    harness: SingleReaderHarness,
    out_dir: Path,
    personas: list[str],
    n_con: int,
    n_pro: int,
    persona_indices: list[int],
    store: TrialStore,
) -> None:
    """Probes every (persona, this mix) not already in `store`, and appends
    one row per trial -- a failure included, never a dropped one."""
    done = store.completed()
    pending = [
        index
        for index in persona_indices
        if mix_key(harness.language, n_con, n_pro, index) not in done
    ]
    if not pending:
        return
    run = with_starter_counts(harness.run, n_con, n_pro)
    mix_harness = SingleReaderHarness(run, out_dir, gateway=harness.gateway)

    async def one(index: int) -> dict:
        key = mix_key(harness.language, n_con, n_pro, index)
        agent = mix_harness.reader(
            key, personas[index], MIDPOINT_LEAN,
            start_key=mix_start_key(n_con, n_pro, index),
        )
        reading: Reading | None = await mix_harness.probe(agent, [], turn=TURN_BASELINE)
        return {
            "key": key,
            "language": harness.language,
            "n_con": n_con,
            "n_pro": n_pro,
            "persona_index": index,
            "mean": reading.mean if reading else None,
            "x": reading.x if reading else None,
            "conf": reading.conf if reading else None,
            "mass": reading.mass if reading else None,
            "status": STATUS_OK if reading else STATUS_FAILED,
        }

    for group in chunked(pending, harness.concurrency):
        rows = await gather_all(one(index) for index in group)
        for row in rows:
            store.append(row)


def summarize(rows: list[dict], languages: list[str]) -> list[dict]:
    """One row per mix: the pooled mean/sd over every language, plus each
    language's own mean -- so a per-language gap is visible without a
    second pass over trials.jsonl."""
    summary = []
    for n_con, n_pro in MIXES:
        ok = [
            r for r in rows
            if r["n_con"] == n_con and r["n_pro"] == n_pro and r["status"] == STATUS_OK
        ]
        row = {
            "n_con": n_con, "n_pro": n_pro,
            "n_ok": len(ok),
            "n_failed": sum(
                1 for r in rows
                if r["n_con"] == n_con and r["n_pro"] == n_pro
            ) - len(ok),
            "pooled_mean": None, "pooled_sd": None,
        }
        if ok:
            means = [r["mean"] for r in ok]
            pooled_mean = sum(means) / len(means)
            variance = sum((m - pooled_mean) ** 2 for m in means) / len(means)
            row["pooled_mean"] = pooled_mean
            row["pooled_sd"] = variance ** 0.5
        for language in languages:
            arm = [r["mean"] for r in ok if r["language"] == language]
            row[f"mean_{language}"] = sum(arm) / len(arm) if arm else None
        summary.append(row)
    return summary


STARTS_COLUMNS = (
    "n_con", "n_pro", "n_ok", "n_failed", "pooled_mean", "pooled_sd",
    *(f"mean_{language}" for language in LANGUAGES),
)


def write_starts_csv(path: Path, summary: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(STARTS_COLUMNS))
        writer.writeheader()
        writer.writerows({col: row.get(col) for col in STARTS_COLUMNS} for row in summary)


def suggest(
    summary: list[dict], targets: list[float], languages: list[str]
) -> list[dict]:
    """For each (lean, target), the mix whose pooled_mean is closest to it,
    and the largest gap between any two languages' means for that mix."""
    if len(targets) != len(SUGGEST_LEANS):
        raise ValueError(
            f"--suggest needs exactly {len(SUGGEST_LEANS)} targets, one per "
            f"lean in {SUGGEST_LEANS}; got {len(targets)}: {targets}"
        )
    scored = [row for row in summary if row["pooled_mean"] is not None]
    if not scored:
        raise ValueError("no mix has a pooled_mean to suggest from -- run the sweep first")
    results = []
    for lean, target in zip(SUGGEST_LEANS, targets, strict=True):
        best = min(scored, key=lambda row: abs(row["pooled_mean"] - target))
        lang_means = [best[f"mean_{language}"] for language in languages]
        lang_means = [m for m in lang_means if m is not None]
        gap = max(lang_means) - min(lang_means) if len(lang_means) >= 2 else None
        results.append({
            "lean": lean, "target": target,
            "n_con": best["n_con"], "n_pro": best["n_pro"],
            "pooled_mean": best["pooled_mean"], "language_gap": gap,
        })
    return results


def write_starter_counts_yaml(path: Path, suggestions: list[dict]) -> str:
    """A complete starter_counts block -- every lean 1..7 -- ready to paste
    into a config. Lean 4 is pinned to MIDPOINT_COUNTS, the module default,
    since no --suggest target names it."""
    by_lean = {row["lean"]: (row["n_con"], row["n_pro"]) for row in suggestions}
    by_lean[MIDPOINT_LEAN] = MIDPOINT_COUNTS
    lines = ["starter_counts:"]
    for lean in sorted(by_lean):
        n_con, n_pro = by_lean[lean]
        lines.append(f"  {lean}: [{n_con}, {n_pro}]")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


async def main_async(args: argparse.Namespace) -> int:
    out_dir = args.runs_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    store = TrialStore(out_dir / "starts_trials.jsonl")

    runs = load_base_configs(args.languages, args.configs)
    print(f"\nstarter-mix calibration -> {out_dir}/")
    print(f"  mixes   {len(MIXES)} ((n_con, n_pro) from (0,0) to "
          f"({MAX_PER_SIDE},{MAX_PER_SIDE}))")

    for language in args.languages:
        run = runs[language]
        check_starter_bank_covers_the_sweep(
            with_starter_counts(run, MAX_PER_SIDE, MAX_PER_SIDE)
        )

    for language in args.languages:
        run = runs[language]
        harness = SingleReaderHarness(run, out_dir)
        personas = personas_for(run)
        indices = apply_sample(list(range(len(personas))), args.sample)
        print(f"  {language:<16} {len(indices)} persona(s) x {len(MIXES)} mixes")
        for n_con, n_pro in MIXES:
            await measure_mix(harness, out_dir, personas, n_con, n_pro, indices, store)

    rows = load_trials(store.path)
    summary = summarize(rows, args.languages)
    starts_path = out_dir / "starts.csv"
    write_starts_csv(starts_path, summary)
    print(f"\n  {starts_path}")

    zero_zero = next((r for r in summary if (r["n_con"], r["n_pro"]) == (0, 0)), None)
    if zero_zero and zero_zero["pooled_mean"] is not None:
        drift = abs(zero_zero["pooled_mean"] - 4.0)
        print(f"    (0, 0) pooled mean = {zero_zero['pooled_mean']:.3f} "
              f"(topic check: |mean - 4| = {drift:.3f})")
        if drift > TOPIC_PRIOR_WARNING_THRESHOLD:
            print(
                f"    WARNING: a reader given NO starter posts still answers "
                f"{drift:.3f} scale points from the midpoint. The model "
                "carries a prior on this topic that no starter-post mix can "
                "correct."
            )

    if args.suggest:
        targets = [float(t) for t in args.suggest.split(",")]
        suggestions = suggest(summary, targets, args.languages)
        print(f"\n  suggestions (closest mix per lean target):")
        for row in suggestions:
            gap = f"{row['language_gap']:.3f}" if row["language_gap"] is not None else "n/a"
            print(
                f"    lean {row['lean']}  target {row['target']:.2f}  "
                f"-> ({row['n_con']}, {row['n_pro']})  "
                f"pooled_mean={row['pooled_mean']:.3f}  "
                f"max language gap={gap}"
            )
        yaml_path = out_dir / "starter_counts.yaml"
        text = write_starter_counts_yaml(yaml_path, suggestions)
        print(f"\n  {yaml_path}\n")
        print(text)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--languages", nargs="+", default=list(LANGUAGES), choices=list(LANGUAGES),
        help="which arms to run (default: all four)",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--configs", type=Path, default=DEFAULT_CONFIG_DIR,
        help=f"directory holding single_<language>.yaml (default: {DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="probe a fixed random sample of N personas per language instead "
             "of the whole pool",
    )
    parser.add_argument(
        "--suggest", type=str, default=None, metavar="TARGETS",
        help="comma-separated probe targets, one per lean in "
             f"{SUGGEST_LEANS} (e.g. '{','.join(str(t) for t in DEFAULT_TARGETS)}'). "
             "Prints the closest mix per target and writes a ready-to-paste "
             "starter_counts YAML block.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
