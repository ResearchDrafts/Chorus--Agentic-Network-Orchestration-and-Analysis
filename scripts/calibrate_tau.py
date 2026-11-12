#!/usr/bin/env python3
"""τ calibration: how much a reading moves when nothing persuasive happened.

    python scripts/calibrate_tau.py --runs-dir runs
    python scripts/calibrate_tau.py --languages english hindi --limit 8

The probe is a measurement, and measurements move on their own. A reader
shown an on-topic sentence that argues nothing, asked to reply to it, and
probed again does not answer with exactly the digit it answered with before:
the history grew, the sampler drew, and the reading shifts. τ is how much.
Every later claim -- that an agent flipped, hardened or softened -- is a claim
that a change exceeded this, so a campaign reported without it cannot
distinguish persuasion from the instrument.

**The trial.** For each language, persona, lean and neutral message:

    1. probe P0        the reader, before it has read anything
    2. reply           it is shown the one neutral message and replies
    3. probe P1        the same reader, with that round in its memory

The only thing between the two readings is a side-free sentence and the
reader's own reply to it. So |x1 - x0| is movement with no argument in it,
and the 95th percentile of that distribution is the threshold a real
movement has to clear.

P0 depends only on (language, persona, lean) -- not on which neutral message
the trial uses, since nothing has been read when it is taken -- so it is
measured once per combination and reused across that combination's trials.
That is a third of the calls saved, and it also makes the baseline a property
of the reader rather than something that could drift between its own trials.

**Why a neutral message rather than nothing at all.** A reader probed twice
with nothing in between is probed with an identical prompt twice, and at
temperature 0 that measures the server's determinism rather than the
instrument's. What the campaign's control arm actually does is read a neutral
message and reply to it, and τ has to be the baseline for that -- which is
why this reads from the same `neutral_bank_path` the E4 control delivers
from.

Writes, under `--runs-dir`:

    calibration/trials.jsonl    one row per trial, including the failures
    calibration/prompts.jsonl   every call, verbatim
    calibration/tau.json        τ and τ_conf per language, with provenance
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):  # `python scripts/calibrate_tau.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sandbox.bank_manager import NeutralBank
from sandbox.prompt_builder import RoundMemory
from sandbox.single_exposure import (
    LANGUAGES,
    STATUS_FAILED,
    STATUS_OK,
    TURN_BASELINE,
    TURN_EXPOSED,
    Reading,
    SingleReaderHarness,
    TrialStore,
    add_common_args,
    apply_limit,
    chunked,
    gather_all,
    gather_in_chunks,
    load_base_configs,
    load_trials,
    percentile,
    personas_for,
)

DEFAULT_LEANS = (1, 2, 3, 4, 5, 6, 7)
DEFAULT_PERCENTILE = 95.0

# A threshold from a handful of trials is not a threshold: its percentile is
# an order statistic of a sample too small to have a 95th. Below this the
# script refuses to write tau.json rather than writing a number every later
# result would be judged against. Overridable only to make a smoke run
# produce a file at all, and a file written that way says so.
MIN_OK_TRIALS = 30

OUTPUT_SUBDIR = "calibration"


class NotEnoughTrials(Exception):
    """Raised instead of writing a τ no one should use."""


def trial_keys(personas: list[str], leans: list[int], neutral_ids: list[str]) -> list[tuple[int, int, str]]:
    """Every (persona index, lean, neutral id), in the nested order above.

    Persona outermost, then lean, then message, so `--limit` takes a slice
    that spans the leans of the first personas rather than one lean across
    everybody -- a smoke test wants to see the lean sentence vary, since that
    is the part of the prompt most likely to be misconfigured.
    """
    return [
        (persona_index, lean, neutral_id)
        for persona_index in range(len(personas))
        for lean in leans
        for neutral_id in neutral_ids
    ]


def trial_key(language: str, persona_index: int, lean: int, neutral_id: str) -> str:
    return f"cal|{language}|p{persona_index:03d}|lean{lean}|{neutral_id}"


def reader_key(persona_index: int, lean: int, neutral_id: str) -> str:
    """The trial's identity with language left out.

    What the reply seed is derived from, so the same trial in four arms hands
    the server the same number and the arms differ in the text of the prompt
    rather than in how the reply to it was sampled. `SeedManager` holds the
    same property for a campaign; this is the single-reader version of it.
    """
    return f"cal|p{persona_index:03d}|lean{lean}|{neutral_id}"


def baseline_key(persona_index: int, lean: int) -> tuple[int, int]:
    """What P0 depends on. Not the neutral message: nothing has been read."""
    return (persona_index, lean)


def baseline_agent_id(persona_index: int, lean: int) -> str:
    return f"cal|p{persona_index:03d}|lean{lean}"


async def measure_baselines(
    harness: SingleReaderHarness,
    personas: list[str],
    combinations: list[tuple[int, int]],
) -> dict[tuple[int, int], Reading | None]:
    """P0 for each (persona, lean), once.

    A None here is a reader whose baseline could not be measured, which makes
    every trial that would have used it a failed row -- without spending a
    reply call on a trial whose difference could never be computed.
    """
    agents = [
        harness.reader(baseline_agent_id(index, lean), personas[index], lean)
        for index, lean in combinations
    ]
    readings = await gather_in_chunks(
        (harness.probe(agent, [], turn=TURN_BASELINE) for agent in agents),
        harness.concurrency,
    )
    return dict(zip(combinations, readings, strict=True))


async def run_trial(
    harness: SingleReaderHarness,
    *,
    persona_index: int,
    persona: str,
    lean: int,
    neutral_id: str,
    neutral_text: str,
    baseline: Reading | None,
) -> dict:
    """One trial: reply to the neutral message, then probe again.

    The reader is a fresh Agent with no stance history, exactly as it was for
    P0 -- the memory it is shown comes from `history_rounds`, not from the
    agent object, so the only difference between the two prompts is the round
    this function puts in it.
    """
    key = trial_key(harness.language, persona_index, lean, neutral_id)
    row: dict = {
        "key": key,
        "language": harness.language,
        "persona_index": persona_index,
        "lean": lean,
        "neutral_id": neutral_id,
    }
    if baseline is None:
        return _row(row, None, None, None)

    agent = harness.reader(
        f"cal|p{persona_index:03d}|lean{lean}|{neutral_id}", persona, lean
    )
    seed = harness.reply_seed(reader_key(persona_index, lean, neutral_id))
    reply_text = await harness.reply(
        agent,
        [neutral_text],
        [],
        trial_key=reader_key(persona_index, lean, neutral_id),
    )
    if reply_text is None:
        return _row(row, baseline, None, seed)

    exposed = await harness.probe(
        agent,
        [RoundMemory(posts_read=[neutral_text], reply_text=reply_text)],
        turn=TURN_EXPOSED,
    )
    return _row(row, baseline, exposed, seed)


def _row(
    base: dict, before: Reading | None, after: Reading | None, seed: int | None
) -> dict:
    """The trial row. Null readings and `failed` when either probe is missing.

    Never a partially filled row that looks measured: if P1 is absent there is
    no dx, and a 0 there would be the single most misleading value this file
    could carry -- a τ computed over fabricated zeros is lower than the truth,
    and a lower τ calls more of the campaign's noise persuasion.
    """
    ok = before is not None and after is not None
    return {
        **base,
        "reply_seed": seed,
        "probe_mean0": before.mean if before else None,
        "probe_mean1": after.mean if after else None,
        "x0": before.x if before else None,
        "x1": after.x if after else None,
        "dx": (after.x - before.x) if ok else None,
        "conf0": before.conf if before else None,
        "conf1": after.conf if after else None,
        "dconf": (after.conf - before.conf) if ok else None,
        "mass0": before.mass if before else None,
        "mass1": after.mass if after else None,
        "status": STATUS_OK if ok else STATUS_FAILED,
    }


async def calibrate_language(
    harness: SingleReaderHarness,
    personas: list[str],
    leans: list[int],
    neutral_bank: NeutralBank,
    store,
    limit: int | None,
) -> int:
    """Runs one arm's pending trials. Returns how many it ran."""
    neutral_ids = neutral_bank.ids()
    all_trials = apply_limit(trial_keys(personas, leans, neutral_ids), limit)
    done = store.completed()
    pending = [
        t
        for t in all_trials
        if trial_key(harness.language, *t) not in done
    ]
    print(
        f"  {harness.language:<16} {len(all_trials)} trials, "
        f"{len(all_trials) - len(pending)} already done, {len(pending)} to run"
    )
    if not pending:
        return 0

    combinations = sorted({baseline_key(index, lean) for index, lean, _ in pending})
    baselines = await measure_baselines(harness, personas, combinations)
    n_failed_baselines = sum(1 for r in baselines.values() if r is None)
    if n_failed_baselines:
        print(
            f"    {n_failed_baselines} of {len(combinations)} baseline probes "
            "failed; their trials are logged as failed"
        )

    for group in chunked(pending, harness.concurrency):
        rows = await gather_all(
            run_trial(
                harness,
                persona_index=index,
                persona=personas[index],
                lean=lean,
                neutral_id=neutral_id,
                neutral_text=neutral_bank.text(neutral_id, harness.language),
                baseline=baselines[baseline_key(index, lean)],
            )
            for index, lean, neutral_id in group
        )
        for row in rows:
            store.append(row)
    return len(pending)


def summarize(
    rows: list[dict], languages: list[str], pct: float, min_ok: int
) -> dict[str, dict]:
    """τ per language, plus the quality numbers that say whether to trust it.

    `mass0`/`mass1` means and minima are reported because τ is only as
    meaningful as the readings it came from: a 95th percentile computed over
    probes that put half their probability off-scale is a threshold for a
    different measurement. The minimum is there because a mean near 0.9 hides
    a handful of 0.51s.
    """
    summary: dict[str, dict] = {}
    short: list[str] = []
    for language in languages:
        arm = [r for r in rows if r["language"] == language]
        ok = [r for r in arm if r["status"] == STATUS_OK]
        n_failed = len(arm) - len(ok)
        if len(ok) < min_ok:
            short.append(f"{language}: {len(ok)} ok trials (need {min_ok})")
            summary[language] = {
                "tau": None,
                "tau_conf": None,
                "n_ok": len(ok),
                "n_failed": n_failed,
            }
            continue
        summary[language] = {
            "tau": percentile([abs(r["dx"]) for r in ok], pct),
            "tau_conf": percentile([abs(r["dconf"]) for r in ok], pct),
            "n_ok": len(ok),
            "n_failed": n_failed,
            "mass0_mean": sum(r["mass0"] for r in ok) / len(ok),
            "mass0_min": min(r["mass0"] for r in ok),
            "mass1_mean": sum(r["mass1"] for r in ok) / len(ok),
            "mass1_min": min(r["mass1"] for r in ok),
        }
    if short:
        raise NotEnoughTrials(
            "refusing to write a threshold from too few trials -- "
            + "; ".join(short)
            + ". Every later claim that an agent flipped is a claim against this "
            "number, so it has to come from a distribution rather than a handful "
            "of draws. Re-run the failed trials, or lower --min-ok-trials "
            "knowingly for a smoke test."
        )
    return summary


def write_tau(
    path: Path,
    summary: dict[str, dict],
    *,
    pct: float,
    min_ok: int,
    git_commit_hash: str,
    smoke: bool,
) -> None:
    """tau.json, with enough provenance to tell two of them apart.

    The commit and the timestamp are here because τ is an instrument
    constant: a tau.json produced before a prompt change does not apply after
    it, and the only way a later reader can notice is if the file says which
    code produced it.
    """
    path.write_text(
        json.dumps(
            {
                "percentile": pct,
                "min_ok_trials": min_ok,
                "git_commit_hash": git_commit_hash,
                "generated_at_utc": datetime.now(UTC).isoformat(),
                # True when --limit or a lowered --min-ok-trials was used, so
                # a file produced for a plumbing check cannot be mistaken for
                # a calibration. analysis/metrics.py prints the flag.
                "smoke": smoke,
                "languages": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


async def main_async(args: argparse.Namespace) -> int:
    out_dir = args.runs_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    store = TrialStore(out_dir / "trials.jsonl")

    runs = load_base_configs(args.languages, args.configs)
    print(f"\nτ calibration -> {out_dir}/")
    print(f"  leans         {args.leans}")
    print(f"  percentile    {args.percentile}")

    for language in args.languages:
        run = runs[language]
        if not run.neutral_bank_path:
            raise ValueError(
                f"the {language!r} base config has no neutral_bank_path. "
                "Calibration reads the same side-free messages the E4 control "
                "delivers, so that τ is the baseline for what the control "
                "actually does rather than for an empty prompt."
            )
        harness = SingleReaderHarness(run, out_dir)
        await calibrate_language(
            harness,
            personas_for(run),
            args.leans,
            NeutralBank(run.neutral_bank_path),
            store,
            args.limit,
        )

    # Summarized over every language the FILE holds, not only the arms this
    # invocation ran. A run with --languages english would otherwise rewrite
    # tau.json with one arm in it and silently drop the three a previous pass
    # had calibrated -- and metrics.py, reading it per run, would then fail on
    # a language that had been measured.
    rows = load_trials(store.path)
    present = [lang for lang in LANGUAGES if any(r["language"] == lang for r in rows)]
    summary = summarize(rows, present, args.percentile, args.min_ok_trials)
    tau_path = out_dir / "tau.json"
    write_tau(
        tau_path,
        summary,
        pct=args.percentile,
        min_ok=args.min_ok_trials,
        git_commit_hash=next(iter(runs.values())).git_commit_hash,
        smoke=args.limit is not None or args.min_ok_trials < MIN_OK_TRIALS,
    )

    print(f"\n  {tau_path}")
    for language, stats in summary.items():
        print(
            f"    {language:<16} tau={stats['tau']:.4f}  "
            f"tau_conf={stats['tau_conf']:.4f}  "
            f"n_ok={stats['n_ok']}  n_failed={stats['n_failed']}  "
            f"min mass={min(stats['mass0_min'], stats['mass1_min']):.3f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser)
    parser.add_argument(
        "--leans",
        nargs="+",
        type=int,
        default=list(DEFAULT_LEANS),
        choices=list(DEFAULT_LEANS),
        help="the starting positions to calibrate (default: all seven)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help=f"which percentile of |dx| becomes τ (default: {DEFAULT_PERCENTILE})",
    )
    parser.add_argument(
        "--min-ok-trials",
        type=int,
        default=MIN_OK_TRIALS,
        # No lower than 1: percentile() raises on an empty sequence, and a
        # threshold from zero observations is not a weaker threshold, it is
        # not one at all.
        metavar="N",
        help=f"refuse to write tau.json below this many ok trials per language "
        f"(default: {MIN_OK_TRIALS}). Lower it only for a smoke test; the file "
        "records that it was lowered.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_ok_trials < 1:
        build_parser().error("--min-ok-trials must be at least 1")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
