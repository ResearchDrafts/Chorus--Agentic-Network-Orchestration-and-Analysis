#!/usr/bin/env python3
"""E1: does one argument move one reader, and does that depend on language?

    python scripts/run_e1.py --runs-dir runs
    python scripts/run_e1.py --with-reply --languages english hinglish

RQ1's primary question, asked in the smallest setting that can answer it. One
reader, one argument, two probes:

    1. probe P0        the reader, before it has read anything
    2. probe P1        the same reader, having read one bank item

The argument is held identical across the four arms -- that is what the bank
is for -- so a difference in how far readers move is a difference in the
rendering and in nothing else. `toward` signs the movement by the item's own
side, so a pro item that pushed a reader up and a con item that pushed one
down are both positive and the four arms can be pooled.

**No reply call, by default.** The reader only reads. A reply is a second
thing that happens between the two readings, and a seeded generation is where
most of the variance in a single-exposure trial comes from -- so the default
trial isolates reading, and `--with-reply` measures the other version
separately, into its own folder. Mixing the two in one file would mean the
language effect and the reply effect were estimated from the same column.

**Leans default to 2, 4 and 6** rather than all seven. E1 asks about
persuasion, and a reader already at 1 or 7 has a ceiling in one direction
that a reader at 4 does not; three positions -- against, undecided, for --
span that without spending a third of the calls on the poles, where a null
result is uninterpretable.

Writes, under `--runs-dir`:

    e1/trials.jsonl        one row per trial, including the failures
    e1/prompts.jsonl       every call, verbatim
    e1_with_reply/...      the same two files, under --with-reply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python scripts/run_e1.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sandbox.bank_manager import BankManager
from sandbox.prompt_builder import RoundMemory
from sandbox.single_exposure import (
    STATUS_FAILED,
    STATUS_OK,
    TURN_BASELINE,
    TURN_EXPOSED,
    Reading,
    SingleReaderHarness,
    TrialStore,
    add_common_args,
    apply_limit,
    apply_sample,
    chunked,
    gather_all,
    gather_in_chunks,
    load_base_configs,
    load_trials,
    personas_for,
)

DEFAULT_LEANS = (2, 4, 6)

# pro is the high end of the scale: the side _resolve_side assigns when a
# reading sits above the midpoint. So a pro item moving a reader up is a
# movement toward what it argued, and a con item moving one down is too --
# which is the whole content of `toward`'s sign rule below.
SIDE_PRO = "pro"

OUTPUT_SUBDIR = "e1"
OUTPUT_SUBDIR_WITH_REPLY = "e1_with_reply"


def trial_grid(
    item_ids: list[str], n_personas: int, leans: list[int]
) -> list[tuple[str, int, int]]:
    """Every (item, persona index, lean), item outermost.

    Item first so `--limit` takes a slice spanning one item's readers rather
    than one reader's items: a smoke test wants to see one argument rendered
    and read, which is the part of the pipeline most likely to be wrong.
    """
    return [
        (item_id, persona_index, lean)
        for item_id in item_ids
        for persona_index in range(n_personas)
        for lean in leans
    ]


def trial_key(language: str, item_id: str, persona_index: int, lean: int) -> str:
    return f"e1|{language}|{item_id}|p{persona_index:03d}|lean{lean}"


def reader_key(item_id: str, persona_index: int, lean: int) -> str:
    """The trial's identity with language left out, for the reply seed.

    Only read under `--with-reply`; derived here regardless so the two
    variants of a trial name the same reader.
    """
    return f"e1|{item_id}|p{persona_index:03d}|lean{lean}"


def baseline_key(persona_index: int, lean: int) -> tuple[int, int]:
    """What P0 depends on. Not the item: nothing has been read when it is
    taken, so one baseline serves every item that reader is shown."""
    return (persona_index, lean)


async def measure_baselines(
    harness: SingleReaderHarness,
    personas: list[str],
    combinations: list[tuple[int, int]],
) -> dict[tuple[int, int], Reading | None]:
    agents = [
        harness.reader(f"e1|p{index:03d}|lean{lean}", personas[index], lean)
        for index, lean in combinations
    ]
    readings = await gather_in_chunks(
        (harness.probe(agent, [], turn=TURN_BASELINE) for agent in agents),
        harness.concurrency,
    )
    return dict(zip(combinations, readings, strict=True))


async def run_trial(
    harness: SingleReaderHarness,
    bank: BankManager,
    *,
    item_id: str,
    persona_index: int,
    persona: str,
    lean: int,
    baseline: Reading | None,
    with_reply: bool,
) -> dict:
    """One exposure: show the reader the item, then probe it.

    The item text is rendered for THIS trial's language by the bank, which is
    the only place the four arms differ. Everything else the prompt carries --
    persona, topic, the lean sentence, the memory heading, the question, the
    anchors -- is byte-identical across arms.
    """
    side = bank.side_of(item_id)
    row: dict = {
        "key": trial_key(harness.language, item_id, persona_index, lean),
        "language": harness.language,
        "item_id": item_id,
        "arg_id": bank.arg_id_of(item_id),
        "strength": bank.strength_of(item_id),
        "side": side,
        "persona_index": persona_index,
        "lean": lean,
        "with_reply": with_reply,
    }
    if baseline is None:
        return _row(row, None, None, side, None)

    item_text = bank.text(item_id, harness.language)
    agent = harness.reader(
        f"e1|{item_id}|p{persona_index:03d}|lean{lean}", persona, lean
    )

    seed = None
    reply_text = ""
    if with_reply:
        seed = harness.reply_seed(reader_key(item_id, persona_index, lean))
        got = await harness.reply(
            agent,
            [item_text],
            [],
            trial_key=reader_key(item_id, persona_index, lean),
        )
        if got is None:
            return _row(row, baseline, None, side, seed)
        reply_text = got

    exposed = await harness.probe(
        agent,
        [RoundMemory(posts_read=[item_text], reply_text=reply_text)],
        turn=TURN_EXPOSED,
    )
    return _row(row, baseline, exposed, side, seed)


def _row(
    base: dict,
    before: Reading | None,
    after: Reading | None,
    side: str,
    seed: int | None,
) -> dict:
    """The trial row, with `toward` signed by the item's side.

        toward =  dx  when the item argues pro (the high end of the scale)
        toward = -dx  when it argues con

    So a positive `toward` always means "moved the way the item argued",
    whichever way that was, and the two sides can be pooled in one model.
    Signed here rather than in the analysis because the sign rule depends on
    which end of the scale `pro` names, and that is a fact about the bank and
    the probe -- not something an analysis script should have to restate.
    """
    ok = before is not None and after is not None
    dx = (after.x - before.x) if ok else None
    return {
        **base,
        "reply_seed": seed,
        "probe_mean0": before.mean if before else None,
        "probe_mean1": after.mean if after else None,
        "x0": before.x if before else None,
        "x1": after.x if after else None,
        "dx": dx,
        "toward": None if dx is None else (dx if side == SIDE_PRO else -dx),
        "conf0": before.conf if before else None,
        "conf1": after.conf if after else None,
        "dconf": (after.conf - before.conf) if ok else None,
        "mass0": before.mass if before else None,
        "mass1": after.mass if after else None,
        "status": STATUS_OK if ok else STATUS_FAILED,
    }


async def run_language(
    harness: SingleReaderHarness,
    bank: BankManager,
    personas: list[str],
    leans: list[int],
    store: TrialStore,
    limit: int | None,
    with_reply: bool,
    sample: int | None = None,
) -> int:
    all_trials = apply_limit(
        apply_sample(trial_grid(bank.all_item_ids(), len(personas), leans), sample),
        limit,
    )
    done = store.completed()
    pending = [
        t for t in all_trials if trial_key(harness.language, *t) not in done
    ]
    print(
        f"  {harness.language:<16} {len(all_trials)} trials, "
        f"{len(all_trials) - len(pending)} already done, {len(pending)} to run"
    )
    if not pending:
        return 0

    combinations = sorted({baseline_key(index, lean) for _, index, lean in pending})
    baselines = await measure_baselines(harness, personas, combinations)
    n_failed = sum(1 for r in baselines.values() if r is None)
    if n_failed:
        print(
            f"    {n_failed} of {len(combinations)} baseline probes failed; "
            "their trials are logged as failed"
        )

    for group in chunked(pending, harness.concurrency):
        rows = await gather_all(
            run_trial(
                harness,
                bank,
                item_id=item_id,
                persona_index=index,
                persona=personas[index],
                lean=lean,
                baseline=baselines[baseline_key(index, lean)],
                with_reply=with_reply,
            )
            for item_id, index, lean in group
        )
        for row in rows:
            store.append(row)
    return len(pending)


def summarize(rows: list[dict]) -> None:
    """Mean `toward` by language and strength: the result, in one table.

    Printed rather than written: `analysis/stats.py` is what fits the model
    and writes the numbers that get reported. This is the look at the data
    that decides whether the model is worth fitting -- a sweep in which
    `toward` is zero everywhere has a plumbing problem, not a null result.
    """
    ok = [r for r in rows if r["status"] == STATUS_OK]
    if not ok:
        print("\n  no ok trials to summarize")
        return
    print(f"\n  mean toward over {len(ok)} ok trials:")
    print(f"    {'language':<16} {'strong':>9} {'weak':>9} {'n':>6}")
    languages = []
    for row in ok:
        if row["language"] not in languages:
            languages.append(row["language"])
    for language in languages:
        arm = [r for r in ok if r["language"] == language]
        cells = []
        for strength in ("strong", "weak"):
            subset = [r["toward"] for r in arm if r["strength"] == strength]
            cells.append(
                f"{sum(subset) / len(subset):>9.4f}" if subset else f"{'-':>9}"
            )
        print(f"    {language:<16} {cells[0]} {cells[1]} {len(arm):>6}")


async def main_async(args: argparse.Namespace) -> int:
    subdir = OUTPUT_SUBDIR_WITH_REPLY if args.with_reply else OUTPUT_SUBDIR
    out_dir = args.runs_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    store = TrialStore(out_dir / "trials.jsonl")

    runs = load_base_configs(args.languages, args.configs)
    print(f"\nE1 single exposure -> {out_dir}/")
    print(f"  leans         {args.leans}")
    print(f"  reply         {'yes, seeded' if args.with_reply else 'no: read only'}")

    for language in args.languages:
        run = runs[language]
        harness = SingleReaderHarness(run, out_dir)
        await run_language(
            harness,
            # Honours exclude_items_path, so an E1 sweep re-run after the
            # comprehension check drops the items no arm could read -- a
            # language effect measured on an item nobody understood is a
            # comprehension failure wearing a persuasion effect's clothes.
            BankManager(run.bank_path, exclude_items_path=run.exclude_items_path),
            personas_for(run),
            args.leans,
            store,
            args.limit,
            args.with_reply,
            args.sample,
        )

    summarize(load_trials(store.path))
    print(f"\n  {store.path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser)
    parser.add_argument(
        "--leans",
        nargs="+",
        type=int,
        default=list(DEFAULT_LEANS),
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="the starting positions to test (default: 2 4 6)",
    )
    parser.add_argument(
        "--with-reply",
        action="store_true",
        help="ask the reader to reply between reading and being probed, and "
        f"write to {OUTPUT_SUBDIR_WITH_REPLY}/ instead of {OUTPUT_SUBDIR}/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
