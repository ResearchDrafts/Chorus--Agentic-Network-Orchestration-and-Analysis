#!/usr/bin/env python3
"""Comprehension: can the model read each item, in each language, at all?

    python scripts/check_comprehension.py --runs-dir runs
    python scripts/check_comprehension.py --languages hindi

E1 can report that the Hindi rendering of an argument moved readers less than
the English one. Two very different things produce that result: the argument
is less persuasive in Hindi, or the model cannot read it in Hindi. The first
is a finding; the second is a broken stimulus, and nothing in E1's output
tells them apart.

So each item is shown, in each language, with no persona and no lean, and the
model is asked which of the two views it supports:

    A user posted: <the item, in this language>

    Which view does this post support?

    A: <one anchor label>
    B: <the other>

    Answer with a single letter, A or B, and nothing else.

**Both orders are run.** A model with a standing preference for the first
option would pass every pro item and fail every con one, and the pass rate
would measure position bias rather than comprehension. An item passes in a
language only when the correct letter carries more than half the probability
in *both* orders, so a position preference cannot produce a pass.

**No persona and no lean.** A reader with a stance to defend confounds "could
not read it" with "would not agree with it", and this check exists precisely
to separate those.

**The question and the labels are English in all four arms**, like every
other part of RQ1's frame. The manipulation under test is the language of the
post. Translating the question would mean asking the Hindi arm a different
question from the English one, which is the confound the whole design is
built to avoid.

Writes, under `--runs-dir`:

    comprehension/trials.jsonl      one row per (item, language, order)
    comprehension/prompts.jsonl     every call, verbatim
    comprehension/summary.csv       item x language: pass, both probabilities
    comprehension/failed_items.txt  every item failing in ANY language, ready
                                    to be passed as exclude_items_path
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python scripts/check_comprehension.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sandbox.bank_manager import BankManager
from sandbox.probe_scorer import ChoiceScore
from sandbox.single_exposure import (
    LANGUAGES,
    STATUS_FAILED,
    STATUS_OK,
    TURN_BASELINE,
    SingleReaderHarness,
    TrialStore,
    add_common_args,
    apply_limit,
    build_comprehension_prompt,
    chunked,
    gather_all,
    load_base_configs,
    load_trials,
)

CHOICES = ("A", "B")

# The two label orders. "high_first" puts the high anchor (the pro pole) at A;
# "low_first" swaps them. Running both is what cancels a standing preference
# for one letter.
ORDER_HIGH_FIRST = "high_first"
ORDER_LOW_FIRST = "low_first"
ORDERS = (ORDER_HIGH_FIRST, ORDER_LOW_FIRST)

# An item passes when the correct letter carries more than this in both
# orders. Half, not the config's probe_mass_floor: the floor is about how much
# of the distribution addressed the question at all, and this is about which
# of two answers it gave. A coin flip is 0.5, so "more than half" is the
# weakest claim that is still a claim.
PASS_THRESHOLD = 0.5

OUTPUT_SUBDIR = "comprehension"

SIDE_PRO = "pro"


def correct_letter(side: str, order: str) -> str:
    """Which letter is right for an item on `side`, in label order `order`.

    `pro` is the high end of the scale, so under `high_first` -- the high
    anchor at A -- a pro item's answer is A and a con item's is B, and under
    `low_first` they swap. Written out as a function rather than inline
    because getting it backwards would invert the entire pass rate while
    still producing a plausible-looking file.
    """
    if order not in ORDERS:
        raise ValueError(f"unknown label order {order!r}, expected one of {list(ORDERS)}")
    high_is_a = order == ORDER_HIGH_FIRST
    item_is_high = side == SIDE_PRO
    return CHOICES[0] if item_is_high == high_is_a else CHOICES[1]


def labels_for(order: str, low: str, high: str) -> tuple[str, str]:
    return (high, low) if order == ORDER_HIGH_FIRST else (low, high)


def trial_key(language: str, item_id: str, order: str) -> str:
    return f"comp|{language}|{item_id}|{order}"


async def run_trial(
    harness: SingleReaderHarness,
    bank: BankManager,
    *,
    item_id: str,
    order: str,
) -> dict:
    run = harness.run
    label_a, label_b = labels_for(order, run.stance_low_label, run.stance_high_label)
    side = bank.side_of(item_id)
    prompt = build_comprehension_prompt(
        bank.text(item_id, harness.language), label_a, label_b, CHOICES
    )
    key = trial_key(harness.language, item_id, order)
    score = await harness.choice_probe(
        prompt, CHOICES, turn=TURN_BASELINE, key=key
    )
    return _row(
        key=key,
        language=harness.language,
        item_id=item_id,
        arg_id=bank.arg_id_of(item_id),
        strength=bank.strength_of(item_id),
        side=side,
        order=order,
        expected=correct_letter(side, order),
        score=score,
    )


def _row(
    *,
    key: str,
    language: str,
    item_id: str,
    arg_id: str,
    strength: str,
    side: str,
    order: str,
    expected: str,
    score: ChoiceScore | None,
) -> dict:
    """One (item, language, order) row.

    A failed row carries null probabilities rather than a zero. A zero would
    read as "the model confidently gave the wrong answer", which is a claim
    about comprehension; what actually happened is that it answered neither
    letter, which is a claim about the call.
    """
    return {
        "key": key,
        "language": language,
        "item_id": item_id,
        "arg_id": arg_id,
        "strength": strength,
        "side": side,
        "order": order,
        "expected": expected,
        "p_correct": None if score is None else score.probability(expected),
        "p_a": None if score is None else score.probability(CHOICES[0]),
        "mass": None if score is None else score.mass,
        "status": STATUS_FAILED if score is None else STATUS_OK,
    }


async def run_language(
    harness: SingleReaderHarness,
    bank: BankManager,
    store: TrialStore,
    limit: int | None,
) -> int:
    trials = apply_limit(
        [(item_id, order) for item_id in bank.all_item_ids() for order in ORDERS],
        limit,
    )
    done = store.completed()
    pending = [t for t in trials if trial_key(harness.language, *t) not in done]
    print(
        f"  {harness.language:<16} {len(trials)} trials, "
        f"{len(trials) - len(pending)} already done, {len(pending)} to run"
    )
    for group in chunked(pending, harness.concurrency):
        rows = await gather_all(
            run_trial(harness, bank, item_id=item_id, order=order)
            for item_id, order in group
        )
        for row in rows:
            store.append(row)
    return len(pending)


# --- summarizing ----------------------------------------------------------


def summarize(rows: list[dict]) -> list[dict]:
    """One record per (item, language): did it pass, and on what numbers.

    An item passes only when both orders put more than half the probability
    on the correct letter. A pair with one missing order, or one failed call,
    does not pass -- an item that could not be checked has not been shown to
    be readable, and treating "unchecked" as "fine" is how a broken stimulus
    reaches a campaign.
    """
    by_pair: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        by_pair.setdefault((row["item_id"], row["language"]), {})[row["order"]] = row

    out: list[dict] = []
    for (item_id, language), orders in sorted(by_pair.items()):
        high = orders.get(ORDER_HIGH_FIRST)
        low = orders.get(ORDER_LOW_FIRST)
        probabilities = [
            None if o is None or o["status"] != STATUS_OK else o["p_correct"]
            for o in (high, low)
        ]
        masses = [
            None if o is None or o["status"] != STATUS_OK else o["mass"]
            for o in (high, low)
        ]
        passed = all(p is not None and p > PASS_THRESHOLD for p in probabilities)
        any_row = high or low
        out.append(
            {
                "item_id": item_id,
                "language": language,
                "arg_id": any_row["arg_id"],
                "strength": any_row["strength"],
                "side": any_row["side"],
                "passed": passed,
                "p_correct_high_first": probabilities[0],
                "p_correct_low_first": probabilities[1],
                "mass_high_first": masses[0],
                "mass_low_first": masses[1],
                "n_failed_orders": sum(1 for p in probabilities if p is None),
            }
        )
    return out


SUMMARY_COLUMNS = (
    "item_id",
    "language",
    "arg_id",
    "strength",
    "side",
    "passed",
    "p_correct_high_first",
    "p_correct_low_first",
    "mass_high_first",
    "mass_low_first",
    "n_failed_orders",
)


def write_summary(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(records)


def failed_items(records: list[dict]) -> list[str]:
    """Every item that fails in ANY language, sorted.

    Any rather than all, because the bank's whole purpose is that one
    argument is held identical across four arms. An item the model reads in
    English and not in Hindi cannot be in the English arm either: dropping it
    from one arm and keeping it in another would make the arms differ in which
    arguments they contain, which is the one difference the design cannot
    have.
    """
    return sorted({r["item_id"] for r in records if not r["passed"]})


def write_failed_items(path: Path, items: list[str], records: list[dict]) -> None:
    """failed_items.txt, with a comment header saying why each item is there.

    Comments and blank lines are skipped by `read_exclusions`, so this file is
    valid as an `exclude_items_path` exactly as written -- the point being
    that the remedy for a failed item is one config line, not a hand-edited
    list.
    """
    lines = [
        "# Items the model could not read in at least one language.",
        "# Produced by scripts/check_comprehension.py. Pass this file as",
        "# exclude_items_path in a run config to drop them from the bank.",
        "#",
        "# An item failing in ANY language is listed: the bank holds one",
        "# argument rendered four ways, so keeping it in the arms that could",
        "# read it would make the arms differ in which arguments they contain.",
        "",
    ]
    for item_id in items:
        where = sorted(
            r["language"] for r in records if r["item_id"] == item_id and not r["passed"]
        )
        lines.append(f"{item_id}  # failed in: {', '.join(where)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_pass_rates(records: list[dict], languages: list[str]) -> None:
    print("\n  pass rate per language:")
    for language in languages:
        arm = [r for r in records if r["language"] == language]
        if not arm:
            continue
        passed = sum(1 for r in arm if r["passed"])
        print(
            f"    {language:<16} {passed}/{len(arm)} = {passed / len(arm):.2%}"
            + (
                f"   ({sum(r['n_failed_orders'] for r in arm)} calls failed)"
                if any(r["n_failed_orders"] for r in arm)
                else ""
            )
        )


async def main_async(args: argparse.Namespace) -> int:
    out_dir = args.runs_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    store = TrialStore(out_dir / "trials.jsonl")

    runs = load_base_configs(args.languages, args.configs)
    print(f"\ncomprehension check -> {out_dir}/")
    print(f"  orders        {ORDERS[0]} and {ORDERS[1]}, to cancel position bias")
    print(f"  threshold     p(correct) > {PASS_THRESHOLD} in both orders")

    for language in args.languages:
        run = runs[language]
        harness = SingleReaderHarness(run, out_dir)
        await run_language(
            harness,
            # Deliberately NOT honouring exclude_items_path: this is the check
            # that decides what belongs in that file, so reading it would mean
            # a once-excluded item could never be re-tested after a prompt or
            # a model change.
            BankManager(run.bank_path),
            store,
            args.limit,
        )

    rows = load_trials(store.path)
    records = summarize(rows)
    present = [lang for lang in LANGUAGES if any(r["language"] == lang for r in records)]

    write_summary(out_dir / "summary.csv", records)
    failed = failed_items(records)
    write_failed_items(out_dir / "failed_items.txt", failed, records)

    print_pass_rates(records, present)
    print(f"\n  {out_dir / 'summary.csv'}  {len(records)} item x language rows")
    print(
        f"  {out_dir / 'failed_items.txt'}  {len(failed)} item(s) failing in "
        "at least one language"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
