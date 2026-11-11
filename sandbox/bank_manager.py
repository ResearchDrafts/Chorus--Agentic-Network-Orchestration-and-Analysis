"""The argument bank (RQ1, bank regime).

RQ1 asks whether the language or script of a post changes how much it moves
an LLM reader, *with the argument held identical*. That last clause is the
whole experiment, and it is not something agents writing their own posts can
provide: an English agent and a Hindi agent asked to argue the same side will
produce different arguments, and any difference in how far their readers move
is then a difference in content as much as in language.

So posts come from here instead. The bank is a CSV of arguments, each
pre-translated into all four conditions, and an agent "posting" means
selecting a bank item for its current side and rendering it in its run's
language. No LLM writes a post.

**The draw must not depend on language.** This module's one hard property is
that `pick` is a pure function of (seed, sender_id, side, how many items this
sender has already posted on that side). Nothing about the rendering, the
turn, or the condition enters it. That is what makes a matched set matched: if
agent_0007 is on the pro side in its third round in all four language arms, it
posts the same argument in all four, and the arms differ in exactly one thing.

The derivation is documented on `pick` itself, since it is the part a reader
has to be able to check rather than trust.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

# The language columns, and the values they must be keyed by: these are
# language_condition values, so a bank whose header says "romanised_hindi"
# cannot be rendered for a run whose condition is "romanized_hindi".
from sandbox.models import LANGUAGE_CONDITIONS

SIDES = ("pro", "con")
STRENGTHS = ("strong", "weak")
REQUIRED_COLUMNS = ("item_id", "arg_id", "strength", "side", *LANGUAGE_CONDITIONS)


class BankValidationError(Exception):
    """Raised for any problem with a bank file, listing every problem found.

    Accumulated rather than fail-fast, for the reason config_loader gives: a
    hand-maintained data file with three mistakes in it should take one run to
    fix, not three.
    """

    def __init__(self, path: Path, errors: list[str]):
        self.path = path
        self.errors = errors
        super().__init__(f"Invalid argument bank at {path}: {'; '.join(errors)}")


class BankManager:
    """Loads and serves one argument bank.

    Stateful in exactly one respect: it counts how many items each sender has
    posted on each side, which is what drives the no-repeat cycle. Those
    counts are the only thing here that a resume has to restore, and
    `restore_post_counts` rebuilds them from interactions.jsonl rather than
    from the checkpoint, the same way the orchestrator rebuilds last posts --
    the log is complete through the last completed turn and is already the
    authority on what was actually posted.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._items = _load_and_validate(self._path)
        self._by_side: dict[str, list[str]] = {
            side: sorted(
                item_id for item_id, row in self._items.items() if row["side"] == side
            )
            for side in SIDES
        }
        # (sender_id, side) -> how many items this sender has posted on that
        # side. Not a per-sender RNG state: see pick's docstring.
        self._post_counts: dict[tuple[str, str], int] = {}

    # --- reading the bank -------------------------------------------------

    def items_for_side(self, side: str) -> list[str]:
        """The item IDs on `side`, sorted.

        Sorted, not in file order: the permutation `pick` derives is a
        function of this sequence, so it has to be a function of the bank's
        contents rather than of how someone happened to order the rows.
        Re-sorting the CSV must not change which argument an agent posts.
        """
        self._require_side(side)
        return list(self._by_side[side])

    def text(self, item_id: str, language: str) -> str:
        """The item's rendering in one language condition.

        Raises for an unknown item or an unknown language rather than falling
        back to English, which would silently turn one arm of a matched set
        into a second copy of another.
        """
        if item_id not in self._items:
            raise KeyError(f"unknown bank item: {item_id!r}")
        if language not in LANGUAGE_CONDITIONS:
            raise ValueError(
                f"unknown language condition {language!r}, expected one of {list(LANGUAGE_CONDITIONS)}"
            )
        return self._items[item_id][language]

    def side_of(self, item_id: str) -> str:
        if item_id not in self._items:
            raise KeyError(f"unknown bank item: {item_id!r}")
        return self._items[item_id]["side"]

    def strength_of(self, item_id: str) -> str:
        if item_id not in self._items:
            raise KeyError(f"unknown bank item: {item_id!r}")
        return self._items[item_id]["strength"]

    # --- picking ----------------------------------------------------------

    def pick(self, sender_id: str, turn: int, side: str, rng: random.Random) -> str:
        """The item `sender_id` posts on `side`, with no repeats within a cycle.

        A sender works through every item on a side before any item on that
        side comes round again, in an order that differs per sender. So two
        agents on the same side do not post the argument in lockstep, and no
        agent repeats itself while it still has something new to say.

        **The derivation, which is the part that matters.** Let

            items  = items_for_side(side)            (sorted, so bank order is irrelevant)
            count  = how many items this sender has already posted on this side
            cycle  = count // len(items)
            index  = count %  len(items)

        then the order for this (sender, side, cycle) is

            random.Random(f"{base}|{side}|{cycle}").shuffle(items)

        and the answer is `items[index]`, where `base` is read out of `rng`'s
        *current state without advancing it*. The orchestrator hands in
        `SeedManager.sender_rng("argument", sender_id)`, which is a fresh
        stream derived from (seed, purpose, sender) and nothing else, so
        `base` is a pure function of (seed, sender_id).

        Three consequences, each of them a requirement rather than a
        side-effect:

        * `turn` does not appear. It is accepted (callers have it, and a
          future variant may want it) and deliberately unused: a per-turn
          derivation would reshuffle the order every round, and "no repeat
          until the side is exhausted" would quietly stop being true.
        * Language does not appear. Two runs differing only in
          language_condition give the same sender the same item sequence on
          the same side, which is what a matched set means.
        * No RNG state is stored, so a resume needs only the counts, which
          `restore_post_counts` rebuilds from the log. Reading `rng`'s state
          without consuming it is what makes the call idempotent: picking
          twice for the same (sender, side, count) returns the same item, so a
          retry or a re-derivation after a crash cannot drift.

        `pick` does not itself advance the count. The orchestrator calls
        `record_post` after the round is written, so a round that was computed
        but never logged leaves no trace -- which is what keeps a resumed run
        identical to an uninterrupted one.
        """
        self._require_side(side)
        if not sender_id:
            raise ValueError("sender_id must be a non-empty string")

        items = self._by_side[side]
        count = self._post_counts.get((sender_id, side), 0)
        cycle, index = divmod(count, len(items))

        order = list(items)
        _cycle_rng(rng, side, cycle).shuffle(order)
        return order[index]

    def record_post(self, sender_id: str, side: str) -> None:
        """Marks one item as spent for (sender, side), advancing the cycle."""
        self._require_side(side)
        self._post_counts[(sender_id, side)] = self._post_counts.get((sender_id, side), 0) + 1

    def post_counts(self) -> dict[tuple[str, str], int]:
        """A copy, for assertions and for checkpoint inspection."""
        return dict(self._post_counts)

    def restore_post_counts(self, counts: dict[tuple[str, str], int]) -> None:
        """Replaces the counts wholesale, for a resume."""
        for (sender_id, side), count in counts.items():
            self._require_side(side)
            if count < 0:
                raise ValueError(f"post count for {sender_id}/{side} is negative: {count}")
        self._post_counts = dict(counts)

    # --- internals --------------------------------------------------------

    def _require_side(self, side: str) -> None:
        if side not in SIDES:
            raise ValueError(f"unknown side {side!r}, expected one of {list(SIDES)}")


def _cycle_rng(rng: random.Random, side: str, cycle: int) -> random.Random:
    """A generator for one (sender, side, cycle), derived without consuming rng.

    `rng` carries the (seed, sender) part of the derivation. Its state is
    copied into a throwaway generator and read there, so the caller's stream
    is left exactly where it was found and `pick` stays idempotent -- calling
    it twice for the same count cannot return two different items.

    The cycle key is a string rather than arithmetic: random.Random seeds a
    string through SHA-512, which is stable across processes and
    collision-free, where `hash((side, cycle))` is neither.
    """
    probe = random.Random()
    probe.setstate(rng.getstate())
    base = probe.getrandbits(64)
    return random.Random(f"{base}|{side}|{cycle}")


def _load_and_validate(path: Path) -> dict[str, dict[str, str]]:
    """Reads the CSV and reports every problem at once.

    Validation is strict for the same reason the rest of this codebase is: a
    bank with one empty Hindi cell would otherwise run to completion and
    produce a Hindi arm in which one argument was silently blank, which is
    indistinguishable in the output from an argument that failed to persuade.
    """
    if not path.exists():
        raise BankValidationError(path, [f"file not found: {path}"])

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing_columns:
            # Nothing further can be checked without the columns, and every
            # per-row error would be a restatement of this one.
            raise BankValidationError(
                path, [f"missing required column(s): {', '.join(missing_columns)}"]
            )
        rows = list(reader)

    errors: list[str] = []
    items: dict[str, dict[str, str]] = {}
    seen_sides: set[str] = set()

    if not rows:
        raise BankValidationError(path, ["bank is empty: no rows after the header"])

    for line_no, row in enumerate(rows, start=2):  # start=2: line 1 is the header
        item_id = (row.get("item_id") or "").strip()
        if not item_id:
            errors.append(f"line {line_no}: empty item_id")
            continue
        if item_id in items:
            errors.append(f"line {line_no}: duplicate item_id {item_id!r}")
            continue

        side = (row.get("side") or "").strip()
        if side not in SIDES:
            errors.append(
                f"line {line_no} ({item_id}): side must be one of {list(SIDES)}, got {side!r}"
            )
        else:
            seen_sides.add(side)

        strength = (row.get("strength") or "").strip()
        if strength not in STRENGTHS:
            errors.append(
                f"line {line_no} ({item_id}): strength must be one of {list(STRENGTHS)}, "
                f"got {strength!r}"
            )

        arg_id = (row.get("arg_id") or "").strip()
        if not arg_id:
            errors.append(f"line {line_no} ({item_id}): empty arg_id")

        renderings = {}
        for language in LANGUAGE_CONDITIONS:
            text = (row.get(language) or "").strip()
            if not text:
                errors.append(f"line {line_no} ({item_id}): empty {language} text")
            renderings[language] = text

        items[item_id] = {"item_id": item_id, "arg_id": arg_id, "side": side,
                          "strength": strength, **renderings}

    for side in SIDES:
        if side not in seen_sides:
            errors.append(f"no items on the {side!r} side: every side must be representable")

    if errors:
        raise BankValidationError(path, errors)
    return items
