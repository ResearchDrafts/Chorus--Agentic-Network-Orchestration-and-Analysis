"""Tests for bank_manager.py.

The load-bearing one is
test_the_same_seed_picks_the_same_items_in_every_language: it is the property
that makes RQ1's matched sets matched, and nothing else in the suite would
notice if a language leaked into the draw.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from sandbox.bank_manager import BankManager, BankValidationError
from sandbox.seed_manager import SeedManager

FIXTURE_BANK = Path("data/bank/test_bank.csv")

HEADER = "item_id,arg_id,strength,side,english,hinglish,romanized_hindi,hindi"


def _write_bank(tmp_path, rows: list[str], header: str = HEADER) -> Path:
    path = tmp_path / "bank.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def _row(item_id="b1", arg_id="a1", strength="strong", side="pro", texts=("e", "hi", "rh", "h")):
    return ",".join([item_id, arg_id, strength, side, *texts])


def _sender_rng(sender_id: str, seed: int = 42) -> random.Random:
    """Exactly what the orchestrator passes in."""
    return SeedManager(seed).sender_rng("argument", sender_id)


# --- the shipped fixture bank --------------------------------------------


def test_the_fixture_bank_loads():
    bank = BankManager(FIXTURE_BANK)
    assert bank.items_for_side("pro") == ["bank_t01", "bank_t02"]
    assert bank.items_for_side("con") == ["bank_t03", "bank_t04"]


def test_the_fixture_bank_has_no_digits_in_any_rendering():
    """The prompt audit asserts no position number reaches an agent. A bank
    whose text contained a digit would make that test unable to tell a leaked
    stance number from the argument's own content."""
    bank = BankManager(FIXTURE_BANK)
    for side in ("pro", "con"):
        for item_id in bank.items_for_side(side):
            for language in ("english", "hinglish", "romanized_hindi", "hindi"):
                text = bank.text(item_id, language)
                assert not any(ch.isdigit() for ch in text), (item_id, language, text)


def test_every_item_renders_in_all_four_languages():
    bank = BankManager(FIXTURE_BANK)
    for item_id in bank.items_for_side("pro") + bank.items_for_side("con"):
        rendered = {
            language: bank.text(item_id, language)
            for language in ("english", "hinglish", "romanized_hindi", "hindi")
        }
        assert all(rendered.values())
        assert len(set(rendered.values())) == 4  # four genuinely distinct renderings


def test_both_strengths_and_both_sides_are_present():
    bank = BankManager(FIXTURE_BANK)
    items = bank.items_for_side("pro") + bank.items_for_side("con")
    assert {bank.strength_of(i) for i in items} == {"strong", "weak"}
    assert {bank.side_of(i) for i in items} == {"pro", "con"}


# --- validation -----------------------------------------------------------


def test_missing_file_raises():
    with pytest.raises(BankValidationError, match="file not found"):
        BankManager(Path("data/bank/does_not_exist.csv"))


def test_missing_columns_are_named(tmp_path):
    path = _write_bank(tmp_path, ["b1,a1,strong,pro,e,hi"], header="item_id,arg_id,strength,side,english,hinglish")
    with pytest.raises(BankValidationError) as excinfo:
        BankManager(path)
    assert "romanized_hindi" in str(excinfo.value)
    assert "hindi" in str(excinfo.value)


def test_an_empty_bank_raises(tmp_path):
    with pytest.raises(BankValidationError, match="bank is empty"):
        BankManager(_write_bank(tmp_path, []))


def test_duplicate_item_ids_are_rejected(tmp_path):
    path = _write_bank(tmp_path, [_row("b1"), _row("b2", side="con"), _row("b1", side="con")])
    with pytest.raises(BankValidationError, match="duplicate item_id"):
        BankManager(path)


def test_a_missing_side_is_rejected(tmp_path):
    """Both sides must be representable: an agent on the con side with no con
    items has nothing to post, and there is no safe thing to do about it at
    dispatch time."""
    path = _write_bank(tmp_path, [_row("b1", side="pro"), _row("b2", side="pro")])
    with pytest.raises(BankValidationError, match="no items on the 'con' side"):
        BankManager(path)


@pytest.mark.parametrize(
    "texts, missing_language",
    [
        (("", "hi", "rh", "h"), "english"),
        (("e", "", "rh", "h"), "hinglish"),
        (("e", "hi", "", "h"), "romanized_hindi"),
        (("e", "hi", "rh", ""), "hindi"),
        (("e", "hi", "rh", "   "), "hindi"),
    ],
)
def test_empty_text_in_any_language_column_is_rejected(tmp_path, texts, missing_language):
    """A blank cell would run to completion and produce an arm in which one
    argument was silently empty, which is indistinguishable in the output from
    an argument that failed to persuade."""
    path = _write_bank(tmp_path, [_row("b1", texts=texts), _row("b2", side="con")])
    with pytest.raises(BankValidationError, match=f"empty {missing_language} text"):
        BankManager(path)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("side", "neutral", "side must be one of"),
        ("strength", "medium", "strength must be one of"),
        ("strength", "", "strength must be one of"),
        ("arg_id", "", "empty arg_id"),
        ("item_id", "", "empty item_id"),
    ],
)
def test_invalid_field_values_are_rejected(tmp_path, field, value, message):
    broken = {"item_id": "b1", field: value}  # dict, so field may be item_id itself
    path = _write_bank(tmp_path, [_row(**broken), _row("b2", side="con"), _row("b3", side="pro")])
    with pytest.raises(BankValidationError, match=message):
        BankManager(path)


def test_every_problem_is_reported_at_once(tmp_path):
    """config_loader's contract, for the same reason: a hand-maintained data
    file with three mistakes should take one run to fix, not three."""
    path = _write_bank(
        tmp_path,
        [
            _row("b1", strength="medium", texts=("e", "", "rh", "h")),
            _row("b2", side="sideways"),
        ],
    )
    with pytest.raises(BankValidationError) as excinfo:
        BankManager(path)

    errors = excinfo.value.errors
    assert any("strength" in e for e in errors)
    assert any("empty hinglish text" in e for e in errors)
    assert any("side must be one of" in e for e in errors)
    assert any("no items on the 'con' side" in e for e in errors)
    assert len(errors) >= 4


def test_errors_name_the_line_and_the_item(tmp_path):
    path = _write_bank(tmp_path, [_row("b1", side="con"), _row("b2", texts=("e", "hi", "rh", ""))])
    with pytest.raises(BankValidationError) as excinfo:
        BankManager(path)
    assert any("line 3" in e and "b2" in e for e in excinfo.value.errors)


# --- lookups --------------------------------------------------------------


def test_unknown_item_or_language_raises_rather_than_falling_back():
    """Falling back to English would silently turn one arm of a matched set
    into a second copy of another."""
    bank = BankManager(FIXTURE_BANK)
    with pytest.raises(KeyError, match="unknown bank item"):
        bank.text("nope", "english")
    with pytest.raises(ValueError, match="unknown language condition"):
        bank.text("bank_t01", "french")
    with pytest.raises(ValueError, match="unknown side"):
        bank.items_for_side("middle")


def test_items_for_side_is_sorted_regardless_of_file_order(tmp_path):
    """The permutation pick derives is a function of this sequence, so
    re-sorting the CSV must not change which argument an agent posts."""
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    forward = _write_bank(tmp_path / "a", [_row("b_x"), _row("b_a"), _row("b_m", side="con")])
    reversed_file = _write_bank(tmp_path / "b", [_row("b_a"), _row("b_x"), _row("b_m", side="con")])

    assert BankManager(forward).items_for_side("pro") == ["b_a", "b_x"]
    assert BankManager(reversed_file).items_for_side("pro") == ["b_a", "b_x"]

    a, b = BankManager(forward), BankManager(reversed_file)
    assert a.pick("agent_0000", 1, "pro", _sender_rng("agent_0000")) == b.pick(
        "agent_0000", 1, "pro", _sender_rng("agent_0000")
    )


# --- picking: the no-repeat cycle ----------------------------------------


def _sequence(bank: BankManager, sender_id: str, side: str, rounds: int, seed: int = 42):
    """What `sender_id` posts on `side` over `rounds` consecutive posts."""
    picked = []
    for turn in range(1, rounds + 1):
        picked.append(bank.pick(sender_id, turn, side, _sender_rng(sender_id, seed)))
        bank.record_post(sender_id, side)
    return picked


def test_no_repeat_within_a_cycle():
    bank = BankManager(FIXTURE_BANK)
    items = bank.items_for_side("pro")
    picked = _sequence(bank, "agent_0000", "pro", rounds=len(items))
    assert sorted(picked) == items


def test_the_cycle_restarts_after_every_item_is_used(tmp_path):
    """With four items on a side, three full cycles must be three
    permutations, each covering every item exactly once."""
    rows = [_row(f"p{i}", side="pro") for i in range(4)] + [_row("c1", side="con")]
    bank = BankManager(_write_bank(tmp_path, rows))
    items = bank.items_for_side("pro")

    picked = _sequence(bank, "agent_0000", "pro", rounds=12)
    for cycle in range(3):
        assert sorted(picked[cycle * 4 : (cycle + 1) * 4]) == items


def test_the_two_sides_cycle_independently():
    """An agent that switches sides must not have its con cycle disturbed by
    what it posted while it was pro."""
    bank = BankManager(FIXTURE_BANK)
    bank.pick("agent_0000", 1, "pro", _sender_rng("agent_0000"))
    bank.record_post("agent_0000", "pro")

    reference = BankManager(FIXTURE_BANK)
    assert bank.pick("agent_0000", 2, "con", _sender_rng("agent_0000")) == reference.pick(
        "agent_0000", 2, "con", _sender_rng("agent_0000")
    )


def test_different_senders_do_not_post_in_lockstep(tmp_path):
    """Per-sender order, so two agents on the same side are not reciting the
    bank in unison."""
    rows = [_row(f"p{i}", side="pro") for i in range(4)] + [_row("c1", side="con")]
    path = _write_bank(tmp_path, rows)

    sequences = {
        sender: _sequence(BankManager(path), sender, "pro", rounds=4)
        for sender in ("agent_0000", "agent_0001", "agent_0002", "agent_0003")
    }
    assert len(set(map(tuple, sequences.values()))) > 1


def test_pick_is_idempotent_for_a_fixed_count():
    """pick does not advance anything, so a re-derivation after a crash, or a
    second call in the same round, cannot drift."""
    bank = BankManager(FIXTURE_BANK)
    first = bank.pick("agent_0000", 1, "pro", _sender_rng("agent_0000"))
    again = bank.pick("agent_0000", 1, "pro", _sender_rng("agent_0000"))
    assert first == again
    assert bank.post_counts() == {}


def test_pick_does_not_consume_the_senders_rng():
    """The caller's stream is left where it was found, which is what makes the
    call a pure function of the count rather than of how many times it has
    been invoked."""
    bank = BankManager(FIXTURE_BANK)
    rng = _sender_rng("agent_0000")
    before = rng.getstate()
    bank.pick("agent_0000", 1, "pro", rng)
    assert rng.getstate() == before


def test_the_turn_argument_does_not_change_the_pick():
    """A per-turn derivation would reshuffle every round and the no-repeat
    guarantee would quietly stop being true."""
    bank = BankManager(FIXTURE_BANK)
    picks = {
        bank.pick("agent_0000", turn, "pro", _sender_rng("agent_0000"))
        for turn in (1, 2, 5, 99)
    }
    assert len(picks) == 1


def test_pick_rejects_an_unknown_side_and_an_empty_sender():
    bank = BankManager(FIXTURE_BANK)
    with pytest.raises(ValueError, match="unknown side"):
        bank.pick("agent_0000", 1, "neutral", _sender_rng("agent_0000"))
    with pytest.raises(ValueError, match="non-empty"):
        bank.pick("", 1, "pro", _sender_rng("agent_0000"))


# --- the matched-set property --------------------------------------------


def test_the_same_seed_picks_the_same_items_in_every_language():
    """RQ1's matched sets rest on exactly this.

    Four runs differing only in language_condition must give the same sender
    the same item on the same side at the same point in its cycle. The bank is
    not even told the language when picking -- `text()` is a separate call --
    and this asserts the consequence rather than the structure, so it would
    fail if a language ever reached the draw by some other route.
    """
    sequences = {}
    for language in ("english", "hinglish", "romanized_hindi", "hindi"):
        bank = BankManager(FIXTURE_BANK)
        side_sequence = ["pro", "pro", "con", "pro", "con", "con"]
        picked = []
        for turn, side in enumerate(side_sequence, start=1):
            item = bank.pick("agent_0007", turn, side, _sender_rng("agent_0007"))
            bank.record_post("agent_0007", side)
            # The rendering differs; the item does not.
            assert bank.text(item, language)
            picked.append(item)
        sequences[language] = picked

    assert len(set(map(tuple, sequences.values()))) == 1


def test_a_different_seed_gives_a_different_order(tmp_path):
    """Compared over three cycles, not one.

    Four items admit only 24 orderings, so two seeds landing on the same
    single-cycle permutation is a 1-in-24 coincidence rather than evidence
    that the seed is being ignored -- and seeds 42 and 43 are such a pair.
    Three consecutive cycles make an accidental match ~1 in 14,000, and the
    failure mode this test exists to catch (a derivation that drops the seed)
    would produce identical output at any length.
    """
    rows = [_row(f"p{i}", side="pro") for i in range(4)] + [_row("c1", side="con")]
    path = _write_bank(tmp_path, rows)

    assert _sequence(BankManager(path), "agent_0000", "pro", 12, seed=42) != _sequence(
        BankManager(path), "agent_0000", "pro", 12, seed=43
    )


# --- resume ---------------------------------------------------------------


def test_restored_counts_give_the_same_next_pick(tmp_path):
    """The resume requirement: a run that stopped after three posts and one
    that never stopped must make the same fourth pick."""
    rows = [_row(f"p{i}", side="pro") for i in range(4)] + [_row("c1", side="con")]
    path = _write_bank(tmp_path, rows)

    uninterrupted = BankManager(path)
    _sequence(uninterrupted, "agent_0000", "pro", rounds=3)
    expected = uninterrupted.pick("agent_0000", 4, "pro", _sender_rng("agent_0000"))

    resumed = BankManager(path)
    resumed.restore_post_counts({("agent_0000", "pro"): 3})
    assert resumed.pick("agent_0000", 4, "pro", _sender_rng("agent_0000")) == expected


def test_restored_counts_round_trip_through_post_counts(tmp_path):
    bank = BankManager(FIXTURE_BANK)
    _sequence(bank, "agent_0000", "pro", rounds=3)
    _sequence(bank, "agent_0001", "con", rounds=1)

    resumed = BankManager(FIXTURE_BANK)
    resumed.restore_post_counts(bank.post_counts())
    assert resumed.post_counts() == {("agent_0000", "pro"): 3, ("agent_0001", "con"): 1}


def test_post_counts_returns_a_copy():
    bank = BankManager(FIXTURE_BANK)
    snapshot = bank.post_counts()
    bank.record_post("agent_0000", "pro")
    assert snapshot == {}


def test_restore_rejects_a_bad_side_or_a_negative_count():
    bank = BankManager(FIXTURE_BANK)
    with pytest.raises(ValueError, match="unknown side"):
        bank.restore_post_counts({("agent_0000", "middle"): 1})
    with pytest.raises(ValueError, match="negative"):
        bank.restore_post_counts({("agent_0000", "pro"): -1})
