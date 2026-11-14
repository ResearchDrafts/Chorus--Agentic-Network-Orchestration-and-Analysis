"""scripts/check_comprehension.py, probe_scorer.score_choices, and exclusions.

The comprehension check is what separates "this rendering is less
persuasive" from "the model cannot read this rendering", so the tests that
matter most here are the ones about the pass rule: both orders, and a
position preference must not be able to produce a pass.
"""

from __future__ import annotations

import csv
import math
import pathlib

import pytest

from sandbox.bank_manager import BankManager, BankValidationError, read_exclusions
from sandbox.config_loader import ConfigLoadError, load_run_config
from sandbox.probe_scorer import ChoiceScore, ProbeMassTooLow, score_choices
from sandbox.single_exposure import build_comprehension_prompt, load_trials
from scripts import check_comprehension as cc
from tests.factories import make_logprob_response

BANK = BankManager("data/bank/test_bank.csv")
ITEM_IDS = BANK.all_item_ids()


@pytest.fixture
def harness_gateway(monkeypatch, bank_gateway):
    monkeypatch.setattr(
        "sandbox.single_exposure.ModelGateway", lambda *a, **k: bank_gateway
    )
    return bank_gateway


def _letters(probabilities: dict[str, float], off_scale: float = 0.0):
    """A next-token response over letters rather than digits."""
    total = sum(probabilities.values())
    scale = (1.0 - off_scale) / total
    pairs = [(k, math.log(v * scale)) for k, v in probabilities.items() if v > 0]
    if off_scale > 0:
        pairs.append((" the", math.log(off_scale)))
    return make_logprob_response(top_logprobs=pairs)


def _run(config_dir, runs_dir, *extra) -> int:
    argv = ["--configs", str(config_dir), "--runs-dir", str(runs_dir), *extra]
    if "--languages" not in argv:
        found = sorted(
            p.stem.removeprefix("single_") for p in config_dir.glob("single_*.yaml")
        )
        argv += ["--languages", *found]
    return cc.main(argv)


def _rows(runs_dir) -> list[dict]:
    return load_trials(runs_dir / "comprehension" / "trials.jsonl")


def _summary(runs_dir) -> list[dict]:
    with (runs_dir / "comprehension" / "summary.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- score_choices --------------------------------------------------------


def test_score_choices_renormalizes_over_the_support():
    score = score_choices(
        [("A", math.log(0.6)), ("B", math.log(0.2)), (" because", math.log(0.2))],
        ("A", "B"),
        0.5,
    )
    assert score.mass == pytest.approx(0.8)
    assert score.probability("A") == pytest.approx(0.75)
    assert score.probability("B") == pytest.approx(0.25)
    assert sum(score.dist) == pytest.approx(1.0)


def test_score_choices_sums_the_same_answer_from_two_tokenizers():
    """" A" and "A" are one answer. Without this, half the providers would
    appear to put no mass on the support at all."""
    score = score_choices(
        [("A", math.log(0.3)), (" A", math.log(0.4)), ("B", math.log(0.3))],
        ("A", "B"),
        0.5,
    )
    assert score.probability("A") == pytest.approx(0.7)


def test_score_choices_refuses_below_the_mass_floor():
    with pytest.raises(ProbeMassTooLow) as excinfo:
        score_choices(
            [("A", math.log(0.2)), (" maybe", math.log(0.8))], ("A", "B"), 0.5
        )
    assert excinfo.value.mass == pytest.approx(0.2)


def test_score_choices_is_case_sensitive():
    """A lower-case "a" is the start of an English sentence far more often
    than it is an answer, so folding case would launder prose into a
    confident choice -- exactly the off-support mass the floor exists to
    catch."""
    with pytest.raises(ProbeMassTooLow):
        score_choices([("a", math.log(0.9)), ("B", math.log(0.1))], ("A", "B"), 0.5)


def test_score_choices_rejects_a_degenerate_support():
    with pytest.raises(ValueError, match="at least 2 choices"):
        score_choices([("A", 0.0)], ("A",), 0.5)
    with pytest.raises(ValueError, match="distinct"):
        score_choices([("A", 0.0)], ("A", "A"), 0.5)


def test_score_choices_distinguishes_nothing_to_read_from_reading_nothing():
    with pytest.raises(ValueError, match="nothing to score"):
        score_choices([], ("A", "B"), 0.5)


def test_score_choices_rejects_an_impossible_floor():
    with pytest.raises(ValueError, match="mass_floor"):
        score_choices([("A", 0.0)], ("A", "B"), 0.0)


def test_probability_of_an_unscored_choice_raises():
    score = ChoiceScore(choices=("A", "B"), dist=[0.5, 0.5], mass=1.0)
    with pytest.raises(KeyError, match="not one of the scored choices"):
        score.probability("C")


# --- the prompt -----------------------------------------------------------


def test_the_prompt_asks_in_english_and_carries_no_persona():
    prompt = build_comprehension_prompt("कुछ लोग कहते हैं", "against", "for")
    assert prompt.startswith("A user posted: कुछ लोग कहते हैं")
    assert "Which view does this post support?" in prompt
    assert "A: against" in prompt and "B: for" in prompt
    assert "Answer with a single letter, A or B, and nothing else." in prompt
    assert "You are:" not in prompt
    assert "hold the view" not in prompt


# --- the pass rule --------------------------------------------------------


def test_correct_letter_follows_the_side_and_the_order():
    """pro is the high end of the scale. Getting this backwards would invert
    the whole pass rate while still producing a plausible file."""
    assert cc.correct_letter("pro", cc.ORDER_HIGH_FIRST) == "A"
    assert cc.correct_letter("con", cc.ORDER_HIGH_FIRST) == "B"
    assert cc.correct_letter("pro", cc.ORDER_LOW_FIRST) == "B"
    assert cc.correct_letter("con", cc.ORDER_LOW_FIRST) == "A"


def test_an_item_read_correctly_in_both_orders_passes(
    tmp_path, single_configs, harness_gateway
):
    def answer(prompt_text):
        high = "A: fares should stay subsidised" in prompt_text
        pro = BANK.text(ITEM_IDS[0], "english") in prompt_text or (
            BANK.text(ITEM_IDS[1], "english") in prompt_text
        )
        right = "A" if pro == high else "B"
        wrong = "B" if right == "A" else "A"
        return _letters({right: 0.9, wrong: 0.1})

    harness_gateway.score_next_token.side_effect = answer
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir) == 0

    records = _summary(runs_dir)
    assert len(records) == len(ITEM_IDS)
    assert all(r["passed"] == "True" for r in records)
    assert (runs_dir / "comprehension" / "failed_items.txt").read_text().strip().endswith(
        "contain."
    )


def test_a_standing_preference_for_one_letter_cannot_produce_a_pass(
    tmp_path, single_configs, harness_gateway
):
    """The whole reason both orders are run. A model that always says A is
    right about half the items in one order and wrong about them in the
    other, so every item fails."""
    harness_gateway.score_next_token.side_effect = lambda _: _letters({"A": 0.95, "B": 0.05})
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir) == 0

    records = _summary(runs_dir)
    assert len(records) == len(ITEM_IDS)
    assert all(r["passed"] == "False" for r in records)
    assert cc.failed_items(cc.summarize(_rows(runs_dir))) == ITEM_IDS


def test_both_orders_are_asked_for_every_item_and_language(
    tmp_path, single_configs, harness_gateway
):
    harness_gateway.score_next_token.side_effect = lambda _: _letters({"A": 0.6, "B": 0.4})
    config_dir = single_configs(("english", "hindi"))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir) == 0

    rows = _rows(runs_dir)
    assert len(rows) == len(ITEM_IDS) * 2 * 2
    assert {r["order"] for r in rows} == set(cc.ORDERS)
    assert harness_gateway.generate.await_count == 0


def test_a_hairline_majority_is_a_pass_and_a_tie_is_not(
    tmp_path, single_configs, harness_gateway
):
    """"More than half" is the weakest claim that is still a claim: a coin
    flip is exactly half, so 0.5 cannot pass."""
    assert cc.PASS_THRESHOLD == 0.5
    records = cc.summarize(
        [
            _trial_row(cc.ORDER_HIGH_FIRST, 0.5000001),
            _trial_row(cc.ORDER_LOW_FIRST, 0.9),
        ]
    )
    assert records[0]["passed"] is True
    records = cc.summarize(
        [_trial_row(cc.ORDER_HIGH_FIRST, 0.5), _trial_row(cc.ORDER_LOW_FIRST, 0.9)]
    )
    assert records[0]["passed"] is False


def _trial_row(order: str, p_correct: float | None, status: str = "ok") -> dict:
    return {
        "key": f"comp|english|{ITEM_IDS[0]}|{order}",
        "language": "english",
        "item_id": ITEM_IDS[0],
        "arg_id": "arg_cost",
        "strength": "strong",
        "side": "pro",
        "order": order,
        "expected": "A",
        "p_correct": p_correct,
        "p_a": p_correct,
        "mass": 1.0,
        "status": status,
    }


def test_an_item_whose_call_failed_does_not_pass():
    """Unchecked is not the same as fine. Treating it as a pass is how a
    stimulus nobody could read reaches a campaign."""
    records = cc.summarize(
        [
            _trial_row(cc.ORDER_HIGH_FIRST, 0.95),
            _trial_row(cc.ORDER_LOW_FIRST, None, status="failed"),
        ]
    )
    assert records[0]["passed"] is False
    assert records[0]["n_failed_orders"] == 1
    assert records[0]["p_correct_low_first"] is None


def test_an_item_failing_in_one_language_is_listed(tmp_path, single_configs, harness_gateway):
    """Any rather than all: the bank holds one argument rendered four ways, so
    keeping it in the arms that could read it would make the arms differ in
    which arguments they contain."""
    target = ITEM_IDS[2]

    def answer(prompt_text):
        if BANK.text(target, "hindi") in prompt_text:
            return _letters({"A": 0.5, "B": 0.5})
        high = "A: fares should stay subsidised" in prompt_text
        pro = any(
            BANK.text(i, lang) in prompt_text
            for i in ITEM_IDS
            for lang in ("english", "hindi")
            if BANK.side_of(i) == "pro"
        )
        right = "A" if pro == high else "B"
        return _letters({right: 0.9, ("B" if right == "A" else "A"): 0.1})

    harness_gateway.score_next_token.side_effect = answer
    config_dir = single_configs(("english", "hindi"))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir) == 0

    text = (runs_dir / "comprehension" / "failed_items.txt").read_text()
    assert target in text
    assert "failed in: hindi" in text
    # And the file it writes is a valid exclusion list, comments and all.
    assert read_exclusions(runs_dir / "comprehension" / "failed_items.txt") == [target]


def test_a_failed_call_is_a_failed_row_not_a_dropped_one(
    tmp_path, single_configs, harness_gateway
):
    harness_gateway.score_next_token.side_effect = lambda _: _letters(
        {"A": 0.6, "B": 0.4}, off_scale=0.8
    )
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir) == 0

    rows = _rows(runs_dir)
    assert len(rows) == len(ITEM_IDS) * 2
    assert all(r["status"] == "failed" for r in rows)
    assert all(r["p_correct"] is None and r["mass"] is None for r in rows)


def test_resumability(tmp_path, single_configs, harness_gateway):
    harness_gateway.score_next_token.side_effect = lambda _: _letters({"A": 0.6, "B": 0.4})
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--limit", "2") == 0
    assert len(_rows(runs_dir)) == 2
    first = harness_gateway.score_next_token.await_count

    assert _run(config_dir, runs_dir) == 0
    assert len(_rows(runs_dir)) == len(ITEM_IDS) * 2
    assert harness_gateway.score_next_token.await_count == first + (len(ITEM_IDS) * 2 - 2)


def test_the_check_does_not_honour_its_own_exclusion_list(
    tmp_path, single_configs, harness_gateway
):
    """It is the check that decides what belongs in that file. Reading it
    would mean an item excluded once could never be re-tested after a prompt
    or a model change."""
    harness_gateway.score_next_token.side_effect = lambda _: _letters({"A": 0.6, "B": 0.4})
    exclusions = tmp_path / "drop.txt"
    exclusions.write_text(f"{ITEM_IDS[0]}\n", encoding="utf-8")
    config_dir = single_configs(("english",), exclude_items_path=str(exclusions))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir) == 0

    assert {r["item_id"] for r in _rows(runs_dir)} == set(ITEM_IDS)


# --- exclude_items_path ---------------------------------------------------


def test_the_bank_drops_the_listed_items(tmp_path):
    path = tmp_path / "drop.txt"
    path.write_text(f"# a comment\n\n{ITEM_IDS[1]}\n", encoding="utf-8")

    bank = BankManager("data/bank/test_bank.csv", exclude_items_path=path)

    assert ITEM_IDS[1] not in bank.all_item_ids()
    assert len(bank.all_item_ids()) == len(ITEM_IDS) - 1
    with pytest.raises(KeyError):
        bank.text(ITEM_IDS[1], "english")
    # And the cycle is over what is left, so nothing can post a dropped item.
    assert ITEM_IDS[1] not in bank.items_for_side(BANK.side_of(ITEM_IDS[1]))


def test_an_unknown_excluded_id_raises(tmp_path):
    """A typo'd exclusion silently keeps the item in the bank, which is the
    opposite of what the config asked for and leaves no trace."""
    path = tmp_path / "drop.txt"
    path.write_text("bank_t99\n", encoding="utf-8")
    with pytest.raises(BankValidationError, match="unknown item_id 'bank_t99'"):
        BankManager("data/bank/test_bank.csv", exclude_items_path=path)


def test_excluding_a_whole_side_raises(tmp_path):
    path = tmp_path / "drop.txt"
    con = [i for i in ITEM_IDS if BANK.side_of(i) == "con"]
    path.write_text("\n".join(con) + "\n", encoding="utf-8")
    with pytest.raises(BankValidationError, match="leaves no 'con' items"):
        BankManager("data/bank/test_bank.csv", exclude_items_path=path)


def test_every_problem_in_an_exclusion_list_is_reported_at_once(tmp_path):
    path = tmp_path / "drop.txt"
    con = [i for i in ITEM_IDS if BANK.side_of(i) == "con"]
    path.write_text("\n".join([*con, *con[:1], "bank_t99"]) + "\n", encoding="utf-8")
    with pytest.raises(BankValidationError) as excinfo:
        BankManager("data/bank/test_bank.csv", exclude_items_path=path)
    joined = "; ".join(excinfo.value.errors)
    assert "unknown item_id" in joined
    assert "duplicate item_id" in joined
    assert "leaves no 'con' items" in joined


def test_a_missing_exclusion_file_raises(tmp_path):
    with pytest.raises(BankValidationError, match="exclusion file not found"):
        BankManager("data/bank/test_bank.csv", exclude_items_path=tmp_path / "nope.txt")


def test_exclusion_happens_after_validation(tmp_path):
    """An excluded item does not make a broken row acceptable: the next run
    may not exclude it."""
    broken = tmp_path / "bank.csv"
    broken.write_text(
        "item_id,arg_id,strength,side,english,hinglish,romanized_hindi,hindi\n"
        "bank_a,arg_a,strong,pro,a,a,a,a\n"
        "bank_b,arg_b,weak,con,b,b,b,\n",
        encoding="utf-8",
    )
    drop = tmp_path / "drop.txt"
    drop.write_text("bank_b\n", encoding="utf-8")
    with pytest.raises(BankValidationError, match="empty hindi text"):
        BankManager(broken, exclude_items_path=drop)


def test_a_config_with_a_broken_exclusion_list_is_rejected_at_load(
    tmp_path, single_configs
):
    drop = tmp_path / "drop.txt"
    drop.write_text("bank_t99\n", encoding="utf-8")
    config_dir = single_configs(("english",), exclude_items_path=str(drop))
    with pytest.raises(ConfigLoadError, match="unknown item_id"):
        load_run_config(config_dir / "single_english.yaml")


def test_a_config_with_a_valid_exclusion_list_loads(tmp_path, single_configs):
    drop = tmp_path / "drop.txt"
    drop.write_text(f"{ITEM_IDS[1]}\n", encoding="utf-8")
    config_dir = single_configs(("english",), exclude_items_path=str(drop))
    run = load_run_config(config_dir / "single_english.yaml")
    assert run.exclude_items_path == str(drop)


def test_no_exclusion_path_means_the_whole_bank():
    assert BankManager("data/bank/test_bank.csv").all_item_ids() == ITEM_IDS


async def test_a_bank_run_never_posts_an_excluded_item(tmp_path, bank_gateway):
    """The field reaches the orchestrator, not only the loader: a setting the
    schema accepts and the wiring drops would be recorded in run_config.json
    as an exclusion the run did not apply."""
    import yaml

    from sandbox.models import Interaction
    from sandbox.simulation_orchestrator import build_orchestrator

    drop = tmp_path / "drop.txt"
    dropped = "bank_t02"
    drop.write_text(f"{dropped}\n", encoding="utf-8")

    raw = yaml.safe_load(pathlib.Path("configs/rq1/smoke.yaml").read_text())
    raw.update(M=8, N=3, K=2, graph_degree=4, run_id="excluded_run",
               persona_pool_id="pool_20", exclude_items_path=str(drop))
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    run = load_run_config(config)
    await build_orchestrator(run, bank_gateway, runs_dir=tmp_path).run()

    rows = [
        Interaction.model_validate_json(line)
        for line in (tmp_path / run.run_id / "interactions.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    posted = {r.posted_item_id for r in rows if r.posted_item_id}
    read = {item for r in rows for item in r.items_read}
    assert posted and dropped not in posted
    assert dropped not in read


def test_read_exclusions_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "drop.txt"
    path.write_text("# header\n\nbank_a\nbank_b  # why\n\n", encoding="utf-8")
    assert read_exclusions(path) == ["bank_a", "bank_b"]
