"""scripts/run_e1.py: one argument, one reader, two probes.

Driven through the real entry point against the mocked gateway, like
test_calibrate_tau.py. The scripted readings key off the prompt rather than
call order, for the same reason.
"""

from __future__ import annotations

import json

import pytest

from sandbox.bank_manager import BankManager
from sandbox.single_exposure import load_trials
from scripts import run_e1
from tests.factories import make_logprob_response

BANK = BankManager("data/bank/test_bank.csv")
ITEM_IDS = BANK.all_item_ids()
MEMORY_MARKER = "Earlier in this discussion:"


@pytest.fixture
def harness_gateway(monkeypatch, bank_gateway):
    monkeypatch.setattr(
        "sandbox.single_exposure.ModelGateway", lambda *a, **k: bank_gateway
    )
    return bank_gateway


def _run(config_dir, runs_dir, *extra) -> int:
    argv = ["--configs", str(config_dir), "--runs-dir", str(runs_dir), *extra]
    if "--languages" not in argv:
        found = sorted(
            p.stem.removeprefix("single_") for p in config_dir.glob("single_*.yaml")
        )
        argv += ["--languages", *found]
    return run_e1.main(argv)


def _rows(runs_dir, subdir="e1") -> list[dict]:
    return load_trials(runs_dir / subdir / "trials.jsonl")


# --- the trial grid -------------------------------------------------------


def test_one_trial_per_item_persona_and_lean(tmp_path, single_configs, harness_gateway):
    config_dir = single_configs(("english", "hinglish"))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "2", "6") == 0

    rows = _rows(runs_dir)
    # 4 bank items, 4 personas, 2 leans, 2 languages.
    assert len(rows) == len(ITEM_IDS) * 4 * 2 * 2
    assert {r["item_id"] for r in rows} == set(ITEM_IDS)
    assert {r["lean"] for r in rows} == {2, 6}
    assert all(r["status"] == "ok" for r in rows)
    # arg_id, strength and side come off the bank, not off the row's key.
    for row in rows:
        assert row["arg_id"] == BANK.arg_id_of(row["item_id"])
        assert row["strength"] == BANK.strength_of(row["item_id"])
        assert row["side"] == BANK.side_of(row["item_id"])


def test_the_baseline_probe_is_measured_once_per_persona_and_lean(
    tmp_path, single_configs, harness_gateway
):
    """P0 does not depend on the item: nothing has been read when it is
    taken, so one baseline serves all four items that reader is shown."""
    config_dir = single_configs(("english",))
    assert _run(config_dir, tmp_path / "out", "--leans", "4") == 0

    n_trials = len(ITEM_IDS) * 4 * 1
    assert harness_gateway.score_next_token.await_count == 4 + n_trials


def test_the_default_trial_makes_no_reply_call(tmp_path, single_configs, harness_gateway):
    """The reader only reads. A reply is a second thing happening between the
    two readings, and E1's default trial isolates reading from it."""
    config_dir = single_configs(("english",))
    assert _run(config_dir, tmp_path / "out", "--leans", "4") == 0

    assert harness_gateway.generate.await_count == 0
    rows = _rows(tmp_path / "out")
    assert all(r["with_reply"] is False for r in rows)
    assert all(r["reply_seed"] is None for r in rows)


def test_with_reply_adds_a_seeded_reply_in_its_own_folder(
    tmp_path, single_configs, harness_gateway
):
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4", "--with-reply") == 0

    n_trials = len(ITEM_IDS) * 4
    assert harness_gateway.generate.await_count == n_trials
    assert not (runs_dir / "e1" / "trials.jsonl").exists()
    rows = _rows(runs_dir, "e1_with_reply")
    assert len(rows) == n_trials
    assert all(r["with_reply"] is True for r in rows)
    assert all(isinstance(r["reply_seed"], int) and r["reply_seed"] >= 0 for r in rows)


def test_the_reply_seed_is_the_same_in_every_arm(tmp_path, single_configs, harness_gateway):
    config_dir = single_configs(("english", "hindi"))
    runs_dir = tmp_path / "out"
    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "1", "--with-reply") == 0

    seeds = {r["language"]: r["reply_seed"] for r in _rows(runs_dir, "e1_with_reply")}
    assert len(seeds) == 2 and len(set(seeds.values())) == 1


# --- toward ---------------------------------------------------------------


def _reading_for(x_after: float):
    """A scripted P1 that lands on a given x, with P0 at x = 0."""

    def probe(prompt_text):
        if MEMORY_MARKER not in prompt_text:
            return make_logprob_response(probabilities={4: 1.0})
        mean = 4 + 3 * x_after
        whole = int(round(mean))
        assert abs(mean - whole) < 1e-9, "pick an x that lands on a scale point"
        return make_logprob_response(probabilities={whole: 1.0})

    return probe


@pytest.mark.parametrize(
    ("x_after", "expected_pro", "expected_con"),
    [
        # The reader moved UP. A pro item (the high end) was moved toward; a
        # con item was moved away from.
        (1 / 3, pytest.approx(1 / 3), pytest.approx(-1 / 3)),
        # And down.
        (-1 / 3, pytest.approx(-1 / 3), pytest.approx(1 / 3)),
    ],
)
def test_toward_is_signed_by_the_items_side(
    tmp_path, single_configs, harness_gateway, x_after, expected_pro, expected_con
):
    """pro is the high end of the scale, so a pro item that moved a reader up
    and a con item that moved one down are both movement TOWARD what was
    argued -- which is what lets the two sides be pooled in one model."""
    harness_gateway.score_next_token.side_effect = _reading_for(x_after)
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4") == 0

    rows = _rows(runs_dir)
    pro = [r for r in rows if r["side"] == "pro"]
    con = [r for r in rows if r["side"] == "con"]
    assert pro and con
    for row in pro:
        assert row["dx"] == pytest.approx(x_after)
        assert row["toward"] == expected_pro
    for row in con:
        assert row["dx"] == pytest.approx(x_after)
        assert row["toward"] == expected_con


def test_toward_is_null_on_a_failed_trial(tmp_path, single_configs, harness_gateway):
    harness_gateway.score_next_token.return_value = make_logprob_response(
        probabilities={5: 1.0}, off_scale=0.9
    )
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "2") == 0

    rows = _rows(runs_dir)
    assert len(rows) == 2
    assert all(r["status"] == "failed" for r in rows)
    assert all(r["toward"] is None and r["dx"] is None for r in rows)


# --- the rendering --------------------------------------------------------


@pytest.mark.parametrize("language", ["english", "hinglish", "romanized_hindi", "hindi"])
def test_the_item_in_the_prompt_is_this_arms_rendering(
    tmp_path, single_configs, harness_gateway, language
):
    """The one thing that differs between the four arms. A trial that showed
    the English text under a Hindi condition would produce a perfectly
    plausible null result for that arm."""
    config_dir = single_configs((language,))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "1") == 0

    item_id = ITEM_IDS[0]
    exposure = [
        json.loads(line)
        for line in (runs_dir / "e1" / "prompts.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if MEMORY_MARKER in json.loads(line)["prompt_text"]
    ]
    assert len(exposure) == 1
    prompt = exposure[0]["prompt_text"]
    assert BANK.text(item_id, language) in prompt
    for other in ("english", "hinglish", "romanized_hindi", "hindi"):
        if other == language:
            continue
        assert BANK.text(item_id, other) not in prompt


def test_no_number_reaches_a_reader(tmp_path, single_configs, harness_gateway):
    """test_bank.csv is digit-free in every language column precisely so this
    audit can tell a leaked position number from an argument's own text."""
    from sandbox.config_loader import load_run_config

    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"
    assert _run(config_dir, runs_dir, "--leans", "2", "6", "--limit", "4") == 0

    run = load_run_config(config_dir / "single_english.yaml")
    allowed = (
        "Answer with a single digit from 1 to 7 and nothing else.",
        f"On this scale, 1 means {run.stance_low_label}, "
        f"and 7 means {run.stance_high_label}.",
    )
    for line in (runs_dir / "e1" / "prompts.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        record = json.loads(line)
        body = record["prompt_text"]
        for phrase in allowed:
            body = body.replace(phrase, "")
        assert not any(ch.isdigit() for ch in body), record["prompt_text"]


# --- resumability ---------------------------------------------------------


def test_a_second_pass_reruns_nothing_that_succeeded(
    tmp_path, single_configs, harness_gateway
):
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4") == 0
    first = harness_gateway.score_next_token.await_count
    rows = _rows(runs_dir)

    assert _run(config_dir, runs_dir, "--leans", "4") == 0

    assert harness_gateway.score_next_token.await_count == first
    assert _rows(runs_dir) == rows


def test_a_resumed_pass_finishes_what_the_first_did_not(
    tmp_path, single_configs, harness_gateway
):
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "3") == 0
    assert len(_rows(runs_dir)) == 3
    after_first = harness_gateway.score_next_token.await_count

    assert _run(config_dir, runs_dir, "--leans", "4") == 0

    total = len(ITEM_IDS) * 4
    assert len(_rows(runs_dir)) == total
    # 13 trials left, plus one baseline per persona still appearing among
    # them -- all four do, since the three finished trials were all persona
    # 0 to 2 of one item. Baselines are not persisted, so a resumed pass
    # re-measures the ones its pending trials need; the trials themselves,
    # which are what the calls were spent on, are not re-run.
    assert harness_gateway.score_next_token.await_count == after_first + (total - 3) + 4


def test_the_two_variants_do_not_share_an_output_file(
    tmp_path, single_configs, harness_gateway
):
    """A read-only trial and a read-then-reply trial have the same key, so one
    file holding both would let the resume rule treat them as the same trial
    and the two effects would be estimated from one column."""
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "1") == 0
    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "1", "--with-reply") == 0

    plain = _rows(runs_dir)
    with_reply = _rows(runs_dir, "e1_with_reply")
    assert len(plain) == len(with_reply) == 1
    assert plain[0]["key"] == with_reply[0]["key"]
    assert plain[0]["with_reply"] is False and with_reply[0]["with_reply"] is True


# --- arg_id_of ------------------------------------------------------------


def test_arg_id_of_names_the_argument_not_the_row():
    assert BANK.arg_id_of("bank_t01") == "arg_cost"
    assert len({BANK.arg_id_of(i) for i in ITEM_IDS}) == len(ITEM_IDS)


def test_arg_id_of_raises_for_an_unknown_item():
    with pytest.raises(KeyError, match="unknown bank item"):
        BANK.arg_id_of("bank_nope")
