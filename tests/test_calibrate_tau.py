"""scripts/calibrate_tau.py: the τ the whole campaign is judged against.

Every test here drives the real entry point -- `main(argv)` -- against the
mocked gateway, so what is exercised is the script a researcher runs rather
than a reimplementation of it. The one thing faked is the LLM call.

The scripted-reading tests key the mock on the PROMPT rather than on call
order. A baseline probe and an exposure probe are distinguishable by whether
the prompt carries a memory block, which is a fact about the instrument;
call order is a fact about asyncio's scheduling, and a test that depended on
it would break for a reason that has nothing to do with calibration.
"""

from __future__ import annotations

import json

import pytest

from sandbox.bank_manager import NeutralBank
from sandbox.config_loader import load_run_config
from sandbox.model_gateway import FatalGatewayError
from sandbox.single_exposure import load_trials, percentile
from scripts import calibrate_tau
from tests.factories import make_backend_response, make_logprob_response

NEUTRAL_IDS = NeutralBank("data/bank/test_neutral.csv").ids()
MEMORY_MARKER = "Earlier in this discussion:"


@pytest.fixture
def harness_gateway(monkeypatch, bank_gateway):
    """Every SingleReaderHarness in the process gets this gateway."""
    monkeypatch.setattr(
        "sandbox.single_exposure.ModelGateway", lambda *a, **k: bank_gateway
    )
    return bank_gateway


def _run(config_dir, runs_dir, *extra) -> int:
    """The script, over whichever arms the fixture wrote configs for.

    --languages is derived from the directory rather than spelled out in
    every test, so a test that asks for two arms gets two arms and nothing
    has to repeat the list.
    """
    argv = ["--configs", str(config_dir), "--runs-dir", str(runs_dir),
            "--min-ok-trials", "1", *extra]
    if "--languages" not in argv:
        found = sorted(p.stem.removeprefix("single_") for p in config_dir.glob("single_*.yaml"))
        argv += ["--languages", *found]
    return calibrate_tau.main(argv)


def _rows(runs_dir) -> list[dict]:
    return load_trials(runs_dir / "calibration" / "trials.jsonl")


def _tau(runs_dir) -> dict:
    return json.loads((runs_dir / "calibration" / "tau.json").read_text())


# --- the trial grid -------------------------------------------------------


def test_one_trial_per_persona_lean_and_neutral_message(
    tmp_path, single_configs, harness_gateway
):
    """persona x lean x neutral message, per language, and nothing dropped."""
    config_dir = single_configs(("english", "hindi"))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "2", "4") == 0

    rows = _rows(runs_dir)
    # 4 personas in test_pool, 2 leans, 3 neutral messages, 2 languages.
    assert len(rows) == 4 * 2 * len(NEUTRAL_IDS) * 2
    assert {r["language"] for r in rows} == {"english", "hindi"}
    assert {r["lean"] for r in rows} == {2, 4}
    assert {r["neutral_id"] for r in rows} == set(NEUTRAL_IDS)
    assert {r["persona_index"] for r in rows} == {0, 1, 2, 3}
    assert all(r["status"] == "ok" for r in rows)


def test_the_baseline_probe_is_measured_once_per_persona_and_lean(
    tmp_path, single_configs, harness_gateway
):
    """P0 does not depend on which neutral message a trial uses -- nothing has
    been read when it is taken -- so measuring it per trial would be three
    identical calls where one does.

    Counted as calls, not as a cache hit: the saving is the point, and only
    the call count shows it.
    """
    config_dir = single_configs(("english",))
    assert _run(config_dir, tmp_path / "out", "--leans", "4", "6") == 0

    n_personas, n_leans, n_messages = 4, 2, len(NEUTRAL_IDS)
    n_trials = n_personas * n_leans * n_messages
    # One probe per (persona, lean), plus one per trial.
    assert harness_gateway.score_next_token.await_count == n_personas * n_leans + n_trials
    assert harness_gateway.generate.await_count == n_trials


def test_limit_takes_the_first_n_trials_of_each_arm(
    tmp_path, single_configs, harness_gateway
):
    config_dir = single_configs(("english", "hinglish"))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "2") == 0

    rows = _rows(runs_dir)
    assert len(rows) == 4  # two arms x two trials
    # The first two of (persona 0, lean 4, each message).
    assert [r["neutral_id"] for r in rows if r["language"] == "english"] == NEUTRAL_IDS[:2]


# --- resumability ---------------------------------------------------------


def test_a_second_pass_reruns_nothing_that_succeeded(
    tmp_path, single_configs, harness_gateway
):
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4") == 0
    first_calls = harness_gateway.score_next_token.await_count
    first_rows = _rows(runs_dir)

    assert _run(config_dir, runs_dir, "--leans", "4") == 0

    assert harness_gateway.score_next_token.await_count == first_calls
    assert harness_gateway.generate.await_count == len(first_rows)
    assert _rows(runs_dir) == first_rows


def test_a_resumed_pass_finishes_the_trials_the_first_pass_did_not(
    tmp_path, single_configs, harness_gateway
):
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "2") == 0
    assert len(_rows(runs_dir)) == 2

    assert _run(config_dir, runs_dir, "--leans", "4") == 0

    rows = _rows(runs_dir)
    assert len(rows) == 4 * len(NEUTRAL_IDS)
    assert all(r["status"] == "ok" for r in rows)
    # The two already-done keys were not re-dispatched: 12 trials, of which
    # 10 ran on the second pass.
    assert harness_gateway.generate.await_count == 2 + 10


# --- the threshold itself -------------------------------------------------


# One persona, one lean, three messages: P0 once, then three trials whose
# second reading is scripted per message. Chosen so |dx| and |dconf| can both
# be worked out by hand below.
SCRIPTED = {
    NEUTRAL_IDS[0]: {5: 0.8, 3: 0.2},   # mean 4.6 -> x = 0.2, conf = 0.8
    NEUTRAL_IDS[1]: {4: 1.0},           # mean 4.0 -> x = 0.0, conf = 1.0
    NEUTRAL_IDS[2]: {6: 0.6, 7: 0.4},   # mean 6.4 -> x = 0.8, conf = 1.0
}
BASELINE = {4: 1.0}                     # mean 4.0 -> x = 0.0, conf = 1.0


def _scripted_probe(prompt_text):
    if MEMORY_MARKER not in prompt_text:
        return make_logprob_response(probabilities=BASELINE)
    for neutral_id, distribution in SCRIPTED.items():
        if NeutralBank("data/bank/test_neutral.csv").text(neutral_id, "english") in prompt_text:
            return make_logprob_response(probabilities=distribution)
    raise AssertionError(f"no scripted reading for prompt:\n{prompt_text}")


def test_tau_is_the_percentile_of_absolute_dx_over_ok_trials(
    tmp_path, single_configs, harness_gateway
):
    """Hand-computed, against the definition in single_exposure.percentile.

        |dx|    = [0.2, 0.0, 0.8]  -> sorted [0.0, 0.2, 0.8]
        position = (3 - 1) * 95/100 = 1.9
        tau      = 0.2 * 0.1 + 0.8 * 0.9 = 0.74

        |dconf| = [0.2, 0.0, 0.0]  -> sorted [0.0, 0.0, 0.2]
        tau_conf = 0.0 * 0.1 + 0.2 * 0.9 = 0.18
    """
    harness_gateway.score_next_token.side_effect = _scripted_probe
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "3") == 0

    arm = _tau(runs_dir)["languages"]["english"]
    assert arm["n_ok"] == 3 and arm["n_failed"] == 0
    assert arm["tau"] == pytest.approx(0.74)
    assert arm["tau_conf"] == pytest.approx(0.18)
    # And the same number the shared helper gives, so the script is not
    # computing a percentile of its own.
    assert arm["tau"] == pytest.approx(
        percentile([abs(r["dx"]) for r in _rows(runs_dir)], 95.0)
    )


def test_tau_json_records_the_percentile_the_commit_and_the_masses(
    tmp_path, single_configs, harness_gateway
):
    harness_gateway.score_next_token.side_effect = _scripted_probe
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"
    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "3") == 0

    tau = _tau(runs_dir)
    assert tau["percentile"] == 95.0
    assert tau["git_commit_hash"]
    assert tau["generated_at_utc"]
    # --limit was used, so this is a plumbing check and says so.
    assert tau["smoke"] is True
    arm = tau["languages"]["english"]
    assert arm["mass0_mean"] == pytest.approx(1.0)
    assert arm["mass1_min"] == pytest.approx(1.0)


def test_a_different_percentile_is_honoured(tmp_path, single_configs, harness_gateway):
    harness_gateway.score_next_token.side_effect = _scripted_probe
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"
    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "3",
                "--percentile", "50") == 0
    # The median of [0.0, 0.2, 0.8].
    assert _tau(runs_dir)["languages"]["english"]["tau"] == pytest.approx(0.2)


# --- failures -------------------------------------------------------------


def test_a_probe_below_the_mass_floor_is_a_failed_row_and_not_in_tau(
    tmp_path, single_configs, harness_gateway
):
    """The trial is kept, with null readings, and excluded from the threshold.

    Dropping it instead would make the failure invisible, and an invisible
    failure in a calibration is the worst kind: τ would be computed over the
    trials that happened to answer, which is not a random subset of the
    trials that were asked.
    """
    def probe(prompt_text):
        if MEMORY_MARKER not in prompt_text:
            return make_logprob_response(probabilities=BASELINE)
        text = NeutralBank("data/bank/test_neutral.csv").text(NEUTRAL_IDS[1], "english")
        if text in prompt_text:
            # 0.2 on the scale, under the config's 0.5 floor.
            return make_logprob_response(probabilities={5: 1.0}, off_scale=0.8)
        return make_logprob_response(probabilities=SCRIPTED[NEUTRAL_IDS[0]])

    harness_gateway.score_next_token.side_effect = probe
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "2") == 0

    rows = _rows(runs_dir)
    failed = [r for r in rows if r["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["neutral_id"] == NEUTRAL_IDS[1]
    assert failed[0]["x1"] is None and failed[0]["dx"] is None
    # x0 survives: the baseline WAS measured, and saying otherwise would
    # throw away a reading that was paid for.
    assert failed[0]["x0"] == pytest.approx(0.0)

    arm = _tau(runs_dir)["languages"]["english"]
    assert (arm["n_ok"], arm["n_failed"]) == (1, 1)
    assert arm["tau"] == pytest.approx(0.2)  # the one ok trial's |dx|


def test_a_fatal_reply_error_is_a_failed_row(tmp_path, single_configs, harness_gateway):
    harness_gateway.generate.side_effect = FatalGatewayError("content policy")
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    # Every trial failed, so there is no threshold to write and the script
    # says so rather than writing one. The rows are still on disk.
    with pytest.raises(calibrate_tau.NotEnoughTrials):
        _run(config_dir, runs_dir, "--leans", "4", "--limit", "1")

    rows = _rows(runs_dir)
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert rows[0]["x1"] is None
    # The prompt is still on disk: a failed call is exactly the one worth
    # being able to read.
    prompts = (runs_dir / "calibration" / "prompts.jsonl").read_text().splitlines()
    assert any(json.loads(line)["call"] == "reply" for line in prompts)


def test_a_failed_baseline_spends_no_reply_call(tmp_path, single_configs, harness_gateway):
    """Three trials share one baseline. If it cannot be read, none of their
    differences can be computed, so none of them is worth a reply call."""
    harness_gateway.score_next_token.return_value = make_logprob_response(
        probabilities={5: 1.0}, off_scale=0.9
    )
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    with pytest.raises(calibrate_tau.NotEnoughTrials):
        _run(config_dir, runs_dir, "--leans", "4", "--limit", "3")

    rows = _rows(runs_dir)
    assert len(rows) == 3 and all(r["status"] == "failed" for r in rows)
    assert all(r["x0"] is None for r in rows)
    assert harness_gateway.generate.await_count == 0
    assert harness_gateway.score_next_token.await_count == 1


def test_too_few_ok_trials_refuses_to_write_a_threshold(
    tmp_path, single_configs, harness_gateway
):
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"

    with pytest.raises(calibrate_tau.NotEnoughTrials, match="too few trials"):
        calibrate_tau.main(
            ["--configs", str(config_dir), "--runs-dir", str(runs_dir),
             "--languages", "english", "--leans", "4", "--limit", "3"]
        )

    # The trials themselves are still on disk -- the refusal is about the
    # threshold, and throwing away the calls already paid for would make a
    # resumed pass start over.
    assert len(_rows(runs_dir)) == 3
    assert not (runs_dir / "calibration" / "tau.json").exists()


# --- the instrument -------------------------------------------------------


def test_no_number_reaches_a_reader(tmp_path, single_configs, harness_gateway):
    """The bank regime's central claim, audited against this script's own log.

    The probe's own scale digits are the one exception, exactly as in a
    campaign: the model cannot answer "a single digit from 1 to 7" without
    being told that range.
    """
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"
    assert _run(config_dir, runs_dir, "--leans", "2", "7", "--limit", "4") == 0

    run = load_run_config(config_dir / "single_english.yaml")
    allowed = (
        "Answer with a single digit from 1 to 7 and nothing else.",
        f"On this scale, 1 means {run.stance_low_label}, "
        f"and 7 means {run.stance_high_label}.",
        "Reply to these posts in 1 to 2 sentences.",
    )
    for line in (runs_dir / "calibration" / "prompts.jsonl").read_text().splitlines():
        record = json.loads(line)
        body = record["prompt_text"]
        for phrase in allowed:
            body = body.replace(phrase, "")
        assert not any(ch.isdigit() for ch in body), record["prompt_text"]


def test_the_lean_sentence_is_in_every_prompt(tmp_path, single_configs, harness_gateway):
    config_dir = single_configs(("english",))
    runs_dir = tmp_path / "out"
    assert _run(config_dir, runs_dir, "--leans", "7", "--limit", "1") == 0

    lines = (runs_dir / "calibration" / "prompts.jsonl").read_text().splitlines()
    assert lines
    for line in lines:
        record = json.loads(line)
        assert "You strongly hold the view that fares should stay subsidised." in (
            record["prompt_text"]
        )


def test_the_reply_seed_is_the_same_in_every_arm(tmp_path, single_configs, harness_gateway):
    """Two arms, same trial, same seed: they differ in the text of the prompt
    and not in how the reply to it was sampled."""
    config_dir = single_configs(("english", "hindi"))
    runs_dir = tmp_path / "out"
    assert _run(config_dir, runs_dir, "--leans", "4", "--limit", "1") == 0

    seeds = {r["language"]: r["reply_seed"] for r in _rows(runs_dir)}
    assert len(seeds) == 2
    assert len(set(seeds.values())) == 1
    assert seeds["english"] >= 0


def test_a_missing_neutral_bank_is_refused(tmp_path, single_configs, harness_gateway):
    config_dir = single_configs(("english",), neutral_bank_path=None)
    with pytest.raises(ValueError, match="no neutral_bank_path"):
        _run(config_dir, tmp_path / "out", "--leans", "4", "--limit", "1")


def test_a_config_whose_condition_contradicts_its_filename_is_refused(
    tmp_path, single_configs, harness_gateway
):
    config_dir = single_configs(("english",))
    path = config_dir / "single_hindi.yaml"
    path.write_text((config_dir / "single_english.yaml").read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="declares language_condition"):
        _run(config_dir, tmp_path / "out", "--languages", "hindi", "--leans", "4")


# --- the shared helpers ---------------------------------------------------


def test_percentile_matches_numpy_on_a_known_sequence():
    import numpy as np

    values = [0.0, 0.2, 0.8, 0.1, 0.35]
    for p in (0, 25, 50, 90, 95, 100):
        assert percentile(values, p) == pytest.approx(np.percentile(values, p))


def test_percentile_of_nothing_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="empty sequence"):
        percentile([], 95)


def test_a_single_observation_is_its_own_percentile():
    assert percentile([0.4], 95) == pytest.approx(0.4)


def test_make_backend_response_is_still_the_reply_shape_these_tests_assume():
    """Guards the fixture, not the code: bank_gateway's reply must be prose
    with no STANCE line, or the no-number audit above fails against the
    fixture rather than against the script."""
    assert "STANCE" not in make_backend_response(text="prose only").text
