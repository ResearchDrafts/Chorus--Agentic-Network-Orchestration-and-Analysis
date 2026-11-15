"""scripts/calibrate_starts.py: what probe reading a starter-post mix produces.

The pure-function tests (with_starter_counts, summarize, suggest) work
against hand-built rows so the arithmetic can be checked by hand; the single
end-to-end test drives the real `main(argv)` against the mocked gateway, the
same convention test_calibrate_tau.py uses.
"""

from __future__ import annotations

import csv
import json

import pytest

from sandbox.bank_manager import BankManager
from scripts import calibrate_starts as cs
from tests.factories import make_run

STARTER_PATH = "data/bank/test_starter.csv"
STARTERS = BankManager(STARTER_PATH)

BANK_RUN = dict(
    regime="bank",
    bank_path="data/bank/test_bank.csv",
    probe_question="Should the fare subsidy continue?",
    topic="whether the fare subsidy should continue",
    stance_low_label="strongly against the fare subsidy",
    stance_high_label="strongly in favour of the fare subsidy",
    starter_bank_path=STARTER_PATH,
)


# --- with_starter_counts ----------------------------------------------------


def test_with_starter_counts_sets_posts_mode_and_every_point():
    run = make_run(**BANK_RUN, start_mode="none")
    out = cs.with_starter_counts(run, 2, 3)
    assert out.start_mode == "posts"
    assert out.starter_counts == {p: (2, 3) for p in (1, 2, 3, 4, 5, 6, 7)}


def test_with_starter_counts_leaves_the_bank_regime_run_otherwise_unchanged():
    run = make_run(**BANK_RUN)
    out = cs.with_starter_counts(run, 1, 1)
    assert out.bank_path == run.bank_path
    assert out.probe_question == run.probe_question


# --- the pre-flight bank-size check ------------------------------------------


def test_check_starter_bank_covers_the_sweep_raises_when_too_small():
    """test_starter.csv has 3 items per side; MAX_PER_SIDE is 4."""
    run = make_run(**BANK_RUN)
    worst = cs.with_starter_counts(run, cs.MAX_PER_SIDE, cs.MAX_PER_SIDE)
    with pytest.raises(ValueError, match="needs at least 4"):
        cs.check_starter_bank_covers_the_sweep(worst)


def test_check_starter_bank_covers_the_sweep_passes_with_enough_items(tmp_path):
    big_bank = tmp_path / "big_starter.csv"
    _write_starter_bank(big_bank, n_con=4, n_pro=4)
    run = make_run(**{**BANK_RUN, "starter_bank_path": str(big_bank)})
    worst = cs.with_starter_counts(run, cs.MAX_PER_SIDE, cs.MAX_PER_SIDE)
    cs.check_starter_bank_covers_the_sweep(worst)  # must not raise


def _write_starter_bank(path, n_con: int, n_pro: int) -> None:
    header = ["item_id", "arg_id", "strength", "side",
              "english", "hinglish", "romanized_hindi", "hindi"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for side, n in (("con", n_con), ("pro", n_pro)):
            for i in range(n):
                item_id = f"starter_{side}{i:02d}"
                text = f"a {side} argument number {i}, with no digit in it"
                writer.writerow([item_id, item_id, "strong", side, text, text, text, text])


# --- summarize ---------------------------------------------------------------


LANGUAGES = ("english", "hinglish", "romanized_hindi", "hindi")


def _row(language, n_con, n_pro, persona_index, mean, status="ok"):
    return {
        "key": cs.mix_key(language, n_con, n_pro, persona_index),
        "language": language, "n_con": n_con, "n_pro": n_pro,
        "persona_index": persona_index, "mean": mean,
        "x": None, "conf": None, "mass": None, "status": status,
    }


def test_summarize_pools_across_languages_and_keeps_each_languages_own_mean():
    rows = [
        _row("english", 1, 0, 0, 2.0),
        _row("english", 1, 0, 1, 4.0),
        _row("hindi", 1, 0, 0, 6.0),
    ]
    summary = cs.summarize(rows, list(LANGUAGES))
    row = next(r for r in summary if (r["n_con"], r["n_pro"]) == (1, 0))
    assert row["n_ok"] == 3
    assert row["pooled_mean"] == pytest.approx(4.0)  # (2 + 4 + 6) / 3
    assert row["mean_english"] == pytest.approx(3.0)  # (2 + 4) / 2
    assert row["mean_hindi"] == pytest.approx(6.0)
    assert row["mean_hinglish"] is None


def test_summarize_counts_a_failed_row_without_including_it_in_the_mean():
    rows = [
        _row("english", 0, 1, 0, 5.0),
        _row("english", 0, 1, 1, None, status="failed"),
    ]
    summary = cs.summarize(rows, list(LANGUAGES))
    row = next(r for r in summary if (r["n_con"], r["n_pro"]) == (0, 1))
    assert row["n_ok"] == 1 and row["n_failed"] == 1
    assert row["pooled_mean"] == pytest.approx(5.0)


def test_summarize_covers_every_mix_even_with_no_data():
    summary = cs.summarize([], list(LANGUAGES))
    assert len(summary) == len(cs.MIXES)
    assert all(row["pooled_mean"] is None for row in summary)


# --- suggest -----------------------------------------------------------------


def _summary_row(n_con, n_pro, pooled_mean, **lang_means):
    row = {"n_con": n_con, "n_pro": n_pro, "pooled_mean": pooled_mean,
           "n_ok": 1, "n_failed": 0, "pooled_sd": 0.0}
    for language in LANGUAGES:
        row[f"mean_{language}"] = lang_means.get(language)
    return row


def test_suggest_picks_the_mix_closest_to_each_target():
    summary = [
        _summary_row(1, 0, 2.0, english=1.8, hindi=2.2),
        _summary_row(2, 0, 1.4, english=1.3, hindi=1.5),
    ]
    targets = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    results = cs.suggest(summary, targets, ["english", "hindi"])
    lean_1 = next(r for r in results if r["lean"] == 1)
    assert (lean_1["n_con"], lean_1["n_pro"]) == (2, 0)  # 1.4 is closer to 1.5 than 2.0
    assert lean_1["language_gap"] == pytest.approx(0.2)  # 1.5 - 1.3


def test_suggest_rejects_the_wrong_number_of_targets():
    summary = [_summary_row(0, 0, 4.0)]
    with pytest.raises(ValueError, match="exactly 6 targets"):
        cs.suggest(summary, [1.5, 2.5], ["english"])


def test_suggest_rejects_when_nothing_has_a_pooled_mean():
    summary = cs.summarize([], list(LANGUAGES))
    with pytest.raises(ValueError, match="run the sweep first"):
        cs.suggest(summary, list(cs.DEFAULT_TARGETS), list(LANGUAGES))


# --- the YAML block -----------------------------------------------------


def test_write_starter_counts_yaml_is_complete_and_pins_the_midpoint(tmp_path):
    suggestions = [
        {"lean": lean, "target": 0.0, "n_con": lean, "n_pro": 7 - lean,
         "pooled_mean": 0.0, "language_gap": None}
        for lean in cs.SUGGEST_LEANS
    ]
    path = tmp_path / "starter_counts.yaml"
    text = cs.write_starter_counts_yaml(path, suggestions)
    assert path.read_text(encoding="utf-8") == text
    assert "4: [1, 1]" in text  # the module default, not a suggested value
    for lean in cs.SUGGEST_LEANS:
        assert f"{lean}: [{lean}, {7 - lean}]" in text


# --- end to end, against the mocked gateway ----------------------------------


@pytest.fixture
def harness_gateway(monkeypatch, bank_gateway):
    monkeypatch.setattr(
        "sandbox.single_exposure.ModelGateway", lambda *a, **k: bank_gateway
    )
    return bank_gateway


def test_main_writes_starts_csv_for_every_mix(tmp_path, single_configs, harness_gateway):
    big_bank = tmp_path / "big_starter.csv"
    _write_starter_bank(big_bank, n_con=4, n_pro=4)
    config_dir = single_configs(
        ("english",), start_mode="posts", starter_bank_path=str(big_bank),
    )
    runs_dir = tmp_path / "out"

    assert cs.main([
        "--configs", str(config_dir), "--runs-dir", str(runs_dir),
        "--languages", "english", "--sample", "2",
    ]) == 0

    starts = list(csv.DictReader((runs_dir / "calibration" / "starts.csv").open()))
    assert len(starts) == len(cs.MIXES)
    assert all(row["n_ok"] == "2" for row in starts)
    # mock_gateway's probe is a constant mean=5.0, so every mix pools to it.
    assert all(float(row["pooled_mean"]) == pytest.approx(5.0) for row in starts)

    trials = (runs_dir / "calibration" / "starts_trials.jsonl").read_text().splitlines()
    assert len(trials) == len(cs.MIXES) * 2  # 2 sampled personas


def test_main_is_resumable(tmp_path, single_configs, harness_gateway):
    big_bank = tmp_path / "big_starter.csv"
    _write_starter_bank(big_bank, n_con=4, n_pro=4)
    config_dir = single_configs(
        ("english",), start_mode="posts", starter_bank_path=str(big_bank),
    )
    runs_dir = tmp_path / "out"
    argv = ["--configs", str(config_dir), "--runs-dir", str(runs_dir),
            "--languages", "english", "--sample", "2"]

    assert cs.main(argv) == 0
    before = (runs_dir / "calibration" / "starts_trials.jsonl").read_text()
    assert cs.main(argv) == 0
    after = (runs_dir / "calibration" / "starts_trials.jsonl").read_text()
    assert before == after  # nothing re-run, nothing re-appended


def test_main_too_small_a_starter_bank_is_refused(tmp_path, single_configs, harness_gateway):
    config_dir = single_configs(
        ("english",), start_mode="posts", starter_bank_path=STARTER_PATH,
    )
    with pytest.raises(ValueError, match="needs at least 4"):
        cs.main([
            "--configs", str(config_dir), "--runs-dir", str(tmp_path / "out"),
            "--languages", "english", "--sample", "1",
        ])


def test_main_suggest_writes_a_yaml_block(tmp_path, single_configs, harness_gateway):
    big_bank = tmp_path / "big_starter.csv"
    _write_starter_bank(big_bank, n_con=4, n_pro=4)
    config_dir = single_configs(
        ("english",), start_mode="posts", starter_bank_path=str(big_bank),
    )
    runs_dir = tmp_path / "out"
    assert cs.main([
        "--configs", str(config_dir), "--runs-dir", str(runs_dir),
        "--languages", "english", "--sample", "1",
        "--suggest", "1.5,2.5,3.5,4.5,5.5,6.5",
    ]) == 0
    text = (runs_dir / "calibration" / "starter_counts.yaml").read_text()
    assert text.startswith("starter_counts:")
    assert "4: [1, 1]" in text
