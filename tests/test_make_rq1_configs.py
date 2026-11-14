"""scripts/make_rq1_configs.py: the campaign's matched sets.

The property that matters is the one the whole design turns on: the four
languages of one seed index differ in exactly three fields, and the seed is
not one of them. A generator that got that wrong would produce a campaign
whose arms are not comparable, and nothing downstream would notice -- which
is why it is asserted here rather than trusted.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from sandbox.config_loader import ConfigLoadError, load_run_config
from scripts import make_rq1_configs as gen

BASE = Path("configs/rq1/base.yaml")
QUESTIONS = Path("configs/rq1/probe_questions.yaml")
LANGUAGES = ("english", "hinglish", "romanized_hindi", "hindi")


@pytest.fixture
def campaign(tmp_path):
    def _make(*extra) -> Path:
        out = tmp_path / "campaign"
        assert gen.main(
            ["--base", str(BASE), "--probe-questions", str(QUESTIONS),
             "--out", str(out), *extra]
        ) == 0
        return out

    return _make


def _manifest(out: Path) -> list[dict]:
    with (out / "manifest.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- the shipped inputs ---------------------------------------------------


def test_the_shipped_base_config_is_a_valid_bank_regime_run():
    """base.yaml is the campaign's single source of every setting. If it stops
    loading, the generator produces nothing and no run can start."""
    run = load_run_config(BASE)
    assert run.regime == "bank"
    assert run.neighbor_strategy == "graph"
    assert run.probe_question and run.bank_path and run.neutral_bank_path
    assert run.max_prompt_tokens and run.max_concurrency


def test_delivery_is_quoted_in_the_base_config():
    """YAML 1.1 reads bare on/off as booleans. The quotes are load-bearing,
    and a well-meaning tidy-up would remove them."""
    raw = yaml.safe_load(BASE.read_text())
    assert isinstance(raw["delivery"], str)


def test_the_shipped_probe_questions_cover_every_arm():
    questions = gen.load_probe_questions(QUESTIONS, list(LANGUAGES))
    assert set(questions) == set(LANGUAGES)
    # Four distinct questions: a file copied without editing one would probe
    # that arm in another arm's language.
    assert len(set(questions.values())) == 4
    assert any("ऀ" <= ch <= "ॿ" for ch in questions["hindi"])


def test_a_missing_probe_question_is_refused(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text("english: Q?\nhinglish: Q?\nhindi: Q?\n", encoding="utf-8")
    with pytest.raises(ValueError, match="romanized_hindi"):
        gen.load_probe_questions(path, list(LANGUAGES))


def test_a_blank_probe_question_is_refused(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        "english: Q?\nhinglish: '  '\nromanized_hindi: Q?\nhindi: Q?\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hinglish"):
        gen.load_probe_questions(path, list(LANGUAGES))


def test_a_probe_question_for_an_unknown_language_is_refused(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        "english: Q?\nhinglish: Q?\nromanized_hindi: Q?\nhindi: Q?\ntamil: Q?\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not language conditions"):
        gen.load_probe_questions(path, list(LANGUAGES))


# --- what gets written ----------------------------------------------------


def test_file_count(campaign):
    out = campaign("--seeds", "3")

    e3 = sorted(out.glob("e3_*.yaml"))
    e4 = sorted(out.glob("e4_*.yaml"))
    single = sorted(out.glob("single_*.yaml"))
    assert len(e3) == 3 * 4
    assert len(e4) == 3 * 4
    assert len(single) == 4
    assert len(list(out.glob("*.yaml"))) == len(e3) + len(e4) + len(single)
    assert (out / "manifest.csv").exists()


def test_the_run_id_is_the_file_stem(campaign):
    out = campaign("--seeds", "2")
    for path in out.glob("*.yaml"):
        assert yaml.safe_load(path.read_text())["run_id"] == path.stem


def test_every_written_config_loads(campaign):
    """Validated by the generator too, so this is a second reader agreeing --
    including that an e4_* file really does carry the neutral bank its
    delivery mode requires."""
    out = campaign("--seeds", "2")
    for path in sorted(out.glob("*.yaml")):
        run = load_run_config(path)
        assert run.regime == "bank"
        if path.stem.startswith("e4_"):
            assert run.delivery == "off" and run.neutral_bank_path
        elif path.stem.startswith("e3_"):
            assert run.delivery == "on" and run.neutral_bank_path is None


def test_the_seed_is_shared_across_languages_and_experiments(campaign):
    """The property the whole design turns on. SeedManager takes the base seed
    and nothing else, so four runs sharing a seed get the same graph, the same
    neighbour draws and the same bank item for the same sender on the same
    side."""
    out = campaign("--seeds", "3", "--base-seed", "7000")

    for seed_index in (1, 2, 3):
        seeds = set()
        for experiment in ("e3", "e4"):
            for language in LANGUAGES:
                path = out / f"{experiment}_{language}_ws_s{seed_index:02d}.yaml"
                config = yaml.safe_load(path.read_text())
                seeds.add(config["seed"])
                assert config["trial_number"] == seed_index
        assert seeds == {7000 + seed_index}


def test_different_seed_indices_get_different_seeds(campaign):
    out = campaign("--seeds", "4", "--base-seed", "100")
    seeds = {
        yaml.safe_load((out / f"e3_english_ws_s{i:02d}.yaml").read_text())["seed"]
        for i in (1, 2, 3, 4)
    }
    assert seeds == {101, 102, 103, 104}


@pytest.mark.parametrize("experiment", ["e3", "e4"])
def test_the_four_languages_differ_in_exactly_three_fields(campaign, experiment):
    """run_id, language_condition and probe_question, and nothing else.

    Not the anchor labels: they are rendered inside English sentences -- the
    probe's scale line and the lean -- so translating them would code-mix the
    frame and leave the four arms holding four different instruments.
    """
    out = campaign("--seeds", "1")
    configs = {
        language: load_run_config(
            out / f"{experiment}_{language}_ws_s01.yaml"
        ).model_dump()
        for language in LANGUAGES
    }
    baseline = configs["english"]
    for language, config in configs.items():
        differing = {k for k in baseline if baseline[k] != config[k]}
        assert differing <= {"run_id", "language_condition", "probe_question"}, language

    assert len({c["probe_question"] for c in configs.values()}) == 4
    assert len({c["stance_low_label"] for c in configs.values()}) == 1
    assert len({c["stance_high_label"] for c in configs.values()}) == 1


def test_e3_and_e4_differ_only_in_delivery_and_its_consequences(campaign):
    out = campaign("--seeds", "1")
    on = load_run_config(out / "e3_english_ws_s01.yaml").model_dump()
    off = load_run_config(out / "e4_english_ws_s01.yaml").model_dump()
    assert {k for k in on if on[k] != off[k]} == {
        "delivery",
        "run_id",
        "neutral_bank_path",
    }


def test_the_single_reader_configs_keep_the_neutral_bank(campaign):
    """calibrate_tau reads the control's own side-free messages, so that τ is
    the baseline for what the control actually does."""
    out = campaign("--seeds", "1")
    for language in LANGUAGES:
        run = load_run_config(out / f"single_{language}.yaml")
        assert run.language_condition == language
        assert run.neutral_bank_path
        assert run.run_id == f"single_{language}"


def test_the_single_reader_configs_differ_in_the_same_three_fields(campaign):
    out = campaign("--seeds", "1")
    configs = {
        language: load_run_config(out / f"single_{language}.yaml").model_dump()
        for language in LANGUAGES
    }
    baseline = configs["english"]
    for language, config in configs.items():
        differing = {k for k in baseline if baseline[k] != config[k]}
        assert differing <= {"run_id", "language_condition", "probe_question"}, language


def test_the_single_reader_configs_are_not_seeded_per_trial(campaign):
    """There is one calibration, not one per seed index, so these sit at the
    base seed rather than borrowing seed index 1's number."""
    out = campaign("--seeds", "3", "--base-seed", "500")
    assert yaml.safe_load((out / "single_english.yaml").read_text())["seed"] == 500


def test_everything_except_the_generated_fields_is_copied_verbatim(campaign):
    out = campaign("--seeds", "1")
    base = yaml.safe_load(BASE.read_text())
    written = yaml.safe_load((out / "e3_hindi_ws_s01.yaml").read_text())
    for field, value in base.items():
        if field in gen.GENERATED_FIELDS or field == "neutral_bank_path":
            continue
        assert written[field] == value, field


def test_a_generated_config_carries_no_stale_commit_hash(campaign):
    """Resolved from the repo at load time. A hash in a file would claim a
    provenance it does not have."""
    out = campaign("--seeds", "1")
    assert "git_commit_hash" not in yaml.safe_load(
        (out / "e3_english_ws_s01.yaml").read_text()
    )


def test_a_generated_file_says_it_is_generated(campaign):
    out = campaign("--seeds", "1")
    assert (out / "e3_english_ws_s01.yaml").read_text().startswith("# GENERATED by")


# --- the manifest ---------------------------------------------------------


def test_the_manifest_lists_every_file_in_run_order(campaign):
    out = campaign("--seeds", "2")
    rows = _manifest(out)

    runs = [r for r in rows if r["experiment"] != "single"]
    assert len(runs) == 2 * 2 * 4
    assert len(rows) == len(runs) + 4
    # Seed index, then experiment, then language -- the order run_campaign.py
    # runs them in, so a campaign cut short leaves complete matched sets.
    assert [(r["seed_index"], r["experiment"]) for r in runs] == [
        ("1", "e3")] * 4 + [("1", "e4")] * 4 + [("2", "e3")] * 4 + [("2", "e4")] * 4
    for row in rows:
        assert (out / f"{row['run_id']}.yaml").exists()


def test_the_manifest_seed_matches_the_config(campaign):
    out = campaign("--seeds", "2", "--base-seed", "3000")
    for row in _manifest(out):
        config = yaml.safe_load((out / f"{row['run_id']}.yaml").read_text())
        assert int(row["seed"]) == config["seed"]
        assert row["language"] == config["language_condition"]


# --- restrictions ---------------------------------------------------------


def test_one_language_can_be_generated_alone(campaign):
    out = campaign("--seeds", "1", "--languages", "hindi")
    assert sorted(p.stem for p in out.glob("*.yaml")) == [
        "e3_hindi_ws_s01", "e4_hindi_ws_s01", "single_hindi"
    ]


def test_zero_seeds_is_refused(tmp_path):
    with pytest.raises(ValueError, match=">= 1"):
        gen.generate(
            base_path=BASE,
            probe_questions_path=QUESTIONS,
            out_dir=tmp_path / "c",
            languages=["english"],
            seeds=0,
            base_seed=1,
        )


def test_a_broken_base_config_is_reported_once_naming_the_base(tmp_path):
    bad = tmp_path / "base.yaml"
    raw = yaml.safe_load(BASE.read_text())
    raw["alpha"] = 5.0
    bad.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigLoadError) as excinfo:
        gen.generate(
            base_path=bad,
            probe_questions_path=QUESTIONS,
            out_dir=tmp_path / "c",
            languages=["english"],
            seeds=1,
            base_seed=1,
        )
    assert excinfo.value.path == bad
    assert not (tmp_path / "c").exists()


# --- the shipped campaign directory ---------------------------------------


SHIPPED = Path("configs/rq1/campaign")


def test_the_shipped_campaign_directory_is_in_sync_with_the_generator(tmp_path):
    """The checked-in campaign must be what the current generator and base
    config produce. A base config edited without re-running the generator
    would leave a campaign nobody could reproduce."""
    out = tmp_path / "campaign"
    assert gen.main(
        ["--base", str(BASE), "--probe-questions", str(QUESTIONS), "--out", str(out)]
    ) == 0

    produced = {p.name: p.read_text(encoding="utf-8") for p in out.iterdir()}
    shipped = {p.name: p.read_text(encoding="utf-8") for p in SHIPPED.iterdir()}
    assert set(produced) == set(shipped)
    for name, text in produced.items():
        assert shipped[name] == text, name
