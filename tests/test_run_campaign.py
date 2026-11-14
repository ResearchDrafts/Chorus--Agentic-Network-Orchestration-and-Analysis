"""scripts/run_campaign.py: ordering, skipping and stopping.

The ordering test is the one that matters. A campaign that runs out of time
mid-way must leave complete matched sets, and whether it does is entirely a
property of the order -- there is nothing to check afterwards that would
reveal a campaign ordered by language instead.

The launcher is injected so the tests drive the real ordering, skip and
failure logic. Two of them inject a launcher that builds a real orchestrator
against the mocked gateway, so the skip rule and the failed-row count are
exercised against output a real run produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sandbox.config_loader import load_run_config
from sandbox.simulation_orchestrator import build_orchestrator
from scripts import make_rq1_configs as gen
from scripts import run_campaign as rc

BASE = Path("configs/rq1/base.yaml")
QUESTIONS = Path("configs/rq1/probe_questions.yaml")


@pytest.fixture
def campaign(tmp_path):
    def _make(seeds: int = 2, **kwargs) -> Path:
        out = tmp_path / "campaign"
        gen.generate(
            base_path=BASE,
            probe_questions_path=QUESTIONS,
            out_dir=out,
            languages=kwargs.get("languages", ["english", "hinglish", "romanized_hindi", "hindi"]),
            seeds=seeds,
            base_seed=1000,
        )
        return out

    return _make


@pytest.fixture
def recorder():
    """A launcher that records what it was asked to run and does nothing."""
    seen: list[str] = []

    async def launcher(run, runs_dir):
        seen.append(run.run_id)

    launcher.seen = seen
    return launcher


def _no_server_check(run):
    return None


def _real_launcher(gateway):
    """A launcher that actually runs the orchestrator against a mock gateway.

    Used where the test needs real output on disk -- the completed-run skip
    reads run_config.json, and the failed-row count reads
    interactions.jsonl.
    """

    async def launcher(run, runs_dir):
        await build_orchestrator(run, gateway, runs_dir=runs_dir).run()

    return launcher


# --- ordering -------------------------------------------------------------


async def test_a_whole_matched_set_finishes_before_the_next_seed_index(
    tmp_path, campaign, recorder
):
    ran = await rc.run_campaign(
        rc.discover(campaign(seeds=2), ["e3", "e4"]),
        tmp_path / "runs",
        launcher=recorder,
        server_check=_no_server_check,
    )

    assert ran == [
        "e3_english_ws_s01",
        "e3_hinglish_ws_s01",
        "e3_romanized_hindi_ws_s01",
        "e3_hindi_ws_s01",
        "e4_english_ws_s01",
        "e4_hinglish_ws_s01",
        "e4_romanized_hindi_ws_s01",
        "e4_hindi_ws_s01",
        "e3_english_ws_s02",
        "e3_hinglish_ws_s02",
        "e3_romanized_hindi_ws_s02",
        "e3_hindi_ws_s02",
        "e4_english_ws_s02",
        "e4_hinglish_ws_s02",
        "e4_romanized_hindi_ws_s02",
        "e4_hindi_ws_s02",
    ]


def test_english_runs_first_within_a_set():
    """The reference arm every other is compared against, so a reader
    watching the campaign sees each set start from its own baseline."""
    parsed = [rc.parse_run_id(f"e3_{lang}_ws_s01") for lang in
              ("hindi", "english", "romanized_hindi", "hinglish")]
    ordered = sorted(parsed, key=rc.sort_key)
    assert [p["language"] for p in ordered] == [
        "english", "hinglish", "romanized_hindi", "hindi"
    ]


def test_the_order_does_not_follow_how_experiments_was_typed(campaign):
    """The run order is a property of the campaign, not of the flag."""
    out = campaign(seeds=1)
    assert rc.discover(out, ["e4", "e3"]) == rc.discover(out, ["e3", "e4"])
    assert rc.discover(out, ["e3", "e4"])[0].stem == "e3_english_ws_s01"


def test_experiments_filters(campaign):
    out = campaign(seeds=2)
    only_e4 = rc.discover(out, ["e4"])
    assert len(only_e4) == 8
    assert all(p.stem.startswith("e4_") for p in only_e4)


def test_the_single_reader_configs_are_not_campaign_runs(campaign):
    """They live in the same directory as the campaign they belong to, and
    this script must not try to run them."""
    out = campaign(seeds=1)
    assert (out / "single_english.yaml").exists()
    assert all("single" not in p.stem for p in rc.discover(out, ["e3", "e4"]))


def test_a_seed_index_above_nine_still_sorts_numerically(campaign):
    out = campaign(seeds=11)
    stems = [p.stem for p in rc.discover(out, ["e3"])]
    assert stems[0] == "e3_english_ws_s01"
    assert stems[-1] == "e3_hindi_ws_s11"
    # s10 before s11, and s02 before s10 -- string order would put s10 first.
    assert stems.index("e3_english_ws_s02") < stems.index("e3_english_ws_s10")


def test_parse_run_id_handles_a_language_with_an_underscore():
    parsed = rc.parse_run_id("e4_romanized_hindi_ws_s07")
    assert parsed == {
        "experiment": "e4",
        "language": "romanized_hindi",
        "structure": "ws",
        "seed_index": 7,
    }


def test_parse_run_id_ignores_what_is_not_a_campaign_run():
    assert rc.parse_run_id("single_english") is None
    assert rc.parse_run_id("manifest") is None
    assert rc.parse_run_id("e5_english_ws_s01") is None


# --- max-runs -------------------------------------------------------------


async def test_max_runs_stops_after_n(tmp_path, campaign, recorder, capsys):
    ran = await rc.run_campaign(
        rc.discover(campaign(seeds=2), ["e3", "e4"]),
        tmp_path / "runs",
        launcher=recorder,
        server_check=_no_server_check,
        max_runs=3,
    )

    assert ran == ["e3_english_ws_s01", "e3_hinglish_ws_s01", "e3_romanized_hindi_ws_s01"]
    # And it says the set it cut into is incomplete, since that is the one
    # thing the ordering guarantee cannot protect against.
    assert "incomplete matched set" in capsys.readouterr().out


async def test_max_runs_larger_than_the_campaign_runs_everything(
    tmp_path, campaign, recorder
):
    configs = rc.discover(campaign(seeds=1), ["e3"])
    ran = await rc.run_campaign(
        configs,
        tmp_path / "runs",
        launcher=recorder,
        server_check=_no_server_check,
        max_runs=99,
    )
    assert len(ran) == len(configs)


# --- skipping and resuming ------------------------------------------------


async def test_a_completed_run_is_skipped(tmp_path, campaign, bank_gateway):
    """Read off the run's own run_config.json: the run directory is already
    the authority on what a run did."""
    out = campaign(seeds=1, languages=["english"])
    runs_dir = tmp_path / "runs"
    configs = rc.discover(out, ["e3"])

    first = await rc.run_campaign(
        configs, runs_dir,
        launcher=_real_launcher(bank_gateway), server_check=_no_server_check,
    )
    assert first == ["e3_english_ws_s01"]
    status = json.loads(
        (runs_dir / "e3_english_ws_s01" / "run_config.json").read_text()
    )["status"]
    assert status == "completed"

    second = await rc.run_campaign(
        configs, runs_dir,
        launcher=_real_launcher(bank_gateway), server_check=_no_server_check,
    )
    assert second == []


async def test_a_run_with_a_checkpoint_but_no_completed_status_is_run(
    tmp_path, campaign, recorder, capsys
):
    """Resuming is the orchestrator's own behaviour. This only has to not
    skip it, and to say where it is picking up."""
    out = campaign(seeds=1, languages=["english"])
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "e3_english_ws_s01"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "latest.json").write_text(
        json.dumps({"run_id": "e3_english_ws_s01", "last_completed_turn": 2,
                    "agent_snapshot": []})
    )
    (run_dir / "run_config.json").write_text(json.dumps({"status": "failed_partial"}))

    ran = await rc.run_campaign(
        rc.discover(out, ["e3"]), runs_dir,
        launcher=recorder, server_check=_no_server_check,
    )

    assert ran == ["e3_english_ws_s01"]
    assert "resuming after turn 2" in capsys.readouterr().out


async def test_a_torn_run_config_is_re_run_not_skipped(tmp_path, campaign, recorder):
    out = campaign(seeds=1, languages=["english"])
    runs_dir = tmp_path / "runs"
    (runs_dir / "e3_english_ws_s01").mkdir(parents=True)
    (runs_dir / "e3_english_ws_s01" / "run_config.json").write_text('{"status": "comp')

    ran = await rc.run_campaign(
        rc.discover(out, ["e3"]), runs_dir,
        launcher=recorder, server_check=_no_server_check,
    )
    assert ran == ["e3_english_ws_s01"]


# --- failures -------------------------------------------------------------


async def test_one_failing_run_stops_the_campaign_and_names_it(
    tmp_path, campaign, recorder
):
    configs = rc.discover(campaign(seeds=2), ["e3", "e4"])

    async def failing(run, runs_dir):
        if run.run_id == "e3_romanized_hindi_ws_s01":
            raise RuntimeError("server disconnected")
        recorder.seen.append(run.run_id)

    with pytest.raises(rc.CampaignRunFailed) as excinfo:
        await rc.run_campaign(
            configs, tmp_path / "runs",
            launcher=failing, server_check=_no_server_check,
        )

    assert excinfo.value.run_id == "e3_romanized_hindi_ws_s01"
    assert "server disconnected" in str(excinfo.value)
    assert "re-run the same campaign command" in str(excinfo.value)
    # It stopped there: the two runs before it happened, nothing after did.
    assert recorder.seen == ["e3_english_ws_s01", "e3_hinglish_ws_s01"]


async def test_an_unreachable_server_stops_the_campaign_before_any_run(
    tmp_path, campaign, recorder
):
    with pytest.raises(rc.CampaignRunFailed, match="cannot reach"):
        await rc.run_campaign(
            rc.discover(campaign(seeds=1), ["e3"]), tmp_path / "runs",
            launcher=recorder, server_check=lambda run: "cannot reach the server",
        )
    assert recorder.seen == []


async def test_the_server_is_checked_once_per_endpoint(tmp_path, campaign, recorder):
    """An HTTP round trip per run would say nothing the first did not, and a
    campaign is forty runs."""
    calls = []

    def check(run):
        calls.append(run.api_base)
        return None

    await rc.run_campaign(
        rc.discover(campaign(seeds=2), ["e3", "e4"]), tmp_path / "runs",
        launcher=recorder, server_check=check,
    )
    assert len(calls) == 1


async def test_the_failed_row_count_is_reported_after_each_run(
    tmp_path, campaign, bank_gateway, capsys
):
    """The one number that says whether a run that completed is usable: a run
    whose probes failed the mass floor completes normally and produces a
    dataset with holes in it."""
    from tests.factories import make_logprob_response

    bank_gateway.score_next_token.return_value = make_logprob_response(
        probabilities={5: 1.0}, off_scale=0.9
    )
    out = campaign(seeds=1, languages=["english"])

    await rc.run_campaign(
        rc.discover(out, ["e3"]), tmp_path / "runs",
        launcher=_real_launcher(bank_gateway), server_check=_no_server_check,
    )

    printed = capsys.readouterr().out
    # M=8 agents x (3 rounds + round 0), every probe below the floor.
    assert "32 failed_logged_null row(s)" in printed
    assert "<-- check this" in printed


async def test_a_clean_run_reports_no_failed_rows(
    tmp_path, campaign, bank_gateway, capsys
):
    out = campaign(seeds=1, languages=["english"])
    await rc.run_campaign(
        rc.discover(out, ["e3"]), tmp_path / "runs",
        launcher=_real_launcher(bank_gateway), server_check=_no_server_check,
    )
    printed = capsys.readouterr().out
    assert "0 failed_logged_null row(s)" in printed
    assert "<-- check this" not in printed


# --- the entry point ------------------------------------------------------


def test_an_empty_campaign_directory_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    assert rc.main([str(tmp_path / "empty"), "--runs-dir", str(tmp_path / "runs")]) == 1


def test_a_missing_campaign_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        rc.discover(tmp_path / "nope", ["e3"])


def test_the_launcher_and_the_check_default_to_run_simulations_own():
    """Reused, not reimplemented: a campaign run must be the same thing as
    `python run_simulation.py` on the same config."""
    import inspect

    import run_simulation

    defaults = inspect.signature(rc.run_campaign).parameters
    assert defaults["launcher"].default is run_simulation.launch
    assert defaults["server_check"].default is run_simulation.check_server


def test_the_shipped_campaign_is_discoverable_in_order():
    configs = rc.discover(Path("configs/rq1/campaign"), ["e3", "e4"])
    assert len(configs) == 40
    assert configs[0].stem == "e3_english_ws_s01"
    assert configs[-1].stem == "e4_hindi_ws_s05"
    for path in configs:
        assert load_run_config(path).regime == "bank"
