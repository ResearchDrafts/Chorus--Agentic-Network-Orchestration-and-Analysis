#!/usr/bin/env python3
"""Runs an RQ1 campaign directory, in matched-set order.

    python scripts/run_campaign.py configs/rq1/campaign --runs-dir runs
    python scripts/run_campaign.py configs/rq1/campaign --runs-dir runs --max-runs 2
    python scripts/run_campaign.py configs/rq1/campaign --experiments e3

**Order: seed index, then experiment, then language.** A whole matched set --
four languages of E3, then four of E4, all sharing one seed -- finishes before
the next seed index starts. So a campaign that runs out of GPU time leaves
complete matched sets rather than three arms of a four-arm comparison, and
every seed index that finished is analysable on its own. Ordering by language
first would leave the opposite: five English runs and nothing to compare them
against.

**Safe to re-run.** A run whose `run_config.json` says `status: completed` is
skipped; a run with a checkpoint resumes from it, because that is what
`SimulationOrchestrator.run` already does with one. Nothing here re-implements
either: the per-run work is `run_simulation.launch` and the reachability check
is `run_simulation.check_server`, called through this module so the campaign
and a single `python run_simulation.py` cannot drift apart.

**One fatal failure stops the campaign.** Not the next run, not a summary at
the end: the most likely causes are a server that died and a config that is
wrong for the hardware, and both make every remaining run fail the same way.
The message names the run, so the resume is one command. The runs already
finished keep their output, and re-running the campaign picks up from the one
that failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

if __package__ in (None, ""):  # `python scripts/run_campaign.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_simulation import check_server, launch
from sandbox.config_loader import load_run_config
from sandbox.models import LANGUAGE_CONDITIONS, ExperimentRun

# The campaign's canonical experiment order. Not the order --experiments was
# typed in: the order runs happen in is a property of the campaign, so
# `--experiments e4 e3` selects both and still runs E3 first.
EXPERIMENT_ORDER = ("e3", "e4")

# e3_romanized_hindi_ws_s04 -> ("e3", "romanized_hindi", 4). The language is
# matched greedily because two of the four conditions contain an underscore.
RUN_ID_PATTERN = re.compile(
    rf"^(?P<experiment>{'|'.join(EXPERIMENT_ORDER)})"
    r"_(?P<language>.+)_(?P<structure>[a-z]+)_s(?P<seed_index>\d+)$"
)

FAILED_STATUS = "failed_logged_null"


class CampaignRunFailed(Exception):
    """One run died. The campaign stops, naming it."""

    def __init__(self, run_id: str, cause: BaseException):
        self.run_id = run_id
        self.cause = cause
        super().__init__(
            f"run {run_id!r} failed with {type(cause).__name__}: {cause}\n"
            f"  The campaign stopped here rather than starting the next run: a "
            f"dead server or a config that does not fit the hardware makes every "
            f"remaining run fail the same way.\n"
            f"  The last completed turn of {run_id!r} is checkpointed. Fix the "
            f"cause and re-run the same campaign command -- finished runs are "
            f"skipped and this one resumes."
        )


def parse_run_id(stem: str) -> dict | None:
    """The experiment, language and seed index in a campaign filename.

    None for anything that is not a campaign run -- `single_english.yaml`, a
    `manifest.csv`, a stray note. Returning None rather than raising is what
    lets the single-reader base configs live in the same directory as the
    campaign they belong to without this script trying to run them.
    """
    match = RUN_ID_PATTERN.match(stem)
    if match is None:
        return None
    return {
        "experiment": match["experiment"],
        "language": match["language"],
        "structure": match["structure"],
        "seed_index": int(match["seed_index"]),
    }


def sort_key(parsed: dict) -> tuple:
    """Seed index, then experiment, then language.

    Language ordering follows LANGUAGE_CONDITIONS -- english first, as the
    reference arm -- rather than alphabetically, so a reader watching the
    campaign sees each set start from the baseline it is compared against.
    An unrecognised language sorts last, alphabetically, so a hand-added arm
    runs after the four the comparison is specified over.
    """
    languages = list(LANGUAGE_CONDITIONS)
    language_rank = (
        languages.index(parsed["language"])
        if parsed["language"] in languages
        else len(languages)
    )
    return (
        parsed["seed_index"],
        EXPERIMENT_ORDER.index(parsed["experiment"]),
        language_rank,
        parsed["language"],
    )


def discover(campaign_dir: Path, experiments: list[str]) -> list[Path]:
    """Every campaign config in the directory, in run order.

    Read off the filenames rather than out of manifest.csv, deliberately: a
    config added or removed by hand should change what runs, and a manifest
    that had drifted from the directory would silently run a stale plan. The
    manifest is a record of what was generated; the directory is what exists.
    """
    if not campaign_dir.is_dir():
        raise NotADirectoryError(f"not a campaign directory: {campaign_dir}")
    selected = [e for e in EXPERIMENT_ORDER if e in experiments]
    found = []
    for path in campaign_dir.glob("*.yaml"):
        parsed = parse_run_id(path.stem)
        if parsed is None or parsed["experiment"] not in selected:
            continue
        found.append((sort_key(parsed), path))
    return [path for _, path in sorted(found)]


def completed_status(runs_dir: Path, run_id: str) -> str | None:
    """The status in this run's own run_config.json, or None if absent.

    Read from the output rather than from a campaign-level ledger: the run
    directory is already the authority on what a run did, and a second record
    of the same fact is a second thing that can be wrong.
    """
    path = runs_dir / run_id / "run_config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except json.JSONDecodeError:
        # A torn run_config.json means the run died while writing it, which
        # is a reason to re-run rather than to skip.
        return None


def resume_turn(runs_dir: Path, run_id: str) -> int | None:
    """The last completed turn, if this run has a checkpoint.

    Reported only. Resuming is the orchestrator's own behaviour, and
    duplicating the decision here would be a second place for the two to
    disagree about where a run restarts.
    """
    path = runs_dir / run_id / "checkpoints" / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["last_completed_turn"]
    except (json.JSONDecodeError, KeyError):
        return None


def count_failed_rows(runs_dir: Path, run_id: str) -> int:
    """`failed_logged_null` rows in this run's log.

    Printed after every run because it is the one number that says whether a
    run that "succeeded" is usable: a run whose probes mostly failed the mass
    floor completes normally and produces a dataset with holes in it, and the
    only moment anyone would notice is here.
    """
    path = runs_dir / run_id / "interactions.jsonl"
    if not path.exists():
        return 0
    failed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if json.loads(line).get("api_call_status") == FAILED_STATUS:
            failed += 1
    return failed


async def run_campaign(
    configs: list[Path],
    runs_dir: Path,
    *,
    launcher: Callable[[ExperimentRun, Path], Awaitable[None]] = launch,
    server_check: Callable[[ExperimentRun], str | None] = check_server,
    max_runs: int | None = None,
) -> list[str]:
    """Runs each config in order. Returns the run_ids it actually ran.

    `launcher` and `server_check` are parameters so the tests can drive the
    real ordering, skip and failure logic against a mocked gateway. Production
    passes `run_simulation`'s own two functions, so nothing about a campaign
    run differs from `python run_simulation.py` on the same config.
    """
    selected = configs if max_runs is None else configs[:max_runs]
    if max_runs is not None and len(configs) > len(selected):
        print(
            f"  --max-runs {max_runs}: running {len(selected)} of "
            f"{len(configs)} configs. A cut inside a seed index leaves an "
            "incomplete matched set; re-run without the cap to finish it."
        )

    checked_endpoints: set[str | None] = set()
    ran: list[str] = []

    for position, config_path in enumerate(selected, start=1):
        run = load_run_config(config_path)
        label = f"[{position}/{len(selected)}] {run.run_id}"

        status = completed_status(runs_dir, run.run_id)
        if status == "completed":
            print(f"{label}  skipped: already completed")
            continue

        # Once per distinct endpoint, not once per run: the check is an HTTP
        # round trip, every run in a campaign normally shares an api_base, and
        # forty identical checks would say nothing the first did not.
        if run.api_base not in checked_endpoints:
            problem = server_check(run)
            checked_endpoints.add(run.api_base)
            if problem is not None:
                raise CampaignRunFailed(run.run_id, RuntimeError(problem))

        resume = resume_turn(runs_dir, run.run_id)
        where = f"resuming after turn {resume}" if resume is not None else "starting"
        print(f"{label}  {where}  (seed {run.seed}, {run.language_condition}, "
              f"delivery {run.delivery})")

        started = time.monotonic()
        try:
            await launcher(run, runs_dir)
        except Exception as e:
            raise CampaignRunFailed(run.run_id, e) from e
        elapsed = time.monotonic() - started

        failed_rows = count_failed_rows(runs_dir, run.run_id)
        print(
            f"{label}  done in {elapsed:6.1f}s  "
            f"{failed_rows} {FAILED_STATUS} row(s)"
            + ("  <-- check this" if failed_rows else "")
        )
        ran.append(run.run_id)

    return ran


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "campaign_dir", type=Path, help="a directory of generated run configs"
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
        help="where run output goes (default: runs/). Point this at persistent "
             "storage when running in a notebook.",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=list(EXPERIMENT_ORDER),
        choices=list(EXPERIMENT_ORDER),
        help="which experiments to run (default: both). The campaign's run "
             "order does not change with the order given here.",
    )
    parser.add_argument(
        "--max-runs", type=int, default=None,
        help="stop after this many runs, for a smoke test. A cap that falls "
             "inside a seed index leaves that matched set incomplete.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configs = discover(args.campaign_dir, args.experiments)
    if not configs:
        print(
            f"no campaign configs in {args.campaign_dir} for "
            f"{args.experiments} (expected files named like "
            f"e3_english_ws_s01.yaml)",
            file=sys.stderr,
        )
        return 1

    print(f"\n{args.campaign_dir}: {len(configs)} run(s) -> {args.runs_dir}/")
    try:
        ran = asyncio.run(
            run_campaign(configs, args.runs_dir, max_runs=args.max_runs)
        )
    except CampaignRunFailed as e:
        print(f"\ncampaign stopped: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "\ninterrupted. The last completed turn of the current run is "
            "checkpointed; re-run the same command to carry on.",
            file=sys.stderr,
        )
        return 130

    print(f"\ncampaign finished: {len(ran)} run(s) executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
