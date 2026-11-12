#!/usr/bin/env python3
"""Expands one base config into the RQ1 campaign's matched sets.

    python scripts/make_rq1_configs.py
    python scripts/make_rq1_configs.py --seeds 10 --base-seed 2000

Writes, for each seed index i in 1..N and each language:

    e3_{language}_ws_s{i:02d}.yaml    delivery: on   -- the treatment
    e4_{language}_ws_s{i:02d}.yaml    delivery: off  -- the control

plus one `single_{language}.yaml` per language, which is what the three
single-reader scripts (calibrate_tau, run_e1, check_comprehension) read
their settings from.

"ws" names the Watts-Strogatz graph the runs use, so a later campaign over a
different structure sits beside this one in the same directory without a
filename collision.

**The seed is shared across the four languages and across E3 and E4.** That
is the whole point of generating these rather than writing them by hand.
`SeedManager` takes the base seed and nothing else -- no language, no
topic, no trial -- so four runs sharing a seed get the same graph, the same
neighbour draws, and the same bank item for the same sender on the same
side. They differ in the rendering of the text and in nothing else, and the
E4 control shares every draw with the E3 arm it is the baseline for.

Eight runs per seed index, and a `--max-runs` cut in `run_campaign.py` falls
on a seed-index boundary, so a campaign stopped early leaves complete
matched sets rather than three arms of a four-arm comparison.

Everything is copied from the base config byte for byte except the six
fields listed below. Nothing is computed per language, deliberately: a
generator that derived, say, a per-language `max_prompt_tokens` would
introduce exactly the kind of uncontrolled difference between arms the
experiment exists to rule out.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

if __package__ in (None, ""):  # `python scripts/make_rq1_configs.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sandbox.config_loader import load_run_config
from sandbox.models import LANGUAGE_CONDITIONS

DEFAULT_BASE = Path("configs/rq1/base.yaml")
DEFAULT_PROBE_QUESTIONS = Path("configs/rq1/probe_questions.yaml")
DEFAULT_OUT = Path("configs/rq1/campaign")
DEFAULT_SEEDS = 5
DEFAULT_BASE_SEED = 1000

# The graph the campaign runs on, as it appears in a filename.
STRUCTURE = "ws"

# delivery on / delivery off. The names are the ones the RQ1 plan uses, and
# analysis/metrics.py reads the experiment back off the run_id prefix.
EXPERIMENTS = {"e3": "on", "e4": "off"}

MANIFEST_COLUMNS = ("run_id", "experiment", "language", "seed_index", "seed")

# Fields the generator sets. Whatever the base config says for these is
# ignored, which is why base.yaml labels them as overwritten: a reader who
# edits `seed` there and sees it have no effect should have been told.
GENERATED_FIELDS = (
    "run_id",
    "language_condition",
    "probe_question",
    "delivery",
    "seed",
    "trial_number",
)


def run_id_for(experiment: str, language: str, seed_index: int) -> str:
    """The file stem, which is also the run_id.

    One string, one function: the campaign runner parses the experiment, the
    language and the seed index back out of this to order its runs, and
    analysis/metrics.py reads the experiment off the prefix. Three readers of
    a format that was written in a fourth place would drift.
    """
    return f"{experiment}_{language}_{STRUCTURE}_s{seed_index:02d}"


def single_run_id_for(language: str) -> str:
    return f"single_{language}"


def load_probe_questions(path: Path, languages: list[str]) -> dict[str, str]:
    """The question per language, with every requested arm required.

    Fails for a missing arm rather than falling back to English or skipping
    it. There is no default probe question anywhere in this codebase, for the
    reason ExperimentRun's own validator gives: a generic English fallback
    would silently probe every arm in English and leave no trace in the
    output.
    """
    if not path.exists():
        raise FileNotFoundError(f"probe questions file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must be a mapping of language condition to question, "
            f"got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - set(LANGUAGE_CONDITIONS))
    if unknown:
        raise ValueError(
            f"{path} names languages that are not language conditions: {unknown}"
        )
    missing = [
        language
        for language in languages
        if not isinstance(raw.get(language), str) or not raw[language].strip()
    ]
    if missing:
        raise ValueError(
            f"{path} has no probe question for {missing}. Every arm must be "
            "probed in its own language, and there is no default: an English "
            "fallback would collapse the comparison the campaign exists to make."
        )
    return {language: raw[language].strip() for language in languages}


def load_base(path: Path) -> dict:
    """The base config as a plain dict, validated as a config first.

    Validated before expansion so a broken base produces one error naming the
    base, rather than 2N identical errors naming generated files.
    """
    load_run_config(path)  # raises ConfigLoadError with every problem at once
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Resolved at load time from the repo, never carried in a file. Leaving a
    # stale hash in a generated config would claim a provenance it does not
    # have.
    raw.pop("git_commit_hash", None)
    return raw


def build_config(
    base: dict,
    *,
    experiment: str,
    language: str,
    probe_question: str,
    seed_index: int,
    base_seed: int,
) -> dict:
    """One campaign config: the base, with the six generated fields set."""
    config = dict(base)
    config["run_id"] = run_id_for(experiment, language, seed_index)
    config["language_condition"] = language
    config["probe_question"] = probe_question
    config["delivery"] = EXPERIMENTS[experiment]
    # base_seed + i, not a hash of anything about the arm. The four languages
    # and the two experiments of one seed index share this number, which is
    # what makes them a matched set.
    config["seed"] = base_seed + seed_index
    config["trial_number"] = seed_index
    if config["delivery"] == "on":
        # An E3 run never opens the neutral bank, so recording one in its
        # run_config.json would claim it read something it did not. The E4
        # control keeps it, and requires it.
        config.pop("neutral_bank_path", None)
    return config


def build_single_config(
    base: dict, *, language: str, probe_question: str, base_seed: int
) -> dict:
    """The base config for the three single-reader scripts, per language.

    `delivery` stays "on" and `neutral_bank_path` is kept: the calibration
    script reads the control's own side-free messages (so that τ is the
    baseline for what the control actually does), while nothing here runs the
    orchestrator, so the delivery mode is never consulted.

    Seeded at the base seed with no index added. These are not a trial in the
    campaign's sense -- there is one calibration, not one per seed index --
    and giving them seed index 1's number would make two different things
    share a seed for no reason.
    """
    config = dict(base)
    config["run_id"] = single_run_id_for(language)
    config["language_condition"] = language
    config["probe_question"] = probe_question
    config["delivery"] = "on"
    config["seed"] = base_seed
    config["trial_number"] = 1
    return config


HEADER = """\
# GENERATED by scripts/make_rq1_configs.py from {base}.
# Do not edit: re-run the generator instead, or the arms of a matched set
# will stop being matched. {note}
"""


def write_config(path: Path, config: dict, base_path: Path, note: str) -> None:
    """One YAML file, sorted, with a header saying where it came from.

    sort_keys so a diff between two generated files shows only the fields
    that differ, which is the check a reviewer actually wants to make;
    allow_unicode so the Devanagari probe question is readable rather than
    escaped.
    """
    path.write_text(
        HEADER.format(base=base_path, note=note)
        + yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def generate(
    *,
    base_path: Path,
    probe_questions_path: Path,
    out_dir: Path,
    languages: list[str],
    seeds: int,
    base_seed: int,
) -> list[dict]:
    """Writes every config, validates each, and returns the manifest rows.

    Ordered by seed index, then experiment, then language -- the same order
    `run_campaign.py` runs them in, so the manifest reads as the plan.
    """
    if seeds < 1:
        raise ValueError(f"--seeds must be >= 1, got {seeds}")
    base = load_base(base_path)
    questions = load_probe_questions(probe_questions_path, languages)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for seed_index in range(1, seeds + 1):
        for experiment in EXPERIMENTS:
            for language in languages:
                config = build_config(
                    base,
                    experiment=experiment,
                    language=language,
                    probe_question=questions[language],
                    seed_index=seed_index,
                    base_seed=base_seed,
                )
                path = out_dir / f"{config['run_id']}.yaml"
                write_config(
                    path,
                    config,
                    base_path,
                    f"Seed {config['seed']} is shared by all four languages "
                    "and by E3 and E4.",
                )
                manifest.append(
                    {
                        "run_id": config["run_id"],
                        "experiment": experiment,
                        "language": language,
                        "seed_index": seed_index,
                        "seed": config["seed"],
                    }
                )

    for language in languages:
        config = build_single_config(
            base,
            language=language,
            probe_question=questions[language],
            base_seed=base_seed,
        )
        path = out_dir / f"{config['run_id']}.yaml"
        write_config(
            path,
            config,
            base_path,
            "Read by calibrate_tau.py, run_e1.py and check_comprehension.py; "
            "never run by the orchestrator.",
        )
        manifest.append(
            {
                "run_id": config["run_id"],
                # Listed in the manifest so it is a complete index of what the
                # generator wrote. run_campaign.py filters on `experiment`, so
                # these are never dispatched as runs.
                "experiment": "single",
                "language": language,
                "seed_index": "",
                "seed": config["seed"],
            }
        )

    # Validated after writing, all of them, so a schema problem in the base is
    # reported once per generated file with the file named -- and so a
    # generator bug cannot leave an unloadable config on disk unremarked.
    for row in manifest:
        load_run_config(out_dir / f"{row['run_id']}.yaml")

    return manifest


def write_manifest(path: Path, manifest: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--probe-questions", type=Path, default=DEFAULT_PROBE_QUESTIONS
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"how many seed indices (default: {DEFAULT_SEEDS})")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED,
                        help=f"seed index i gets base_seed + i "
                             f"(default: {DEFAULT_BASE_SEED})")
    parser.add_argument(
        "--languages", nargs="+", default=list(LANGUAGE_CONDITIONS),
        choices=list(LANGUAGE_CONDITIONS),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate(
        base_path=args.base,
        probe_questions_path=args.probe_questions,
        out_dir=args.out,
        languages=args.languages,
        seeds=args.seeds,
        base_seed=args.base_seed,
    )
    manifest_path = args.out / "manifest.csv"
    write_manifest(manifest_path, manifest)

    runs = [row for row in manifest if row["experiment"] != "single"]
    print(f"\n{args.out}/")
    print(f"  {len(runs)} campaign configs "
          f"({args.seeds} seed indices x {len(EXPERIMENTS)} experiments "
          f"x {len(args.languages)} languages)")
    print(f"  {len(manifest) - len(runs)} single-reader base configs")
    print(f"  {manifest_path}")
    print(
        f"\n  seeds {args.base_seed + 1}..{args.base_seed + args.seeds}, "
        "each shared by every language and both experiments"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
