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
from sandbox.seed_manager import SeedManager
from sandbox.simulation_orchestrator import _build_strategy

DEFAULT_BASE = Path("configs/rq1/base.yaml")
DEFAULT_PROBE_QUESTIONS = Path("configs/rq1/probe_questions.yaml")
DEFAULT_OUT = Path("configs/rq1/campaign")
DEFAULT_SEEDS = 5
DEFAULT_BASE_SEED = 1000

# The graph the campaign runs on, as it appears in a filename.
STRUCTURE = "ws"

# Default rewire_rate for E5's three arms, overridable with --rewire-rate.
# 0.3 is the one number the design originally asked the generator to set,
# named after base.yaml's own comment on rewire_rate ("the campaign's
# echo_rewire_rate") -- it cannot be a literal key IN base.yaml, because
# base.yaml validates as a complete ExperimentRun on its own
# (ExperimentRun.model_config forbids unknown fields, and
# test_the_shipped_base_config_is_a_valid_bank_regime_run loads it exactly
# that way), and an echo_rewire_rate field the schema does not define would
# fail that load. So it lives here, where e3/e4's own delivery values
# already live, rather than in the base config.
DEFAULT_REWIRE_RATE = 0.3

# e8/e9's own N and graph_degree, overridable only by a future flag -- the
# design calls for N=5 specifically (Jo et al.'s "up to five") and a WS
# graph of degree 6 (their k=6), neither of which base.yaml's own smoke-test
# N=3/graph_degree=4 can satisfy (N cannot exceed graph_degree). Raising
# graph_degree only for e8/e9 -- rather than in base.yaml, which every other
# experiment also inherits -- is what keeps e3 through e7's generated files
# byte-identical to before this feature existed.
RECSYS_N = 5
RECSYS_GRAPH_DEGREE = 6

# Default recsys_threshold for e8, overridable with --recsys-threshold.
DEFAULT_RECSYS_THRESHOLD = 1.0

# delivery / rewire_target per experiment. e3/e4 are unchanged: neither key
# is in their overrides, so build_config leaves base.yaml's own
# rewire_rate=0.0 in place, byte for byte. e5/e6/e7 are E5's echo-chamber
# arms: e5 is the treatment (rewiring on, delivery on), e6 is its control
# (rewiring on the shadow items under delivery off -- the same mirror E4 is
# for E3), and e7 is rewiring without selection (rewire_target="random"), a
# control for whether rewiring ALONE -- without choosing same-side
# neighbours -- produces any sorting. analysis/metrics.py reads the
# experiment back off the run_id prefix for all five.
#
# rewire_rate itself is NOT in this dict: every one of these three shares
# whatever --rewire-rate the generator invocation asked for (see
# REWIRE_EXPERIMENTS below and build_config), so a sweep over rates does not
# require three separate entries that would drift from each other.
EXPERIMENTS: dict[str, dict] = {
    "e3": {"delivery": "on"},
    "e4": {"delivery": "off"},
    "e5": {"delivery": "on", "rewire_target": "same_side"},
    "e6": {"delivery": "off", "rewire_target": "same_side"},
    "e7": {"delivery": "on", "rewire_target": "random"},
    # Fix 9: Jo et al.'s similarity recommender (e8) and its same-feed-size
    # random control (e9). Both read N=5 posts a round instead of the
    # smoke-scale base config's N=3, which needs graph_degree raised to 6
    # (Jo et al.'s own k=6) so N=5 <= graph_degree still holds -- e3/e4/e5/
    # e6/e7 are untouched and keep base.yaml's graph_degree. e9 needs no
    # neighbor_strategy override: base.yaml's own "graph" is already the
    # random-neighbours strategy e3/e5/e7 use, just at a different N and
    # graph_degree, so it IS the "same feed size" control for e8.
    "e8": {
        "delivery": "on", "neighbor_strategy": "graph_similarity",
        "N": RECSYS_N, "graph_degree": RECSYS_GRAPH_DEGREE,
    },
    "e9": {
        "delivery": "on",
        "N": RECSYS_N, "graph_degree": RECSYS_GRAPH_DEGREE,
        # Fix 1: at N=5/degree=6, Watts-Strogatz construction strands enough
        # nodes below N that most seeds cannot build at all under the
        # default strict check. allow_short_feed makes e9 mean "up to 5"
        # the same way e8's recommender already does, so the two arms stay
        # the matched-feed-size control pair they are meant to be.
        "allow_short_feed": True,
    },
}

# Experiments that set rewire_rate/rewire_target at all. e3/e4 never read
# rewire_rate from here regardless of what --rewire-rate was passed, since
# the design holds rewiring absent from the pre-E5 arms.
REWIRE_EXPERIMENTS = frozenset({"e5", "e6", "e7"})

# e8 is the only experiment whose recsys_threshold is ever set to anything
# but the schema default -- e9 uses the "graph" strategy, where the field is
# inert, so leaving it at the default there (rather than copying --recsys-
# threshold onto e9 too) keeps e9's run_config.json from claiming a filter
# it never applies.
RECSYS_EXPERIMENTS = frozenset({"e8"})

# Experiments whose starting leans are balanced into two equal camps rather
# than drawn uniformly at random (see ExperimentRun.lean_assignment): every
# echo-chamber arm, so a population that happens to start lopsided toward
# one pole -- or with a midpoint camp of its own tipping the count -- does
# not confound "formed an echo chamber faster" with "started closer to one
# already" or "one side started with more agents than the other".
# "balanced_two_camps", not plain "balanced": M=20 over a 7-point scale
# under "balanced" gives lean 4 three agents of its own and splits the
# remaining 17 as 9 below and 8 above, a 9-vs-11 population a weaker camp
# could be absorbed out of before it ever sorts. e8/e9 need the same
# guarantee for the same reason, even though neither one rewires.
BALANCED_LEAN_EXPERIMENTS = frozenset({"e5", "e6", "e7", "e8", "e9"})

# The default --experiments: unchanged from before E5 existed, so a
# generator invocation with no flag produces exactly the E3/E4 campaign it
# always has.
DEFAULT_EXPERIMENTS = ("e3", "e4")

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


def rewire_rate_token(rate: float) -> str:
    """'' at the default rate, so the default campaign's filenames are
    unaffected; '_r0p1' etc. otherwise, so a sweep over rates can share one
    runs folder without filenames colliding."""
    if rate == DEFAULT_REWIRE_RATE:
        return ""
    return "_r" + f"{rate:g}".replace(".", "p").replace("-", "neg")


def recsys_threshold_token(threshold: float) -> str:
    """'' at the default threshold, so the default campaign's filenames are
    unaffected; '_t0p5' etc. otherwise -- the same pattern
    rewire_rate_token uses, so a --recsys-threshold sweep can share one runs
    folder without filenames colliding."""
    if threshold == DEFAULT_RECSYS_THRESHOLD:
        return ""
    return "_t" + f"{threshold:g}".replace(".", "p").replace("-", "neg")


def probe_language_token(probe_language: str | None) -> str:
    """'' when the probe asks in each arm's own language (the default);
    '_probe-<lang>' when every arm's probe is pinned to one language
    regardless of what it posts in."""
    return f"_probe-{probe_language}" if probe_language else ""


def run_id_for(
    experiment: str,
    language: str,
    seed_index: int,
    *,
    rewire_rate: float = DEFAULT_REWIRE_RATE,
    recsys_threshold: float = DEFAULT_RECSYS_THRESHOLD,
    probe_language: str | None = None,
) -> str:
    """The file stem, which is also the run_id.

    One string, one function: the campaign runner parses the experiment, the
    language and the seed index back out of this to order its runs, and
    analysis/metrics.py reads the experiment off the prefix. Three readers of
    a format that was written in a fourth place would drift.

    rewire_rate is only consulted for e5/e6/e7 (see REWIRE_EXPERIMENTS);
    recsys_threshold only for e8 (see RECSYS_EXPERIMENTS); every other
    experiment never carries that token, since it never sets the field away
    from its default at all.
    """
    rate_token = rewire_rate_token(rewire_rate) if experiment in REWIRE_EXPERIMENTS else ""
    threshold_token = (
        recsys_threshold_token(recsys_threshold) if experiment in RECSYS_EXPERIMENTS else ""
    )
    return (
        f"{experiment}_{language}_{STRUCTURE}{rate_token}{threshold_token}"
        f"{probe_language_token(probe_language)}_s{seed_index:02d}"
    )


def single_run_id_for(language: str, *, probe_language: str | None = None) -> str:
    return f"single_{language}{probe_language_token(probe_language)}"


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
    rewire_rate: float = DEFAULT_REWIRE_RATE,
    recsys_threshold: float = DEFAULT_RECSYS_THRESHOLD,
    rounds: int | None = None,
    probe_language: str | None = None,
) -> dict:
    """One campaign config: the base, with the generated fields set.

    Six fields for every experiment (run_id, language_condition,
    probe_question, delivery, seed, trial_number, per GENERATED_FIELDS),
    plus -- for e5/e6/e7 only -- rewire_rate, rewire_target and
    lean_assignment, which base.yaml does not set at all (rewire_rate stays
    its schema default of 0.0 there, and e3/e4 inherit that unchanged); and
    -- for e8/e9 only -- N, graph_degree and (e8 alone) neighbor_strategy
    and recsys_threshold, per EXPERIMENTS' own overrides.

    rounds, when given, overrides K for every experiment -- not only
    e5/e6/e7 -- since K is a campaign-wide dimension, not an echo-chamber
    one. probe_language, when given, is passed straight through to
    run_id_for; the caller is responsible for resolving probe_question to
    that language's question (see generate()).
    """
    config = dict(base)
    overrides = EXPERIMENTS[experiment]
    config["run_id"] = run_id_for(
        experiment, language, seed_index,
        rewire_rate=rewire_rate, recsys_threshold=recsys_threshold,
        probe_language=probe_language,
    )
    config["language_condition"] = language
    config["probe_question"] = probe_question
    config["delivery"] = overrides["delivery"]
    # base_seed + i, not a hash of anything about the arm. The four languages
    # and the experiments of one seed index share this number, which is what
    # makes them a matched set.
    config["seed"] = base_seed + seed_index
    config["trial_number"] = seed_index
    if experiment in REWIRE_EXPERIMENTS:
        config["rewire_rate"] = rewire_rate
        config["rewire_target"] = overrides["rewire_target"]
    if "neighbor_strategy" in overrides:
        config["neighbor_strategy"] = overrides["neighbor_strategy"]
    if "N" in overrides:
        config["N"] = overrides["N"]
    if "graph_degree" in overrides:
        config["graph_degree"] = overrides["graph_degree"]
    if "allow_short_feed" in overrides:
        config["allow_short_feed"] = overrides["allow_short_feed"]
    if experiment in RECSYS_EXPERIMENTS:
        config["recsys_threshold"] = recsys_threshold
    if experiment in BALANCED_LEAN_EXPERIMENTS:
        config["lean_assignment"] = "balanced_two_camps"
    if rounds is not None:
        config["K"] = rounds
    if config["delivery"] == "on":
        # A delivery-on run never opens the neutral bank, so recording one in
        # its run_config.json would claim it read something it did not. A
        # delivery-off control (e4, e6) keeps it, and requires it.
        config.pop("neutral_bank_path", None)
    return config


def build_single_config(
    base: dict,
    *,
    language: str,
    probe_question: str,
    base_seed: int,
    probe_language: str | None = None,
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
    config["run_id"] = single_run_id_for(language, probe_language=probe_language)
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
    experiments: list[str] = DEFAULT_EXPERIMENTS,
    rewire_rate: float = DEFAULT_REWIRE_RATE,
    recsys_threshold: float = DEFAULT_RECSYS_THRESHOLD,
    rounds: int | None = None,
    probe_language: str | None = None,
) -> list[dict]:
    """Writes every config, validates each, and returns the manifest rows.

    Ordered by seed index, then experiment, then language -- the same order
    `run_campaign.py` runs them in, so the manifest reads as the plan.

    `experiments` defaults to e3/e4 only, so a generator invocation with no
    `--experiments` flag produces exactly the campaign it always has.
    Passing e5/e6/e7 adds E5's echo-chamber arms alongside (or instead of)
    them; every seed index still shares one seed across whichever
    experiments are requested, the same matched-set guarantee e3/e4 already
    give.

    `rewire_rate` defaults to DEFAULT_REWIRE_RATE, so an invocation that
    never passes --rewire-rate produces byte-identical e5/e6/e7 files to
    before this flag existed. `rounds` defaults to None, meaning "whatever
    K the base config says" -- every experiment's K, not only e5/e6/e7's.
    `probe_language`, when given, pins every arm's probe_question to that
    one language's question (an E1 control: posts stay in the arm's own
    language, the opinion question does not) and is folded into every
    run_id so the campaign directory cannot confuse it with the default.
    """
    if seeds < 1:
        raise ValueError(f"--seeds must be >= 1, got {seeds}")
    unknown = sorted(set(experiments) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(
            f"unknown experiment(s) {unknown}; expected one or more of "
            f"{list(EXPERIMENTS)}"
        )
    base = load_base(base_path)
    # probe_language may name a language outside `languages` (an English
    # probe over a Hindi-only generation, say), so its question is loaded
    # even when that language's own arm is not being generated.
    needed_languages = sorted(set(languages) | ({probe_language} if probe_language else set()))
    questions = load_probe_questions(probe_questions_path, needed_languages)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The exact original phrasing when experiments is the original default,
    # so the --experiments flag's mere existence does not change a byte of
    # what `make_rq1_configs.py` with no flag produces (and so the checked-in
    # configs/rq1/campaign/ -- generated with no flag -- stays in sync).
    experiments_note = (
        "and by E3 and E4."
        if list(experiments) == list(DEFAULT_EXPERIMENTS)
        else f"and by every experiment in this campaign ({', '.join(experiments)})."
    )

    manifest: list[dict] = []
    for seed_index in range(1, seeds + 1):
        for experiment in experiments:
            for language in languages:
                config = build_config(
                    base,
                    experiment=experiment,
                    language=language,
                    probe_question=questions[probe_language or language],
                    seed_index=seed_index,
                    base_seed=base_seed,
                    rewire_rate=rewire_rate,
                    recsys_threshold=recsys_threshold,
                    rounds=rounds,
                    probe_language=probe_language,
                )
                path = out_dir / f"{config['run_id']}.yaml"
                write_config(
                    path,
                    config,
                    base_path,
                    f"Seed {config['seed']} is shared by all four languages "
                    f"{experiments_note}",
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
            probe_question=questions[probe_language or language],
            base_seed=base_seed,
            probe_language=probe_language,
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
    #
    # Fix 1: loading a config as a schema (load_run_config) is not the same
    # as the run being able to start -- a graph/graph_similarity config can
    # pass every schema check and still fail to build its Watts-Strogatz
    # graph for this seed (the e8/e9 bug this fix exists for). So every
    # generated config is also built as a strategy, the same object the
    # orchestrator would construct at run start, with the offending file
    # named on failure. A config that loads but cannot start should never be
    # committed to the repo again.
    for row in manifest:
        path = out_dir / f"{row['run_id']}.yaml"
        run = load_run_config(path)
        try:
            _build_strategy(run, SeedManager(run.seed))
        except Exception as exc:
            raise ValueError(f"{path}: strategy failed to build: {exc}") from exc

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
    parser.add_argument(
        "--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS),
        choices=list(EXPERIMENTS),
        help="which experiments to generate "
             f"(default: {' '.join(DEFAULT_EXPERIMENTS)}). Add e5/e6/e7 for "
             "E5's echo-chamber arms: e5 (rewiring on), e6 (its shadow-item "
             "control) and e7 (rewiring without selection). Add e8/e9 for "
             "Fix 9's similarity recommender (e8) and its same-feed-size "
             "random control (e9).",
    )
    parser.add_argument(
        "--rewire-rate", type=float, default=DEFAULT_REWIRE_RATE,
        help=f"rewire_rate for e5/e6/e7 (default: {DEFAULT_REWIRE_RATE}). "
             "Appears in the run_id only when it differs from the default, "
             "so the default campaign's filenames are unaffected and a "
             "sweep over rates can share one runs folder.",
    )
    parser.add_argument(
        "--recsys-threshold", type=float, default=DEFAULT_RECSYS_THRESHOLD,
        help=f"recsys_threshold for e8 (default: {DEFAULT_RECSYS_THRESHOLD}), "
             "in x units. Appears in the run_id only when it differs from "
             "the default, so a --recsys-threshold sweep can share one runs "
             "folder the same way --rewire-rate's sweep does.",
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="override K for every generated config (default: base.yaml's own K).",
    )
    parser.add_argument(
        "--probe-language", choices=list(LANGUAGE_CONDITIONS), default=None,
        help="ask every arm's probe in this one language instead of its own "
             "(posts still render in each arm's own language). Adds "
             "'_probe-<lang>' to every run_id. For an E1 control arm.",
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
        experiments=args.experiments,
        rewire_rate=args.rewire_rate,
        recsys_threshold=args.recsys_threshold,
        rounds=args.rounds,
        probe_language=args.probe_language,
    )
    manifest_path = args.out / "manifest.csv"
    write_manifest(manifest_path, manifest)

    runs = [row for row in manifest if row["experiment"] != "single"]
    print(f"\n{args.out}/")
    print(f"  {len(runs)} campaign configs "
          f"({args.seeds} seed indices x {len(args.experiments)} experiments "
          f"({', '.join(args.experiments)}) x {len(args.languages)} languages)")
    print(f"  {len(manifest) - len(runs)} single-reader base configs")
    print(f"  {manifest_path}")
    print(
        f"\n  seeds {args.base_seed + 1}..{args.base_seed + args.seeds}, "
        f"each shared by every language and every experiment requested"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
