#!/usr/bin/env python3
"""Launch one opinion-CSV experiment run from a config file (plan.md).

    python scripts/run_opinion_experiment.py configs/opinion/example.yaml
    python scripts/run_opinion_experiment.py configs/opinion/example.yaml --dry-run

Separate from run_simulation.py / ExperimentRun on purpose -- see plan.md's
Context section for why this pipeline doesn't fit the bank/generated
regimes' invariants. No resume/checkpoint support: a crash restarts the
whole run from scratch.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from sandbox.opinion_models import OpinionExperimentConfig
from sandbox.opinion_orchestrator import resolve_languages, run
from sandbox.opinion_pool import load_opinion_persona_pool


def load_opinion_config(path: Path) -> OpinionExperimentConfig:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    try:
        return OpinionExperimentConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"invalid config at {path}:\n{e}") from e


def describe(config: OpinionExperimentConfig) -> None:
    pool = load_opinion_persona_pool(config.persona_pool_path)
    M = config.M if config.M is not None else len(pool)
    active = sorted(lang for lang, on in config.languages.items() if on)

    print(f"  run_id        {config.run_id}")
    print(f"  topic         {config.topic}")
    print(f"  population    M={M}  N={config.N}  K={config.K}")
    print(f"  graph         WS(M={M}, degree={config.graph_degree}, p={config.graph_rewire_p})")
    print(f"  similarity    threshold={config.similarity_threshold} (belief/v scale, [-2, 2])")
    print(f"  languages     {active}")
    print(f"  model         {config.model_backend_id}")
    print(f"  api_base      {config.api_base or '(hosted provider)'}")
    print(f"  seed          {config.seed}")
    print(f"  output        {config.output_dir}/")
    calls = M * (config.K + 1)
    print(f"  calls         at most {calls} LLM calls (M x (K+1))")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, help="path to an opinion-experiment config YAML")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the config and print the plan without calling any model",
    )
    args = parser.parse_args()

    try:
        config = load_opinion_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"config invalid: {e}", file=sys.stderr)
        return 1

    print(f"\n{args.config}")
    try:
        describe(config)
        # Fails loudly before any call is dispatched if the language toggle
        # combination is unsupported, or a persona line has no recognizable
        # language preference.
        pool = load_opinion_persona_pool(config.persona_pool_path)
        if config.M is not None:
            pool = pool[: config.M]
        resolve_languages(config.languages, pool)
    except (NotImplementedError, ValueError) as e:
        print(f"\nconfig will not run: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\ndry run, nothing dispatched.")
        return 0

    print()
    try:
        output_dir = asyncio.run(run(config))
    except Exception as e:
        print(f"\nrun failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    print(f"\ndone. output -> {output_dir}/")
    print("  initial_stance.csv        one row per agent, growing by 4 "
          "columns (opinion, belief, confidence, status) each round")
    print("  opinion_round_metrics.csv one row per round "
          "(polarization_index, global_disagreement, nci, mean_belief_shift, "
          "mean_abs_individual_shift)")
    print("  run_config.json           the exact config used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
