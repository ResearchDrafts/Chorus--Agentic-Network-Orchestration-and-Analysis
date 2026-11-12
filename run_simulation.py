#!/usr/bin/env python3
"""Launch one simulation run from a config file.

    python run_simulation.py configs/smoke_test.yaml
    python run_simulation.py configs/smoke_test.yaml --dry-run

A stopgap until Phase 6 builds the real CLI (`python -m sandbox run`) and
`sandbox/api.py`. Deliberately thin: it wires the four objects the
orchestrator needs and gets out of the way, so there is no behaviour here
that the eventual CLI would have to reimplement differently.

Re-running the same config resumes from the last checkpoint rather than
starting over, and a resumed run produces output byte-identical to an
uninterrupted one. To start fresh, delete runs/{run_id}/ or change run_id.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sandbox.config_loader import ConfigLoadError, load_run_config
from sandbox.cost_tracker import CostTracker
from sandbox.model_gateway import ModelGateway
from sandbox.models import ExperimentRun
from sandbox.rate_limiter import RateLimiter
from sandbox.simulation_orchestrator import build_orchestrator


def describe(run: ExperimentRun, runs_dir: Path) -> None:
    """What this run will do, before it does it."""
    print(f"  run_id        {run.run_id}")
    print(f"  topic         {run.topic}")
    print(f"  population    M={run.M}  N={run.N}  K={run.K}")
    print(f"  neighbours    {_describe_neighbours(run)}")
    print(f"  regime        {_describe_regime(run)}")
    print(f"  language      {run.language_condition}")
    print(f"  model         {run.model_backend_id}")
    print(f"  api_base      {run.api_base or '(hosted provider)'}")
    print(f"  memes         {'enabled' if run.meme_injection.enabled else 'disabled'}")
    print(f"  seed          {run.seed}")
    print(f"  output        {runs_dir / run.run_id}/")
    print()
    for line in _describe_calls(run):
        print(f"  {line}")

    if run.regime == "bank":
        print()
        print(f"  probe         {run.probe_question!r}")
        print(f"                top_logprobs={run.probe_top_logprobs}  "
              f"mass_floor={run.probe_mass_floor}")
        print(f"  anchors       1 = {run.stance_low_label}")
        print(f"                7 = {run.stance_high_label}")
        print(f"  memory        {run.memory_rounds} rounds of reading and replies")
        if run.delivery == "off":
            print(f"  neutral       {run.neutral_message!r}")
        for problem in _probe_language_warnings(run):
            print()
            print(f"  WARNING: {problem}")


def _describe_neighbours(run: ExperimentRun) -> str:
    if run.neighbor_strategy == "graph":
        return (
            f"graph  WS(M={run.M}, degree={run.graph_degree}, p={run.graph_rewire_p})"
            f"  reading {run.N} of each neighbourhood"
        )
    return f"alpha  alpha={run.alpha}  sampling {run.N} from the whole population"


def _describe_regime(run: ExperimentRun) -> str:
    if run.regime != "bank":
        return "generated  agents write their own posts, stance self-reported"
    delivery = (
        "delivery ON"
        if run.delivery == "on"
        else "delivery OFF (E4 control: reads one side-free message)"
    )
    return f"bank  posts from {run.bank_path}, stance by probe  [{delivery}]"


def _describe_calls(run: ExperimentRun) -> list[str]:
    """What this run will cost in calls, counted per regime.

    A bank round is two calls per agent, not one, plus one probe each for
    round 0 -- so a bank run of the same M and K is about twice the traffic of
    a generated one, which is worth seeing before starting rather than after.

    The delivery-off control costs the same as a treatment arm: it reads the
    neutral message and replies to it, so it makes M x K reply calls too.
    That is the point of it -- the two arms differ in the persuasive content
    of what agents read, not in how much work they do.
    """
    if run.regime != "bank":
        return [f"at most {run.M * run.K} LLM calls (M x K), fewer if memes replace turns"]

    replies = run.M * run.K
    probes = run.M * (run.K + 1)  # +1 for the round-0 probe
    return [
        f"{replies} reply calls  (M x K"
        + ("; the control replies to the neutral message)" if run.delivery == "off" else ")"),
        f"{probes} probe calls  (M x (K+1), including the round-0 probe)",
        f"{replies + probes} LLM calls in total",
    ]


def _probe_language_warnings(run: ExperimentRun) -> list[str]:
    """Warns when the probe text is visibly in the wrong script.

    The probe question and the two anchor labels have to be written in the
    run's own language -- there is no default, deliberately, because an
    English fallback would probe every arm in English and leave no trace of it
    in the output. But a config copied from another arm can still carry the
    wrong arm's question, and the run would look perfectly valid.

    Only the Devanagari arm is checkable. english, hinglish and
    romanized_hindi are all Roman script, so nothing short of language
    identification could tell a Hinglish question from an English one, and a
    heuristic that guessed would cry wolf on every Hinglish config -- which is
    worse than silence, because a warning nobody believes is a warning nobody
    reads. For those three arms this says nothing, and the reviewer has to
    read the question, which describe() prints in full for that reason.
    """
    if run.language_condition != "hindi":
        return []
    fields = {
        "probe_question": run.probe_question,
        "stance_low_label": run.stance_low_label,
        "stance_high_label": run.stance_high_label,
    }
    wrong = [name for name, text in fields.items() if not _has_devanagari(text)]
    if not wrong:
        return []
    return [
        f"language_condition is hindi but {', '.join(wrong)} contains no "
        "Devanagari. Every arm must be probed in its own language, or the "
        "comparison between arms is lost."
    ]


def _has_devanagari(text: str | None) -> bool:
    """Any character in the Devanagari block."""
    return bool(text) and any("\u0900" <= ch <= "\u097f" for ch in text)



def check_server(run: ExperimentRun) -> str | None:
    """Confirm the endpoint is up before dispatching anything.

    Without this, a server that is not running costs a long wait for nothing:
    the connection error is classed as transient, so every agent retries three
    times with exponential backoff, concurrently, and the real cause ends up
    buried in litellm's output. Forgetting to start the server is the most
    likely first failure, so it should be the fastest and clearest one.

    Returns an error string, or None if reachable.
    """
    if not run.api_base:
        return None  # hosted provider: nothing local to check

    import urllib.error
    import urllib.request

    # Ollama and vLLM expose different listing endpoints, and litellm's
    # ollama provider talks to the native API rather than the /v1 shim.
    is_ollama = run.model_backend_id.startswith("ollama/")
    base = run.api_base.rstrip("/")
    url = f"{base}/api/tags" if is_ollama else f"{base}/models"
    wanted = run.model_backend_id.split("/", 1)[-1]

    if is_ollama:
        start_hint = f"    ollama serve        # then: ollama pull {wanted}"
    else:
        start_hint = f"    vllm serve {wanted} --dtype half --port 8000"

    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        return (
            f"cannot reach {url}\n"
            f"  {e.reason}\n\n"
            f"  Start the server first:\n"
            f"{start_hint}\n"
            f"  then confirm: curl {url}"
        )
    except Exception as e:  # noqa: BLE001 - any failure here is a failed check
        return f"cannot reach {url}: {type(e).__name__}: {e}"

    if wanted not in body:
        pulled = "  Pull it with: ollama pull " + wanted if is_ollama else ""
        return (
            f"{url} is up, but does not report serving {wanted!r}.\n"
            f"  It returned: {body[:300]}\n\n"
            f"  model_backend_id must match the served name exactly, after the\n"
            f"  provider prefix, including any :tag suffix.\n{pulled}"
        )
    return None


async def launch(run: ExperimentRun, runs_dir: Path) -> None:
    tracker = CostTracker(run)
    gateway = ModelGateway(
        run.model_backend_id,
        RateLimiter(run.rate_limits),
        tracker,
        api_base=run.api_base,
        # Both were settable on the gateway and configurable on the run, and
        # neither was passed: a config that asked for 5 top logprobs got the
        # gateway's default of 20, and max_concurrency was unreachable
        # outside the notebook. A setting the schema accepts and the launcher
        # drops is worse than one that does not exist, because
        # run_config.json records the value the run did not use.
        max_concurrency=run.max_concurrency,
        top_logprobs=run.probe_top_logprobs,
    )
    await build_orchestrator(
        run, gateway, cost_tracker=tracker, runs_dir=runs_dir
    ).run()

    run_dir = runs_dir / run.run_id
    rows = (run_dir / "interactions.jsonl").read_text().splitlines()
    print(f"\ndone. {len(rows)} interactions -> {run_dir}/")
    print("  interactions.jsonl   the dataset every analysis reads")
    print("  agents_final.jsonl   per-agent state, including stance_history")
    print("  run_config.json      the exact config used, with git_commit_hash")
    print("  checkpoints/         resume point")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, help="path to a run config YAML")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the config and print the plan without calling any model",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
        help="where run output goes (default: runs/). Point this at persistent "
             "storage when running in a notebook, whose filesystem is ephemeral.",
    )
    args = parser.parse_args()

    try:
        run = load_run_config(args.config)
    except ConfigLoadError as e:
        # Every config problem surfaces here, with all errors at once rather
        # than one per attempt.
        print(f"config invalid: {e}", file=sys.stderr)
        return 1

    print(f"\n{args.config}")
    describe(run, args.runs_dir)

    if args.dry_run:
        print("\ndry run, nothing dispatched.")
        return 0

    problem = check_server(run)
    if problem is not None:
        print(f"\n{problem}", file=sys.stderr)
        return 1

    print()
    try:
        asyncio.run(launch(run, args.runs_dir))
    except KeyboardInterrupt:
        print("\ninterrupted. the last completed turn is checkpointed; "
              "re-run the same config to resume.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nrun failed: {type(e).__name__}: {e}", file=sys.stderr)
        print("the last completed turn is checkpointed; re-run to resume.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
