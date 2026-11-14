"""Single-reader experiments: one reader, one thing read, two probes.

Three scripts in `scripts/` share this module -- τ calibration
(`calibrate_tau.py`), E1 single exposure (`run_e1.py`) and the comprehension
check (`check_comprehension.py`). None of them has a population, a graph or a
second round; each asks what one reading does to one reader, which is the
measurement the network runs are interpreted against.

**They must use the same instrument the network runs use.** That is the whole
reason this module exists rather than three scripts with their own prompt
strings:

* every prompt is built by `PromptBuilder` -- `build_probe_prompt`,
  `build_discussion_prompt`, `RoundMemory`. There is no second copy of any
  prompt text anywhere in `scripts/`. A τ measured against a slightly
  different probe prompt is a threshold for a different instrument, and
  nothing downstream could tell.
* every probe is scored by `probe_scorer.score`, against the run's own
  `probe_mass_floor` and `probe_top_logprobs`.
* every setting -- topic, anchors, probe question, bank, persona pool, model,
  `api_base`, temperature, seed -- comes from one `ExperimentRun` YAML per
  language, loaded by `config_loader.load_run_config`. Nothing is hardcoded
  here that a config could carry, because run_config.json is what makes a
  measurement reproducible and a constant in a script is not in it.

**A single-exposure trial is a one-round run, and is numbered like one.** The
baseline probe is turn 0 -- the same round number the orchestrator's
`_probe_round_zero` uses, and for the same reason: it is the reading taken
before anything is read. The exposure, the reply and the second probe are
turn 1. So `prompts.jsonl` from a calibration run and from a campaign run are
keyed the same way, and the same audit reads both.

**Failures are logged, never filled in.** A probe whose mass falls below the
floor, or a call the gateway calls fatal, produces a row with
`status="failed"` and null readings. The trial is not dropped -- a dropped
trial is invisible, and an invisible failure in a τ calibration shifts the
threshold every later result is judged against. A context-length error is
fatal for the whole script, exactly as it is for a run (see
`ContextLengthExceeded`): the prompts that overflow first are the ones whose
language tokenizes least efficiently, so absorbing it per trial would thin
one arm and keep another.

**Everything is resumable.** Each script appends one JSONL row per trial as
it finishes, and on start skips the trials whose key already carries
`status="ok"`. A re-run therefore costs only what the first pass did not
finish. Re-logged keys are read back last-write-wins, the rule
`export_csv.py` applies to a resumed run's `interactions.jsonl`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sandbox.agent_manager import load_persona_pool
from sandbox.config_loader import load_run_config
from sandbox.cost_tracker import CostTracker
from sandbox.logging_writer import LoggingWriter
from sandbox.model_gateway import FatalGatewayError, ModelGateway
from sandbox.models import LANGUAGE_CONDITIONS, Agent, ExperimentRun, PromptRecord
from sandbox.probe_scorer import ChoiceScore, ProbeMassTooLow, ProbeScore
from sandbox.probe_scorer import score as score_probe
from sandbox.probe_scorer import score_choices, to_x
from sandbox.prompt_builder import PromptBuilder, RoundMemory
from sandbox.rate_limiter import RateLimiter
from sandbox.seed_manager import SeedManager

# The four arms, in the order every output file lists them.
LANGUAGES: tuple[str, ...] = tuple(LANGUAGE_CONDITIONS)

# Where step 4 writes the per-language base configs these scripts read.
DEFAULT_CONFIG_DIR = Path("configs/rq1/campaign")

# Turn numbers, named rather than written as literals at eight call sites.
# They are the orchestrator's own round numbers: 0 is the reading taken
# before anything was read, 1 is the round in which something was.
TURN_BASELINE = 0
TURN_EXPOSED = 1

# Fallback concurrency when a config sets no max_concurrency. 32 is a common
# vLLM --max-num-seqs; a config that knows its server's batch size overrides
# it, and that is the value recorded in run_config.json.
DEFAULT_CONCURRENCY = 32

STATUS_OK = "ok"
STATUS_FAILED = "failed"


# --- readings -------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """One probe, as the trial rows record it.

    A thin projection of ProbeScore onto the four numbers every single-reader
    row carries: `x` rather than the raw mean, because every RQ1 model is
    specified in those units, and `conf` because how decided a reader is and
    where it stands move independently (a τ for one is not a τ for the
    other).

    `mean` is kept alongside `x` so a row can be re-derived if `to_x` is ever
    questioned, and `mass` because a reading off 0.51 of the distribution and
    the same reading off 0.95 are not equally trustworthy.
    """

    mean: float
    x: float
    conf: float
    mass: float

    @classmethod
    def of(cls, score: ProbeScore) -> "Reading":
        return cls(
            mean=score.mean,
            x=to_x(score.mean),
            conf=score.side_conf,
            mass=score.mass,
        )


# --- the harness ----------------------------------------------------------


class SingleReaderHarness:
    """One language arm's instrument: a gateway, a prompt builder, a log.

    Constructed per language, from that language's base config, and used for
    every trial in that arm. Languages are driven one at a time rather than
    interleaved, so the gateway's concurrency cap means what the config says
    it means -- four harnesses dispatching at once would put four times
    `max_concurrency` connections on one server, which is the pile-up the cap
    exists to prevent.
    """

    def __init__(self, run: ExperimentRun, out_dir: Path, gateway: ModelGateway | None = None):
        self._run = run
        self._out_dir = out_dir
        self._cost_tracker = CostTracker(run)
        self._gateway = gateway or ModelGateway(
            run.model_backend_id,
            RateLimiter(run.rate_limits),
            self._cost_tracker,
            api_base=run.api_base,
            max_concurrency=run.max_concurrency,
            top_logprobs=run.probe_top_logprobs,
        )
        # meme_lookup raises: no single-exposure trial has a meme in it, and a
        # lookup that returned something would hide a wiring mistake that can
        # only happen by passing a meme-bearing Interaction to the generated
        # path, which these scripts never build.
        self._builder = PromptBuilder(
            run,
            supports_vision=self._gateway.supports_vision,
            meme_lookup=_no_memes,
        )
        self._seeds = SeedManager(run.seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._logger = LoggingWriter(out_dir)

    # Read-only handles the scripts need. Exposed as properties rather than
    # reaching into the private attributes, so a script cannot reconfigure the
    # instrument half-way through a campaign.

    @property
    def run(self) -> ExperimentRun:
        return self._run

    @property
    def language(self) -> str:
        return self._run.language_condition

    @property
    def gateway(self) -> ModelGateway:
        return self._gateway

    @property
    def builder(self) -> PromptBuilder:
        return self._builder

    @property
    def cost_usd(self) -> float:
        return self._cost_tracker.total_usd

    @property
    def concurrency(self) -> int:
        return self._run.max_concurrency or DEFAULT_CONCURRENCY

    def reader(
        self, agent_id: str, persona: str, lean: float, start_key: str | None = None
    ) -> Agent:
        """One reader, built from the trial's own factors.

        `initial_stance` is the lean being tested, which is what
        `PromptBuilder._lean_section` renders in words -- the permanent trait
        every prompt in this arm shows. `agent_id` is a stable string built
        from the trial's factors so a resumed script, a prompts.jsonl audit
        and a trials.jsonl row all name the same reader.

        Constructed directly rather than through AgentManager: there is no
        population here to assign personas across and no seeded draw to make,
        and a manager would invent an agent_0000-style id that said nothing
        about which trial it belonged to.
        """
        return Agent(
            agent_id=agent_id,
            run_id=self._run.run_id,
            persona=persona,
            language_condition=self._run.language_condition,
            model_backend_id=self._run.model_backend_id,
            initial_stance=float(lean),
            start_key=start_key,
        )

    def reply_seed(self, trial_key: str) -> int:
        """The seed the reply call for this trial carries.

        `trial_key` excludes language, so the same trial in four arms hands
        the server the same number and the arms differ in the text of the
        prompt rather than in how the reply to it was sampled -- the same
        property `SeedManager` holds for a campaign's matched set.
        """
        return self._seeds.call_seed("reply", TURN_BASELINE, trial_key)

    # --- the three calls ---------------------------------------------------

    async def probe(
        self,
        agent: Agent,
        history: Sequence[RoundMemory],
        *,
        turn: int,
    ) -> Reading | None:
        """One probe. None means a logged failure, never a substituted value.

        Identical in shape to the orchestrator's `_dispatch_probe`: the prompt
        comes from `build_probe_prompt`, the call from `score_next_token`, the
        reading from `probe_scorer.score` against the run's own mass floor.
        `ProbeMassTooLow` and `FatalGatewayError` are absorbed into None; a
        context-length error is not caught and ends the script.
        """
        prompt_text = self._builder.build_probe_prompt(agent, list(history))
        try:
            response = await self._gateway.score_next_token(prompt_text)
        except FatalGatewayError:
            await self._log(turn, agent.agent_id, "probe", prompt_text, "")
            return None
        # The raw distribution is logged whether or not it scores: a probe
        # that answered off-scale is diagnosable only from the logprobs it
        # answered with.
        await self._log(
            turn,
            agent.agent_id,
            "probe",
            prompt_text,
            response.text or "",
            top_logprobs=response.top_logprobs,
        )
        try:
            return Reading.of(
                score_probe(response.top_logprobs or [], self._run.probe_mass_floor)
            )
        except ProbeMassTooLow:
            return None

    async def reply(
        self,
        agent: Agent,
        delivered_posts: Sequence[str],
        history: Sequence[RoundMemory],
        *,
        trial_key: str,
        turn: int = TURN_EXPOSED,
    ) -> str | None:
        """The reader's reply to what it was shown. None means a logged failure.

        `build_discussion_prompt` on the bank path, so the reader is shown the
        same feed heading, the same directive line and the same "reply in 1 to
        2 sentences" a campaign agent is shown. Seeded, so a rerun of the same
        trial asks the server for the same reply.
        """
        built = self._builder.build_discussion_prompt(
            agent,
            [],
            turn,
            delivered_posts=list(delivered_posts),
            history_rounds=list(history),
        )
        seed = self.reply_seed(trial_key)
        try:
            response = await self._gateway.generate(
                built.text,
                image=None,
                temperature=self._run.temperature,
                seed=seed,
            )
        except FatalGatewayError:
            await self._log(turn, agent.agent_id, "reply", built.text, "", seed=seed)
            return None
        await self._log(
            turn, agent.agent_id, "reply", built.text, response.text or "", seed=seed
        )
        return response.text or ""

    async def choice_probe(
        self,
        prompt_text: str,
        choices: Sequence[str],
        *,
        turn: int,
        key: str,
    ) -> ChoiceScore | None:
        """A probe over letters rather than scale points. None means a failure.

        The comprehension check's call. Same gateway method, same mass floor,
        same absorb-or-raise rule as `probe`; only the support differs, and
        `probe_scorer.score_choices` owns that difference.

        `key` stands in for an agent id in the log, because this call has no
        reader: there is no persona and no lean, deliberately, so that a
        failure to answer is a failure to read the post rather than a persona
        declining to.
        """
        try:
            response = await self._gateway.score_next_token(prompt_text)
        except FatalGatewayError:
            await self._log(turn, key, "probe", prompt_text, "")
            return None
        await self._log(
            turn,
            key,
            "probe",
            prompt_text,
            response.text or "",
            top_logprobs=response.top_logprobs,
        )
        try:
            return score_choices(
                response.top_logprobs or [], choices, self._run.probe_mass_floor
            )
        except ProbeMassTooLow:
            return None

    async def _log(
        self,
        turn: int,
        agent_id: str,
        call: str,
        prompt_text: str,
        response_text: str,
        top_logprobs: list[tuple[str, float]] | None = None,
        seed: int | None = None,
    ) -> None:
        """One line of this output folder's prompts.jsonl.

        Written unconditionally, not behind `run.log_prompts`: these scripts
        produce the numbers the whole of RQ1's inference is calibrated
        against, and the audit that no position number ever reached a reader
        has to be possible against their output and not only against a
        campaign's.
        """
        await self._logger.write_prompt(
            PromptRecord(
                run_id=self._run.run_id,
                turn=turn,
                agent_id=agent_id,
                call=call,
                prompt_text=prompt_text,
                response_text=response_text,
                top_logprobs=list(top_logprobs) if top_logprobs else None,
                seed=seed,
            )
        )


def _no_memes(meme_id: str):
    raise AssertionError(
        f"single-exposure trials have no memes, but one was looked up: {meme_id!r}"
    )


# --- the comprehension prompt --------------------------------------------


def build_comprehension_prompt(
    post_text: str, label_a: str, label_b: str, choices: Sequence[str] = ("A", "B")
) -> str:
    """The post, then a two-way forced choice about what it supports.

    One function, one copy of this text, for `PromptBuilder`'s reason: a
    comprehension check written slightly differently in two places would be
    two different checks, and the one that passed would be the one quoted.

    **The question is in English in all four arms**, and so are the labels.
    They are the run's own anchor labels, which are English everywhere by
    design -- they sit inside English sentences in the probe and the lean --
    and translating the question would mean asking the Hindi arm a different
    question from the English one. The manipulation under test is the
    language of the *post*; everything framing it is held byte-identical, as
    it is everywhere else in RQ1.

    No persona and no lean: this asks whether the model can read the post, so
    a reader with a stance to defend would confound a failure to comprehend
    with a refusal to agree.

    The two labels are passed in rather than ordered here, because the caller
    runs the prompt twice with them swapped -- a model with a standing
    preference for "A" would otherwise pass every pro item and fail every con
    one, and the pass rate would be a measure of position bias.
    """
    letters = ", ".join(choices[:-1]) + f" or {choices[-1]}"
    return "\n\n".join(
        [
            f"A user posted: {post_text}",
            "Which view does this post support?",
            f"{choices[0]}: {label_a}\n{choices[1]}: {label_b}",
            f"Answer with a single letter, {letters}, and nothing else.",
        ]
    )


# --- configs --------------------------------------------------------------


def base_config_path(config_dir: Path, language: str) -> Path:
    return Path(config_dir) / f"single_{language}.yaml"


def load_base_configs(languages: Sequence[str], config_dir: Path) -> dict[str, ExperimentRun]:
    """One validated ExperimentRun per language, keyed by language.

    Raises for a language whose config declares a different
    `language_condition` than its filename claims. A copied config with an
    unedited condition would probe one arm in another arm's language and
    produce a perfectly plausible file with the arms silently collapsed --
    the single most destructive thing that can happen to this experiment, and
    the cheapest to check.
    """
    runs: dict[str, ExperimentRun] = {}
    for language in languages:
        path = base_config_path(config_dir, language)
        run = load_run_config(path)
        if run.language_condition != language:
            raise ValueError(
                f"{path} is the {language!r} base config but declares "
                f"language_condition={run.language_condition!r}"
            )
        if run.regime != "bank":
            raise ValueError(
                f"{path} must be a bank-regime config (regime={run.regime!r}): the "
                "single-exposure scripts measure stance by probe, which only the "
                "bank regime defines"
            )
        runs[language] = run
    return runs


def with_start_mode(
    runs: dict[str, ExperimentRun], start_mode: str | None
) -> dict[str, ExperimentRun]:
    """The base configs, with start_mode overridden when one was asked for.

    E1 and the τ calibration read the same single_<language>.yaml, but need
    not use the same starting position: E1 asks how far one argument moves a
    reader and is cleanest with none, while τ must match the network runs it
    is the threshold for. Re-validated through the model, so "posts" without a
    starter bank fails here rather than on the first trial.
    """
    if start_mode is None:
        return runs
    out = {}
    for language, run in runs.items():
        data = run.model_dump()
        data["start_mode"] = start_mode
        out[language] = ExperimentRun.model_validate(data)
    return out


def check_one_start_mode(rows: Iterable[dict], start_mode: str, path: Path) -> None:
    """Refuses to add trials under one start_mode to a file holding another.

    The trial key does not include the mode, so a second pass under a
    different mode would see every trial as already done and silently
    report the first mode's numbers as the second's. Rows written before this
    column existed were all run under "sentence".
    """
    found = {row.get("start_mode", "sentence") for row in rows}
    if found and found != {start_mode}:
        raise ValueError(
            f"{path} holds trials run with start_mode {sorted(found)}, and this "
            f"sweep uses {start_mode!r}. Write to a different --runs-dir: the "
            "trial key does not include the mode, so they would be mixed."
        )


def start_key_for(persona_index: int, lean: int) -> str:
    """The reader, without the trial: what its starter posts are drawn from.

    Shared by E1 and τ, so the same persona at the same lean is shown the
    same starter posts in both, and by the baseline and the exposed probe of
    one trial, which must differ in the trial's own content alone.
    """
    return f"p{persona_index:03d}|lean{lean}"


def personas_for(run: ExperimentRun) -> list[str]:
    """The run's persona pool, indexed the way the trial rows index it.

    The same loader the population uses, so persona index 7 in a calibration
    row and persona index 7 in a campaign run are the same persona.
    """
    return load_persona_pool(run.persona_pool_id)


# --- resumable output -----------------------------------------------------


class TrialStore:
    """One trials.jsonl, append-only, with a resume rule.

    `completed()` is the set of keys whose *last* written row says
    `status="ok"`. Last rather than first for the reason `export_csv.py`
    deduplicates that way: a key written twice was retried, and the later row
    is the one the run that produced the file actually ended on. A key whose
    last row is `failed` is retried on the next pass -- a failure is worth one
    more attempt, and if it fails again the file simply carries a third row
    saying so.
    """

    def __init__(self, path: Path, key_field: str = "key"):
        self._path = Path(path)
        self._key_field = key_field
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def completed(self) -> set[str]:
        return {
            key
            for key, row in self.rows().items()
            if row.get("status") == STATUS_OK
        }

    def rows(self) -> dict[str, dict]:
        """Every row, keyed, last write winning. Empty if the file is absent."""
        out: dict[str, dict] = {}
        if not self._path.exists():
            return out
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out[row[self._key_field]] = row
        return out

    def append(self, row: dict) -> None:
        if self._key_field not in row:
            raise ValueError(f"trial row has no {self._key_field!r}: {row}")
        if row.get("status") not in (STATUS_OK, STATUS_FAILED):
            raise ValueError(
                f"trial row status must be {STATUS_OK!r} or {STATUS_FAILED!r}, "
                f"got {row.get('status')!r}"
            )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_trials(path: Path, key_field: str = "key") -> list[dict]:
    """A trials.jsonl as a list, deduplicated on key, in first-seen order.

    Used by the scripts' own summary steps and by `analysis/stats.py`. The
    order is the order keys first appear, so a resumed file reads in the same
    order the first pass wrote it rather than in retry order.
    """
    store = TrialStore(path, key_field=key_field)
    rows = store.rows()
    seen: list[str] = []
    if path.exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            key = json.loads(line)[key_field]
            if key not in seen:
                seen.append(key)
    return [rows[key] for key in seen]


# --- dispatch -------------------------------------------------------------


async def gather_in_chunks(tasks: Iterable[Awaitable], chunk_size: int) -> list:
    """Awaits `tasks` in chunks, preserving order, re-raising the first error.

    Chunked rather than one `asyncio.gather` over everything: a calibration
    sweep is hundreds of trials and the gateway's semaphore would hold all of
    them as pending tasks, which is a lot of live coroutines for no gain over
    a bounded window the size of the server's batch.

    `return_exceptions=True` then re-raise, rather than letting `gather`
    cancel the rest: an unexpected exception must still surface, and it should
    surface after the chunk's other calls have been awaited rather than
    leaving them as un-awaited coroutines.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    out: list = []
    chunk: list[Awaitable] = []
    for task in tasks:
        chunk.append(task)
        if len(chunk) == chunk_size:
            out.extend(await _gather_chunk(chunk))
            chunk = []
    if chunk:
        out.extend(await _gather_chunk(chunk))
    return out


async def gather_all(tasks: Iterable[Awaitable]) -> list:
    """One `asyncio.gather`, in order, re-raising the first exception.

    `return_exceptions=True` then re-raise, rather than letting `gather`
    cancel the rest: an unexpected exception must still surface, and it should
    surface after the other calls have been awaited rather than leaving them
    as un-awaited coroutines. The absorbed failures -- a fatal gateway error,
    a probe below the mass floor -- never reach here; they are already rows.
    """
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return list(results)


async def _gather_chunk(chunk: list[Awaitable]) -> list:
    return await gather_all(chunk)


def chunked(items: Sequence, size: int) -> Iterable[Sequence]:
    """`items` in consecutive groups of at most `size`.

    The scripts dispatch one group, write its rows, then dispatch the next, so
    a script killed half-way through has every finished trial on disk rather
    than only the trials of the last chunk. Writing per chunk rather than per
    trial also keeps the rows in trial order instead of completion order,
    which makes two passes over the same sweep diffable.
    """
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --- statistics -----------------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float:
    """The p-th percentile by linear interpolation between order statistics.

        sorted(values)[k] interpolated at k = (n - 1) * p / 100

    Written out rather than taken from numpy so the definition is readable at
    the point it is used and a test can hand-compute it. It is numpy's default
    ("linear") method, so a reader who checks against `np.percentile` gets the
    same number.

    Raises on an empty sequence rather than returning 0 or nan: a threshold
    computed from no observations is not a threshold, and a 0 here would
    classify every later movement as significant.
    """
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {p}")
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("percentile of an empty sequence is undefined")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# --- command line ---------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The three options every single-exposure script takes.

    `--runs-dir` for the same reason `run_simulation.py` has one: a notebook's
    filesystem is ephemeral while these outputs are what every later analysis
    reads, so it has to be pointable at mounted storage.
    """
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(LANGUAGES),
        choices=list(LANGUAGES),
        help="which arms to run (default: all four)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="where output goes (default: runs/). Point this at persistent "
        "storage when running in a notebook.",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"directory holding single_<language>.yaml (default: {DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N trials of each arm, for a smoke test",
    )
    parser.add_argument(
        "--start-mode",
        choices=("sentence", "none", "posts"),
        default=None,
        help="override the base configs' start_mode for this sweep: how the "
             "reader's starting position is set (see ExperimentRun.start_mode)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="run a fixed random sample of N trials of each arm, spread over "
             "the whole grid (items, personas, leans). Unlike --limit, which "
             "takes the first N and so covers only the first item or persona",
    )
    return parser


# Fixed, and not the run's seed: the sample names which trials are run, and
# must name the same ones in all four arms and on every pass. A sample that
# moved with the seed would give each language a different subset of items.
SAMPLE_SEED = 20261114


def apply_sample(trials: list, sample: int | None) -> list:
    """A fixed random `sample` of the trials, in their original order.

    `--limit` takes the first N trials, which is what a plumbing check wants
    and is useless for anything read as a number: with items outermost the
    first 120 E1 trials are two items, and with personas outermost the first
    120 calibration trials are one persona. A sample spreads the same budget
    over the whole grid.

    Random rather than every k-th, because a stride aliases with the inner
    loops -- a stride equal to the number of personas picks persona 0 every
    time. The draw depends only on the grid's length, so the four language
    arms, whose grids are identical, get the same trials, and a second pass
    names the same ones and resumes rather than widening.
    """
    if sample is None:
        return trials
    if sample < 1:
        raise ValueError(f"--sample must be >= 1, got {sample}")
    if sample >= len(trials):
        return trials
    import random

    chosen = sorted(random.Random(SAMPLE_SEED).sample(range(len(trials)), sample))
    return [trials[i] for i in chosen]


def apply_limit(trials: list, limit: int | None) -> list:
    """The first `limit` trials, or all of them.

    Applied to the full ordered trial list before the resume filter, not
    after, so `--limit 2` names the same two trials on every pass -- a limit
    applied to what is left to do would walk through the whole sweep two at a
    time, which is not what a smoke test asks for.
    """
    if limit is None:
        return trials
    if limit < 1:
        raise ValueError(f"--limit must be >= 1, got {limit}")
    return trials[:limit]
