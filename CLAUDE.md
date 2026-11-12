# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`sandbox` is a Python simulation of populations of LLM-backed agents discussing a topic over multiple turns. It exists to answer three research questions: whether the language or script of a post changes how much it moves an LLM reader (RQ1); whether the quality of an agent's stance-updating reasoning differs by language and correlates with polarization speed (RQ2, a post-hoc analysis of RQ1's data); and whether injecting memes from a labeled dataset into the discussion accelerates polarization convergence (RQ3).

There is **one simulation mode** (multi-turn) and **two regimes**, selected by `regime`:

- **`generated`** (the original, default). Agents write their own posts and self-report a stance as a `STANCE: n` line. Memes are a content-replacement mechanism within it — a meme-bearing agent's turn uses a dataset stance label instead of an LLM call, then gets sampled like any other agent. This is what RQ3 uses.
- **`bank`** (added for RQ1). Posts come from a fixed, pre-translated argument CSV, so the argument is held identical across language conditions and only its rendering varies; agents never see a position number or an agent id; and stance is measured by a **separate probe call** whose next-token logprobs over the digits 1-7 give a continuous reading. A round is two phases against one frozen snapshot — every reply, then every probe — and the probe never enters any agent's history. Exactly three things vary across the four language arms: the bank text, `probe_question`, and the one directive line. The rest of the prompt — persona, topic, the lean sentence, the anchor labels, every heading — is byte-identical in all four, and `tests/test_integration_run.py` compares whole runs line by line to keep it that way.

The two regimes are independent of `neighbor_strategy`, which selects `alpha` (homophilous sampling over the whole population) or `graph` (a fixed Watts-Strogatz social graph, which RQ1 uses so that who an agent reads is structural rather than a function of where it already stands).

The project was built in tiers, bottom-up, one phase per commit. Each phase was planned in a doc before implementation; those docs were removed in `c0facd4` and the build log now lives in git history.

## Commands

Activate the venv first: `source .venv/bin/activate` (a `.venv` already exists at repo root with all deps installed).

```bash
# Run the full test suite
python -m pytest

# Run a single test file
python -m pytest tests/test_agent_manager.py

# Run a single test
python -m pytest tests/test_agent_manager.py::test_name -q

# Reinstall the package in editable mode (after pyproject.toml changes)
pip install -e ".[dev]"
```

There is no lint/format command configured yet. `asyncio_mode = "auto"` is set in `pyproject.toml`, so `async def test_*` functions run without `@pytest.mark.asyncio`.

## Architecture

### Layered dependency tiers

The system is built as strict dependency tiers — a module only depends on tiers below it, never sideways or up. This ordering is intentional and drives both the build sequence and how you should reason about changes:

- **Tier 0** (`sandbox/models.py`, `config_loader.py`, `seed_manager.py`, `cost_tracker.py`, `rate_limiter.py`, `probe_scorer.py`) — pure data models and standalone infrastructure. No dependency on each other. `probe_scorer.py` is pure arithmetic: top-logprobs in, `(mean, dist, mass, side_conf)` out, no I/O. (`config_loader` imports `bank_manager` *inside a function* to validate a bank at load time, deliberately, so the tier direction is not inverted at module scope.)
- **Tier 1** (`agent_manager.py`, `meme_pool_manager.py`, `model_gateway.py`, `bank_manager.py`) — depends only on Tier 0.
- **Tier 2** (`interaction_engine.py`, `stance_parser.py`, `vision_fallback.py`, `logging_writer.py`, `checkpoint_manager.py`) — depends only on Tier 0/1, and not on each other. Built and unit-tested.
- **Tier 3** (`simulation_orchestrator.py`, `prompt_builder.py`) — the integration tier; the only module that calls every other component, wiring them into the per-turn loop. Built and unit-tested, including one end-to-end integration test. `prompt_builder.py` has no design in either `Architecture-docs/` file: it was designed from scratch, and the reasoning lives in its module docstring and in git history.
- **Post-hoc analysis** (**not yet built**: `stance_regression.py`, `coherence_scorer.py`, `codemix_ratio_tracker.py`, `sentiment_proxy.py`, `bimodality_analysis.py`, `embedding_cluster.py`) — offline, operates only on a completed run's `interactions.jsonl`, no orchestrator dependency.

Tier status drifts in prose. Don't trust a status line here; check the filesystem and `python -m pytest`. Current: **17 modules, 606 passing tests, Tiers 0-3 complete**, including the RQ1 bank regime. The post-hoc analysis tier is the only unbuilt part of the original design.

Check `Architecture-docs/full_design_doc.md` §3 for the authoritative module-by-module LLD (function signatures, error-handling contract, concurrency notes) before implementing any new module — the design doc's sample code is the spec each module is a direct port of, not just background reading. `Architecture-docs/sandbox_hld.md` carries the HLD, the schema derivation, and three passes of adversarial self-critique with numbered Fixes (Fix A–N) — when a design decision seems arbitrary (e.g. why the memory window is turn-count-based, not token-budget-based; why the same RNG seed is deliberately reused across language conditions), it's almost certainly resolving a specific numbered Fix in that document, not an accident.

### The bank regime's per-round flow (RQ1)

Round 0 probes every agent once, before anything is read, so the starting stance is a measurement in the dataset rather than something inferred from `initial_stance`. Then for each of K rounds, all against one frozen snapshot:

1. each agent's **side** (`pro` above the 7-point midpoint, `con` below, a seeded draw exactly on it);
2. the **bank item** it posts for that side — a pure function of (seed, sender, side, how many it has posted on that side), never of language or turn, which is what makes four language arms post the same argument;
3. its **neighbours**, from the graph;
4. what it is **delivered** (under `delivery: off`, the E4 control, exactly one post: `run.neutral_message`, logged as the reserved item id `__neutral__`);
5. **Phase A** — every reply, concurrently (`gateway.generate`);
6. **Phase B** — every probe, concurrently (`gateway.score_next_token`), each seeing this round's own reply;
7. one `Interaction` per agent written, then the successful ones applied.

Phase B depending on Phase A is not a Fix F violation: it is the agent's *own* output, read by nobody else. Fix F forbids one agent's turn-t output reaching another agent's turn-t stimulus, and steps 1-4 all complete before either phase dispatches.

The `delivery: off` control runs both phases too. It reads one on-topic message that takes no side and replies to it, so it costs the same M × K reply calls a treatment arm does and differs from it in the persuasive content of what agents read and in nothing else. A control that read nothing and replied to nothing had a history that never changed, so it could not show drift even in principle.

Two pieces of bank state are **not in the checkpoint** and are rebuilt from `interactions.jsonl` on resume: each sender's position in its bank cycle, and the rounds each agent remembers. The rebuild is **as of `last_completed_turn`** and ignores every later row — the log runs ahead of the checkpoint when a round died part-way through, and those rows describe a round the resumed pass is about to re-run. Within that cutoff, rows are keyed on `(turn, speaker_agent_id)` keeping the last write, the same rule `export_csv.py` applies, so a re-logged turn cannot double-count the cycle.

### Core data flow (implemented in `simulation_orchestrator.py`)

Config Loader → `ExperimentRun` → Agent Manager creates M agents → for each of K turns: freeze snapshot → Interaction Engine resolves N neighbors per agent (`AlphaSampling`) → Meme Pool Manager resolves which scheduled speakers post a meme this turn → dispatch concurrently (generated-text speakers → Model Gateway; meme speakers → free dataset lookup, no LLM call) → collect results → write `Interaction` records → advance snapshot → checkpoint → next turn.

The frozen-snapshot ordering guarantee (neighbor and meme-injection resolution both complete *before* any dispatch begins, so no agent's turn-t output is ever read by another agent's turn-t neighbor/meme resolution) is load-bearing for reproducibility and is referred to throughout the docs and codebase as "Fix F."

### Key design invariants (apply when touching any module)

- **Never silently clamp or default.** Invalid `alpha`/`N`, out-of-range parsed stances, unrecognized model ids, missing pricing entries — all raise immediately (`ValueError`/custom exceptions) rather than being coerced to a safe value. This is a deliberate, repeated pattern across the codebase (see Fix I: "never fabricate a placeholder stance") — don't add fallback/default behavior to "fix" a raised error unless the design doc says to.
- **Seeded RNGs only, never the global `random` module.** `seed_manager.py`'s `SeedManager` hands out three independently-seeded `random.Random` instances (`neighbor_sampling_rng`, `meme_injection_rng`, `persona_assignment_rng`), deliberately offset (`seed`, `seed+1`, `seed+2`) so the three streams never correlate, plus three derived families: `turn_rng(purpose, turn)`, `sender_rng(purpose, sender_id)` and `graph_seed`. Reproducibility depends on every stochastic choice consuming from the correct purpose-specific instance.
- **No condition may enter a seed.** `SeedManager` takes the base seed and nothing else — no language, topic or trial — and every derivation is a function of that alone. This is what makes a *matched set*: four runs differing only in `language_condition` share a graph, the same neighbour draws, and the same bank item for the same sender on the same side, so the arms differ in the rendering of the text and in nothing else. `tests/test_seed_manager.py` pins it from both directions, including that the constructor has no condition parameter to mix in.
- **Never truncate silently, and never lose an agent to a limit.** Bank-regime agents keep their **full** history by default (`memory_rounds: None`); a window bounds the effect the experiment can measure, so it is opt-in. The two ways unbounded history can end are both loud and both fatal for the whole run: `max_prompt_tokens` (checked against the provider's reported count, raising `PromptBudgetExceeded`) and `ContextLengthExceeded`, which is deliberately **not** a `FatalGatewayError` so the orchestrator cannot absorb it per agent. The agents that overflow first are the ones whose language tokenizes least efficiently, so a per-agent failure would drop one arm's agents and keep another's.
- **In the bank regime, no number reaches an agent.** Not its own position, not a neighbour's, not an agent id. An agent that can see a position number does not have to read the arguments — it can move its own number towards the ones it was shown, and the result is indistinguishable from persuasion. The probe exists so stance can be read without ever showing one, and it never writes into any agent's history. A prompt audit in `tests/test_integration_run.py` checks every prompt of a whole run, and `prompts.jsonl` (on by default for this regime, via `log_prompts`) makes the same audit possible against a real run's output rather than only in the test suite. The scale's own digits (the probe's `1 means .../7 means ...` anchors, and its "answer with a single digit from 1 to 7") are the one exception, and are necessary for the model to answer at all.
- **The lean sentence is a permanent trait, in every prompt.** `LEAN_SENTENCES` renders where an agent *started*, in words, directly after the topic line of every bank reply prompt and every probe prompt. Not round 1 and round 0 only: a sentence that vanishes afterwards makes the round-0 probe and every later probe two different instruments, so the step between the first two readings partly measures the removal of a sentence — and under `delivery: off`, where nothing else about a probe changes between rounds, it was the only thing that changed at all.
- **Fatal vs. per-agent-recoverable errors are distinguished deliberately.** `FatalGatewayError`/`StanceParseFailure` (after retry budget exhausted) become a `failed_logged_null` `Interaction` for that one agent and the turn continues for everyone else (`asyncio.gather(..., return_exceptions=True)`); anything else propagates and aborts the whole run. Don't broaden what gets caught — an uncaught, unexpected exception type is meant to surface loudly as a bug, not be absorbed into a soft-failure record.
- **`CostTracker.record()` is synchronous**, not async, even though it's called from `ModelGateway.generate()` (async) — this was a deliberate early decision, not an oversight. Don't add `await` to that call site.
- **Pydantic v2 models throughout** (`sandbox/models.py`), not dataclasses — the HLD's own sketches use dataclasses, but the codebase supersedes that with `pydantic.BaseModel` for validation (regex-constrained enums via `pattern=`, cross-field `@model_validator`). When porting a design-doc snippet, translate its dataclasses to the existing Pydantic models rather than introducing a parallel dataclass.
- **Vision support is a static lookup table, not a runtime probe**, keyed by `pricing_table.yaml`'s model set — keep the vision-capable/text-only lists in `model_gateway.py` and the pricing table in sync; an unrecognized `model_backend_id` should raise, never silently default to text-only.

### Data and fixtures

- `data/memes/*.jsonl` — one JSON object per line, loaded via `MemeContent.model_validate_json()`.
- `data/personas/*.jsonl` — **despite the extension, these are plain text, one free-text persona description per line, not JSON**. `agent_manager.py`'s `_load_persona_pool()` just strips and splits lines; nothing parses JSON. Don't add `model_validate_json()` here.
- `test_pool.jsonl` variants are small (3-4 entries) fixtures for tests; `pool_20.jsonl`/`empty_pool.jsonl` exist for specific acceptance-criteria tests (cycling, empty-pool error path). All three images referenced by `data/memes/test_pool.jsonl` now exist; two were missing until a pre-flight audit found the meme path would raise `FileNotFoundError` for roughly two sampled memes in three.
- `data/bank/*.csv` — argument banks: `item_id, arg_id, strength, side, english, hinglish, romanized_hindi, hindi`. The language column names ARE `language_condition` values, so a header typo makes the bank unrenderable for that arm. Validated in full at config load (not merely stat()ed), reporting every problem at once: a bank with one empty Hindi cell would otherwise run to completion and produce an arm in which one argument was silently blank, which is indistinguishable in the output from an argument that failed to persuade. `test_bank.csv` is a 4-item fixture, both sides, both strengths, and deliberately **digit-free in every language column** so the prompt audit can tell a leaked position number from the argument's own text. The item id `__neutral__` is reserved for the `delivery: off` control's side-free message and is rejected in a bank.
- `pricing_table.yaml` — hand-maintained `provider -> model_id -> modality -> {prompt_per_1k, completion_per_1k}`, keyed by the BARE model id (`ModelGateway._bare_model_id` strips the provider prefix). `CostTracker` raises on a missing entry *after* the call completes, so every model actually used must appear here. The openai/anthropic/google entries remain illustrative placeholders; the `hosted_vllm`, `ollama`, `groq` and `cerebras` entries are genuinely 0.0, since self-hosted compute is billed by the hour rather than per token.
- `runs/` — output directory, one run = one subdirectory (`run_config.json`, `interactions.jsonl`, `agents_final.jsonl`, `checkpoints/`, plus `prompts.jsonl` when `log_prompts` is on and `graph_edges.csv` under the graph strategy), per `logging_writer.py`. Gitignored. Completed campaigns are kept under `outputs/` instead; `outputs/chorus_run1/` holds the first real dataset, 8 runs and 8,000 interactions.
- `configs/` — run configs. `example_run.yaml` documents every field; `smoke_test.yaml` (vLLM) and `smoke_local.yaml` (Ollama on Apple Silicon) are minimal five-call runs for checking plumbing. `configs/rq1/` holds the bank-regime smoke configs: one per language arm (`smoke.yaml`, `smoke_hinglish.yaml`, `smoke_romanized_hindi.yaml`, `smoke_hindi.yaml`, all sharing a seed) plus `smoke_off.yaml` (the `delivery: off` control, which also needs a `neutral_message`). The four arms differ in exactly `run_id`, `language_condition` and `probe_question`; a test in `tests/test_config_loader.py` pins that, including that the anchor labels stay English. **`delivery` must be quoted in YAML** — YAML 1.1 reads bare `off` as the boolean `False`, so an unquoted value is rejected by the schema; a test asserts every shipped config parses it as a string.

### Where to look before starting new work

The two `Architecture-docs/` files are the stable spec: `full_design_doc.md` §3
for the module-by-module LLD, `sandbox_hld.md` for the HLD and the numbered
Fixes. **Both predate the code and contain errors that were found by building
it** (see the invariants above), so treat them as design intent, not as truth
about current behaviour.

The phase-by-phase build log now lives in **git history** rather than the
working tree; the phase docs were deleted in `c0facd4`. `git log --oneline`
walks the tiers in order, and the commit messages carry the design decisions and
the defects found, with their reasoning. Twelve defects were found and fixed
during Phase 4 and a pre-flight audit, several of them faithful ports of errors
in the design doc's own sample code.

### Running things

```bash
python run_simulation.py configs/smoke_test.yaml --dry-run   # validate, no calls
python run_simulation.py configs/smoke_test.yaml             # a real run
python export_csv.py outputs/chorus_run1                     # runs -> analysis CSVs
```

`notebooks/chorus_colab.ipynb` runs a full campaign on a Colab GPU: vLLM
serving, the topic x language grid, and the combining step. Inference is
self-hosted on open-weight models, so a campaign costs GPU time and nothing
else.

Two hazards worth knowing before touching the Colab path: `asyncio.run()` raises
inside a notebook cell, so runs launch as subprocesses or via top-level `await`;
and the Colab filesystem is ephemeral while `runs/` holds the checkpoint resume
depends on, so `--runs-dir` must point at mounted Drive.

### Known gaps

- **Post-hoc analysis does not exist.** No regression, bimodality, code-mix
  index, sentiment, embedding, or network analysis, and no τ calibration or
  assortativity computation over `graph_edges.csv`. `export_csv.py` is the only
  analysis tooling.
- **One language per run.** The bank regime renders a delivered post in its
  *sender's* language while the resume path renders remembered posts in the
  *run's* language, since the log records item IDs and not who posted each one.
  Those agree exactly while every agent shares the run's condition, which
  `_verify_one_language_per_run` checks at run start. A mixed-language arm (E5)
  would need the sender recorded per delivered item first.
- **Resumed runs can duplicate rows.** The orchestrator writes interactions one
  at a time and checkpoints only after a full turn, so a crash mid-turn leaves
  rows with no checkpoint and the resumed pass re-logs that turn (the same is
  true of `prompts.jsonl`). `export_csv.py` deduplicates on read, keeping the
  later write, which `agents_final.jsonl` confirms is authoritative. The
  duplication itself is unfixed; what *is* fixed is that the resume no longer
  reads those rows back as state — see the per-round flow above.
- **Prompt length differs by language.** Measured at 693 tokens for English and
  809 for Hinglish, a 17% gap, because code-mixed `reason_text` tokenizes less
  efficiently and feeds back into memory and neighbour blocks. Fix A's
  turn-count guarantee still holds; the information content per turn does not.
  The bank regime reduces but does not remove this: the posts are now identical
  arguments rather than freely written text, but a Hinglish rendering of the
  same argument still tokenizes to more than its English one, and the agents'
  own replies are still free text. Full history (the default) makes the gap
  compound over a run rather than sit at a fixed window, which is why
  `reply_prompt_tokens` and `probe_prompt_tokens` are logged per call —
  reporting the gap is part of the result, not a diagnostic — and why
  `max_prompt_tokens` exists. The four language directives are held to a
  42-character spread for the same reason, and `test_prompt_builder.py` bounds
  it — keep any new directive short.
- **No real persona pool.** Only `test_pool.jsonl` (4) and `pool_20.jsonl` (20).
  At M=100 each persona is shared by five agents.
