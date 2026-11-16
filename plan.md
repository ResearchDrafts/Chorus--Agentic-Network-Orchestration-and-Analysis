# New opinion-CSV experiment pipeline (language-toggle echo chamber study)

## Context

The existing codebase (`sandbox/`) is built around two tightly-invariant
regimes — `bank` (probe-measured stance, no free number ever reaches an
agent) and `generated` (free-text posts, self-reported `STANCE: n`) — both
logging to `interactions.jsonl` and both carrying a long list of load-bearing
invariants (Fix F's frozen-snapshot ordering, "no number reaches an agent" in
the bank regime, seeded reproducibility, `extra="forbid"` schemas, etc.),
pinned by an extensive test suite.

What's being asked for here is a different experiment shape entirely:
agents write free-text opinions to a **wide CSV** (one row per agent, one new
trio of columns per round), self-report a numeric belief, opinion **and** a new
**confidence** value every round, sample neighbours' opinions filtered by a
**belief-similarity threshold**, always sample opinions from the last round i.e. if you're in round 5, you sample opinions from round 4, no other round. 
and the population's language is controlled
by **independent yes/no toggles** (not the existing single `language_condition`
enum) — with the special case that turning on *all four* toggles together
produces a genuinely mixed-language population, each agent speaking its own
`language preference` from `data/personas/pool_100.jsonl`.

This doesn't fit inside the existing regimes without compromising their
invariants, so the plan builds it as a **new, additive, self-contained
pipeline** that reuses the right low-level building blocks (model gateway,
RNG manager, WS graph construction, the Jo-et-al. metrics functions) but
touches none of the existing regime/tier code, config schema, or tests. The
goal is a working instrument to compare belief-similarity-filtered echo
chamber formation across language conditions, per the "Decoding Echo
Chambers" / Jo et al. metrics (`Pz`, `DG`, `NCI`, mean belief shift) already
implemented in `analysis/metrics.py`.

## What already exists and will be reused as-is

- `sandbox/model_gateway.py` — `ModelGateway.generate(prompt_text, temperature, *, seed=None, max_tokens=None)` for every free-text LLM call (initial opinion + every round's update). `seed`/`max_tokens` follow the existing not-sent-when-`None` convention.
- `sandbox/seed_manager.py` — `SeedManager(seed)`, its `turn_rng(purpose, turn)` / `sender_rng(purpose, sender_id)` / `graph_seed` / `call_seed(purpose, turn, agent_id)`. No changes needed to this file — `purpose` is already an arbitrary string, so new purposes (`"initial_belief"`, `"opinion_reply"`, `"opinion_recsys"`) need no code change there.
- `sandbox/interaction_engine.py`'s `GraphSampling` — reused **only** for its constructor (`nx.connected_watts_strogatz_graph` built once from `graph_seed`) and its `neighbors_of(agent_id)` method, to get each agent's fixed candidate neighbourhood. Its own `select_neighbors`/`GraphSimilaritySampling` are **not** reused directly — they're hard-wired to `Agent.current_stance()`/`probe_scorer.to_x`'s 1-7 probe scale, which doesn't exist in this pipeline (confirmed by exploration: "not reusable as-is").
- `sandbox/prompt_builder.py`'s `LANGUAGE_DIRECTIVES` dict (line 78) — imported verbatim for the four language directive lines (`"Write your reply in English."`, Hinglish, Romanized Hindi, Hindi/Devanagari), so wording stays consistent with the rest of RQ1 rather than being re-translated.
- `sandbox/stance_parser.py` — not imported, but its exact pattern (a compiled regex tag, `StanceParseFailure`-style exception, never clamp) is the template for the new parser below.
- `analysis/metrics.py` — `population_variance(values)`, `global_disagreement(graph, x_by_agent)`, `nci(graph, x_by_agent)`. All three take `x` in `[-1, 1]` and do the `v = 2x` conversion internally. Since this pipeline's belief is already on the `[-2, 2]` (`v`) scale, the adapter is one line: `x_by_agent = {aid: belief / 2.0 for aid, belief in beliefs.items()}`. Mean belief shift needs no adapter at all — since `shift_v = 2*(x - x0) = belief - belief0` algebraically, it's computed directly from the raw belief values.
- `data/personas/pool_100.jsonl` — the persona + language source. **Nothing is generated or added here.** One free-text line per agent, already containing age, gender, Big Five (5-level ordinal: Low/Low-Medium/Medium/Medium-high/High per trait), and a `language preference: <English|Hindi|Hinglish|Romanized Hindi>` field. Confirmed: exactly 100 lines, exactly these 4 language values, no missing cases.

## New files (nothing existing is modified)

1. **`sandbox/opinion_pool.py`** (Tier 1 — depends only on stdlib)
   - `load_opinion_using_persona_pool(path) -> list[PersonaRecord]`. `PersonaRecord` = `(agent_id, raw_text, language_preference)`. Parses `language preference: ([A-Za-z ]+)` out of each line via regex and normalizes to one of `{"english", "hindi", "hinglish", "romanized_hindi"}`; raises `ValueError` naming the line number if the field is missing or doesn't match one of the four — never silently drops a persona, per the codebase's "never silently default" convention. The rest of the line is kept as `raw_text` and passed into prompts verbatim (not field-by-field parsed) — avoids a brittle 10-field parser for data that's otherwise just descriptive context.

2. **`sandbox/opinion_models.py`** (Tier 0 — pure Pydantic models, `extra="forbid"`)
   - `OpinionExperimentConfig`: `run_id`, `topic`, `M: int | None` (default `None` = use the whole pool), `K: int` (rounds), `languages: dict[str, bool]` with exactly the 4 keys, `similarity_threshold: float = 2.0` (on the belief/`v` scale directly — no `to_x` conversion anywhere in this pipeline), `N: int` (neighbours sampled per round), `graph_degree: int`, `graph_rewire_p: float`, `seed: int`, `model_backend_id: str`, `api_base: str | None`, `persona_pool_path: str`, `max_reply_tokens: int = 512`, `output_dir: str`.
   - `OpinionAgentState`: `agent_id`, `persona_text`, `language`, latest `belief`, `confidence`, `opinion_text`.

3. **`sandbox/opinion_llm.py`** (Tier 1/2 — prompt + parse, no I/O beyond string building)
   - `build_initial_prompt(topic, persona_text, initial_belief, language) -> str` — tells the agent its topic, persona (raw line), and that it holds belief `initial_belief` on the `-2..+2` scale **in words, not as the literal number shown to it as an instruction to echo back** (same "lean sentence" spirit as the bank regime: the agent is pointed at a position, not handed a digit to parrot). Appends the language directive from `LANGUAGE_DIRECTIVES` (imported from `prompt_builder.py`).
   - `build_update_prompt(topic, persona_text, own_state, neighbours: list[OpinionAgentState], language) -> str` — shows the agent its own last opinion/belief/confidence and each sampled neighbour's opinion text + belief, asks it to write an updated opinion and re-report `BELIEF:` and `CONFIDENCE:`. Note that the LLM can also choose to not update if it likes.
   - `parse_opinion_response(raw_text) -> (opinion_text, belief, confidence)` — two compiled regexes, `BELIEF:\s*(-?\d+(?:\.\d+)?)` and `CONFIDENCE:\s*(\d+)`, modeled directly on `stance_parser._STANCE_PATTERN`. Raises `OpinionParseFailure` (new exception, same role as `StanceParseFailure`) if either tag is missing, `belief` falls outside `[-2, 2]`, or `confidence` isn't an integer `1..5` — never clamps.

4. **`sandbox/opinion_csv.py`** (Tier 2 — the CSV I/O)
   - `write_initial_rows(path, rows: list[dict])` — creates `initial_stance.csv` with header `agent_id, language, persona, initial_belief, initial_opinion, initial_confidence`.
   - `append_round_columns(path, round_num, updates: dict[agent_id, (opinion, belief, confidence, status)])` — reads the CSV (pandas, same library `export_csv.py` already uses), adds `round_{k}_opinion`, `round_{k}_belief`, `round_{k}_confidence`, `round_{k}_status` columns, rewrites the file. A widening-columns CSV can't be done as a pure append (unlike the JSONL files elsewhere), so this is a full rewrite each round — fine at the pool's scale (~100 agents × a handful of rounds).

5. **`sandbox/opinion_metrics.py`** (Tier 2/3 — the analysis adapter)
   - `compute_round_metrics(graph, beliefs: dict[agent_id, float], beliefs_round0: dict[agent_id, float]) -> dict` — builds `x_by_agent = {aid: b/2 for aid, b in beliefs.items()}`, calls `analysis.metrics.population_variance` (→ `Pz = 4 * var`), `analysis.metrics.global_disagreement(graph, x_by_agent)` (→ `DG`), `analysis.metrics.nci(graph, x_by_agent)` (→ `NCI`), and computes `mean_belief_shift`/`mean_abs_individual_shift` directly from `beliefs - beliefs_round0` (no `x` round-trip needed, see above).
   - `write_round_metrics_csv(path, round_num, metrics)` — same append-a-row-per-round pattern as `LoggingWriter.write_graph_edges_round`, writing to `opinion_round_metrics.csv`.

6. **`sandbox/opinion_orchestrator.py`** (Tier 3 — the per-round loop)
   - Loads `OpinionExperimentConfig` from YAML, loads the persona pool via `opinion_pool.load_opinion_persona_pool`.
   - **Language resolution** (the toggle logic):
     - Exactly one of the 4 `languages` keys `True` → every agent's language is forced to that one value, overriding the pool's own `language preference`.
     - All 4 `True` → each agent's language is its own pool `language_preference` (mixed population, no RNG — it's a persona attribute).
     - Any other combination (2 or 3 `True`) → raises `NotImplementedError` naming the unsupported combination — deliberately unhandled per your "ignore that edge case for now," rather than silently guessing a resolution rule.
   - **Initial belief**: drawn per agent, uniform random in `[-2, 2]`, from a dedicated seeded stream (`seed_manager.turn_rng("initial_belief", 0)`) — no code change to `seed_manager.py` needed.
   - **Graph**: one `GraphSampling(agent_ids, graph_degree, graph_rewire_p, N, graph_seed)`, built once, reused for every round (`neighbors_of`).
   - **Round 0**: for each agent concurrently, `gateway.generate(build_initial_prompt(...), seed=call_seed("reply", 0, agent_id), max_tokens=max_reply_tokens)`, parse, write `initial_stance.csv`.
   - **Rounds 1..K**: for each agent, candidates = `graph.neighbors_of(agent_id)`; eligible = those with `abs(belief_i - belief_j) <= similarity_threshold` using the previous round's beliefs; ranked via a private per-reader RNG seeded `f"{seed}|opinion_recsys|{turn}|{agent_id}"` (mirrors the existing Fix-11 isolation pattern: shuffle for tie-break, stable sort by gap ascending, take up to `N`, shuffle again for delivery order — **never pad back up to `N`** if fewer qualify, same rule as the existing `GraphSimilaritySampling`). Zero eligible neighbours → skip that agent's reply call for the round, copy its previous opinion/belief/confidence forward, log `status="no_feed"` for that round's columns (mirrors the bank regime's Fix 9 `no_feed` handling). Otherwise call `gateway.generate(build_update_prompt(...), seed=call_seed("reply", k, agent_id), max_tokens=...)`, parse, `status="success"`.
   - After each round: `opinion_csv.append_round_columns(...)` then `opinion_metrics.compute_round_metrics(...)` + `write_round_metrics_csv(...)`.
   - Writes a `run_config.json`-equivalent (the resolved config, dumped once) for provenance.

7. **`configs/opinion/example.yaml`** — template config with all the fields above, PLACEHOLDER-marked smoke values (small `M`, `K=3`), modeled on `configs/rq1/base.yaml`'s own commenting style.

8. **`scripts/run_opinion_experiment.py`** — thin CLI entry point: `python scripts/run_opinion_experiment.py configs/opinion/example.yaml` → loads config, calls `opinion_orchestrator.run(config)`.

## What is explicitly out of scope / deliberately simple

- **No resume/checkpoint support.** Unlike the existing orchestrator, a crash restarts the whole run from scratch. Flagging this since a long K might make that costly.
- **Confidence is logged, not mechanically load-bearing.** It doesn't weight the similarity filter or the belief update in this version — it's an output variable for post-hoc analysis, same role Jo et al. give a self-reported-conviction-adjacent signal. Can be wired in later if the RQ calls for it.
- **2-or-3-active-language combination is unimplemented** (raises loudly) per your "ignore that edge case for now."
- **No test suite changes**, per your instruction.

## Decisions required (my recommendation first in each)

1. **Initial belief distribution.** *Recommended:* uniform random float in `[-2, 2]` per agent, seeded (matches the existing `lean_assignment: "random"` default in `base.yaml`). *Alternative:* a symmetric two-camp split like `balanced_two_camps` if you want guaranteed polarized starting conditions rather than a random spread.
2. **Zero-eligible-neighbour behavior.** *Recommended:* freeze that agent's opinion/belief/confidence for the round (`status="no_feed"`), consistent with Fix 9's existing precedent. *Alternative:* still force a reply call with no neighbour content (pure self-reflection, no exposure) — this would keep the agent "active" every round but changes what a frozen row means for the metrics.
3. **Number of neighbours sampled per round (`N`).** *Recommended:* a configurable field, default matching `base.yaml`'s `N: 6`. Confirm this default is fine, or give me a different number.
4. **Where metrics get written.** *Recommended:* `opinion_round_metrics.csv` inside the same run's `output_dir`, one row per round (`Pz, DG, NCI, mean_belief_shift, mean_abs_individual_shift`). *Alternative:* route through `results/` the way `analysis/metrics.py`'s existing CSVs do, if you want it to live alongside the RQ1 analysis outputs instead.
5. **Persona rendering.** *Recommended:* pass each pool line's full raw text into the prompt as-is (it already reads as one natural sentence covering age/gender/Big Five/etc.) rather than parsing out only "age, gender, Big Five" and dropping occupation/political-orientation/other fields. Say so if you'd rather I strip it down to just the three attributes the paper names.

## Verification

- `python scripts/run_opinion_experiment.py configs/opinion/example.yaml` against a small `M`/`K` smoke config (mirroring how `configs/smoke_test.yaml` is used today), checking: `initial_stance.csv` is created with one row per agent and the right initial columns; each round adds the expected 4 new columns; `opinion_round_metrics.csv` has one row per round with non-null `Pz`/`DG`/`NCI` once at least 2 rounds have run; a single-language-toggle run has every agent's `language` column equal to that language regardless of its pool preference; an all-4-toggles run has `language` matching each agent's own pool `language preference` exactly; a 2-or-3-toggle config raises `NotImplementedError` at startup rather than running.
- Manually inspect a few rows' `opinion_text` to confirm the directive's language is actually what came back (eyeball check, not automated, since no test suite is being touched).
