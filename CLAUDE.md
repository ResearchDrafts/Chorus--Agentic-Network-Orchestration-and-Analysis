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
- **Tier 3** (`simulation_orchestrator.py`, `prompt_builder.py`, `single_exposure.py`) — the integration tier; the only modules that call every other component. `prompt_builder.py` has no design in either `Architecture-docs/` file: it was designed from scratch, and the reasoning lives in its module docstring and in git history. `single_exposure.py` is the shared machinery for the three single-reader scripts in `scripts/`; it depends on the same tiers the orchestrator does and on nothing the orchestrator imports.
- **`analysis/`** (`metrics.py`, `stats.py`, `figures.py`) — RQ1's post-hoc analysis, offline, reading only a completed run's `interactions.jsonl`, `graph_edges.csv` and `run_config.json` plus the single-reader scripts' own output. Nothing the orchestrator touches imports it, so a change to an analysis cannot change a dataset.
- **RQ2/RQ3 analysis** (**not yet built**: `coherence_scorer.py`, `codemix_ratio_tracker.py`, `sentiment_proxy.py`, `bimodality_analysis.py`, `embedding_cluster.py`) — the parts of the original design that are not RQ1's.

Tier status drifts in prose. Don't trust a status line here; check the filesystem and `python -m pytest`. Current: **18 `sandbox/` modules plus `analysis/` (3) and `scripts/` (6), Tiers 0-3 and RQ1's analysis complete**, including E5's echo-chamber rewiring (`interaction_engine.GraphSampling.rewire`/`restore_edges`, `simulation_orchestrator._rewire_round`, and the `t_echo`/`assort_excess` metrics) and its follow-on fixes: a reply token cap, the campaign generator's `--rewire-rate`/`--rounds`/`--probe-language` sweeps, `balanced_two_camps`/`starter_counts`/`calibrate_starts.py`, the Wang-et-al.-derived metrics and their figures, the e5-vs-e7 contrast family and censoring, off-by-default structured persona attributes, and Fix 9's similarity-based recommender (`interaction_engine.GraphSimilaritySampling`, `neighbor_strategy="graph_similarity"`, e8/e9, `--recsys-threshold`). Check `python -m pytest` for the current count rather than trusting a number pinned here. RQ2's and RQ3's own analysis modules are the unbuilt part of the original design.

Check `Architecture-docs/full_design_doc.md` §3 for the authoritative module-by-module LLD (function signatures, error-handling contract, concurrency notes) before implementing any new module — the design doc's sample code is the spec each module is a direct port of, not just background reading. `Architecture-docs/sandbox_hld.md` carries the HLD, the schema derivation, and three passes of adversarial self-critique with numbered Fixes (Fix A–N) — when a design decision seems arbitrary (e.g. why the memory window is turn-count-based, not token-budget-based; why the same RNG seed is deliberately reused across language conditions), it's almost certainly resolving a specific numbered Fix in that document, not an accident.

### The bank regime's per-round flow (RQ1)

Round 0 probes every agent once, before anything is read, so the starting stance is a measurement in the dataset rather than something inferred from `initial_stance`. Then for each of K rounds, all against one frozen snapshot:

1. each agent's **side** (`pro` above the 7-point midpoint, `con` below, a seeded draw exactly on it);
2. the **bank item** it posts for that side — a pure function of (seed, sender, side, how many it has posted on that side), never of language or turn, which is what makes four language arms post the same argument;
3. its **neighbours**, from the graph;
4. what it is **delivered** (under `delivery: off`, the E4 control, N side-free posts from the neutral bank, one standing in for each bank item it would have been delivered);
5. **Phase A** — every reply, concurrently (`gateway.generate`), each carrying a seed from `SeedManager.call_seed("reply", turn, agent_id)`;
6. **Phase B** — every probe, concurrently (`gateway.score_next_token`), each seeing this round's own reply;
7. one `Interaction` per agent written, then the successful ones applied.

**How to read `posted_item_id`.** It is the bank argument the feed shows on this agent's behalf to its neighbours, matched to the side its last probe put it on. The agent is never shown that item and is never told it posted anything — "posted" names the item's role in the delivery, not an act the agent performed. The field keeps its name because every existing analysis and the whole log history read it; what changes is only how it should be interpreted. On a `delivery: off` row it is still set, which is what keeps the control's draws comparable to the treatment's row by row even though no argument was delivered.

Phase B depending on Phase A is not a Fix F violation: it is the agent's *own* output, read by nobody else. Fix F forbids one agent's turn-t output reaching another agent's turn-t stimulus, and steps 1-4 all complete before either phase dispatches.

The `delivery: off` control runs both phases too, and its feed **mirrors the treatment's shape**. Each agent reads N on-topic messages that take no side — one standing in for each bank item it would have been delivered — so it costs the same M × K reply calls a treatment arm does, reads the same number of posts, repeats a post exactly when the treatment would have, and differs in the persuasive content of what agents read and in nothing else. Two earlier versions were each wrong in a different direction: a control that read nothing and replied to nothing had a history that never changed, so it could not show drift even in principle; a control that read one fixed message read less than the treatment did, and never twice.

`NeutralBank.stand_in_for(item_id, all_item_ids)` is `ids()[sorted(all_item_ids).index(item_id) % len(ids())]` — a pure function of the two files' contents, with no seed, turn, agent or language in it. That is what makes the mirror hold: equal items give equal messages, so the repeat structure matches; a stand-in drawn from a stream would match the count and not the repeats, and one that varied by language would give four arms four different feeds. It also means a resume rebuilds the same mapping with nothing to restore. The two files need not be the same size — a smaller neutral bank wraps, several arguments share a message, and the mirror survives it.

**Replies are seeded.** A bank reply call carries `seed=SeedManager.call_seed("reply", turn, agent_id)`, a non-negative int32 derived from the base seed, the turn and the agent id and from nothing else. Without it the provider sampled freshly every call, so a rerun of one config produced different replies and — since the probe reads a stance out of the history those replies build — different readings: the run was reproducible in every draw this process makes and in nothing the server does. The probe is **not** seeded; it runs at temperature 0, and a seed there would suggest the measurement had a draw in it. A transient retry reuses the same seed. The generated regime sends no seed at all and its traffic is byte-identical to what it was.

Two pieces of bank state are **not in the checkpoint** and are rebuilt from `interactions.jsonl` on resume: each sender's position in its bank cycle, and the rounds each agent remembers. The rebuild is **as of `last_completed_turn`** and ignores every later row — the log runs ahead of the checkpoint when a round died part-way through, and those rows describe a round the resumed pass is about to re-run. Within that cutoff, rows are keyed on `(turn, speaker_agent_id)` keeping the last write, the same rule `export_csv.py` applies, so a re-logged turn cannot double-count the cycle.

**Replies are capped.** Every bank-regime reply call carries `max_tokens=run.max_reply_tokens` (default 512), following `seed`'s own not-sent-when-`None` pattern in `ModelGateway.generate`; without one, a live run saw a single reply loop until litellm's 600s timeout and stop the whole campaign. `Interaction.reply_truncated` records whether `completion_token_count >= max_reply_tokens`, exported in `export_csv.py` and summarized as `rate_reply_truncated` per run in `analysis/metrics.py`. The probe's `max_tokens=1` is fixed inside the gateway and is never affected by this field.

### E5: echo-chamber formation (RQ1, graph strategy only)

E3/E4 give agents influence but no selection: who an agent reads is a fixed graph, never a function of where it or its neighbours stand. E5 adds selection on top, as a step 8 appended to the per-round flow above, run once the round has fully committed:

8. **rewiring** — for each agent `i`, in sorted `agent_id` order, and for each post it read this round from neighbour `j`, in the order read: `side_i` is the sign of `i`'s **own probe this round** (pro above the midpoint, con below, no side exactly on it); `side_j` is `side_map[j]` — the side `j`'s post argued, a property of what was posted, read identically under `delivery: on` and `delivery: off` (see below). If `side_j != side_i`, a draw from the `"rewire"` stream (seeded, `turn_rng("rewire", turn)`) decides whether `i` drops `(i, j)` for a same-side (or, under `rewire_target: random`, any) non-neighbour `k`. `i`'s own degree never changes — one edge leaves, one arrives, both at `i`'s end — and no rewire may drop a neighbour below `N`.

The draw is taken for **every** read post, opposite-side or not, for `_resolve_side`'s own reason: making it conditional would let the stream's position depend on how many reads happened to be opposite-side, and two matched-set arms whose probes disagree on one agent's side would then draw everything after it differently. A reader exactly at the midpoint has no side and never rewires, and neither does a same-side read — both still consume the draw.

**`(i, j)` can already be gone by the time `i`'s own entry is processed.** `neighbor_map` is frozen before rewiring starts, so `i` and `j` can each have sampled the other as one of their own `N` reads this round; whichever is processed first (sorted order) may drop the shared edge, and the second attempt then finds `j` no longer a neighbour at all. `GraphSampling.rewire` raises if asked to drop a non-edge, so the orchestrator checks `neighbors_of` first and folds this case into `skipped_no_candidate` — it is not a bug, just nothing left for the second read to act on.

**`delivery: off` rewires identically to `delivery: on`.** `post_map`/`side_map` are computed once, before delivery branches on mode, and `side_j` for a shadow item is `side_map[sender]` either way — there is no separate code path to keep in sync, which is also why E6 (E5's `delivery: off` control) needs no special-casing anywhere in `_rewire_round`.

A rewired graph is state the checkpoint must carry, unlike the bank's post counts and remembered rounds: `interactions.jsonl` records what an agent *read*, never who it is currently *linked to*, so there is nothing to rebuild an edge list from on resume. `CheckpointState.graph_edges` and `GraphSampling.restore_edges` exist for exactly this, and `rewire_rate == 0.0` (every pre-E5 config) skips the whole step — no stream draw, no file touched, no checkpoint field populated — so existing runs are unaffected byte for byte.

`graph_edges_by_round.csv` (`turn,agent_a,agent_b`, the graph **used for reading** in that turn, plus a final `turn = K + 1` block for the graph after the last rewiring) and `rewires.jsonl` (`turn, agent_id, dropped, added, reader_side, status` — `rewired`, `skipped_no_candidate`, or `skipped_min_degree`) are both written only when `rewire_rate > 0`, from `LoggingWriter.write_graph_edges_round`/`write_rewire`. `analysis/metrics.py` reads the per-round file with a fallback to the static `graph_edges.csv` for every round when it is absent (`_graph_for_round`), which is also what every pre-E5 run's analysis still does.

**Metrics.** Per round: `assort_null_mean`/`assort_null_hi` (`assortativity_null` — a 1,000-draw permutation null of `x` over the round's graph, seeded by `f"{seed}|{turn}|assort_null"` so it is reproducible from the run alone), `assort_excess` (assortativity minus the null mean), `cross_side_edge_share` (share of edges whose ends disagree in `sign_of(x)`, with `x == 0` agents excluded on either end — the same exclusion `sign_of` uses everywhere), `minority_share` (the smaller side's population share, `None` when nobody has a decided side), `n_rewires` (from `rewires.jsonl`, 0 when absent). Per run: `t_echo` (`echo_round` — the later of the first `ECHO_WINDOW=2` **consecutive** rounds where `assortativity > assort_null_hi` **and** `minority_share >= MINORITY_SHARE_FLOOR(0.2)`; `K + 1` with `echo_censored=True` if no such window exists — the `minority_share` floor is what keeps a population that collapsed onto one side, consensus rather than an echo chamber, from reporting a `t_echo`) and `auc_assort_excess` (mean `assort_excess` over rounds 1..K, dropping `None` rounds rather than zeroing them — a speed measure that needs no threshold, unlike `t_stationary`/`t_echo`).

`scripts/make_rq1_configs.py --experiments e3 e4 e5 e6 e7` is how a campaign gets E5's three arms (default stays `e3 e4`): `e5` is the treatment (`delivery: on`, `rewire_rate: 0.3`, `rewire_target: same_side`, `lean_assignment: balanced_two_camps`), `e6` mirrors `e4` the same way `e5` mirrors `e3` (`delivery: off`, same rewiring), `e7` is rewiring without selection (`rewire_target: random`) — a control for whether rewiring *alone* sorts anything. `DEFAULT_REWIRE_RATE` lives in the generator, not in `base.yaml`: `base.yaml` validates as a complete `ExperimentRun` on its own, and a literal `echo_rewire_rate` key would be an unknown field under `extra="forbid"`. `--rewire-rate` sweeps it (a non-default rate is folded into the run_id, e.g. `e5_english_ws_r0p1_s01`, so a sweep shares one runs folder); `--rounds` overrides `K` for every experiment generated, not only e5/e6/e7's; `--probe-language` pins every arm's probe question to one language while posts stay in each arm's own, for an E1-style control, and is likewise folded into the run_id (`_probe-<lang>`).

**`lean_assignment: "balanced_two_camps"`** (`agent_manager._balanced_two_camps_leans`) is what e5/e6/e7 actually use, not plain `"balanced"`: on a 7-point scale `"balanced"` gives the midpoint (lean 4) agents of its own, which both unbalances the two sides' counts and tips the population toward whichever side the midpoint's probe reading happens to round to. `balanced_two_camps` draws equal-sized camps from the points strictly below and above the midpoint, with no agent at the midpoint — for odd `M` the one extra agent's side is itself drawn from the RNG stream, so even that is not hard-coded to one side. `ExperimentRun.starter_counts` (an optional override of `bank_manager.STARTER_COUNTS`, validated at load time against the override's own maximum via `bank_manager.min_per_side`, never the module constant's) exists for the matching problem one level deeper: the module default's starter-post composition is itself asymmetric between the con and pro leans unless the population already is. `scripts/calibrate_starts.py` sweeps every `(n_con, n_pro)` mix cold (`start_mode="posts"`, no item, no reply) to find which mix lands a reader near which lean, and `--suggest` writes a ready-to-paste `starter_counts` block.

**Echo-chamber metrics beyond `t_echo`/`assort_excess`** (`analysis/metrics.py`, following Wang et al.): `polarization_index` (`Pz`, equal to `var_x`), `global_disagreement` (`DG`), `pairwise_disagreement`/`local_disagreement`/`disagreement_ratio`, `nci`/`nci_null_mean`/`nci_null_hi`/`nci_excess` (a permutation null built exactly like assortativity's), `bimodality` (Sarle's coefficient), `mean_belief_shift`/`mean_signed_shift`, `n_side_switches`/`total_side_switches` (read next to `n_rewires` as the conversion-versus-sorting race), and run-level `auc_nci_excess`/`delta_polarization_index`/`delta_global_disagreement`/`delta_nci`/`total_rewires`. Wang et al. use `x` in `[-2, 2]`; this codebase's `x` is `[-1, 1]`, so their `Pz`/`DG` read 4x ours for the same state. **`auc_nci_excess` is the primary echo-chamber outcome** (`analysis/stats.py`'s `PRIMARY_ECHO_METRIC`): unlike `t_echo` it needs no threshold and is defined for every run, censored or not. `analysis/stats.py` runs e5-vs-e6 and e5-vs-e7 as two separate Holm-corrected families (`_echo_contrasts_for_pair`, requiring both experiments of a pair present), once per `rewire_rate` actually present so a sweep is never pooled; every `t_echo` contrast row carries `n_censored_treatment`/`n_censored_control`/`censored_warning`, and `formation_summary.csv` reports per experiment/language/rate how many runs formed an echo chamber and the median `t_echo` among the ones that did (never counting a censored `K + 1` into that median). `analysis/figures.py` adds `nci_excess`/`disagreement_ratio`/`minority_share`/`n_side_switches` line figures, `e7` as a dotted style distinct from `e5`'s solid and `e6`'s dashed, and one `network_snapshots` file per experiment present rather than one file always showing whichever ran last.

### E8/E9: a similarity-based recommender (Fix 9, graph strategy family)

A third `neighbor_strategy`, `"graph_similarity"` (`GraphSimilaritySampling` in `sandbox/interaction_engine.py`, a **subclass of `GraphSampling`** so a seed's WS graph is byte-identical under either strategy — add a test for that, as the Fix did). Where E5's rewiring changes *who is a neighbour*, this changes *which of a reader's existing neighbours it is shown*: among the reader's current graph neighbours with a valid probe reading, within `recsys_threshold` (x units, `[-1, 1]`; Jo et al.'s own threshold is `2.0` on `[-2, 2]`, i.e. `1.0` here) of the reader's own, deliver up to `N` of the closest, ranked by `|x_i - x_j|`, ties and final delivery order both broken by the round's own `"neighbor"` RNG stream (shuffle, then stable sort; shuffle again before delivery). **Fewer than `N` qualify: deliver what there is, never pad back up** — backfilling would make every feed exactly `N` long regardless of the threshold, which is the one way the threshold would have no visible effect. A reader with an empty `stance_history` (never successfully probed, not even round 0) has nothing to rank against and falls back to `N` random graph neighbours (`recsys_fallback=True`, logged). An empty feed skips that reader's reply call for the round — nothing to reply to — but still runs its probe and writes an `Interaction` with `api_call_status="no_feed"`, a real applicable round rather than Fix I's `failed_logged_null`.

Opinion comes from `Agent.current_stance()` — the same field `_resolve_side` reads — **not** the module-level `_current_stance(agent, turn)` helper `AlphaSampling` uses: that helper special-cases `turn == 1` to `initial_stance` unconditionally, which is correct for the generated regime (Ohagi's mechanism never probes round 0) and wrong here, since the bank regime's round-0 probe is already applied to `stance_history` before turn 1 runs. Reading `initial_stance` instead at turn 1 would silently rank against an unmeasured, configured value rather than "the round 0 baseline probe in round 1," which is what the design actually specifies.

Rewiring and this rule are two different selection mechanisms and are never combined in one run. `ExperimentRun.check_graph_similarity_requirements` requires `regime="bank"` unconditionally for `neighbor_strategy="graph_similarity"` (the rule needs a probe reading, which only the bank regime has); `check_rewire_requirements`'s existing `neighbor_strategy != "graph"` check already rejects `rewire_rate > 0` under `graph_similarity` for free, since the two strings are never equal.

**Logging.** `Interaction.feed_size`/`n_neighbors_over_threshold`/`recsys_fallback` (all `None` outside this strategy). `GraphSimilaritySampling.feed_info(agent_id)` is how the orchestrator reads these back right after calling `select_neighbors` for every agent in a round — state kept on the strategy instance rather than threaded through `RecommendationStrategy`'s shared return type, so `AlphaSampling`/`GraphSampling` need no signature change. `round_metrics.csv` adds `mean_feed_size`/`share_empty_feed` (over the logged `feed_size` column, any strategy), `share_neighbors_filtered`/`mean_neighbor_gap` (over the round's own graph edges — the full candidate neighbourhood, needs `config["recsys_threshold"]`; `mean_neighbor_gap` is `local_disagreement` under a second name, kept for the side-by-side comparison), and `mean_feed_gap` (over the actually-delivered reader/sender pairs in `neighbor_agent_ids`, any strategy). All five are `None` without the backing column/threshold, the same convention every other optional metric here follows.

`scripts/make_rq1_configs.py --experiments e8 e9` adds: `e8` (`neighbor_strategy: graph_similarity`, `N: 5`, `recsys_threshold` from `--recsys-threshold`), `e9` (`neighbor_strategy: graph` — base.yaml's own default, unchanged — at the same `N: 5`, the control at matched feed size; `e3`'s `N=3` cannot serve that role). Both override `graph_degree` to `RECSYS_GRAPH_DEGREE=6` (Jo et al.'s own `k=6`) **only for e8/e9**, never touching e3–e7's `graph_degree=4`: `base.yaml`'s own `N=3` cannot reach `5`, and raising `graph_degree` in `base.yaml` itself would change every other experiment's byte-identical output, which the Fix holds fixed. `--recsys-threshold` sweeps `e8`'s own field the same way `--rewire-rate` sweeps `rewire_rate`, folded into the run_id only off its own default (`e8_english_ws_t0p5_s01`); `e9` never carries the token or the field, since the "graph" strategy never reads it. `analysis/metrics.py`'s `EXPERIMENT_PREFIXES` and `analysis/stats.py`'s `RECSYS_METRICS`/`_recsys_contrasts_for_pair` (grouped by `recsys_threshold`, mirroring `_echo_contrasts_for_pair`'s grouping by `rewire_rate`) extend the same way E5's did. Jo et al.'s "neighbour correlation" is, by construction, the Pearson correlation `analysis/metrics.nci` already computes (correlation is invariant to the `x`-vs-`v=2x` rescaling), so there is no separate `final_neighbor_corr` column — `final_nci` is it.

At `configs/rq1/base.yaml`'s own smoke-scale `M=8`, `graph_degree=6`/`N=5` is tight enough that one seed index (5, of the shipped campaign's six) produces a Watts-Strogatz graph with a node short of `k=5` neighbours, which raises at `GraphSimilaritySampling`/`GraphSampling` construction time — not at `--dry-run`, which never builds the graph. `e3` through `e7` never hit this at their own `N=3`/`graph_degree=4`. This is a property of the smoke-scale placeholder population, not of the Fix 9 logic; see Known Limitations in the README.

### Core data flow (implemented in `simulation_orchestrator.py`)

Config Loader → `ExperimentRun` → Agent Manager creates M agents → for each of K turns: freeze snapshot → Interaction Engine resolves N neighbors per agent (`AlphaSampling`) → Meme Pool Manager resolves which scheduled speakers post a meme this turn → dispatch concurrently (generated-text speakers → Model Gateway; meme speakers → free dataset lookup, no LLM call) → collect results → write `Interaction` records → advance snapshot → checkpoint → next turn.

The frozen-snapshot ordering guarantee (neighbor and meme-injection resolution both complete *before* any dispatch begins, so no agent's turn-t output is ever read by another agent's turn-t neighbor/meme resolution) is load-bearing for reproducibility and is referred to throughout the docs and codebase as "Fix F."

### The RQ1 pipeline (`scripts/` and `analysis/`)

Nine commands (`calibrate_starts.py` is step 0, optional, ahead of the
generator), each consuming what the one before it wrote; README's "The
RQ1 pipeline" lists them with their outputs. Three things about them are
load-bearing enough to restate here:

**The three single-reader scripts use the same instrument the network runs
use.** `calibrate_tau.py`, `run_e1.py` and `check_comprehension.py` build
every prompt with `PromptBuilder`, score every probe with
`probe_scorer.score`, and take every setting from one `ExperimentRun` YAML per
language (`single_<language>.yaml`, written by `make_rq1_configs.py`). There
is **no second copy of any prompt text in `scripts/`** — a τ measured against
a slightly different probe prompt is a threshold for a different instrument,
and nothing downstream could tell. The shared machinery is
`sandbox/single_exposure.py`.

A single-exposure trial is numbered like a one-round run: the baseline probe
is **turn 0** (the orchestrator's own round number for a reading taken before
anything is read) and the exposure, reply and second probe are **turn 1**. So
one `prompts.jsonl` audit reads a calibration sweep and a campaign alike.

**Every script is resumable and every failure is a row.** Each appends one
JSONL row per trial as it finishes and skips keys already carrying
`status="ok"`; a probe below the mass floor or a fatal gateway error produces
`status="failed"` with null readings, never a dropped trial. An invisible
failure in a τ calibration shifts the threshold every later result is judged
against. A `ContextLengthExceeded` is fatal for the script, as it is for a
run.

**`x = (probe_mean - 4) / 3` is defined once**, in `probe_scorer.to_x`, and
imported by `export_csv.py`, the scripts and the analysis tier. A sign error
there does not raise — it reverses every reported effect.

Three definitions the analysis turns on, each pinned by a test in
`tests/test_analysis_metrics.py`:

- **`sign(x)` makes 0 its own sign.** A move off the midpoint is a change of
  sign, and (if it clears τ) a flip. Folding 0 in with a pole would make the
  midpoint arbitrarily belong to one side and turn half of the undecided
  population's movements into hardening.
- **`var_x` is the population variance (ddof=0).** These agents are the
  population; the inferential step happens later, across runs, in
  `analysis/stats.py`. ddof=1 is also undefined at n=1, which is real when
  every probe but one failed in a round.
- **An undefined round-to-round change is not within ε.** A null
  assortativity at either end of a step makes that step uncomputable, and
  "we cannot tell whether this moved" is not evidence that it stopped —
  otherwise a run would become stationary precisely because its population
  collapsed to one position.

Two refusals rather than defaults: a run whose language has no τ raises
`MissingTau` (classifying an uncalibrated arm against another arm's noise
would look exactly like a calibrated result), and `analysis.metrics` requires
exactly one of `--epsilon` and `--epsilon-from-e4` (a default tolerance would
set the threshold for every reported `t_stationary`).

In `analysis/stats.py`, the unit of analysis differs between the two halves
and that is deliberate: **E1 is trial-level** (mixed model, random intercept
per item, variance component for persona) while **the networks are run-level,
paired by seed** — a hundred agents inside one run are not a hundred
independent observations. Holm is applied **within families**, not across
everything, so one question's tests are not penalised for another being asked
alongside; `p_raw` survives in every row so a reader who wants a different
family can recompute. A non-converged `MixedLM` is never reported as a
result; the fallback is cluster-robust OLS and `e1_model.txt` leads with
which one you have.

### Key design invariants (apply when touching any module)

- **Never silently clamp or default.** Invalid `alpha`/`N`, out-of-range parsed stances, unrecognized model ids, missing pricing entries — all raise immediately (`ValueError`/custom exceptions) rather than being coerced to a safe value. This is a deliberate, repeated pattern across the codebase (see Fix I: "never fabricate a placeholder stance") — don't add fallback/default behavior to "fix" a raised error unless the design doc says to.
- **Seeded RNGs only, never the global `random` module.** `seed_manager.py`'s `SeedManager` hands out three independently-seeded `random.Random` instances (`neighbor_sampling_rng`, `meme_injection_rng`, `persona_assignment_rng`), deliberately offset (`seed`, `seed+1`, `seed+2`) so the three streams never correlate, plus three derived families: `turn_rng(purpose, turn)`, `sender_rng(purpose, sender_id)` and `graph_seed`. Reproducibility depends on every stochastic choice consuming from the correct purpose-specific instance.
- **No condition may enter a seed.** This covers the seed sent *out* to the provider as well: `call_seed(purpose, turn, agent_id)` derives a non-negative int32 for a reply call from the base seed alone, so two language arms hand the server the same number for the same (turn, agent) and differ in the text of the prompt rather than in how the reply to it was sampled. `SeedManager` takes the base seed and nothing else — no language, topic or trial — and every derivation is a function of that alone. This is what makes a *matched set*: four runs differing only in `language_condition` share a graph, the same neighbour draws, and the same bank item for the same sender on the same side, so the arms differ in the rendering of the text and in nothing else. `tests/test_seed_manager.py` pins it from both directions, including that the constructor has no condition parameter to mix in.
- **Never truncate silently, and never lose an agent to a limit.** Bank-regime agents keep their **full** history by default (`memory_rounds: None`); a window bounds the effect the experiment can measure, so it is opt-in. The two ways unbounded history can end are both loud and both fatal for the whole run: `max_prompt_tokens` (checked against the provider's reported count, raising `PromptBudgetExceeded`) and `ContextLengthExceeded`, which is deliberately **not** a `FatalGatewayError` so the orchestrator cannot absorb it per agent. The agents that overflow first are the ones whose language tokenizes least efficiently, so a per-agent failure would drop one arm's agents and keep another's.
- **In the bank regime, no number reaches an agent.** Not its own position, not a neighbour's, not an agent id. An agent that can see a position number does not have to read the arguments — it can move its own number towards the ones it was shown, and the result is indistinguishable from persuasion. The probe exists so stance can be read without ever showing one, and it never writes into any agent's history. A prompt audit in `tests/test_integration_run.py` checks every prompt of a whole run, and `prompts.jsonl` (on by default for this regime, via `log_prompts`) makes the same audit possible against a real run's output rather than only in the test suite. The scale's own digits (the probe's `1 means .../7 means ...` anchors, and its "answer with a single digit from 1 to 7") are the one exception, and are necessary for the model to answer at all.
- **The lean sentence is a permanent trait, in every prompt.** `LEAN_SENTENCES` renders where an agent *started*, in words, directly after the topic line of every bank reply prompt and every probe prompt. Not round 1 and round 0 only: a sentence that vanishes afterwards makes the round-0 probe and every later probe two different instruments, so the step between the first two readings partly measures the removal of a sentence — and under `delivery: off`, where nothing else about a probe changes between rounds, it was the only thing that changed at all.
- **Fatal vs. per-agent-recoverable errors are distinguished deliberately.** `FatalGatewayError`/`StanceParseFailure` (after retry budget exhausted) become a `failed_logged_null` `Interaction` for that one agent and the turn continues for everyone else (`asyncio.gather(..., return_exceptions=True)`); anything else propagates and aborts the whole run. Don't broaden what gets caught — an uncaught, unexpected exception type is meant to surface loudly as a bug, not be absorbed into a soft-failure record.
- **`CostTracker.record()` is synchronous**, not async, even though it's called from `ModelGateway.generate()` (async) — this was a deliberate early decision, not an oversight. Don't add `await` to that call site.
- **Pydantic v2 models throughout** (`sandbox/models.py`), not dataclasses — the HLD's own sketches use dataclasses, but the codebase supersedes that with `pydantic.BaseModel` for validation (regex-constrained enums via `pattern=`, cross-field `@model_validator`). When porting a design-doc snippet, translate its dataclasses to the existing Pydantic models rather than introducing a parallel dataclass.
- **Vision support is a static lookup table, not a runtime probe**, keyed by `pricing_table.yaml`'s model set — keep the vision-capable/text-only lists in `model_gateway.py` and the pricing table in sync; an unrecognized `model_backend_id` should raise, never silently default to text-only.

### Data and fixtures

- `data/memes/*.jsonl` — one JSON object per line, loaded via `MemeContent.model_validate_json()`.
- `data/personas/*.jsonl` — **despite the extension, these are plain text, one free-text persona description per line, not JSON**, when `ExperimentRun.persona_attributes` is `"none"` (the default, and every pool written before Fix 7). `agent_manager.py`'s `_load_persona_pool()` just strips and splits lines; nothing parses JSON there. With `persona_attributes` set to `"demographic"` or `"demographic_big5"`, `agent_manager.load_persona_rows()` reads the SAME extension as actual JSON Lines instead — one object per line, `{"persona": ..., "age": ..., "gender": ..., "education": ...}` — and `config_loader._persona_attributes_errors` requires every field non-empty on every row, naming the row if one is missing. Which parse a pool gets is a property of the run, not the file, so a plain-text pool pointed at by a demographic-mode run fails loudly naming the line rather than three fields silently `None`. `test_pool_demographic.jsonl`/`test_pool_demographic_missing.jsonl` are the JSON-object fixtures for this path. With `"demographic_big5"`, the five Big Five traits are drawn `"high"`/`"low"` by `agent_manager.draw_big5`, keyed by **persona pool index** via `seed_manager.persona_trait_rng` (not a sequential draw off the live `persona_assignment_rng`, which would make the traits depend on `lean_assignment`/`M`) — two agents sharing a persona (the pool wraps) also share its traits, the same way they already share its text. The attribute sentence itself is rendered from `configs/persona_attribute_templates.yaml`, not in code, by `prompt_builder.PromptBuilder._persona_section`, shared by both regimes and the probe prompt.
- `test_pool.jsonl` variants are small (3-4 entries) fixtures for tests; `pool_20.jsonl`/`empty_pool.jsonl` exist for specific acceptance-criteria tests (cycling, empty-pool error path). All three images referenced by `data/memes/test_pool.jsonl` now exist; two were missing until a pre-flight audit found the meme path would raise `FileNotFoundError` for roughly two sampled memes in three.
- `data/bank/*.csv` — argument banks: `item_id, arg_id, strength, side, english, hinglish, romanized_hindi, hindi`. The language column names ARE `language_condition` values, so a header typo makes the bank unrenderable for that arm. Validated in full at config load (not merely stat()ed), reporting every problem at once: a bank with one empty Hindi cell would otherwise run to completion and produce an arm in which one argument was silently blank, which is indistinguishable in the output from an argument that failed to persuade. `test_bank.csv` is a 4-item fixture, both sides, both strengths, and deliberately **digit-free in every language column** so the prompt audit can tell a leaked position number from the argument's own text. An item id starting with `neutral_` is rejected: that prefix names the neutral bank's messages.
- An item ID listed in a run's `exclude_items_path` is dropped from the bank **after** validation, never instead of it: an excluded item does not make an empty Hindi cell in its row acceptable, because the next run may not exclude it. An unknown ID, a duplicate, or an exclusion that leaves a side empty all raise, and `config_loader` applies the list too, so a bank-emptying exclusion is a config error rather than something found on the first round of a paid run. `scripts/check_comprehension.py` writes `failed_items.txt` in exactly that format (comments and blank lines are skipped) — and deliberately does **not** read it, since it is the check that decides what belongs in it.
- `data/bank/test_neutral.csv` — the **neutral bank**, read by the `delivery: off` control: `neutral_id, english, hinglish, romanized_hindi, hindi`, every id prefixed `neutral_`, no side and no strength. Validated in full at config load like the argument bank, and digit-free for the same reason. One column (`items_read`) carries both kinds of id, so the prefix is how a reader — and `_remembered_text` on the resume path — knows which file to look an id up in.
- `pricing_table.yaml` — hand-maintained `provider -> model_id -> modality -> {prompt_per_1k, completion_per_1k}`, keyed by the BARE model id (`ModelGateway._bare_model_id` strips the provider prefix). `CostTracker` raises on a missing entry *after* the call completes, so every model actually used must appear here. The openai/anthropic/google entries remain illustrative placeholders; the `hosted_vllm`, `ollama`, `groq` and `cerebras` entries are genuinely 0.0, since self-hosted compute is billed by the hour rather than per token.
- `export_csv.py` columns added for the bank regime, beyond the probe and prompt-length ones: `reply_seed` (the seed that reply call carried; empty at round 0 and on a generated row), `shadow_items` (semicolon-joined, the bank items a control row's neutral messages stood in for, in `items_read`'s order — this is what lines a control row up against the treatment row it is a baseline for), and three **exposure counts** over this agent's earlier rounds in this run: `n_repeat_reads`, `max_prior_exposures` and `n_distinct_items_read_so_far`. The three count only bank rows that are valid for analysis and not superseded duplicates; round 0 is `0, 0, 0` (it genuinely read nothing), while a failed row, a superseded duplicate and a generated row are null and contribute nothing. Both repeat columns are counted against the state *before* the round, so an item two neighbours both posted is one round's stimulus rather than a repeat of itself — though it still counts twice toward later rounds. `derive()` must run after `mark_duplicates()` or it raises.
- `runs/` — output directory, one run = one subdirectory (`run_config.json`, `interactions.jsonl`, `agents_final.jsonl`, `checkpoints/`, plus `prompts.jsonl` when `log_prompts` is on, `graph_edges.csv` under the graph strategy, and — once `rewire_rate > 0` (E5/E6/E7) — `graph_edges_by_round.csv` and `rewires.jsonl`), per `logging_writer.py`. Gitignored. Completed campaigns are kept under `outputs/` instead; `outputs/chorus_run1/` holds the first real dataset, 8 runs and 8,000 interactions.
- `configs/rq1/base.yaml` — the campaign template `scripts/make_rq1_configs.py` expands, and a complete valid config on its own. Every smoke value in it is marked PLACEHOLDER. Six fields are overwritten by the generator (`run_id`, `language_condition`, `probe_question`, `delivery`, `seed`, `trial_number`); `neutral_bank_path` is kept on the `e4_*` and `single_*` files and dropped from the `e3_*` ones, which never read it. `configs/rq1/probe_questions.yaml` holds the one piece of prompt text that is translated, with no default and no fallback.
- `configs/rq1/campaign/` — generated, and checked in. A test asserts it is byte-identical to what the current generator and base config produce: a base edited without re-running the generator would leave a campaign nobody could reproduce. `scripts/run_campaign.py` discovers runs from the **filenames** rather than from `manifest.csv`, so a config added or removed by hand changes what runs.
- `configs/` — run configs. `example_run.yaml` documents every field; `smoke_test.yaml` (vLLM) and `smoke_local.yaml` (Ollama on Apple Silicon) are minimal five-call runs for checking plumbing. `configs/rq1/` holds the bank-regime smoke configs: one per language arm (`smoke.yaml`, `smoke_hinglish.yaml`, `smoke_romanized_hindi.yaml`, `smoke_hindi.yaml`, all sharing a seed) plus `smoke_off.yaml` (the `delivery: off` control, which also needs a `neutral_bank_path`). The four arms differ in exactly `run_id`, `language_condition` and `probe_question`; a test in `tests/test_config_loader.py` pins that, including that the anchor labels stay English. **`delivery` must be quoted in YAML** — YAML 1.1 reads bare `off` as the boolean `False`, so an unquoted value is rejected by the schema; a test asserts every shipped config parses it as a string.

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

The whole RQ1 pipeline, in order (README's "The RQ1 pipeline" has the outputs
of each):

```bash
python scripts/make_rq1_configs.py --seeds 5 --out configs/rq1/campaign
python scripts/check_comprehension.py --runs-dir runs
python scripts/calibrate_tau.py --runs-dir runs
python scripts/run_e1.py --runs-dir runs
python scripts/run_campaign.py configs/rq1/campaign --runs-dir runs
python -m analysis.metrics --runs-dir runs --tau runs/calibration/tau.json \
        --out results/ --epsilon-from-e4
python -m analysis.stats --e1 runs/e1/trials.jsonl --results results/ --out results/
python -m analysis.figures --results results/ --e1 runs/e1/trials.jsonl \
        --calibration runs/calibration/ --out results/figures/
```

Each script runs both as `python scripts/x.py` and as an import (`from scripts
import run_e1`), which is how the tests drive the real entry points against a
mocked gateway. `scripts/` is deliberately excluded from
`[tool.setuptools.packages.find]`: these are scripts run from a checkout, not
library code.

`notebooks/chorus_colab.ipynb` runs a full campaign on a Colab GPU: vLLM
serving, the topic x language grid, and the combining step. Inference is
self-hosted on open-weight models, so a campaign costs GPU time and nothing
else.

Two hazards worth knowing before touching the Colab path: `asyncio.run()` raises
inside a notebook cell, so runs launch as subprocesses or via top-level `await`;
and the Colab filesystem is ephemeral while `runs/` holds the checkpoint resume
depends on, so `--runs-dir` must point at mounted Drive.

### Known gaps

- **RQ2's and RQ3's analyses do not exist.** No coherence scoring, code-mix
  index, sentiment, bimodality or embedding analysis. RQ1's analysis is built
  (`analysis/`), including τ calibration and assortativity over
  `graph_edges.csv`.
- **`scripts/run_campaign.py` does not dispatch E5/E6/E7/E8/E9.** Its
  `EXPERIMENT_ORDER` and `--experiments` choices are still `e3`/`e4` only, so
  a config `make_rq1_configs.py --experiments e5 e6 e7 e8 e9` generated has to
  be launched individually with `run_simulation.py`. Left unchanged rather
  than extended here because `tests/test_run_campaign.py::test_parse_run_id_ignores_what_is_not_a_campaign_run`
  pins `"e5_english_ws_s01"` as *not* a campaign-run filename, and the task's
  "add tests, do not edit existing ones" rules out updating that pin as part
  of this change — see the echo-chamber PR description.
- **Nothing has been run against a real server yet.** Every script and every
  analysis module is exercised against the mocked gateway in the test suite
  and against output the orchestrator produced under it; none has met a
  live vLLM endpoint, so the operational numbers (how long a matched set
  takes, what `probe_mass` a real model puts on the scale, whether the four
  arms' τ are comparable) are unknown.
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
