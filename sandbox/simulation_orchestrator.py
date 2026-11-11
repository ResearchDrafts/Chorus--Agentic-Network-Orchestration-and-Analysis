"""The per-turn simulation loop (full_design_doc.md Sec 3.9, Tier 3).

The only module that calls every other component. Everything below the
integration tier is deliberately unaware of this one.

Ported from Sec 3.9's sample, with the gaps that sample leaves open resolved
per phase4.md Section 2: run-directory creation, the exception taxonomy, the
run-lifecycle fields nothing previously read, SeedManager wiring, and passing
the multimodal prompt through to the gateway.

**Two regimes.** The generated regime is the original loop: one LLM call per
agent per turn, which both writes a post and states a stance. The bank regime
(RQ1) splits those apart into two phases per round, because they were never
the same thing:

    Phase A   every reader is shown three bank posts and writes a reply
    Phase B   every agent is privately probed for its stance

Both phases are computed against the same frozen snapshot, and no agent state
changes until both have finished. The probe is what makes the split worth it:
stance is read out of a separate call's next-token distribution, in a context
the agent's own history never sees, so measuring an agent cannot change what
it or anyone else is shown next round -- and no agent is ever shown a position
number at all.

Phase B depends on Phase A within a round (the probe shows the agent the reply
it just wrote), which is not a Fix F violation: it is the agent's *own* output,
read by nobody else. What Fix F forbids is one agent's turn-t output reaching
another agent's turn-t stimulus, and neighbour resolution, side assignment and
item selection all complete before either phase dispatches.

**Fix F, stated as an invariant because nothing structurally enforces it.**
`AgentManager.all_agents()` hands back live `Agent` references, so the frozen
snapshot is a convention, not a guarantee. The rule: steps 2 and 3 (neighbour
resolution, meme-injection resolution) run to completion before any dispatch
begins, and no dispatch coroutine mutates agent state. All mutation happens in
step 6, after `asyncio.gather` returns. Break that and runs stop being
reproducible, silently.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sandbox.agent_manager import AgentManager, agent_id_for
from sandbox.bank_manager import BankManager
from sandbox.checkpoint_manager import CheckpointManager
from sandbox.cost_tracker import CostTracker
from sandbox.interaction_engine import AlphaSampling, GraphSampling, RecommendationStrategy
from sandbox.logging_writer import LoggingWriter
from sandbox.meme_pool_manager import MemePoolManager
from sandbox.model_gateway import BackendResponse, FatalGatewayError, ModelGateway
from sandbox.models import Agent, ExperimentRun, Interaction, MemeContent
from sandbox.probe_scorer import MIDPOINT, ProbeMassTooLow, ProbeScore
from sandbox.probe_scorer import score as score_probe
from sandbox.prompt_builder import BuiltPrompt, PromptBuilder, RoundMemory
from sandbox.seed_manager import SeedManager
from sandbox.stance_parser import (
    StanceParseFailure,
    clamp_and_validate_scale,
    parse_stance,
)

# Fix I: one initial attempt plus two retries with an emphasised format block.
MAX_STANCE_ATTEMPTS = 3

# The round-0 probe's turn number. Not a turn: nothing is read, nothing is
# posted, and no reply is written. It exists so every agent's starting stance
# is a measurement in the dataset rather than something inferred afterwards
# from initial_stance, which is a configured value and not a probe reading.
ROUND_ZERO = 0

# Which side each agent argues, decided from its last probe reading.
SIDE_PRO = "pro"
SIDE_CON = "con"
SIDES = (SIDE_CON, SIDE_PRO)  # sorted, so a tie-break draw is order-stable


class SimulationOrchestrator:
    def __init__(
        self,
        run: ExperimentRun,
        agent_manager: AgentManager,
        interaction_engine: RecommendationStrategy,
        meme_pool: MemePoolManager,
        gateway: ModelGateway,
        logger: LoggingWriter,
        checkpoint_mgr: CheckpointManager,
        prompt_builder: PromptBuilder,
        seeds: SeedManager,
        cost_tracker: CostTracker | None = None,
        runs_dir: Path = Path("runs"),
        bank: BankManager | None = None,
    ):
        self._run = run
        self._agent_manager = agent_manager
        self._interaction_engine = interaction_engine
        self._meme_pool = meme_pool
        self._gateway = gateway
        self._logger = logger
        self._checkpoint_mgr = checkpoint_mgr
        self._prompt_builder = prompt_builder
        # Per-turn streams are derived from this, rather than the long-lived
        # neighbor_sampling_rng / meme_injection_rng properties. Those advance
        # as turns consume them and their position is not checkpointed, so a
        # resumed run would rebuild them at turn 1's position and replay
        # earlier draws (phase4.md D8).
        self._seeds = seeds
        self._cost_tracker = cost_tracker
        self._run_dir = runs_dir / run.run_id
        # agent_id -> that agent's most recent Interaction. This is what
        # prompt_builder renders as neighbour posts (Q4.1). Mutated only in
        # step 6.
        self._last_posts: dict[str, Interaction] = {}

        # --- bank regime ---------------------------------------------------
        self._bank = bank
        if self._is_bank and self._bank is None:
            raise ValueError("regime='bank' requires a BankManager")
        # agent_id -> the rounds it remembers, oldest first. Held here rather
        # than on StanceRecord: the memory section needs the TEXT of what an
        # agent read, and putting that on the agent would copy those strings
        # into every checkpoint, every agents_final.jsonl line and every
        # snapshot deep-copy, for data this object already has. The log keeps
        # the item IDs, so interactions.jsonl stays the authority after the
        # fact and _rebuild_history can reconstruct this from it.
        self._history: dict[str, list[RoundMemory]] = {}

    @property
    def _is_bank(self) -> bool:
        return self._run.regime == "bank"

    async def run(self) -> None:
        """Entry point. Resumes from a checkpoint when one exists, else starts
        fresh at turn 1 (preceded, in the bank regime, by the round-0 probe)."""
        self._run_dir.mkdir(parents=True, exist_ok=True)

        resume_state = self._checkpoint_mgr.load(self._run.run_id)
        if resume_state is not None:
            start_turn = resume_state.last_completed_turn + 1
            self._agent_manager.restore_from_snapshot(resume_state.agent_snapshot)
            self._last_posts = self._rebuild_last_posts()
            # Round 0 is checkpointed as last_completed_turn=0, so any
            # checkpoint at all means it has already happened and start_turn
            # is at least 1. There is no path that re-probes it.
            needs_round_zero = False
            self._restore_bank_state()
        else:
            self._agent_manager.initialize_population()
            self._run.status = "running"
            self._run.started_at_utc = _now_utc_iso()
            self._logger.write_run_config(self._run)
            start_turn = 1
            needs_round_zero = self._is_bank

        self._verify_strategy_covers_the_population()
        self._verify_one_language_per_run()
        self._write_graph_edges()

        last_completed = start_turn - 1
        try:
            if needs_round_zero:
                await self._probe_round_zero()
                # Checkpointed immediately, so a crash in round 1 does not
                # cost the starting measurement -- and so a resume cannot
                # probe round 0 twice, which would put two turn-0 rows in the
                # log and two turn-0 records in every agent's history.
                self._save_checkpoint(ROUND_ZERO)

            for turn in range(start_turn, self._run.K + 1):
                await self._run_turn(turn)
                last_completed = turn
                if turn % self._run.checkpoint_every_n_turns == 0:
                    self._save_checkpoint(turn)
        except BaseException:
            # Any exception reaching here is fatal by construction: the two
            # recoverable failure modes are absorbed per-agent below. Record
            # what was reached before re-raising, so a resume knows where to
            # pick up and the run does not look "running" forever.
            self._finalize("failed_partial" if last_completed else "failed_total")
            raise

        # The loop's modulo can skip the final turn (K=5 with
        # checkpoint_every_n_turns=2 checkpoints 2 and 4, never 5), which
        # would make a completed run resume from the wrong place.
        if last_completed and last_completed % self._run.checkpoint_every_n_turns != 0:
            self._save_checkpoint(last_completed)

        self._logger.write_agents_final(self._agent_manager.all_agents())
        self._finalize("completed")

    async def _run_turn(self, turn: int) -> None:
        """One turn, in whichever regime the run is configured for."""
        if self._is_bank:
            await self._run_bank_turn(turn)
        else:
            await self._run_generated_turn(turn)

    async def _run_generated_turn(self, turn: int) -> None:
        """One turn, in the exact order Fix F requires."""
        all_agents = self._agent_manager.all_agents()

        # Steps 2 and 3 are sequential with respect to each other, not merely
        # before dispatch: they draw from two different seeded Random
        # instances, and a single Random is not safe under concurrent use
        # (full_design_doc.md:1528).
        #
        # Each stream is derived fresh from (seed, purpose, turn), so this
        # turn's draws are identical whether the run reached here uninterrupted
        # or resumed from a checkpoint.
        neighbor_rng = self._seeds.turn_rng("neighbor", turn)
        meme_rng = self._seeds.turn_rng("meme", turn)
        neighbor_map = {
            agent.agent_id: self._interaction_engine.select_neighbors(
                agent, all_agents, turn, neighbor_rng
            )
            for agent in all_agents
        }
        meme_map = self._meme_pool.resolve_injections_for_turn(
            all_agents, turn, rng=meme_rng
        )

        # The frozen snapshot of what everyone last said. Copied so step 6's
        # writes cannot be observed by a dispatch task still in flight.
        frozen_posts = dict(self._last_posts)

        results = await asyncio.gather(
            *(
                self._process_one_agent_turn(
                    agent,
                    neighbor_map[agent.agent_id],
                    meme_map[agent.agent_id],
                    frozen_posts,
                    turn,
                )
                for agent in all_agents
            ),
            return_exceptions=True,
        )

        # Step 6.
        for agent, result in zip(all_agents, results, strict=True):
            if isinstance(result, BaseException):
                # Sec 3.9's sample `continue`s here, which silently swallows
                # CostCeilingExceeded and contradicts Sec 3.11's statement
                # that it is fatal. Re-raise: StanceParseFailure and
                # FatalGatewayError are already absorbed below, so anything
                # arriving here is an unanticipated bug and must surface.
                raise result
            # The record is always logged; a failed turn is data, not a gap.
            await self._logger.write_interaction(result)

            if not result.is_valid_for_analysis:
                # Fix I: a failed agent is EXCLUDED, not given a null turn.
                # Sec 3.9 is explicit that no stance_history entry is
                # appended, "so _current_stance() falls back to the agent's
                # last known stance automatically". Applying it anyway would
                # append a StanceRecord with empty reason_text and make this
                # blank row the agent's most recent post, so neighbours would
                # be shown an empty contribution and the agent would spend one
                # of its five Fix A memory slots on nothing.
                #
                # That is not cosmetic. If one language condition has a higher
                # parse-failure rate than the other (plausible: consistent
                # formatting is harder in code-mixed output), its agents
                # accumulate more blank posts and more blank memory slots, and
                # therefore systematically receive less context. That is
                # precisely the language-driven asymmetry Fix A exists to
                # prevent, arriving through a path Fix A does not guard.
                continue

            self._agent_manager.apply_interaction(agent.agent_id, result)
            self._last_posts[agent.agent_id] = result

    async def _process_one_agent_turn(
        self,
        agent: Agent,
        neighbors: list[Agent],
        meme: MemeContent | None,
        frozen_posts: dict[str, Interaction],
        turn: int,
    ) -> Interaction:
        stance_before = _current_stance_for_logging(agent, turn)

        if meme is not None:
            return self._build_meme_interaction(
                agent, neighbors, meme, turn, stance_before
            )

        # Neighbours who have not spoken yet (turn 1, or an agent whose every
        # turn so far failed) simply contribute nothing.
        neighbor_posts = [
            frozen_posts[n.agent_id] for n in neighbors if n.agent_id in frozen_posts
        ]

        # Only two distinct prompts exist across the retry budget (plain, and
        # format-emphasised), so build each at most once. Rebuilding per
        # attempt re-ran vision_fallback._load_image, re-reading the meme image
        # from disk on every retry for no benefit.
        prompt_cache: dict[bool, BuiltPrompt] = {}

        def prompt_for(emphasize: bool) -> BuiltPrompt:
            if emphasize not in prompt_cache:
                prompt_cache[emphasize] = self._prompt_builder.build_discussion_prompt(
                    agent, neighbor_posts, turn, emphasize_format=emphasize
                )
            return prompt_cache[emphasize]

        emphasize_format = False
        for attempt in range(MAX_STANCE_ATTEMPTS):
            built = prompt_for(emphasize_format)
            try:
                response = await self._gateway.generate(
                    built.text, image=built.image, temperature=self._run.temperature
                )
                stance_after, reason = parse_stance(response.text)
                stance_after = clamp_and_validate_scale(
                    stance_after, self._run.stance_scale
                )
            except StanceParseFailure:
                # Fix I: re-ask with a louder format block. Nothing else about
                # the prompt changes, so a retry stays a response to the same
                # stimulus (Q4.6).
                emphasize_format = True
                continue
            except FatalGatewayError:
                return self._failed_interaction(agent, neighbors, turn, stance_before)

            return Interaction(
                interaction_id=str(uuid.uuid4()),
                run_id=self._run.run_id,
                turn=turn,
                speaker_agent_id=agent.agent_id,
                neighbor_agent_ids=[n.agent_id for n in neighbors],
                stance_before=stance_before,
                stance_after=stance_after,
                reason_text=reason,
                content_type="generated_text",
                meme_id=None,
                used_vision_fallback=built.used_vision_fallback,
                prompt_token_count=response.prompt_tokens,
                completion_token_count=response.completion_tokens,
                model_backend_id=self._run.model_backend_id,
                latency_ms=response.latency_ms,
                api_call_status="success" if attempt == 0 else "retried_success",
                timestamp_utc=_now_utc_iso(),
            )

        # Retry budget exhausted. Fix I: log and exclude, never fabricate a
        # placeholder stance. stance_after repeats stance_before, and because
        # no StanceRecord is appended, interaction_engine's _current_stance
        # keeps reading this agent's last known value on later turns.
        return self._failed_interaction(agent, neighbors, turn, stance_before)

    def _build_meme_interaction(
        self,
        agent: Agent,
        neighbors: list[Agent],
        meme: MemeContent,
        turn: int,
        stance_before: float,
    ) -> Interaction:
        """A meme turn costs no LLM call: the dataset label is the stance.

        used_vision_fallback is False here by definition. It describes what a
        *receiving* speaker saw of someone else's meme, and this agent's own
        posting turn never builds a prompt at all.
        """
        return Interaction(
            interaction_id=str(uuid.uuid4()),
            run_id=self._run.run_id,
            turn=turn,
            speaker_agent_id=agent.agent_id,
            neighbor_agent_ids=[n.agent_id for n in neighbors],
            stance_before=stance_before,
            stance_after=meme.stance_label,
            reason_text=meme.caption_text,
            content_type="meme",
            meme_id=meme.meme_id,
            used_vision_fallback=False,
            prompt_token_count=0,
            completion_token_count=0,
            model_backend_id=self._run.model_backend_id,
            latency_ms=0,
            api_call_status="success",
            timestamp_utc=_now_utc_iso(),
        )

    def _failed_interaction(
        self, agent: Agent, neighbors: list[Agent], turn: int, stance_before: float
    ) -> Interaction:
        return Interaction(
            interaction_id=str(uuid.uuid4()),
            run_id=self._run.run_id,
            turn=turn,
            speaker_agent_id=agent.agent_id,
            neighbor_agent_ids=[n.agent_id for n in neighbors],
            stance_before=stance_before,
            stance_after=stance_before,
            reason_text="",
            content_type="generated_text",
            meme_id=None,
            used_vision_fallback=False,
            prompt_token_count=0,
            completion_token_count=0,
            model_backend_id=self._run.model_backend_id,
            latency_ms=0,
            api_call_status="failed_logged_null",
            timestamp_utc=_now_utc_iso(),
        )

    # --- bank regime (RQ1) ----------------------------------------------

    async def _run_bank_turn(self, turn: int) -> None:
        """One round: resolve, dispatch Phase A, dispatch Phase B, then apply.

        Everything the two phases read is computed from the frozen snapshot
        before either dispatches, which is Fix F. In order:

            1. each agent's side, from its last probe reading
            2. the bank item each agent posts, for that side
            3. each agent's neighbours
            4. what each agent is delivered (nothing, under delivery: off)
            5. Phase A: every reply, concurrently
            6. Phase B: every probe, concurrently
            7. one Interaction per agent, written, then applied

        Steps 1-4 touch no network and mutate nothing. Step 7 is the only
        place agent state, last posts, remembered rounds or bank cycle counts
        change.
        """
        all_agents = self._sorted_agents()
        by_id = {agent.agent_id: agent for agent in all_agents}

        # 1. Sides. Iterated in sorted id order so the tie-break draws come
        # off the stream in a fixed sequence.
        argument_rng = self._seeds.turn_rng("argument", turn)
        side_map = {
            agent.agent_id: self._resolve_side(agent, argument_rng) for agent in all_agents
        }

        # 2. What each agent posts. Before any dispatch, so an agent's own
        # turn-t item is never a function of anybody's turn-t output.
        post_map = {
            agent.agent_id: self._bank.pick(
                agent.agent_id,
                turn,
                side_map[agent.agent_id],
                self._seeds.sender_rng("argument", agent.agent_id),
            )
            for agent in all_agents
        }

        # 3. Neighbours.
        neighbor_rng = self._seeds.turn_rng("neighbor", turn)
        neighbor_map = {
            agent.agent_id: self._interaction_engine.select_neighbors(
                agent, all_agents, turn, neighbor_rng
            )
            for agent in all_agents
        }

        # 4. Delivery. Under `delivery: off` nobody is shown anything, but
        # sides, items and neighbours are still resolved above and the
        # neighbour ids are still logged. That is deliberate: it keeps every
        # seeded stream consuming identically to a delivery-on arm, so the
        # control differs from the treatment in what agents SAW and in
        # nothing else -- which is the only way its drift measurement is a
        # baseline for the treatment rather than for a different run.
        delivered_items: dict[str, list[str]] = {}
        delivered_texts: dict[str, list[str]] = {}
        for agent in all_agents:
            if self._run.delivery == "off":
                delivered_items[agent.agent_id] = []
                delivered_texts[agent.agent_id] = []
                continue
            senders = neighbor_map[agent.agent_id]
            delivered_items[agent.agent_id] = [post_map[n.agent_id] for n in senders]
            # Rendered in the SENDER's language. Identical to the run's
            # language today, since initialize_population gives every agent
            # run.language_condition -- which _verify_one_language_per_run
            # checks, so this and the resume path that renders from
            # run.language_condition provably agree. Reading it off the sender
            # is what a future mixed-language arm would need.
            delivered_texts[agent.agent_id] = [
                self._bank.text(post_map[n.agent_id], by_id[n.agent_id].language_condition)
                for n in senders
            ]

        # 5. Phase A. Skipped entirely under delivery: off -- an agent shown
        # nothing has nothing to reply to, and inventing a reply for it would
        # make the control arm a treatment arm with an empty feed.
        if self._run.delivery == "off":
            replies: list[BackendResponse | None] = [None] * len(all_agents)
        else:
            raw_replies = await asyncio.gather(
                *(
                    self._dispatch_reply(
                        agent,
                        turn,
                        delivered_texts[agent.agent_id],
                        self._history.get(agent.agent_id, []),
                    )
                    for agent in all_agents
                ),
                return_exceptions=True,
            )
            replies = self._unwrap(raw_replies)

        # 6. Phase B. Each probe sees this round's reply, built in memory:
        # _history is not touched until step 7.
        probe_tasks = []
        for agent, reply in zip(all_agents, replies, strict=True):
            if self._run.delivery == "on" and reply is None:
                # The reply failed, so this agent will not be applied this
                # round whatever the probe says. Probing it anyway would spend
                # a call on a reading nothing can consume.
                probe_tasks.append(_no_probe())
                continue
            probe_tasks.append(
                self._dispatch_probe(
                    self._prompt_builder.build_probe_prompt(
                        agent,
                        self._history_with_this_round(
                            agent.agent_id,
                            delivered_texts[agent.agent_id],
                            _reply_text(reply),
                        ),
                    )
                )
            )
        probes = self._unwrap(await asyncio.gather(*probe_tasks, return_exceptions=True))

        # 7. Apply. One row per agent, always written; applied only when both
        # phases produced something.
        for agent, reply, probe in zip(all_agents, replies, probes, strict=True):
            agent_id = agent.agent_id
            interaction = self._build_bank_interaction(
                agent=agent,
                turn=turn,
                neighbors=neighbor_map[agent_id],
                items_read=delivered_items[agent_id],
                posted_item_id=post_map[agent_id],
                reply=reply,
                probe=probe,
            )
            await self._logger.write_interaction(interaction)

            # The item was selected and shown to this agent's neighbours
            # whether or not its own round succeeded, so the cycle advances
            # for everyone. Counted from the logged rows on resume, which is
            # what keeps the two paths identical.
            self._bank.record_post(agent_id, side_map[agent_id])

            if not interaction.is_valid_for_analysis:
                # Fix I, unchanged: a failed round is logged, never applied.
                # It leaves no stance record and spends no memory slot, so
                # this agent's neighbours keep seeing its last real post and
                # its own next prompt is not padded with a blank round.
                continue

            self._agent_manager.apply_interaction(agent_id, interaction)
            self._last_posts[agent_id] = interaction
            self._remember_round(
                agent_id, delivered_texts[agent_id], interaction.reply_text
            )

    async def _probe_round_zero(self) -> None:
        """Measures every agent's starting stance before it reads anything.

        Not a turn: no neighbours, no posts, no replies. Without it the only
        available baseline would be initial_stance, which is a value the
        config chose rather than one the model reports -- and the whole point
        of the probe is that what an agent will answer is not what it was
        told it believes. Round 0 is also the only probe that shows the lean
        sentence, since there is no history for it to read instead.
        """
        all_agents = self._sorted_agents()
        probes = self._unwrap(
            await asyncio.gather(
                *(
                    self._dispatch_probe(
                        self._prompt_builder.build_probe_prompt(
                            agent, [], include_lean=True
                        )
                    )
                    for agent in all_agents
                ),
                return_exceptions=True,
            )
        )

        for agent, probe in zip(all_agents, probes, strict=True):
            interaction = self._build_bank_interaction(
                agent=agent,
                turn=ROUND_ZERO,
                neighbors=[],
                items_read=[],
                posted_item_id=None,
                reply=None,
                probe=probe,
            )
            await self._logger.write_interaction(interaction)
            if not interaction.is_valid_for_analysis:
                continue
            self._agent_manager.apply_interaction(agent.agent_id, interaction)

    def _resolve_side(self, agent: Agent, argument_rng) -> str:
        """pro above the midpoint, con below, a seeded draw exactly on it.

        The draw is taken for EVERY agent and used only by the ones sitting
        exactly on the midpoint. Drawing only when needed would make the
        sequence depend on how many agents happened to land on 4, so two arms
        of a matched set that differ in one agent's reading would get
        different tie-breaks for every agent after it. Drawing unconditionally
        makes each agent's tie-break a function of its position alone.
        """
        tie_break = argument_rng.choice(SIDES)
        stance = agent.current_stance()
        if stance > MIDPOINT:
            return SIDE_PRO
        if stance < MIDPOINT:
            return SIDE_CON
        return tie_break

    async def _dispatch_reply(
        self,
        agent: Agent,
        turn: int,
        delivered_texts: list[str],
        history: list[RoundMemory],
    ) -> BackendResponse | None:
        """Phase A for one agent. None means a logged failure, not a retry.

        There is no parse-retry loop here, unlike the generated regime: a bank
        reply is not parsed for anything, so there is no malformed output to
        re-ask for. Transient provider errors are still retried inside the
        gateway.
        """
        built = self._prompt_builder.build_discussion_prompt(
            agent,
            [],
            turn,
            delivered_posts=delivered_texts,
            history_rounds=history,
        )
        try:
            return await self._gateway.generate(
                built.text, image=None, temperature=self._run.temperature
            )
        except FatalGatewayError:
            return None

    async def _dispatch_probe(self, prompt_text: str) -> tuple[BackendResponse, ProbeScore] | None:
        """Phase B for one agent. None means a logged failure.

        ProbeMassTooLow is absorbed here exactly like FatalGatewayError: the
        model answered something other than a scale point, which is a
        per-agent event and not a reason to end the run. It is never
        retried -- the probe is deterministic at temperature 0, so a second
        identical call would return the same off-scale answer.
        """
        try:
            response = await self._gateway.score_next_token(prompt_text)
            return response, score_probe(
                response.top_logprobs or [], self._run.probe_mass_floor
            )
        except (FatalGatewayError, ProbeMassTooLow):
            return None

    def _build_bank_interaction(
        self,
        *,
        agent: Agent,
        turn: int,
        neighbors: list[Agent],
        items_read: list[str],
        posted_item_id: str | None,
        reply: BackendResponse | None,
        probe: tuple[BackendResponse, ProbeScore] | None,
    ) -> Interaction:
        """One row. Token counts and latency are summed over both calls.

        Summed rather than kept in separate columns: the fields already exist,
        every analysis reads them as "what this agent-round cost", and a round
        is two calls. The split is recoverable from the raw responses if it is
        ever needed; the total is what budgeting and the Fix A prompt-length
        diagnostic actually use.

        A failed row repeats stance_before as stance_after and leaves every
        probe column null, per Fix I: no fabricated stance, and nothing that
        could be mistaken for a reading.
        """
        stance_before = agent.current_stance()
        reply_text = _reply_text(reply)
        failed = probe is None or (reply is None and self._run.delivery == "on" and turn != ROUND_ZERO)

        prompt_tokens = (reply.prompt_tokens if reply else 0) + (probe[0].prompt_tokens if probe else 0)
        completion_tokens = (
            (reply.completion_tokens if reply else 0) + (probe[0].completion_tokens if probe else 0)
        )
        latency_ms = (reply.latency_ms if reply else 0) + (probe[0].latency_ms if probe else 0)

        if failed:
            return Interaction(
                interaction_id=str(uuid.uuid4()),
                run_id=self._run.run_id,
                turn=turn,
                speaker_agent_id=agent.agent_id,
                neighbor_agent_ids=[n.agent_id for n in neighbors],
                stance_before=stance_before,
                stance_after=stance_before,
                reason_text=reply_text,
                content_type="bank_item",
                meme_id=None,
                used_vision_fallback=False,
                items_read=items_read,
                posted_item_id=posted_item_id,
                reply_text=reply_text,
                probe_mean=None,
                probe_dist=None,
                probe_mass=None,
                side_conf=None,
                prompt_token_count=prompt_tokens,
                completion_token_count=completion_tokens,
                model_backend_id=self._run.model_backend_id,
                latency_ms=latency_ms,
                api_call_status="failed_logged_null",
                timestamp_utc=_now_utc_iso(),
            )

        _, reading = probe
        return Interaction(
            interaction_id=str(uuid.uuid4()),
            run_id=self._run.run_id,
            turn=turn,
            speaker_agent_id=agent.agent_id,
            neighbor_agent_ids=[n.agent_id for n in neighbors],
            stance_before=stance_before,
            stance_after=reading.mean,
            # reason_text mirrors reply_text so every existing analysis that
            # reads reason_text keeps working across both regimes.
            reason_text=reply_text,
            content_type="bank_item",
            meme_id=None,
            used_vision_fallback=False,
            items_read=items_read,
            posted_item_id=posted_item_id,
            reply_text=reply_text,
            probe_mean=reading.mean,
            probe_dist=list(reading.dist),
            probe_mass=reading.mass,
            side_conf=reading.side_conf,
            prompt_token_count=prompt_tokens,
            completion_token_count=completion_tokens,
            model_backend_id=self._run.model_backend_id,
            latency_ms=latency_ms,
            api_call_status="success",
            timestamp_utc=_now_utc_iso(),
        )

    # --- bank regime: remembered rounds ---------------------------------

    def _history_with_this_round(
        self, agent_id: str, posts_read: list[str], reply_text: str
    ) -> list[RoundMemory]:
        """This agent's memory plus the round in progress, without mutating.

        The probe has to see the reply the agent just wrote -- that is the
        point of running it after Phase A -- but the round is not committed
        until step 7, and a round whose probe then fails must leave no trace.
        So the extra round is appended to a copy.
        """
        previous = self._history.get(agent_id, [])
        if not posts_read and not reply_text:
            return list(previous)
        return [*previous, RoundMemory(posts_read=list(posts_read), reply_text=reply_text)]

    def _remember_round(self, agent_id: str, posts_read: list[str], reply_text: str) -> None:
        """Commits one round to memory, trimmed to the window.

        A round with neither reads nor a reply is not remembered at all, which
        is every round of a `delivery: off` arm. Recording them would spend
        memory slots on empty blocks; the slice then counts rounds that had
        something in them, which is what the window is for.

        Trimmed here as well as in prompt_builder because this dict lives for
        the whole run: without it a K=200 run would carry 200 rounds per agent
        in memory to render three.
        """
        if not posts_read and not reply_text:
            return
        rounds = max(self._run.memory_rounds, 1)
        remembered = [
            *self._history.get(agent_id, []),
            RoundMemory(posts_read=list(posts_read), reply_text=reply_text),
        ]
        self._history[agent_id] = remembered[-rounds:]

    # --- resume and lifecycle -------------------------------------------

    def _rebuild_last_posts(self) -> dict[str, Interaction]:
        """Reconstructs each agent's most recent post after a resume.

        CheckpointState carries only agent_snapshot, and StanceRecord has no
        content_type or meme_id, so agent state alone cannot say whether a
        past post was a meme. Without this, a resumed run would render every
        previous meme as though it had been ordinary text, quietly changing
        the stimulus mid-run. interactions.jsonl is append-only and complete
        through last_completed_turn, so it is the right source. Not a schema
        change; phase4.md Section 2 did not anticipate this.
        """
        path = self._run_dir / "interactions.jsonl"
        if not path.exists():
            return {}
        last: dict[str, Interaction] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            interaction = Interaction.model_validate_json(line)
            if not interaction.is_valid_for_analysis:
                # Same exclusion as step 6: a failed turn never becomes an
                # agent's most recent post, so a resumed run shows neighbours
                # the same last-known post an uninterrupted one would.
                continue
            previous = last.get(interaction.speaker_agent_id)
            if previous is None or interaction.turn >= previous.turn:
                last[interaction.speaker_agent_id] = interaction
        return last

    def _logged_rows(self) -> list[Interaction]:
        """Every logged row, with re-logged duplicates resolved.

        A run interrupted mid-turn can leave rows on disk with no checkpoint,
        and the resumed pass re-runs that turn and appends again. The later
        write is authoritative (agents_final.jsonl's stance_history references
        it), so rows are keyed on (turn, speaker) with the last write kept --
        the same rule export_csv.py applies on read. Without it, a rebuilt
        bank cycle count would double-count that turn and the resumed run
        would post different items from an uninterrupted one.
        """
        path = self._run_dir / "interactions.jsonl"
        if not path.exists():
            return []
        latest: dict[tuple[int, str], Interaction] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            interaction = Interaction.model_validate_json(line)
            latest[(interaction.turn, interaction.speaker_agent_id)] = interaction
        return [latest[key] for key in sorted(latest)]

    def _restore_bank_state(self) -> None:
        """Rebuilds the bank cycle counts and remembered rounds from the log.

        Neither is in the checkpoint. The counts could have been, but the log
        is already the authority on what was posted and is already being read
        to rebuild last posts; deriving both from one source means they cannot
        disagree with each other. The remembered rounds genuinely cannot be
        checkpointed without putting every post's text into agent state.
        """
        if not self._is_bank:
            return
        rows = self._logged_rows()

        counts: dict[tuple[str, str], int] = {}
        for row in rows:
            if row.posted_item_id is None:
                continue  # the round-0 probe posts nothing
            key = (row.speaker_agent_id, self._bank.side_of(row.posted_item_id))
            counts[key] = counts.get(key, 0) + 1
        self._bank.restore_post_counts(counts)

        rounds = max(self._run.memory_rounds, 1)
        history: dict[str, list[RoundMemory]] = {}
        for row in rows:
            if row.turn == ROUND_ZERO or not row.is_valid_for_analysis:
                # Round 0 is a measurement with nothing to remember, and a
                # failed round was never applied, so neither is in the memory
                # an uninterrupted run would be carrying.
                continue
            remembered = RoundMemory(
                posts_read=[
                    self._bank.text(item_id, self._run.language_condition)
                    for item_id in row.items_read
                ],
                reply_text=row.reply_text,
            )
            if not remembered.posts_read and not remembered.reply_text:
                continue
            history.setdefault(row.speaker_agent_id, []).append(remembered)
        self._history = {
            agent_id: entries[-rounds:] for agent_id, entries in history.items()
        }

    def _verify_strategy_covers_the_population(self) -> None:
        """Every agent must be a node in the graph, checked before dispatch.

        GraphSampling is constructed from ids generated by agent_id_for before
        the population exists, so a mismatch is possible in principle. It
        would otherwise surface as every agent raising on turn 1, after the
        run directory, the config and the round-0 probe had all been written.
        """
        strategy = self._interaction_engine
        if not isinstance(strategy, GraphSampling):
            return
        nodes = {a for edge in strategy.edges() for a in edge}
        population = {agent.agent_id for agent in self._agent_manager.all_agents()}
        if not population <= nodes:
            raise ValueError(
                "graph does not cover the population; agents absent from it: "
                f"{sorted(population - nodes)}"
            )

    def _verify_one_language_per_run(self) -> None:
        """Every agent shares the run's language condition.

        True by construction -- initialize_population assigns
        run.language_condition to all M agents -- and checked because two
        places depend on it. Live delivery renders a post in its SENDER's
        language, while the resume path renders remembered posts in the RUN's
        language (the log records item IDs, not who posted each one). Those
        agree exactly while this holds, and would diverge silently if a future
        mixed-language arm broke it: a resumed run would show agents different
        text from the one it was continuing.
        """
        if not self._is_bank:
            return
        wrong = sorted(
            agent.agent_id
            for agent in self._agent_manager.all_agents()
            if agent.language_condition != self._run.language_condition
        )
        if wrong:
            raise ValueError(
                "the bank regime assumes one language per run, but these agents "
                f"differ from run.language_condition={self._run.language_condition!r}: {wrong}"
            )

    def _write_graph_edges(self) -> None:
        """Writes graph_edges.csv once, at run start.

        Analysis needs the structure the run actually used -- assortativity
        asks whether agents ended up adjacent to others on their own side --
        and rebuilding the graph afterwards from the config would depend on
        this networkx version producing the same graph for the same seed,
        which is not something to rely on across a reinstall.

        Idempotent, so a resumed run still has the file if its first pass
        died before writing it, and an existing file is never rewritten.
        """
        strategy = self._interaction_engine
        if not isinstance(strategy, GraphSampling):
            return
        path = self._run_dir / "graph_edges.csv"
        if path.exists():
            return
        lines = ["agent_a,agent_b"]
        lines += [f"{a},{b}" for a, b in strategy.edges()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _sorted_agents(self) -> list[Agent]:
        """The population in agent-id order.

        Every per-round draw consumes its stream while iterating this list, so
        the order is part of the seed's meaning. AgentManager's dict happens
        to be in creation order, which is already sorted -- sorting explicitly
        means a future change to how agents are stored cannot silently
        reorder every draw in the project.
        """
        return sorted(self._agent_manager.all_agents(), key=lambda a: a.agent_id)

    @staticmethod
    def _unwrap(results: list) -> list:
        """Passes gathered values through, re-raising anything unexpected.

        asyncio.gather(..., return_exceptions=True) flattens real bugs into
        return values. The two recoverable failures are already absorbed into
        None by the dispatch helpers, so an exception arriving here -- a cost
        ceiling, a TypeError, anything -- is not a designed failure mode and
        must surface loudly rather than becoming a soft-failure row.
        """
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results

    def _save_checkpoint(self, turn: int) -> None:
        self._checkpoint_mgr.save(
            self._run.run_id, turn, self._agent_manager.snapshot()
        )

    def _finalize(self, status: str) -> None:
        """Re-persists run_config.json with the terminal lifecycle fields.

        Sec 4.5 requires this, but nothing in the codebase read or wrote
        status / started_at_utc / completed_at_utc / total_cost_usd before
        now. This is the one record that is mutated and rewritten rather than
        appended.
        """
        self._run.status = status
        self._run.completed_at_utc = _now_utc_iso()
        if self._cost_tracker is not None:
            self._run.total_cost_usd = self._cost_tracker.total_usd
        self._logger.write_run_config(self._run)


def build_orchestrator(
    run: ExperimentRun,
    gateway: ModelGateway,
    cost_tracker: CostTracker | None = None,
    runs_dir: Path = Path("runs"),
) -> SimulationOrchestrator:
    # Note on concurrency: this dispatches all M agents per turn at once. For a
    # self-hosted server, construct the gateway with max_concurrency set to the
    # server's batch size (vLLM's --max-num-seqs), or the surplus connections
    # accumulate until the server starts closing them.
    """Wires a full orchestrator from a validated ExperimentRun.

    Exists so the SeedManager routing lives in exactly one place. Each stream
    is deliberately offset from the others, and handing a consumer the wrong
    one produces a run that still completes and is still deterministic but is
    no longer the run the config describes, which is close to undetectable
    after the fact.
    """
    seeds = SeedManager(run.seed)
    meme_pool = MemePoolManager(run.meme_injection, seeds.meme_injection_rng)
    return SimulationOrchestrator(
        run=run,
        agent_manager=AgentManager(run, seeds.persona_assignment_rng),
        interaction_engine=_build_strategy(run, seeds),
        meme_pool=meme_pool,
        gateway=gateway,
        logger=LoggingWriter(runs_dir / run.run_id),
        checkpoint_mgr=CheckpointManager(runs_dir),
        prompt_builder=PromptBuilder(
            run, gateway.supports_vision, meme_lookup=meme_pool.get_meme
        ),
        seeds=seeds,
        cost_tracker=cost_tracker,
        runs_dir=runs_dir,
        bank=BankManager(run.bank_path) if run.regime == "bank" else None,
    )


def _build_strategy(run: ExperimentRun, seeds: SeedManager) -> RecommendationStrategy:
    """Alpha sampling or a fixed graph, per run.neighbor_strategy.

    The graph is built from graph_seed, which is derived from the base seed
    alone, so the four language arms of one trial share a structure. Its nodes
    come from agent_id_for rather than from the population, which does not
    exist yet at wiring time; the orchestrator verifies the two agree before
    it dispatches anything.
    """
    if run.neighbor_strategy == "graph":
        return GraphSampling(
            [agent_id_for(i) for i in range(run.M)],
            degree=run.graph_degree,
            rewire_p=run.graph_rewire_p,
            k=run.N,
            graph_seed=seeds.graph_seed,
        )
    return AlphaSampling(run.alpha, run.N)


async def _no_probe() -> None:
    """A placeholder coroutine for an agent whose reply already failed.

    asyncio.gather needs an awaitable per position so the results line up
    with the agent list; this returns the same None a failed probe does.
    """
    return None


def _reply_text(reply: BackendResponse | None) -> str:
    """The reply body, or an empty string when there was no reply.

    Empty covers two distinct cases on purpose -- a `delivery: off` agent that
    was never asked, and an agent whose call failed -- because the prompt
    treats them identically (nothing to show) while the log distinguishes
    them: the first is api_call_status="success" with no items_read, the
    second is failed_logged_null.
    """
    if reply is None or reply.text is None:
        return ""
    return reply.text.strip()


def _current_stance_for_logging(agent: Agent, turn: int) -> float:
    """Mirrors interaction_engine._current_stance exactly.

    Deliberately duplicated rather than imported: if these two ever disagree,
    stance_before in the log would not be the value neighbour sampling
    actually used, and every regression fitted on that column would be wrong.
    Keeping the logic visible here makes that coupling explicit.
    """
    if turn == 1 or not agent.stance_history:
        return agent.initial_stance
    return agent.stance_history[-1].stance_value


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
