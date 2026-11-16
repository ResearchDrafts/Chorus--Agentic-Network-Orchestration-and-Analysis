"""Per-round loop for the opinion-CSV experiment (plan.md).

Deliberately simple relative to simulation_orchestrator.py: no checkpoint/
resume support, no per-agent soft-failure handling -- a parse failure or
gateway error aborts the whole run. Round 0 writes initial_stance.csv;
each round 1..K appends three columns per agent to it and one row to
opinion_round_metrics.csv.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from sandbox.cost_tracker import CostTracker
from sandbox.interaction_engine import GraphSampling
from sandbox.model_gateway import ModelGateway
from sandbox.opinion_csv import write_initial_rows, append_round_columns
from sandbox.opinion_llm import build_initial_prompt, build_update_prompt, parse_opinion_response
from sandbox.opinion_metrics import build_graph, compute_round_metrics, write_round_metrics_row
from sandbox.opinion_models import LANGUAGES, OpinionAgentState, OpinionExperimentConfig
from sandbox.opinion_pool import PersonaRecord, load_opinion_persona_pool
from sandbox.rate_limiter import RateLimiter
from sandbox.seed_manager import SeedManager


def resolve_languages(
    config_languages: dict[str, bool], pool_records: list[PersonaRecord]
) -> dict[str, str]:
    """agent_id -> resolved language.

    exactly one toggle True -> every agent forced to that language,
    overriding the pool's own preference.
    all four toggles True   -> each agent's own pool language_preference
    (a genuinely mixed population).
    anything else (2 or 3)  -> NotImplementedError, deliberately unhandled
    per the "ignore that edge case for now" scope decision in plan.md.
    """
    active = sorted(lang for lang, on in config_languages.items() if on)
    if len(active) == 1:
        return {r.agent_id: active[0] for r in pool_records}
    if len(active) == len(LANGUAGES):
        return {r.agent_id: r.language_preference for r in pool_records}
    raise NotImplementedError(
        f"languages={active}: only exactly one language on, or all "
        f"{len(LANGUAGES)} on, is supported. Partial combinations (2 or 3 "
        "active) are deliberately unimplemented for now."
    )


def _similarity_filtered_feed(
    config: OpinionExperimentConfig,
    graph: GraphSampling,
    agent_id: str,
    turn: int,
    prev_beliefs: dict[str, float],
) -> list[str]:
    """Up to config.N of agent_id's current graph neighbours within
    similarity_threshold of its own previous-round belief, nearest first.

    Mirrors GraphSimilaritySampling's algorithm (interaction_engine.py) --
    a private per-reader RNG keyed by (seed, turn, agent_id) so this draw
    never desyncs across language arms that happen to filter a different
    number of candidates for one agent; shuffle for tie-break, stable sort
    by gap ascending, shuffle again for delivery order. Never pads back up
    to N if fewer qualify.
    """
    own_belief = prev_beliefs[agent_id]
    candidates = graph.neighbors_of(agent_id)
    eligible = [
        c for c in candidates
        if abs(prev_beliefs[c] - own_belief) <= config.similarity_threshold
    ]

    reader_rng = random.Random(f"{config.seed}|opinion_recsys|{turn}|{agent_id}")
    reader_rng.shuffle(eligible)
    eligible.sort(key=lambda c: abs(prev_beliefs[c] - own_belief))

    chosen = eligible[: config.N]
    reader_rng.shuffle(chosen)
    return chosen


async def run(config: OpinionExperimentConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = load_opinion_persona_pool(config.persona_pool_path)
    if config.M is not None:
        if config.M > len(pool):
            raise ValueError(f"M={config.M} exceeds pool size {len(pool)}")
        pool = pool[: config.M]

    agent_ids = [r.agent_id for r in pool]
    persona_by_agent = {r.agent_id: r.persona_text for r in pool}
    language_by_agent = resolve_languages(config.languages, pool)

    seed_mgr = SeedManager(config.seed)
    belief_rng = random.Random(f"{config.seed}|initial_belief")
    initial_belief = {aid: belief_rng.uniform(-2.0, 2.0) for aid in agent_ids}

    graph = GraphSampling(
        agent_ids, config.graph_degree, config.graph_rewire_p, config.N, seed_mgr.graph_seed,
        allow_short_feed=config.allow_short_feed,
    )
    metrics_graph = build_graph(agent_ids, graph.edges())

    tracker = CostTracker(config)
    gateway = ModelGateway(
        config.model_backend_id,
        RateLimiter({}),
        tracker,
        api_base=config.api_base,
        max_concurrency=config.max_concurrency,
    )

    csv_path = output_dir / "initial_stance.csv"
    metrics_path = output_dir / "opinion_round_metrics.csv"

    states: dict[str, OpinionAgentState] = {}

    async def _initial_call(agent_id: str) -> tuple[str, str, float, int]:
        prompt = build_initial_prompt(
            config.topic, persona_by_agent[agent_id], initial_belief[agent_id],
            language_by_agent[agent_id],
        )
        response = await gateway.generate(
            prompt,
            seed=seed_mgr.call_seed("reply", 0, agent_id),
            max_tokens=config.max_reply_tokens,
        )
        opinion_text, belief, confidence = parse_opinion_response(response.text)
        return agent_id, opinion_text, belief, confidence

    for agent_id, opinion_text, belief, confidence in await asyncio.gather(
        *(_initial_call(aid) for aid in agent_ids)
    ):
        states[agent_id] = OpinionAgentState(
            agent_id=agent_id,
            persona_text=persona_by_agent[agent_id],
            language=language_by_agent[agent_id],
            belief=belief,
            confidence=confidence,
            opinion_text=opinion_text,
        )

    write_initial_rows(
        csv_path,
        [
            {
                "agent_id": aid,
                "language": states[aid].language,
                "persona": persona_by_agent[aid],
                "initial_belief": states[aid].belief,
                "initial_opinion": states[aid].opinion_text,
                "initial_confidence": states[aid].confidence,
            }
            for aid in agent_ids
        ],
    )

    beliefs_round0 = {aid: states[aid].belief for aid in agent_ids}
    write_round_metrics_row(
        metrics_path, 0, compute_round_metrics(metrics_graph, beliefs_round0, beliefs_round0)
    )

    for turn in range(1, config.K + 1):
        prev_beliefs = {aid: states[aid].belief for aid in agent_ids}

        async def _round_call(agent_id: str) -> tuple[str, str, float, int, str]:
            chosen = _similarity_filtered_feed(config, graph, agent_id, turn, prev_beliefs)
            if not chosen:
                prev = states[agent_id]
                return agent_id, prev.opinion_text, prev.belief, prev.confidence, "no_feed"

            prompt = build_update_prompt(
                config.topic, persona_by_agent[agent_id], states[agent_id],
                [states[c] for c in chosen], language_by_agent[agent_id],
            )
            response = await gateway.generate(
                prompt,
                seed=seed_mgr.call_seed("reply", turn, agent_id),
                max_tokens=config.max_reply_tokens,
            )
            opinion_text, belief, confidence = parse_opinion_response(response.text)
            return agent_id, opinion_text, belief, confidence, "success"

        results = await asyncio.gather(*(_round_call(aid) for aid in agent_ids))

        updates = {}
        for agent_id, opinion_text, belief, confidence, status in results:
            states[agent_id] = OpinionAgentState(
                agent_id=agent_id,
                persona_text=persona_by_agent[agent_id],
                language=language_by_agent[agent_id],
                belief=belief,
                confidence=confidence,
                opinion_text=opinion_text,
            )
            updates[agent_id] = (opinion_text, belief, confidence, status)

        append_round_columns(csv_path, turn, updates)

        beliefs_now = {aid: states[aid].belief for aid in agent_ids}
        write_round_metrics_row(
            metrics_path, turn, compute_round_metrics(metrics_graph, beliefs_now, beliefs_round0)
        )

    (output_dir / "run_config.json").write_text(json.dumps(config.model_dump(), indent=2))
    return output_dir
