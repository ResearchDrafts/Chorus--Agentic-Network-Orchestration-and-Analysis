"""
Creates and holds the Agent population for a run.

never fully specified in the source
docs): memory_window is read as the agent's own most recent up-to-5
interaction_ids -- a bounded pointer to its own last 5 stance_history
entries, recomputed on every apply_interaction() call, not a record of
which neighbor posts it saw this turn.
"""

from __future__ import annotations
import copy
import random
from pathlib import Path
from sandbox.models import Agent, ExperimentRun, Interaction, StanceRecord


class AgentManager:
    def __init__(self, run: ExperimentRun, rng: random.Random):
        self._run = run
        self._rng = rng
        self._agents: dict[str, Agent] = {}

    def initialize_population(self) -> list[Agent]:
        """
        Creates run.M agents: assigns each a persona (drawn from the
        persona pool referenced by run.persona_pool_id), an initial_stance
        (uniform draw across the finite stance scale), and
        run.language_condition / run.model_backend_id (identical for every
        agent in this scope).

        Deterministic given the same seed: persona assignment order and
        initial-stance draws both consume from self._rng in a fixed order
        (persona first, then stance).
        """
        personas = self._load_persona_pool()
        agents = []
        for i in range(self._run.M):
            persona = personas[i % len(personas)]  # cycle if M > pool size
            stance = self._draw_initial_stance()
            agent = Agent(
                agent_id=f"agent_{i:04d}",
                run_id=self._run.run_id,
                persona=persona,
                language_condition=self._run.language_condition,
                model_backend_id=self._run.model_backend_id,
                initial_stance=stance,
                stance_history=[],
                memory_window=[],
                created_at_turn=0,
            )
            self._agents[agent.agent_id] = agent
            agents.append(agent)
        return agents

    def _load_persona_pool(self) -> list[str]:
        """Raises FileNotFoundError if the pool file is missing -- should
        never happen here since config_loader already validated this path
        exists, but this does not trust that check blindly."""
        path = Path("data/personas") / f"{self._run.persona_pool_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"persona pool vanished: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def _draw_initial_stance(self) -> float:
        # Uniform draw across the configured finite stance scale. Uses
        # self._rng, NOT the global random module, for reproducibility.
        scale = self._run.stance_scale
        return float(self._rng.choice(scale))

    def get_agent(self, agent_id: str) -> Agent:
        """Raises KeyError if agent_id is unknown -- treated as a
        programmer error, so it is NOT caught/wrapped here."""
        return self._agents[agent_id]

    def all_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def apply_interaction(self, agent_id: str, interaction: Interaction) -> None:
        """Appends a StanceRecord derived from `interaction` to the agent's
        stance_history, then recomputes memory_window as the agent's own
        most recent up-to-5 interaction_ids (schema-capped at 5, per
        Agent.memory_window's Field(max_length=5))."""
        agent = self._agents[agent_id]
        agent.stance_history.append(
            StanceRecord(
                turn=interaction.turn,
                stance_value=interaction.stance_after,
                reason_text=interaction.reason_text,
                interaction_id=interaction.interaction_id,
            )
        )
        agent.memory_window = [r.interaction_id for r in agent.stance_history[-5:]]

    def snapshot(self) -> list[Agent]:
        """For Checkpointing (Feature 13). Deep copy so later mutation of
        the manager's own agents never affects the returned snapshot."""
        return copy.deepcopy(self.all_agents())

    def restore_from_snapshot(self, snapshot: list[Agent]) -> None:
        """Rebuilds the internal dict from a list[Agent], deep-copying so
        later mutation of the manager never affects the passed-in list."""
        self._agents = {agent.agent_id: agent for agent in copy.deepcopy(snapshot)}
