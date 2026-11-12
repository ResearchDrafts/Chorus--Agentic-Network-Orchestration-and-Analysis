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


def load_persona_pool(persona_pool_id: str) -> list[str]:
    """The personas in one pool, in file order.

    Despite the .jsonl extension these files are plain text, one free-text
    description per line -- nothing parses JSON here.

    Module-level rather than a method because the single-exposure scripts
    (sandbox/single_exposure.py) need the same pool the population is drawn
    from, and read it by persona index. A second reader would be a second
    chance to disagree about blank lines or ordering, and a persona index
    that meant different things in a calibration run and in a campaign run
    would silently mismatch the two.

    Raises FileNotFoundError if the pool file is missing -- config_loader
    already validated the path for a run, but this does not trust that check
    blindly, and a script may be pointed at a pool no config mentions.
    """
    path = Path("data/personas") / f"{persona_pool_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"persona pool vanished: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def agent_id_for(index: int) -> str:
    """The canonical agent id for a population index.

    Exists because GraphSampling has to be constructed from the population's
    ids before the population exists -- build_orchestrator wires the graph,
    and initialize_population runs later -- so two places need to agree on
    this format exactly. A drift between them would build a graph whose nodes
    are not the run's agents, and the first symptom would be every agent
    raising "not a node in this run's graph" on turn 1.
    """
    return f"agent_{index:04d}"


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
                agent_id=agent_id_for(i),
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
        """Delegates to load_persona_pool, so the population and the
        single-exposure scripts read one pool the same way."""
        return load_persona_pool(self._run.persona_pool_id)

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
        Agent.memory_window's Field(max_length=5)).

        In the bank regime the record's stance_value is the probe mean and its
        reason_text is the reply the agent wrote. Both come from the same two
        columns the generated regime uses, so everything downstream that reads
        stance_history -- checkpointing, agents_final.jsonl, the engine's
        current-stance lookup -- keeps working without knowing which regime
        produced the run.

        What `items_read` is NOT stored here. The bank regime's memory section
        needs the text of the posts an agent read, and StanceRecord has no
        field for it; adding one would put the same strings in the checkpoint,
        in agents_final.jsonl and in every snapshot deep-copy, for data the
        orchestrator already holds in its own per-round history. The
        orchestrator keeps it (see _history_for), and the Interaction row
        carries the item IDs so the log remains the authority after the fact.
        """
        agent = self._agents[agent_id]
        agent.stance_history.append(
            StanceRecord(
                turn=interaction.turn,
                stance_value=self._stance_value_for(interaction),
                reason_text=self._reason_text_for(interaction),
                interaction_id=interaction.interaction_id,
            )
        )
        agent.memory_window = [r.interaction_id for r in agent.stance_history[-5:]]

    def _stance_value_for(self, interaction: Interaction) -> float:
        """The bank regime's stance is the probe mean, not a self-report.

        Raises when a bank row reaches here with no probe_mean, rather than
        falling back to stance_after. A row without a probe is a row whose
        probe failed, and Fix I's rule is that such a row is logged and never
        applied -- so arriving here with one is an orchestrator bug, and a
        fallback would hide it by writing the agent's previous stance into
        history as though it had been measured again.
        """
        if self._run.regime != "bank":
            return interaction.stance_after
        if interaction.probe_mean is None:
            raise ValueError(
                f"bank-regime interaction {interaction.interaction_id} has no "
                "probe_mean: a failed probe must be logged, never applied"
            )
        return interaction.probe_mean

    def _reason_text_for(self, interaction: Interaction) -> str:
        """reply_text in the bank regime, reason_text otherwise.

        The orchestrator writes the reply into both columns, so these agree
        today; reading reply_text here keeps the dependency pointing at the
        field that actually holds a bank reply, rather than at the one that
        happens to mirror it.
        """
        if self._run.regime == "bank":
            return interaction.reply_text
        return interaction.reason_text

    def snapshot(self) -> list[Agent]:
        """For Checkpointing (Feature 13). Deep copy so later mutation of
        the manager's own agents never affects the returned snapshot."""
        return copy.deepcopy(self.all_agents())

    def restore_from_snapshot(self, snapshot: list[Agent]) -> None:
        """Rebuilds the internal dict from a list[Agent], deep-copying so
        later mutation of the manager never affects the passed-in list."""
        self._agents = {agent.agent_id: agent for agent in copy.deepcopy(snapshot)}
