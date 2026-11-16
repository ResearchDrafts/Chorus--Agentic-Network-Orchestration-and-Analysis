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
import json
import random
from pathlib import Path
from sandbox.models import Agent, ExperimentRun, Interaction, StanceRecord
from sandbox.seed_manager import persona_trait_rng


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


# The three demographic fields ExperimentRun.persona_attributes="demographic"
# (or above) requires every persona-pool row to carry. Named here, not
# inlined, so config_loader's load-time check and this module's own parse
# agree on the set without repeating it.
PERSONA_ATTRIBUTE_FIELDS = ("age", "gender", "education")

# The Big Five, in OCEAN order -- fixed so the five draws from one
# persona_trait_rng call always land on the same trait in the same
# position, regardless of which order a caller happens to read them in.
BIG_FIVE_TRAITS = (
    "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
)


class PersonaRow:
    """One line of a demographic-mode persona pool: the free-text
    description plus whatever of age/gender/education that line carried.

    A plain class, not a Pydantic model: this is an internal loader
    artifact describing one line on disk, not a value that crosses a
    process or API boundary the way Agent and ExperimentRun do.
    """

    __slots__ = ("persona", "age", "gender", "education")

    def __init__(self, persona: str, age=None, gender=None, education=None):
        self.persona = persona
        self.age = age
        self.gender = gender
        self.education = education


def load_persona_rows(persona_pool_id: str) -> list[PersonaRow]:
    """Every row of the pool, read as a JSON object per line.

    Used instead of load_persona_pool when ExperimentRun.persona_attributes
    is "demographic" or "demographic_big5", which need age/gender/education
    alongside the free-text description -- a plain-text pool's lines parse
    as neither JSON nor an object, so pointing a demographic-mode run at
    one fails here with the line number, not three fields silently None.

    This loader itself only requires a non-empty "persona" key on each
    line; whether age/gender/education are actually present and non-empty
    is a validation question for config_loader (Fix 7's own "a missing
    field is a load-time error naming the row"), not a parsing one -- a
    row missing "education" is valid JSON and a valid PersonaRow, just one
    config_loader must refuse to run a demographic-mode population from.
    """
    path = Path("data/personas") / f"{persona_pool_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"persona pool vanished: {path}")
    rows: list[PersonaRow] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{path} line {line_no}: not valid JSON ({e}) -- a "
                "demographic persona pool needs one JSON object per line"
            ) from e
        if not isinstance(raw, dict):
            raise ValueError(f"{path} line {line_no}: not a JSON object")
        persona = raw.get("persona")
        if not isinstance(persona, str) or not persona.strip():
            raise ValueError(f"{path} line {line_no}: no non-empty 'persona' field")
        rows.append(PersonaRow(
            persona=persona.strip(),
            age=raw.get("age"), gender=raw.get("gender"), education=raw.get("education"),
        ))
    if not rows:
        raise ValueError(f"{path}: no persona rows found")
    return rows


def draw_big5(seed: int, persona_index: int) -> dict[str, str]:
    """This persona index's five Big Five draws: trait name -> "high" or
    "low", each independently at probability 0.5.

    Keyed by persona_index via persona_trait_rng (see its own docstring for
    why this is not a sequential draw off the live persona-assignment
    stream), so the same persona index gets the same five draws in every
    language arm of a seed, and in every lean_assignment mode of it.
    """
    rng = persona_trait_rng(seed, persona_index)
    return {trait: ("high" if rng.random() < 0.5 else "low") for trait in BIG_FIVE_TRAITS}


def _evenly_repeated(points: list[float], n: int) -> list[float]:
    """`points` repeated whole as many times as it divides into n, plus the
    first `remainder` points in `points`' own order.

    Shared by _balanced_leans (over the whole scale) and
    _balanced_two_camps_leans (over one camp's half of it): both need "as
    even a multiset of n draws from this list as possible, with no
    randomness in which points make up the remainder" and would otherwise
    duplicate each other's arithmetic.
    """
    full_blocks, remainder = divmod(n, len(points))
    return list(points) * full_blocks + list(points)[:remainder]


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
        (persona first, then stance) under lean_assignment="random"; under
        "balanced" the whole population's leans are drawn once, up front,
        in the same stream position the per-agent draws would otherwise
        occupy (see _balanced_leans).

        With persona_attributes="none" (every existing config), this is
        byte-for-byte what it always has been. With "demographic" or above,
        personas come from load_persona_rows instead of load_persona_pool,
        and each agent also carries the row's age/gender/education; with
        "demographic_big5" it additionally carries five Big Five draws from
        draw_big5, keyed by the persona's POOL index -- not the population
        index i -- so two agents sharing a persona (the pool wraps before M
        agents are built) also share its demographic and trait profile,
        the same way they already share its free-text description.
        """
        attributes = self._run.persona_attributes
        persona_rows: list[PersonaRow] | None = None
        if attributes == "none":
            personas = self._load_persona_pool()
        else:
            persona_rows = load_persona_rows(self._run.persona_pool_id)
            personas = [row.persona for row in persona_rows]
        if self._run.lean_assignment == "balanced":
            leans = self._balanced_leans()
        elif self._run.lean_assignment == "balanced_two_camps":
            leans = self._balanced_two_camps_leans()
        else:
            leans = None
        agents = []
        for i in range(self._run.M):
            pool_index = i % len(personas)  # cycle if M > pool size
            persona = personas[pool_index]
            stance = leans[i] if leans is not None else self._draw_initial_stance()
            extra: dict = {}
            if persona_rows is not None:
                row = persona_rows[pool_index]
                extra["age"] = row.age
                extra["gender"] = row.gender
                extra["education"] = row.education
            if attributes == "demographic_big5":
                for trait, level in draw_big5(self._run.seed, pool_index).items():
                    extra[f"trait_{trait}"] = level
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
                **extra,
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

    def _balanced_leans(self) -> list[float]:
        """M leans with every stance_scale point represented as evenly as
        possible, shuffled -- so a "balanced" run starts with the same MIX
        of camps on every seed, and only the ASSIGNMENT of that mix to
        particular agents varies with the seed.

        Built as stance_scale repeated whole as many times as it divides
        into M, plus the first `remainder` points of the scale, in the
        scale's own order (e.g. for M=20 over a 7-point scale: two full
        passes of all 7 points, plus points 1-6 once more), then shuffled
        as one list. The remainder is deterministic -- never drawn from
        self._rng -- which is what makes the multiset itself independent
        of the seed: a seeded `sample` of the remainder would let different
        seeds exclude different scale points from their third pass, so two
        seeds could disagree about which points are even represented
        3 times, and "every seed starts with the same mix of leans" would
        be false.

        The shuffle is the only step that consumes self._rng, in the same
        stream position _draw_initial_stance's own choice() calls would
        otherwise occupy -- a "balanced" run and a "random" run of the same
        seed start from the same point in the stream; they just use what
        comes off it differently.
        """
        leans = _evenly_repeated(self._run.stance_scale, self._run.M)
        self._rng.shuffle(leans)
        return leans

    def _balanced_two_camps_leans(self) -> list[float]:
        """M leans split into two equal-sized camps, one drawn evenly from
        the scale points strictly below the midpoint and the other its
        mirror image about the midpoint, with nobody at the midpoint
        itself.

        "balanced" alone can leave a population at 9 agents on one side and
        11 on the other (M=20 over a 7-point scale gives lean 4, the
        midpoint, 3 agents of its own, splitting the other 17 as 9 below
        and 8 above) and that midpoint camp's own probe reading tips it
        toward whichever side it is already closer to -- an echo-chamber
        measurement that depends on two roughly matched camps existing at
        all needs to rule both of those out before round 1.

        Fix 7: the pro camp is NOT drawn independently from `above` by its
        own call to _evenly_repeated. _evenly_repeated's remainder always
        favours the first point of whichever list it is given, and `below`
        and `above` are each sorted ascending -- so the con camp's extra
        agent landed on 1 (farthest from the midpoint) while the pro camp's
        landed on 5 (closest to it), making con systematically more
        extreme than pro on every even M. Mirroring con's own leans about
        the midpoint (lean L <-> 2*midpoint - L) instead guarantees the two
        camps are exact reflections of each other, independent of seed.
        This assumes the configured stance_scale is itself symmetric about
        its midpoint (true of every Likert-style scale this codebase uses);
        a scale where a mirrored point does not exist raises rather than
        silently producing an out-of-scale lean.

        Camp sizes are M // 2 each; for odd M, the RNG stream this method
        itself shuffles with decides which camp gets the extra agent, which
        means -- unlike "balanced"'s own deterministic remainder -- an odd
        M's exact camp sizes are the one part of this multiset that is
        allowed to depend on the seed. The extra agent's own lean is drawn
        from `below`/`above` directly (not mirrored -- there is no partner
        to mirror against for an unpaired agent), the same way it always
        has been.
        """
        scale = sorted(self._run.stance_scale)
        midpoint = scale[len(scale) // 2]
        below = [point for point in scale if point < midpoint]
        above = [point for point in scale if point > midpoint]
        if not below or not above:
            raise ValueError(
                "lean_assignment='balanced_two_camps' needs at least one "
                f"stance_scale point below and above the midpoint {midpoint}, "
                f"got {scale}"
            )
        half, remainder = divmod(self._run.M, 2)
        con_leans = _evenly_repeated(below, half)
        pro_leans = [2 * midpoint - lean for lean in con_leans]
        unmirrored = sorted(set(pro_leans) - set(above))
        if unmirrored:
            raise ValueError(
                "lean_assignment='balanced_two_camps' needs stance_scale "
                f"symmetric about the midpoint {midpoint} to mirror the con "
                f"camp into a pro camp, but {unmirrored} (below's mirror) "
                f"is not in the scale's own above-midpoint points {above}"
            )
        leans = con_leans + pro_leans
        if remainder:
            extra_side = below if self._rng.choice(["con", "pro"]) == "con" else above
            leans += _evenly_repeated(extra_side, 1)
        self._rng.shuffle(leans)
        return leans

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
