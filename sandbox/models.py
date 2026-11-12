"""Pydantic data models shared across the sandbox.

Per full_design_doc.md §4: all models are Pydantic BaseModel subclasses,
superseding the HLD's original dataclass sketches.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The four RQ1 conditions. romanized_hindi is Hindi written in Roman script,
# which is a distinct condition from hinglish (Hindi-English code-mixed) and
# from hindi (Devanagari): RQ1 asks whether language and script each move a
# reader independently, so collapsing the two Roman-script conditions into one
# would merge exactly the contrast the experiment manipulates. Shared by Agent
# and ExperimentRun so the two can never drift apart -- an agent whose
# condition its own run rejects would be unreachable through the loader and
# constructible through the models.
LANGUAGE_CONDITIONS = ("english", "hinglish", "romanized_hindi", "hindi")
LANGUAGE_CONDITIONS_PATTERN = "^(" + "|".join(LANGUAGE_CONDITIONS) + ")$"


class StanceRecord(BaseModel):
    # ge=0, not ge=1: the bank regime probes every agent once BEFORE round 1
    # (the "round 0" probe) to measure its starting stance, and that probe is
    # applied as a StanceRecord like any other. Nothing else in the system
    # writes turn 0.
    turn: int = Field(ge=0)
    stance_value: float
    reason_text: str
    interaction_id: str


class Agent(BaseModel):
    # validate_assignment re-checks field constraints on assignment, not just
    # at construction. This exists for memory_window's max_length=5 (Fix A):
    # without it, `agent.memory_window = [8 items]` silently succeeds and the
    # cap is enforced only by agent_manager.py's [-5:] slice, i.e. by
    # convention rather than by the schema -- which is precisely what
    # full_design_doc.md Sec 4.1 claims is NOT the case.
    #
    # Known limit: this catches assignment, not in-place mutation. Pydantic
    # cannot see `agent.memory_window.append(...)`. Fix A is therefore also
    # re-enforced at the point of use, where prompt_builder.py slices [-5:]
    # when rendering memory into a prompt -- the prompt is the only thing
    # that affects experimental results, so that is the cap that matters.
    model_config = ConfigDict(validate_assignment=True)

    agent_id: str
    run_id: str
    persona: str = Field(min_length=1)
    language_condition: str = Field(pattern=LANGUAGE_CONDITIONS_PATTERN)
    model_backend_id: str
    initial_stance: float
    stance_history: list[StanceRecord] = Field(default_factory=list)
    memory_window: list[str] = Field(default_factory=list, max_length=5)
    created_at_turn: int = 0

    def current_stance(self) -> float:
        """Last stance_history entry's value, or initial_stance if empty."""
        return self.stance_history[-1].stance_value if self.stance_history else self.initial_stance


class MemeContent(BaseModel):
    meme_id: str
    image_path: str
    caption_text: str = Field(min_length=1)
    stance_label: float
    offensiveness_label: float | None = None
    source_dataset: str


class Interaction(BaseModel):
    interaction_id: str
    run_id: str
    # ge=0, not ge=1: see StanceRecord above. The bank regime's round-0 probe
    # logs one row per agent at turn 0, before any reading or replying has
    # happened, so the starting stance is in the dataset rather than inferred
    # from initial_stance.
    turn: int = Field(ge=0)
    speaker_agent_id: str
    # No min_length here, but a validator below still requires at least one
    # neighbour for generated_text and meme rows, which is where the original
    # min_length=1 was load-bearing. Two bank-regime rows legitimately carry
    # none: the round-0 probe (nobody has been sampled yet) and -- were the
    # strategy ever to hand back an empty set -- a row that must still be
    # loggable rather than crashing the turn.
    neighbor_agent_ids: list[str] = Field(default_factory=list)
    stance_before: float
    stance_after: float
    reason_text: str
    content_type: str = Field(pattern="^(generated_text|meme|bank_item)$")
    meme_id: str | None = None
    used_vision_fallback: bool = False
    prompt_token_count: int = Field(ge=0)
    completion_token_count: int = Field(ge=0)
    model_backend_id: str
    latency_ms: int = Field(ge=0)
    api_call_status: str = Field(pattern="^(success|retried_success|failed_logged_null)$")
    timestamp_utc: str

    # --- bank regime (RQ1) ------------------------------------------------
    #
    # All optional, all defaulted, so a generated_text row is unchanged and
    # every existing config and fixture still validates. They are populated
    # only by the bank regime, where a round is two LLM calls rather than one:
    # a reply the agent writes, and a private probe that reads its stance.

    # The bank item IDs this agent was shown this round, in delivery order.
    # Empty for the round-0 probe, which is the only row in the dataset where
    # an agent genuinely read nothing. A `delivery: off` round carries the one
    # reserved ID "__neutral__" (bank_manager.NEUTRAL_ITEM_ID), so a control
    # row says "read the side-free message" rather than "read nothing", and a
    # bank is rejected at load if it uses that ID itself.
    items_read: list[str] = Field(default_factory=list)
    # The bank item THIS agent posted this round, for its current side. None
    # at round 0, when nobody has posted yet.
    posted_item_id: str | None = None
    # What the agent wrote in reply. Duplicated into reason_text, which stays
    # the column every existing analysis reads; this one says what it is.
    reply_text: str = ""
    # The two calls' prompt lengths, kept separately as well as summed into
    # prompt_token_count. Both are needed because RQ1 has to report prompt
    # length per language -- Hinglish tokenizes less efficiently than English,
    # measured at a 17% gap -- and the reply and the probe are different
    # prompts whose lengths grow for different reasons: the reply carries the
    # feed, the probe carries the question, and both carry the history. The
    # sum cannot be split back apart after the fact.
    #
    # 0 on a call that never happened: the round-0 probe has no reply, and a
    # failed call has no reported count.
    reply_prompt_tokens: int = Field(default=0, ge=0)
    probe_prompt_tokens: int = Field(default=0, ge=0)
    # The probe result. probe_mean is s-bar, the expected value on 1..7 under
    # the next-token distribution over the digits -- a continuous stance read
    # out of the model rather than a number it was asked to state. None on a
    # failed row, never a filled-in placeholder (Fix I).
    probe_mean: float | None = None
    probe_dist: list[float] | None = None
    probe_mass: float | None = None
    side_conf: float | None = None

    @model_validator(mode="after")
    def check_meme_id_consistency(self) -> "Interaction":
        if self.content_type == "meme" and self.meme_id is None:
            raise ValueError("content_type='meme' requires a non-null meme_id")
        if self.content_type == "generated_text" and self.meme_id is not None:
            raise ValueError("content_type='generated_text' requires meme_id to be null")
        # A bank item is not a meme. Allowing a meme_id here would make the
        # RQ3 meme mechanism and the RQ1 bank mechanism indistinguishable in
        # the log, and content_type is what every analysis filters on.
        if self.content_type == "bank_item" and self.meme_id is not None:
            raise ValueError("content_type='bank_item' requires meme_id to be null")
        return self

    @model_validator(mode="after")
    def check_neighbors_present_for_posted_content(self) -> "Interaction":
        """Preserves the old Field(min_length=1) for the rows it protected.

        A generated_text or meme row with no neighbours means the speaker was
        shown nothing and the sampling step silently produced an empty set,
        which the original min_length=1 existed to catch. Bank rows are
        exempt: the round-0 probe genuinely has no neighbours.
        """
        if self.content_type != "bank_item" and not self.neighbor_agent_ids:
            raise ValueError(
                f"content_type={self.content_type!r} requires at least one neighbor_agent_id"
            )
        return self

    @model_validator(mode="after")
    def check_probe_dist_is_a_full_scale(self) -> "Interaction":
        """A probe distribution is over exactly the seven scale points.

        Anything else means the scorer renormalized over a different support
        than the one probe_mean was computed on, so probe_mean would not be
        the mean of this distribution and the two columns would disagree.
        """
        if self.probe_dist is not None and len(self.probe_dist) != 7:
            raise ValueError(
                f"probe_dist must hold exactly 7 values, got {len(self.probe_dist)}"
            )
        return self

    @property
    def is_valid_for_analysis(self) -> bool:
        return self.api_call_status != "failed_logged_null"


class MemeInjectionConfig(BaseModel):
    # extra="forbid" for the same reason as ExperimentRun below: this block is
    # hand-written YAML, and a typo'd key here silently disables the exact
    # mechanism RQ3 manipulates.
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    meme_pool_id: str | None = None
    injection_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    injection_schedule: str = Field(default="random", pattern="^(random|fixed_turn)$")
    fixed_turns: set[int] = Field(default_factory=set)

    @model_validator(mode="after")
    def check_enabled_requirements(self) -> "MemeInjectionConfig":
        if self.enabled:
            if self.meme_pool_id is None:
                raise ValueError("meme_pool_id is required when meme_injection.enabled is True")
            if self.injection_schedule == "fixed_turn" and not self.fixed_turns:
                raise ValueError("fixed_turns must be non-empty when injection_schedule='fixed_turn'")
        return self


class ExperimentRun(BaseModel):
    # extra="forbid" so an unrecognized YAML key raises instead of vanishing.
    # Twelve of this model's fields are optional, so without it a typo is
    # silently absorbed and the run proceeds on the default. The dangerous
    # case is `meme_injections:` (plural): the config loads, meme injection
    # stays off, and RQ3's meme-present arm runs as a second meme-absent arm,
    # producing a clean null result that reads as "memes do not matter".
    # `max_cost_usd` and `rate_limits` fail the same way, just less quietly.
    model_config = ConfigDict(extra="forbid")

    run_id: str
    rq_target: str = Field(pattern="^(RQ1_RQ2|RQ3|RQ1_RQ2_RQ3)$")
    mode: str = Field(default="multi_turn", pattern="^(multi_turn)$")
    topic: str = Field(min_length=1)
    alpha: float = Field(ge=0.0, le=1.0)
    M: int = Field(ge=2)
    N: int = Field(ge=1)
    K: int = Field(ge=1)
    trial_number: int = Field(ge=1)
    language_condition: str = Field(pattern=LANGUAGE_CONDITIONS_PATTERN)
    model_backend_id: str = Field(min_length=1)
    # Endpoint for a self-hosted backend, e.g. "http://localhost:8000/v1" for
    # vLLM. Lives on the run rather than in an environment variable so it is
    # captured in run_config.json and the run stays reproducible: which server
    # produced a dataset is part of its provenance.
    api_base: str | None = None
    stance_scale: list[float] = Field(min_length=2)
    # Anchor text for the two ends of stance_scale, rendered into the prompt
    # by prompt_builder.py. An unanchored numeric scale is interpreted
    # inconsistently by an LLM across turns and personas, so these exist to
    # make a stance number mean the same thing to every agent. Optional with
    # generic defaults, so existing configs stay valid; set them to something
    # topic-appropriate for a real run (phase4.md Q4.4).
    stance_low_label: str = Field(default="strongly opposed", min_length=1)
    stance_high_label: str = Field(default="strongly in favour", min_length=1)
    persona_pool_id: str = Field(min_length=1)
    meme_injection: MemeInjectionConfig = Field(default_factory=MemeInjectionConfig)

    # --- RQ1 bank regime --------------------------------------------------
    #
    # Every field here defaults to the pre-existing behaviour, so every config
    # written before this block still loads and still means what it meant.
    # The two switches are neighbor_strategy (who an agent reads) and regime
    # (what an agent posts); they are independent, and either can be changed
    # without the other.

    # "alpha" is Ohagi's homophilous sampling over the whole population.
    # "graph" fixes a Watts-Strogatz social graph once per run and samples N
    # neighbours from each agent's own graph neighbourhood, so an agent's
    # candidate set is a structural property of the run rather than a function
    # of everyone's current stance. RQ1 needs the latter: with alpha sampling,
    # who you read depends on where you already stand, and a stance effect and
    # a selection effect become impossible to separate.
    neighbor_strategy: str = Field(default="alpha", pattern="^(alpha|graph)$")
    # WS(M, graph_degree, graph_rewire_p). Even because the ring lattice the
    # construction starts from connects each node to degree/2 neighbours on
    # each side; an odd degree is not representable and networkx would round
    # it silently.
    graph_degree: int = Field(default=6, ge=2)
    graph_rewire_p: float = Field(default=0.1, ge=0.0, le=1.0)

    # "generated": agents write their own posts (the original mechanism).
    # "bank": posts come from a fixed, pre-translated argument bank, so the
    # argument is held identical across language conditions and only its
    # rendering varies. RQ1's whole question is whether the rendering moves a
    # reader, which cannot be asked while the agents are also writing
    # different arguments in each condition.
    regime: str = Field(default="generated", pattern="^(generated|bank)$")
    bank_path: str | None = None
    # "off" is the E4 control: agents read one message that takes no side,
    # reply to it, and are probed exactly like a treatment arm. It measures
    # how much stance drifts from taking part at all -- reading something,
    # writing something, being asked -- which is the baseline every
    # delivery-on arm is compared against.
    #
    # It used to deliver nothing and ask for no reply. That made the control
    # a different activity from the treatment rather than the same activity
    # without the manipulated factor: an agent with nothing to read and
    # nothing to say has a history that never changes, so the arm could not
    # show drift even in principle.
    delivery: str = Field(default="on", pattern="^(on|off)$")
    # The one post a delivery-off agent reads, every round, in the run's own
    # language. On-topic and taking no side: it holds the activity constant
    # against the treatment while holding the persuasive content at zero.
    #
    # Required when regime="bank" and delivery="off", never defaulted, for
    # the same reason probe_question is not: a generic English fallback would
    # silently make every arm's control read English, and the control is what
    # every treatment effect is measured against.
    neutral_message: str | None = None

    # The 7-point survey question, written in this run's language. Asked in a
    # fresh context that never enters the agent's history, so the agent is
    # never told its own position as a number and never anchors on one.
    probe_question: str | None = None
    # How many top logprobs to ask the provider for. The seven digits must all
    # be reachable within this many, or probe_mass_floor rejects the read.
    probe_top_logprobs: int = Field(default=20, ge=1)
    # Minimum share of next-token probability that must land on the digits
    # 1..7 for a probe to count. Below this the model is answering some other
    # question (prose, a refusal, a different scale), and renormalizing the
    # remainder would manufacture a stance out of noise.
    probe_mass_floor: float = Field(default=0.5, gt=0.0, le=1.0)
    # How many past rounds of reading and replies appear in a prompt, for the
    # bank regime. None -- the default -- means the agent keeps its FULL
    # history and nothing is ever dropped.
    #
    # Full history is the default because a window silently changes what the
    # experiment measures: with a three-round window, round 10's stimulus is
    # rounds 7-9, so an agent cannot be moved by an argument it read early and
    # the measured effect is bounded by the window rather than by persuasion.
    # An int >= 1 still means a window, so the option remains available for a
    # run that deliberately wants one; 0 is rejected rather than read as
    # "none", since a zero-round window and no window at all are opposite
    # instructions.
    #
    # When a window IS used it is counted in rounds, not tokens, for exactly
    # the reason Fix A gives: a token budget would hand the language whose
    # text tokenizes less efficiently systematically less context. Full
    # history keeps that guarantee trivially -- every arm keeps everything --
    # and shifts the risk to prompt length, which max_prompt_tokens bounds
    # loudly rather than by truncating.
    memory_rounds: Annotated[int, Field(ge=1)] | None = None
    # A hard ceiling on the prompt length the provider reports, in tokens.
    # None means no ceiling.
    #
    # The point is that nothing is EVER truncated silently. With full history
    # a prompt grows without bound, and the two ways that can end are a
    # provider context-length error (fatal, see model_gateway) or a quiet
    # server-side truncation that nothing in the output records. This bounds
    # it in the one place a run can still be stopped and restarted smaller:
    # exceeding it aborts the run, naming the agent, the round and the count.
    # A run where one language arm's agents lose the front of their history is
    # confounded exactly where RQ1 is measuring.
    max_prompt_tokens: int | None = Field(default=None, ge=1)

    seed: int
    temperature: float = Field(ge=0.0, le=2.0)
    max_cost_usd: float | None = Field(default=None, gt=0.0)
    rate_limits: dict[str, int] = Field(default_factory=dict)
    checkpoint_every_n_turns: int = Field(default=1, ge=1)
    git_commit_hash: str = ""
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    status: str = Field(default="pending", pattern="^(pending|running|completed|failed_partial|failed_total)$")
    total_cost_usd: float = 0.0

    @model_validator(mode="after")
    def check_n_less_than_m(self) -> "ExperimentRun":
        if self.N >= self.M:
            raise ValueError(f"N ({self.N}) must be less than M ({self.M})")
        return self

    @model_validator(mode="after")
    def check_graph_degree(self) -> "ExperimentRun":
        """Only enforced against M when the graph strategy is actually used.

        graph_degree defaults to 6 and is meaningless under alpha sampling, so
        validating `degree < M` unconditionally would reject every existing
        small-population config (M=5 smoke tests included) over a field those
        runs never read. Evenness is checked unconditionally, since it is a
        property of the number itself rather than of the population.
        """
        if self.graph_degree % 2 != 0:
            raise ValueError(
                f"graph_degree must be even, got {self.graph_degree} "
                "(the Watts-Strogatz ring connects degree/2 neighbours per side)"
            )
        if self.neighbor_strategy == "graph":
            if self.graph_degree >= self.M:
                raise ValueError(
                    f"graph_degree ({self.graph_degree}) must be less than M ({self.M})"
                )
            if self.N > self.graph_degree:
                # Every node starts with exactly graph_degree neighbours and
                # rewiring can only reduce some of them, so N > degree can
                # never be satisfied for any node. Raising here beats raising
                # per-agent on turn 1 after the population is already built.
                raise ValueError(
                    f"N ({self.N}) cannot exceed graph_degree ({self.graph_degree}): "
                    "no node has that many graph neighbours"
                )
        return self

    @model_validator(mode="after")
    def check_bank_regime_requirements(self) -> "ExperimentRun":
        """The bank regime cannot run without a bank or a probe question.

        Both are required rather than defaulted. A missing bank_path has no
        safe default at all; a missing probe_question is worse, because a
        generic English fallback would silently probe every language arm in
        English and destroy the comparison the run exists to make.
        """
        if self.regime != "bank":
            return self
        missing = [
            name
            for name, value in (("bank_path", self.bank_path), ("probe_question", self.probe_question))
            if value is None or (isinstance(value, str) and not value.strip())
        ]
        if missing:
            raise ValueError(
                f"regime='bank' requires {' and '.join(missing)} to be set"
            )
        return self

    @model_validator(mode="after")
    def check_neutral_message_for_the_off_control(self) -> "ExperimentRun":
        """The E4 control cannot run without the message it delivers.

        Required rather than defaulted, and non-empty rather than merely
        present: an empty string would deliver a blank post to every agent in
        the control arm every round, which is indistinguishable in the output
        from a message that failed to move anyone.
        """
        if self.regime != "bank" or self.delivery != "off":
            return self
        if self.neutral_message is None or not self.neutral_message.strip():
            raise ValueError(
                "regime='bank' with delivery='off' requires a non-empty "
                "neutral_message: the control reads one on-topic message that "
                "takes no side, written in this run's language"
            )
        return self

    @model_validator(mode="after")
    def check_stance_scale_distinct(self) -> "ExperimentRun":
        """full_design_doc.md Sec 7.1 requires at least 2 DISTINCT values.

        Field(min_length=2) only counts entries, so [3, 3] passed. Then
        min == max, clamp_and_validate_scale accepts exactly one value, every
        reply that is not that value raises StanceParseFailure, and the whole
        population ends up failed_logged_null. Loud once it happens, but
        cheaper to reject at config load.
        """
        if len(set(self.stance_scale)) < 2:
            raise ValueError(
                f"stance_scale needs at least 2 distinct values, got {self.stance_scale}"
            )
        return self


class CheckpointState(BaseModel):
    run_id: str
    last_completed_turn: int
    agent_snapshot: list[Agent]
