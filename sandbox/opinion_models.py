"""Pure data models for the opinion-CSV experiment (plan.md).

Separate from sandbox/models.py's ExperimentRun on purpose: this pipeline's
data shape (free-text opinions, a self-reported confidence, a wide CSV with
one new column-trio per round) does not fit the bank/generated regimes'
invariants, so it gets its own schema rather than bolting new optional
fields onto ExperimentRun.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Mirrors sandbox.models.LANGUAGE_CONDITIONS -- the four values this
# pipeline's language toggles and pool_100.jsonl's "language preference"
# field are both restricted to.
LANGUAGES = ("english", "hindi", "hinglish", "romanized_hindi")


class OpinionExperimentConfig(BaseModel):
    # extra="forbid" so a typo'd YAML key (e.g. "simlarity_threshold") raises
    # at load time instead of silently taking a default -- same reasoning as
    # ExperimentRun's own extra="forbid" (see models.py).
    model_config = ConfigDict(extra="forbid")

    run_id: str
    topic: str = Field(min_length=1)

    # None means "use every row in the persona pool".
    M: int | None = Field(default=None, ge=2)
    K: int = Field(ge=1)

    # Exactly the four LANGUAGES keys, each True/False. Resolution rule
    # (see opinion_orchestrator.resolve_languages):
    #   exactly one True  -> every agent forced to that language
    #   all four True     -> each agent uses its own pool language preference
    #   anything else     -> raises NotImplementedError (unsupported for now)
    languages: dict[str, bool]

    # On the belief scale directly ([-2, 2], Jo et al.'s own v scale) -- this
    # pipeline never converts through probe_scorer.to_x, so there is no x vs
    # v ambiguity to track here the way the bank regime has.
    similarity_threshold: float = Field(default=2.0, gt=0.0)

    N: int = Field(ge=1)
    graph_degree: int = Field(ge=2)
    graph_rewire_p: float = Field(ge=0.0, le=1.0)
    # Off by default, matching every pre-existing graph-strategy config in
    # this repo (e3-e7): GraphSampling normally refuses to start if any node
    # ends up with fewer than N neighbours after Watts-Strogatz rewiring.
    # True opts into delivering fewer than N for just the short nodes
    # instead -- the same escape hatch e9 already uses for this exact
    # problem, which gets more likely to bite as M grows even with 1 slack
    # (degree = N + 1).
    allow_short_feed: bool = False

    seed: int
    model_backend_id: str = Field(min_length=1)
    api_base: str | None = None

    persona_pool_path: str = "data/personas/pool_100.jsonl"
    max_reply_tokens: int = Field(default=512, ge=1)

    # Caps simultaneous in-flight requests, same reasoning as base.yaml's own
    # max_concurrency: unbounded dispatch of M agents at once against a local
    # vLLM server leaves connections beyond its batch size sitting open until
    # the run dies with "Server disconnected". None = unbounded (hosted
    # providers that pool connections themselves).
    max_concurrency: int | None = None

    output_dir: str

    # Read by CostTracker.record() via self._run.max_cost_usd -- kept here,
    # same name, so CostTracker can be reused unmodified against this config
    # type (duck-typed; CostTracker's type hint says ExperimentRun but never
    # checks it at runtime).
    max_cost_usd: float | None = None

    @model_validator(mode="after")
    def check_languages(self) -> "OpinionExperimentConfig":
        missing = set(LANGUAGES) - set(self.languages)
        extra = set(self.languages) - set(LANGUAGES)
        if missing or extra:
            raise ValueError(
                f"languages must have exactly the keys {LANGUAGES}; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        if not any(self.languages.values()):
            raise ValueError("at least one language must be turned on")
        return self

    @model_validator(mode="after")
    def check_graph_degree(self) -> "OpinionExperimentConfig":
        if self.graph_degree % 2 != 0:
            raise ValueError(f"graph_degree must be even, got {self.graph_degree}")
        return self


class OpinionAgentState(BaseModel):
    """One agent's current, mutable round-to-round state.

    Not logged directly -- opinion_csv.py writes rows, this is just what the
    orchestrator threads through the per-round loop and what opinion_llm.py
    renders into prompts.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    persona_text: str
    language: str = Field(pattern="^(" + "|".join(LANGUAGES) + ")$")
    belief: float = Field(ge=-2.0, le=2.0)
    confidence: int = Field(ge=1, le=5)
    opinion_text: str
