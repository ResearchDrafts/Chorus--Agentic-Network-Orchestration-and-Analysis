"""Neighbor selection (full_design_doc.md Sec 3.3).

Two strategies live here, both satisfying RecommendationStrategy and neither
aware of the other.

AlphaSampling is Ohagi's mechanism, below, and is unchanged.

GraphSampling fixes a social graph once per run and samples from each agent's
own neighbourhood. RQ1 needs it because under alpha sampling who an agent
reads is a function of where that agent already stands, so a stance effect
and a selection effect cannot be separated afterwards: an agent that moved
towards its feed and an agent whose feed moved towards it produce the same
rows. A graph makes the candidate set a structural property of the run, held
constant while language varies.

Ohagi's alpha-sampling mechanism: alpha in [0,1] blends uniform-random
neighbor selection (alpha=0) with maximally homophilous, stance-distance-
weighted selection (alpha=1). ValueError on invalid alpha/N at
construction, and on an insufficient candidate population at call time --
never silently clamped, per the HLD's Critique Pass 3: silently clamping
alpha or N would itself be exactly the kind of implementation-level
confound the design warns against.
"""

from __future__ import annotations

import math
import random
from typing import Protocol

import networkx as nx

from sandbox.models import Agent


class RecommendationStrategy(Protocol):
    def select_neighbors(
        self,
        speaker: Agent,
        all_agents: list[Agent],
        turn: int,
        rng: random.Random,
    ) -> list[Agent]:
        """Must return exactly N distinct agents, never including speaker itself."""
        ...


class AlphaSampling:
    """Ohagi's mechanism. alpha in [0,1]: 0 = uniform random sampling,
    1 = maximally homophilous (always prefers closest-stance candidates)."""

    def __init__(self, alpha: float, N: int):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        self.alpha = alpha
        self.N = N

    def select_neighbors(
        self,
        speaker: Agent,
        all_agents: list[Agent],
        turn: int,
        rng: random.Random,
    ) -> list[Agent]:
        candidates = [a for a in all_agents if a.agent_id != speaker.agent_id]
        if len(candidates) < self.N:
            raise ValueError(f"population too small: need {self.N} candidates, have {len(candidates)}")

        current_stance = _current_stance(speaker, turn)
        weights = []
        for c in candidates:
            c_stance = _current_stance(c, turn)
            distance = abs(current_stance - c_stance)
            similarity_weight = 1.0 / (1.0 + distance)  # closer stance -> higher weight
            uniform_weight = 1.0  # equal weight for every candidate
            blended = self.alpha * similarity_weight + (1 - self.alpha) * uniform_weight
            weights.append(blended)

        # weighted sample without replacement, using rng (never the global
        # random module -- reproducibility depends on this)
        return _weighted_sample_without_replacement(candidates, weights, self.N, rng)


def _current_stance(agent: Agent, turn: int) -> float:
    """Reads the agent's stance as of the end of turn-1 (the frozen
    snapshot, per Fix F). turn=1 reads initial_stance; turn>1 reads the
    last entry in stance_history.

    Deliberately duplicates Agent.current_stance()'s logic rather than
    calling it (per phase3.md's resolution of this fidelity-vs-consolidation
    question): the design doc's own sample code never routes through
    Agent.current_stance(), so this stays a standalone port rather than a
    silent deviation from the spec.
    """
    if turn == 1 or not agent.stance_history:
        return agent.initial_stance
    return agent.stance_history[-1].stance_value


def _weighted_sample_without_replacement(
    items: list[Agent], weights: list[float], k: int, rng: random.Random
) -> list[Agent]:
    """Efraimidis-Spirakis weighted reservoir sampling -- O(n log k),
    deterministic given rng's state.

    Weights must be strictly positive and finite. The key formula divides by
    w, so w == 0 raises ZeroDivisionError, and w < 0 is worse than an error:
    random() ** negative > 1, and keys sort descending, so a negative-weight
    item is *guaranteed* to be selected first. That is a silent wrong answer
    in the module whose whole contract is "never silently clamp", so it is
    checked rather than assumed. AlphaSampling cannot currently produce a
    non-positive weight (blended is strictly positive for any alpha in
    [0,1]), but a NaN stance would propagate through to NaN weights and make
    the sort order meaningless.
    """
    if len(weights) != len(items):
        raise ValueError(f"weights/items length mismatch: {len(weights)} vs {len(items)}")
    for w in weights:
        if not math.isfinite(w) or w <= 0.0:
            raise ValueError(f"sampling weights must be finite and > 0, got {w}")

    keyed = [(rng.random() ** (1.0 / w), item) for w, item in zip(weights, items, strict=True)]
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in keyed[:k]]


class GraphSampling:
    """Watts-Strogatz neighbour sampling: WS(M, degree, rewire_p).

    The graph is built once, at construction, from graph_seed alone -- never
    per turn and never from anything that varies across a matched set. Two
    runs differing only in language therefore share an identical graph, which
    is what makes "the same reader saw the same neighbours in all four arms"
    exactly true rather than approximately true.

    Node indices map to agent IDs in the order given, so agent_ids must arrive
    in a stable order (the orchestrator passes them sorted). The mapping is
    positional rather than by label because the ring lattice WS starts from is
    defined over 0..n-1, and a shuffled mapping would silently change which
    agents are structurally adjacent without changing the graph.

    connected_watts_strogatz_graph, not watts_strogatz_graph: rewiring can
    disconnect a plain WS graph, and a disconnected population is two smaller
    experiments sharing an output file. The connected variant retries
    construction until the graph is connected, deterministically given a seed.
    """

    def __init__(
        self,
        agent_ids: list[str],
        degree: int,
        rewire_p: float,
        k: int,
        graph_seed: int,
    ):
        if degree < 2:
            raise ValueError(f"degree must be >= 2, got {degree}")
        if degree % 2 != 0:
            raise ValueError(f"degree must be even, got {degree}")
        if degree >= len(agent_ids):
            raise ValueError(
                f"degree ({degree}) must be less than the population ({len(agent_ids)})"
            )
        if not (0.0 <= rewire_p <= 1.0):
            raise ValueError(f"rewire_p must be in [0,1], got {rewire_p}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > degree:
            # Every node starts with exactly `degree` neighbours and rewiring
            # only ever removes some of them, so a larger k cannot be
            # satisfied for any node. Raised here rather than per-agent at
            # call time: this is a config error, not a run-time condition.
            raise ValueError(
                f"k ({k}) cannot exceed degree ({degree}): no node has that many neighbours"
            )
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agent_ids must be unique")

        self.k = k
        self._agent_ids = list(agent_ids)
        self._graph = nx.connected_watts_strogatz_graph(
            len(agent_ids), degree, rewire_p, seed=graph_seed
        )
        # agent_id -> that agent's graph neighbours, sorted. Sorted because
        # rng.sample's output depends on the order of the sequence it is given
        # and networkx's adjacency order is an implementation detail; sorting
        # makes the draw a function of the seed alone.
        self._neighbors: dict[str, list[str]] = {
            self._agent_ids[node]: sorted(
                self._agent_ids[other] for other in self._graph.neighbors(node)
            )
            for node in self._graph.nodes
        }

    def select_neighbors(
        self,
        speaker: Agent,
        all_agents: list[Agent],
        turn: int,
        rng: random.Random,
    ) -> list[Agent]:
        """k distinct graph neighbours of speaker, sampled without replacement.

        `turn` is unused: the graph does not change over the run, and which of
        an agent's neighbours it reads this round comes from rng, which the
        orchestrator derives per turn. Kept in the signature to satisfy
        RecommendationStrategy, so the orchestrator can hold either strategy
        without knowing which.
        """
        if speaker.agent_id not in self._neighbors:
            raise ValueError(
                f"{speaker.agent_id!r} is not a node in this run's graph "
                "(the graph is built once from the initial population)"
            )
        candidates = self._neighbors[speaker.agent_id]
        if len(candidates) < self.k:
            # Never return fewer. A short feed is a different stimulus, and an
            # agent that quietly read two posts where everyone else read three
            # would be a confound nothing in the output records.
            raise ValueError(
                f"{speaker.agent_id} has {len(candidates)} graph neighbours, "
                f"need {self.k}"
            )
        # Sampled exactly once: a second rng.sample call would both advance
        # the stream twice per agent and return a different set than the one
        # just validated.
        chosen = rng.sample(candidates, self.k)
        by_id = {a.agent_id: a for a in all_agents}
        missing = set(chosen) - by_id.keys()
        if missing:
            raise ValueError(
                f"graph neighbours absent from the population: {sorted(missing)}"
            )
        # Returned in the sampled order, not sorted: delivery order is part of
        # the stimulus (it is the order posts appear in the feed) and must be
        # reproducible from the seed, which rng.sample's output already is.
        return [by_id[agent_id] for agent_id in chosen]

    def edges(self) -> list[tuple[str, str]]:
        """The edge list, as sorted (agent_id, agent_id) pairs.

        Exposed so analysis can compute assortativity -- whether agents are
        adjacent to others who ended up on their own side -- which needs the
        structure the run actually used, not a graph rebuilt afterwards from
        the config.
        """
        return sorted(
            tuple(sorted((self._agent_ids[a], self._agent_ids[b])))
            for a, b in self._graph.edges
        )
