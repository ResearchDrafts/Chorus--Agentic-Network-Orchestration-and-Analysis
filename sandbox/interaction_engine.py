"""Neighbor selection (full_design_doc.md Sec 3.3).

Three strategies live here, all satisfying RecommendationStrategy and none
aware of the others.

AlphaSampling is Ohagi's mechanism, below, and is unchanged.

GraphSampling fixes a social graph once per run and samples from each agent's
own neighbourhood. RQ1 needs it because under alpha sampling who an agent
reads is a function of where that agent already stands, so a stance effect
and a selection effect cannot be separated afterwards: an agent that moved
towards its feed and an agent whose feed moved towards it produce the same
rows. A graph makes the candidate set a structural property of the run, held
constant while language varies.

GraphSimilaritySampling (Fix 9) is a third mechanism on top of the same
graph: Jo et al.'s recommendation rule, which does not change who a reader's
graph neighbours are (that is E5's rewiring, a separate mechanism the two
are never combined with) but changes which of them it is actually shown,
ranked by belief similarity within a threshold. See its own docstring.

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
from dataclasses import dataclass
from typing import Protocol

import networkx as nx

from sandbox.models import Agent
from sandbox.probe_scorer import to_x


class RecommendationStrategy(Protocol):
    def select_neighbors(
        self,
        speaker: Agent,
        all_agents: list[Agent],
        turn: int,
        rng: random.Random,
    ) -> list[Agent]:
        """AlphaSampling and GraphSampling always return exactly N distinct
        agents, never including speaker itself. GraphSimilaritySampling is
        the one exception: it may return FEWER than N (its threshold has no
        effect otherwise -- see its own docstring), including zero."""
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
        allow_short_feed: bool = False,
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
        # Fix 11: GraphSimilaritySampling.select_neighbors keys its own
        # per-reader RNG off this, so it never consumes the shared
        # per-round stream it is handed. Stored on the base class since
        # GraphSampling itself has no need of it after construction.
        self.graph_seed = graph_seed
        # Fix 1: at N=5/degree=6 (e8/e9), Watts-Strogatz rewiring during
        # construction strands enough nodes below k that most seeds cannot
        # build at all (48 of the originally committed e8/e9 configs failed
        # this way). allow_short_feed, off by default, is the opt-in that
        # lets both the construction check and select_neighbors below
        # deliver fewer than k rather than raising -- every pre-Fix-1 config
        # leaves this False and keeps today's error byte for byte.
        self.allow_short_feed = allow_short_feed
        self._agent_ids = list(agent_ids)
        self._graph = nx.connected_watts_strogatz_graph(
            len(agent_ids), degree, rewire_p, seed=graph_seed
        )
        self._verify_every_node_has_k_neighbours(graph_seed)
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
        # agent_id -> its node index, the inverse of self._agent_ids. Built
        # once rather than searched per call: rewire() and the resume-time
        # restore_edges() both need it, and both run once per rewire attempt
        # or once per resume, which over a long campaign is often enough that
        # an O(n) list.index() per lookup would matter.
        self._node_of_id: dict[str, int] = {
            agent_id: node for node, agent_id in enumerate(self._agent_ids)
        }

    def _verify_every_node_has_k_neighbours(self, graph_seed: int) -> None:
        """Rejects a graph in which any node has fewer than k neighbours.

        `k <= degree` is necessary but not sufficient. Every node starts the
        ring lattice with exactly `degree` neighbours, and rewiring moves
        edges rather than adding them, so individual nodes can end up below
        `degree` -- WS(8, 4, 0.1) leaves a node with 2 for some seeds.

        Without this, select_neighbors raises for that one agent on round 1,
        after the population is built, the run directory and config are
        written, and the round-0 probe has already been paid for -- M calls
        for a graph that was never going to work. Worse, the failure is a
        function of the seed, so a matched set either fails in all four arms
        or in none, and discovering it after the first arm has run means
        discarding that arm too.

        Raised rather than patched up: adding an edge to satisfy k would make
        this node structurally unlike every other, and dropping to k-1 for it
        would give one agent a shorter feed, which is a confound nothing in
        the output records. The fix is a different seed, a higher degree or a
        lower N, and all three are the caller's to choose -- unless
        allow_short_feed opted in to a short feed being the point rather than
        an error (Fix 1), in which case this check does not run at all.
        """
        if self.allow_short_feed:
            return
        short = sorted(
            (self._agent_ids[node], degree)
            for node, degree in self._graph.degree
            if degree < self.k
        )
        if not short:
            return
        detail = ", ".join(
            f"{agent_id} has {degree}" for agent_id, degree in short
        )
        raise ValueError(
            f"graph_seed={graph_seed} produced a graph in which "
            f"{len(short)} node(s) have fewer than k={self.k} neighbours "
            f"({detail}). Watts-Strogatz rewiring moves edges rather than "
            "adding them, so k <= degree does not guarantee it per node: "
            "raise graph_degree, lower N, or change the run's seed."
        )

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
        if len(candidates) < self.k and not self.allow_short_feed:
            # Never return fewer. A short feed is a different stimulus, and an
            # agent that quietly read two posts where everyone else read three
            # would be a confound nothing in the output records. (Unless
            # allow_short_feed opted into exactly that -- Fix 1.)
            raise ValueError(
                f"{speaker.agent_id} has {len(candidates)} graph neighbours, "
                f"need {self.k}"
            )
        # Sampled exactly once: a second rng.sample call would both advance
        # the stream twice per agent and return a different set than the one
        # just validated.
        chosen = rng.sample(candidates, min(self.k, len(candidates)))
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
        the config. Live: once rewire() has run, this is the CURRENT edge
        set, not the one the graph was constructed with.
        """
        return sorted(
            tuple(sorted((self._agent_ids[a], self._agent_ids[b])))
            for a, b in self._graph.edges
        )

    def neighbors_of(self, agent_id: str) -> list[str]:
        """The CURRENT neighbours of one agent, sorted.

        Live, like edges() -- this reflects every rewire() applied so far,
        which is the point: E5's rewiring rule has to check "is k already a
        neighbour of i" against the graph as it stands partway through a
        round's pass, not as it stood at the round's start.
        """
        if agent_id not in self._neighbors:
            raise ValueError(
                f"{agent_id!r} is not a node in this run's graph"
            )
        return list(self._neighbors[agent_id])

    def degree(self, agent_id: str) -> int:
        """How many CURRENT neighbours one agent has.

        Every agent starts at graph_degree; rewire() can only move edges
        between agents it is not asked to touch, never remove one without
        adding another, so this only ever changes for the two agents at an
        applied rewire's non-reader ends (the one dropped, the one added).
        """
        return len(self.neighbors_of(agent_id))

    def rewire(self, drop: tuple[str, str], add: tuple[str, str]) -> None:
        """Replaces edge `drop` with edge `add`, keeping both ends' other
        neighbours untouched.

        `drop` and `add` need not share an endpoint -- the caller decides
        that -- but E5's rule always calls this as drop=(reader, old) and
        add=(reader, new), which is what keeps the reader's own degree fixed
        (Design point 3): one edge leaves it, one edge arrives, and it is the
        SAME end of both.

        Raises rather than silently no-opping on every way this could be
        asked to do something that is not a well-defined rewire: dropping an
        edge that is not there, adding one that already is, a self-loop, or
        an unknown agent. A caller that reaches any of these has a bug in its
        candidate selection, and a silent no-op would leave the graph
        looking rewired in the log while the structure never moved.
        """
        drop_a, drop_b = drop
        add_a, add_b = add
        for agent_id in (drop_a, drop_b, add_a, add_b):
            if agent_id not in self._neighbors:
                raise ValueError(f"{agent_id!r} is not a node in this run's graph")
        if drop_b not in self._neighbors[drop_a]:
            raise ValueError(f"{drop_a!r} and {drop_b!r} are not neighbours; cannot drop")
        if add_a == add_b:
            raise ValueError(f"cannot add a self-loop at {add_a!r}")
        if add_b in self._neighbors[add_a]:
            raise ValueError(f"{add_a!r} and {add_b!r} are already neighbours; cannot add")

        self._graph.remove_edge(self._node_of_id[drop_a], self._node_of_id[drop_b])
        self._graph.add_edge(self._node_of_id[add_a], self._node_of_id[add_b])

        self._neighbors[drop_a].remove(drop_b)
        self._neighbors[drop_b].remove(drop_a)
        # Re-sorted after insertion, not just appended: select_neighbors'
        # rng.sample draws over this list, and its order is part of what a
        # seed means (see the sorting note on self._neighbors' construction
        # above).
        self._neighbors[add_a] = sorted(self._neighbors[add_a] + [add_b])
        self._neighbors[add_b] = sorted(self._neighbors[add_b] + [add_a])

    def restore_edges(self, edges: list[tuple[str, str]]) -> None:
        """Replaces the current edge set wholesale, for resuming a rewired
        run.

        A rewired graph's edges are not recoverable from interactions.jsonl
        the way the bank's post counts and remembered rounds are (see
        SimulationOrchestrator._restore_bank_state): the log records what an
        agent read, not who it is currently linked to. So the checkpoint
        carries the edge list directly, and this is how it goes back in --
        called once, at resume, before any turn runs.

        Validated rather than trusted: the node set must be exactly this
        graph's agent_ids (same population, same run), and every agent must
        still have at least k neighbours, which is the invariant
        select_neighbors depends on. A checkpoint from an incompatible run is
        a resume error, not a run that quietly starts in a broken state.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self._agent_ids)))
        for a, b in edges:
            if a not in self._node_of_id or b not in self._node_of_id:
                raise ValueError(
                    f"checkpoint edge {(a, b)!r} names an agent not in this "
                    "run's graph"
                )
            graph.add_edge(self._node_of_id[a], self._node_of_id[b])

        neighbors = {
            self._agent_ids[node]: sorted(
                self._agent_ids[other] for other in graph.neighbors(node)
            )
            for node in graph.nodes
        }
        short = sorted(
            agent_id for agent_id, ns in neighbors.items() if len(ns) < self.k
        )
        if short:
            raise ValueError(
                f"checkpoint graph_edges give {short} fewer than k={self.k} "
                "neighbours; this checkpoint does not match this run"
            )
        self._graph = graph
        self._neighbors = neighbors


@dataclass(frozen=True)
class RecsysFeedInfo:
    """What GraphSimilaritySampling.select_neighbors decided for one reader
    in one round -- logged onto that reader's Interaction row by the
    orchestrator, which reads it back via feed_info() right after calling
    select_neighbors."""

    feed_size: int
    n_neighbors_over_threshold: int | None
    fallback: bool


class GraphSimilaritySampling(GraphSampling):
    """Jo et al.'s belief-similarity recommendation rule (Fix 9), over the
    SAME Watts-Strogatz graph GraphSampling builds -- this subclasses it
    rather than duplicating the construction, so a seed produces byte-
    identical graphs under both strategies and only select_neighbors itself
    differs.

    Where GraphSampling samples k of a reader's graph neighbours at random,
    this instead ranks them by belief similarity and delivers the closest
    ones, within a threshold -- and, critically, does NOT pad a short feed
    back up to k when fewer than k neighbours qualify. Filling it back up
    with the next-closest candidates would make every feed exactly k long
    regardless of the threshold, which would make the threshold invisible
    in the output: "up to k" is what lets a tighter threshold show up as a
    shorter average feed.

    Rewiring (E5) and this rule are two different selection mechanisms --
    one changes who is a neighbour, this changes which neighbours are read
    -- and ExperimentRun.check_graph_similarity_requirements /
    check_rewire_requirements refuse to combine them in one config, so this
    class never has to reconcile a candidate set that rewire() has moved
    mid-round.

    Opinion comes from Agent.current_stance() -- the same field
    _resolve_side reads in the orchestrator -- and NOT from the module-level
    _current_stance() helper above, which AlphaSampling uses: that helper
    special-cases turn == 1 to always return initial_stance, which is right
    for the generated regime (Ohagi's own mechanism, which never probes
    round 0) but wrong here. The bank regime probes every agent at round 0
    and applies that reading before turn 1 ever runs, so by turn 1
    Agent.current_stance() already returns the round-0 probe out of
    stance_history -- which is exactly "the round 0 baseline probe in round
    1" this rule is specified to use, and reading initial_stance instead
    would silently use an unmeasured, configured value in its place.

    current_stance() is on the raw 1-7 probe scale, the same units
    stance_scale and MIDPOINT use -- but recsys_threshold is specified in x
    units ([-1, 1], probe_scorer.to_x's signed scale, where 1.0 is Jo et
    al.'s own 2.0). Every comparison below converts through to_x() before
    measuring a distance, which is what makes a threshold of 1.0 mean "half
    the scale's radius" regardless of which raw digits it is computed from.

    "No valid probe" (step 6 of the design) means stance_history is empty:
    this agent has never been successfully probed, not even at round 0 (Fix
    I never applies a failed round, so a string of failures leaves
    stance_history untouched and current_stance() falls back to
    initial_stance -- a configured value, not a measurement). That is the
    one case this class treats specially, both for the reader (whose
    candidate ranking has nothing real to rank against) and, independently,
    for each candidate (dropped from consideration rather than ranked
    against an unmeasured value).

    Nothing here reads language_condition, or any other condition.
    """

    def __init__(
        self,
        agent_ids: list[str],
        degree: int,
        rewire_p: float,
        k: int,
        graph_seed: int,
        threshold: float,
    ):
        super().__init__(agent_ids, degree, rewire_p, k, graph_seed)
        if not (math.isfinite(threshold) and threshold > 0.0):
            raise ValueError(f"recsys_threshold must be > 0, got {threshold}")
        self.threshold = threshold
        self._feed_info: dict[str, RecsysFeedInfo] = {}

    def _verify_every_node_has_k_neighbours(self, graph_seed: int) -> None:
        """No-op (Fix 1): this class already allows a feed shorter than k --
        select_neighbors below delivers "up to k", never padding back up --
        so a node with fewer than k graph neighbours is not a construction-
        time error here the way it is for plain GraphSampling."""
        return

    def feed_info(self, agent_id: str) -> RecsysFeedInfo:
        """This round's selection diagnostics for one reader, read back by
        the orchestrator right after select_neighbors ran for every agent.

        Raises for an agent select_neighbors has not yet been called for
        this round (or ever) -- there is no sensible default to hand back,
        and a caller reaching this before step 3 has a bug.
        """
        if agent_id not in self._feed_info:
            raise ValueError(
                f"no feed info recorded for {agent_id!r}: select_neighbors "
                "must run for this agent first"
            )
        return self._feed_info[agent_id]

    def select_neighbors(
        self,
        speaker: Agent,
        all_agents: list[Agent],
        turn: int,
        rng: random.Random,
    ) -> list[Agent]:
        """Up to k adjacent neighbours, closest belief first, within
        self.threshold -- see the class docstring for the full rule.

        Fix 11: does NOT consume `rng`, the shared per-round stream every
        other strategy draws from. That stream is one sequence per round,
        shared across every agent processed in it, so how many draws THIS
        reader's own selection makes depends on how many of its candidates
        have a probe reading and how many are eligible -- both a function
        of opinions, which differ by language after round 1. Consuming the
        shared stream therefore left every later agent's tie-break and feed
        order drawing from a different position in four different language
        arms of the same seed, even though the arms are meant to share
        every non-opinion random draw. Instead, every draw here -- the
        fallback sample, the tie-break shuffle, and the final feed-order
        shuffle -- comes from a private stream keyed by this graph's own
        seed, the round, and this reader's own id, so it is unaffected by
        what any other agent in the round did or how many draws its own
        selection took.

        Candidate order: shuffled with the private stream, then stably
        sorted by |x_i - x_j| -- so ties at the same distance land in a
        seeded-random relative order rather than always breaking toward
        whichever agent_id sorts first. The chosen set is then shuffled
        again with the SAME private stream before being returned, so a
        post's position in the feed is not itself a clue to how similar it
        was.
        """
        reader_rng = random.Random(f"{self.graph_seed}|recsys|{turn}|{speaker.agent_id}")
        by_id = {a.agent_id: a for a in all_agents}
        candidate_ids = self.neighbors_of(speaker.agent_id)

        if not speaker.stance_history:
            # No valid probe at all: rank against nothing, fall back to up to
            # k random graph neighbours -- the same thing GraphSampling
            # itself would deliver under allow_short_feed (Fix 1: a reader
            # with fewer than k graph neighbours cannot sample k of them).
            chosen_ids = reader_rng.sample(candidate_ids, min(self.k, len(candidate_ids)))
            self._feed_info[speaker.agent_id] = RecsysFeedInfo(
                feed_size=len(chosen_ids),
                n_neighbors_over_threshold=None,
                fallback=True,
            )
            return [by_id[agent_id] for agent_id in chosen_ids]

        x_i = to_x(speaker.current_stance())
        with_probe = [
            agent_id for agent_id in candidate_ids if by_id[agent_id].stance_history
        ]
        gap = {
            agent_id: abs(x_i - to_x(by_id[agent_id].current_stance()))
            for agent_id in with_probe
        }
        n_over_threshold = sum(1 for d in gap.values() if d > self.threshold)
        eligible = [agent_id for agent_id in with_probe if gap[agent_id] <= self.threshold]

        reader_rng.shuffle(eligible)
        eligible.sort(key=lambda agent_id: gap[agent_id])
        chosen_ids = eligible[: self.k]
        reader_rng.shuffle(chosen_ids)

        self._feed_info[speaker.agent_id] = RecsysFeedInfo(
            feed_size=len(chosen_ids),
            n_neighbors_over_threshold=n_over_threshold,
            fallback=False,
        )
        return [by_id[agent_id] for agent_id in chosen_ids]
