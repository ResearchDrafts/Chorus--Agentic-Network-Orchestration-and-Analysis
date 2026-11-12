"""Feature 2: Seed Management.

Provides three independently-seeded random.Random instances (neighbor
sampling, meme injection, persona assignment), deterministically offset
from one top-level seed, per Fix M. Reusing the same top-level seed across
two ExperimentRun configs (e.g. an English-condition run and a Hinglish-
condition run in a combined sweep) reproduces identical meme-injection and
neighbor-sampling draw sequences across both runs.

Four derived families sit alongside those: turn_rng (per purpose and
turn), sender_rng (per purpose and sender, with no turn in it), graph_seed
(per run) and call_seed (per purpose, turn and agent, handed to the provider
rather than drawn from here). None of them takes language, topic, trial or
any other condition as input. That omission is the point: it is what makes
two runs differing only in language draw identically, and it is load-bearing for
RQ1's matched sets, where the same argument must reach the same reader in
four languages.

This feature makes no claim about LLM sampling determinism -- provider
APIs do not guarantee deterministic output even at temperature=0, and
call_seed asks a server to be reproducible rather than making it so.
"""

from __future__ import annotations

import random


class SeedManager:
    """One instance per run. Each property returns the SAME long-lived
    random.Random instance on every call -- not a fresh one -- since each
    stream advances as consuming features draw from it.

    Note (Question B2, per the plan's Phase 6 checkpoint): neighbor-sampling
    and meme-injection resolution are assumed to happen strictly sequentially
    within a turn (never concurrently with each other), so these
    random.Random instances are never accessed concurrently. If a future
    change ever parallelizes those two resolution steps against each other,
    this assumption breaks and per-call locking would be needed here.
    """

    def __init__(self, seed: int):
        self._base_seed = seed
        self._neighbor_rng = random.Random(seed)
        self._meme_rng = random.Random(seed + 1)
        self._persona_rng = random.Random(seed + 2)

    @property
    def neighbor_sampling_rng(self) -> random.Random:
        return self._neighbor_rng

    @property
    def meme_injection_rng(self) -> random.Random:
        return self._meme_rng

    @property
    def persona_assignment_rng(self) -> random.Random:
        return self._persona_rng

    @property
    def graph_seed(self) -> int:
        """The seed for the run's Watts-Strogatz social graph.

        A plain int, not a Random: networkx takes an integer seed and builds
        its own generator, and the graph is constructed exactly once per run,
        so there is no stream to advance and nothing to checkpoint.

        Offset far from every other stream for the same reason the three
        long-lived offsets are spaced: a graph built from the same integer a
        neighbour draw consumes would correlate structure with sampling, and
        the correlation would be invisible in the output.

        Variety is deliberately absent from the derivation -- it is a function
        of the base seed alone. Two runs differing only in language_condition
        must therefore share a graph, which is what makes "the same agent read
        the same neighbours in both arms" true rather than approximately true.
        """
        return self._base_seed + 9_000_003

    # Purpose offsets for turn_rng, kept distinct from the three long-lived
    # stream offsets above, from each other, and from graph_seed, so no
    # derived stream can collide with another.
    #
    # "argument" resolves which bank item a sender posts, and breaks the tie
    # when an agent's probe lands exactly at the scale midpoint and has no
    # side. "probe" is reserved for any stochastic choice inside the probe
    # path; the probe itself runs at temperature 0 and consumes nothing today,
    # and the purpose exists so that adding one later cannot be done by
    # borrowing another purpose's stream.
    #
    # "reply" seeds the provider's own sampler for a bank-regime reply call.
    # It is not drawn from here as a stream -- call_seed derives an integer
    # the server is handed -- but it lives in this table so a reply seed
    # cannot collide with any other purpose's derivation.
    _TURN_PURPOSES = {
        "neighbor": 1_000_003,
        "meme": 2_000_003,
        "argument": 3_000_003,
        "probe": 4_000_003,
        "reply": 5_000_003,
    }

    def turn_rng(self, purpose: str, turn: int) -> random.Random:
        """A fresh stream for one (purpose, turn), derived from the base seed.

        The long-lived streams above cannot survive a resume. Their state
        advances as turns consume them, but CheckpointState stores only agent
        state, so a resumed run rebuilds them at their *initial* position and
        replays earlier turns' draws: resuming at turn 2 gives turn 2 the
        numbers turn 1 already used. The run still completes and is still
        internally deterministic, which is precisely why this was invisible
        until a test compared a resumed run against an uninterrupted one
        (phase4.md D8).

        Deriving per turn makes each turn's draws a pure function of
        (seed, purpose, turn), so a resumed turn is identical to the same turn
        in a run that never stopped, and no state has to be serialized.

        Fix M is preserved: two runs sharing a seed still draw identically,
        which is what holds neighbour sampling and meme injection constant
        while language varies.
        """
        if purpose not in self._TURN_PURPOSES:
            raise ValueError(f"unknown rng purpose: {purpose!r}")
        if turn < 1:
            raise ValueError(f"turn must be >= 1, got {turn}")
        return random.Random(self._base_seed + self._TURN_PURPOSES[purpose] * turn)

    def sender_rng(self, purpose: str, sender_id: str) -> random.Random:
        """A stream for one (purpose, sender), with no turn in the derivation.

        This exists for the argument bank. A sender must not repeat an item on
        a side until it has exhausted that side, which means the draw has to
        be a function of how many times that sender has posted on that side --
        not of which turn it happens to be posting on. Deriving per turn, as
        turn_rng does, would re-shuffle the sender's order every round and the
        no-repeat guarantee would be gone.

        Keyed by a string rather than arithmetic on the seed: an agent id is
        not a number, and `base_seed + hash(agent_id)` is neither stable
        across processes (PYTHONHASHSEED) nor collision-free. random.Random
        seeds a string through SHA-512, which is both.

        As with every other derivation here, language is absent: two runs
        sharing a seed draw the same items for the same sender on the same
        side, which is precisely what lets a matched set post the same
        argument in four languages.
        """
        if purpose not in self._TURN_PURPOSES:
            raise ValueError(f"unknown rng purpose: {purpose!r}")
        if not sender_id:
            raise ValueError("sender_id must be a non-empty string")
        return random.Random(f"{self._base_seed}|{purpose}|{sender_id}")

    def call_seed(self, purpose: str, turn: int, agent_id: str) -> int:
        """An integer seed to hand the provider for one (purpose, turn, agent).

        Not a stream. The other derivations here produce a random.Random this
        process draws from; this produces a number that leaves the process and
        is given to the server, so that the server's own sampler is a function
        of the run's seed rather than of whatever it felt like that afternoon.
        Without it a bank reply is sampled afresh on every rerun, and since the
        probe reads the stance out of a history made of those replies, a rerun
        of the same config produced different readings and the run was
        reproducible in its draws and in nothing else.

        Keyed by a string, like sender_rng and for the same reason: an agent id
        is not a number, and arithmetic on hash(agent_id) is neither stable
        across processes nor collision-free, while random.Random seeds a string
        through SHA-512.

        31 bits, so the result is a non-negative int32. That is the range every
        OpenAI-compatible server accepts; a wider integer is rejected by some
        and silently truncated by others, which would make the seed a function
        of the provider as well as of the run.

        Language, topic and trial are absent, as everywhere else in this class.
        Two runs differing only in language_condition therefore send the same
        seed for the same (turn, agent), so the four arms of a matched set
        differ in the text of the prompt and not in how it was sampled.
        """
        if purpose not in self._TURN_PURPOSES:
            raise ValueError(f"unknown rng purpose: {purpose!r}")
        # turn >= 0, not >= 1 as turn_rng requires: round 0 is a real turn
        # number in the bank regime, and a caller asking for its seed should
        # get one rather than an error about a round that exists.
        if turn < 0:
            raise ValueError(f"turn must be >= 0, got {turn}")
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        return random.Random(
            f"{self._base_seed}|{purpose}|{turn}|{agent_id}"
        ).getrandbits(31)
