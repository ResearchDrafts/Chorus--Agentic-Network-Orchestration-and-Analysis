"""Reading a stance out of next-token logprobs (RQ1, bank regime).

The generated regime asks an agent to state its position as a number, which
has two problems RQ1 cannot tolerate. The agent sees its own previous number
in its prompt and anchors on it, so a measurement becomes part of the
stimulus; and the number it writes is a token it chose for discursive reasons,
not a calibrated read of where it stands.

So stance is measured by a separate call instead. A fresh context holds the
agent's persona and recent history plus a 7-point survey question, the model
is asked for one token at temperature 0, and the distribution over the digits
1 to 7 is what gets recorded. The probe never enters the agent's history, so
nothing it produces can anchor a later turn.

This module is the pure part of that: the arithmetic from a provider's
top-logprobs list to (mean, dist, mass, side_conf). No I/O, no provider
knowledge, no retry logic -- the gateway owns the call, the orchestrator owns
what to do with a failure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# The 7-point Likert scale the probe question is written against. Fixed rather
# than configurable: the question text, the anchor labels and this support all
# have to agree, and a scale parameter would let them drift apart silently.
SCALE_POINTS = (1, 2, 3, 4, 5, 6, 7)
MIDPOINT = 4


class ProbeMassTooLow(Exception):
    """Too little of the next-token distribution landed on the seven digits.

    Carries the observed mass so the caller can log how far short it fell.

    This is a refusal to measure, not a measurement failure to paper over. If
    most of the probability is on prose, a refusal, or a number outside the
    scale, the model is answering some other question, and renormalizing the
    remainder would manufacture a confident stance out of whatever scraps
    happened to be in range. Fix I's rule -- never fabricate a stance --
    applies here more sharply than anywhere else, because the fabricated value
    would look exactly like a real one in the output.
    """

    def __init__(self, mass: float, mass_floor: float):
        self.mass = mass
        self.mass_floor = mass_floor
        super().__init__(
            f"only {mass:.4f} of next-token probability fell on the digits 1-7, "
            f"below the floor of {mass_floor:.4f}"
        )


@dataclass(frozen=True)
class ProbeScore:
    """One probe read.

    mean      s-bar, the expected scale point under dist. The stance value.
    dist      the seven probabilities, renormalized to sum to 1.
    mass      how much of the RAW next-token distribution was on the digits,
              before renormalizing. A quality measure, not a stance: a mean of
              5.0 read off 0.95 mass and the same mean read off 0.51 are not
              equally trustworthy, and only this column distinguishes them.
    side_conf how concentrated the distribution is on the side the mean falls
              on -- how decided the agent is, as distinct from where it stands.
    """

    mean: float
    dist: list[float]
    mass: float
    side_conf: float


def score(top_logprobs: list[tuple[str, float]], mass_floor: float) -> ProbeScore:
    """Turns a provider's top-logprobs list into one stance read.

    top_logprobs is [(token_text, logprob), ...]. Tokens whose stripped text
    is exactly one of "1".."7" count; everything else is ignored but still
    counted against the mass floor, since it is probability the model spent
    not answering.

    Tokenizers differ on leading whitespace -- the same answer arrives as "3"
    from one and " 3" from another -- so both are read as 3 and their
    probabilities summed. Without that, half the providers would appear to put
    no mass on the scale at all and every probe would fail the floor.
    """
    if not (0.0 < mass_floor <= 1.0):
        raise ValueError(f"mass_floor must be in (0, 1], got {mass_floor}")
    if not top_logprobs:
        # Distinct from "the mass was too low": there was nothing to read at
        # all, which is a gateway or wiring problem rather than a model that
        # answered off-scale.
        raise ValueError("top_logprobs is empty: nothing to score")

    # Probability per scale point, summing duplicates (" 3" and "3" are one
    # answer), plus the total probability seen across every token.
    on_scale = {point: 0.0 for point in SCALE_POINTS}
    total_seen = 0.0
    for token, logprob in top_logprobs:
        probability = math.exp(logprob)
        total_seen += probability
        stripped = token.strip()
        if len(stripped) == 1 and stripped.isdigit():
            point = int(stripped)
            if point in on_scale:
                on_scale[point] += probability

    mass = sum(on_scale.values())
    if mass < mass_floor:
        raise ProbeMassTooLow(mass, mass_floor)
    # Unreachable while mass >= mass_floor > 0, since mass is a sum of
    # non-negative terms, but the division below is the one place a zero would
    # become a NaN stance rather than an error.
    if mass <= 0.0:
        raise ProbeMassTooLow(mass, mass_floor)

    dist = [on_scale[point] / mass for point in SCALE_POINTS]
    mean = sum(point * probability for point, probability in zip(SCALE_POINTS, dist, strict=True))

    return ProbeScore(
        mean=mean,
        dist=dist,
        mass=mass,
        side_conf=_side_confidence(mean, dist),
        # total_seen is deliberately not returned. It is close to 1 when the
        # provider sent its whole top-k and far below 1 when it sent a slice,
        # so it measures the request rather than the answer; `mass` is the
        # share of the distribution that addressed the question, which is what
        # the floor is about and what analysis needs.
    )


@dataclass(frozen=True)
class ChoiceScore:
    """One forced-choice read: a distribution over a short list of answers.

    The comprehension check's counterpart to ProbeScore. There is no mean and
    no side here -- the answers are letters, not scale points, so the only
    questions are which one the model picked and how much of its probability
    went to answering at all.

    choices   the answers scored, in the order `dist` reports them
    dist      their probabilities, renormalized to sum to 1
    mass      how much of the RAW next-token distribution landed on them,
              before renormalizing -- the same quality measure ProbeScore.mass
              is, and what the floor is checked against
    """

    choices: tuple[str, ...]
    dist: list[float]
    mass: float

    def probability(self, choice: str) -> float:
        """The renormalized probability of one choice.

        Raises for a choice that was not scored, rather than returning 0: a
        caller asking about "C" when the prompt offered "A" and "B" has a bug,
        and a 0 would read as "the model never said C", which is a different
        and much more interesting claim.
        """
        try:
            index = self.choices.index(choice)
        except ValueError:
            raise KeyError(
                f"{choice!r} was not one of the scored choices {list(self.choices)}"
            ) from None
        return self.dist[index]


def score_choices(
    top_logprobs: list[tuple[str, float]],
    choices: Sequence[str],
    mass_floor: float,
) -> ChoiceScore:
    """Aggregates next-token probability over a short list of literal answers.

    Same contract as `score`, with a different support. Tokens whose stripped
    text equals one of `choices` count; everything else is ignored but still
    counts against the floor, since it is probability the model spent not
    answering. Duplicates are summed, so a tokenizer that emits " A" and one
    that emits "A" are read the same way.

    **The match is case-sensitive, and that is deliberate.** The prompt asks
    for a capital letter, and folding case would also fold in the "a" that
    begins an English sentence -- so a model answering "a post like this
    supports..." would be scored as having chosen A with high confidence. That
    is precisely the off-support probability the mass floor exists to catch,
    and case folding would launder it into an answer.

    Raises ProbeMassTooLow below the floor, for `score`'s reason: renormalizing
    whatever scraps happened to be in range would manufacture a confident
    answer out of a model that was answering some other question, and the
    fabricated value would look exactly like a real one in the output.
    """
    if not (0.0 < mass_floor <= 1.0):
        raise ValueError(f"mass_floor must be in (0, 1], got {mass_floor}")
    if len(choices) < 2:
        raise ValueError(f"need at least 2 choices to score, got {list(choices)}")
    if len(set(choices)) != len(choices):
        raise ValueError(f"choices must be distinct, got {list(choices)}")
    if not top_logprobs:
        # Distinct from "the mass was too low", exactly as in `score`: there
        # was nothing to read at all, which is a gateway or wiring problem
        # rather than a model that answered off-support.
        raise ValueError("top_logprobs is empty: nothing to score")

    on_support = {choice: 0.0 for choice in choices}
    for token, logprob in top_logprobs:
        stripped = token.strip()
        if stripped in on_support:
            on_support[stripped] += math.exp(logprob)

    mass = sum(on_support.values())
    if mass < mass_floor or mass <= 0.0:
        raise ProbeMassTooLow(mass, mass_floor)

    return ChoiceScore(
        choices=tuple(choices),
        dist=[on_support[choice] / mass for choice in choices],
        mass=mass,
    )


def _side_confidence(mean: float, dist: list[float]) -> float:
    """How concentrated the read is on the side the mean falls on.

    mean says where the agent stands; this says how decided it is. The two
    come apart in the case that matters: a distribution split hard between 1
    and 7 and one piled entirely on 4 both have a mean of 4, and only this
    tells them apart.

    Computed from the renormalized dist, so it is already a share of the
    on-scale mass rather than of the raw distribution -- dividing a raw
    tail probability by the raw mass gives the same number.

    At exactly the midpoint there is no side to be confident about, so the
    value becomes P(s = 4): how much of the read is genuinely "undecided"
    rather than two opposite convictions averaging out.
    """
    above = sum(p for point, p in zip(SCALE_POINTS, dist, strict=True) if point > MIDPOINT)
    below = sum(p for point, p in zip(SCALE_POINTS, dist, strict=True) if point < MIDPOINT)
    at = dist[SCALE_POINTS.index(MIDPOINT)]

    if mean > MIDPOINT:
        return above
    if mean < MIDPOINT:
        return below
    return at


def to_x(mean: float) -> float:
    """The probe mean on the signed scale every RQ1 model is specified in.

    -1 is the con pole, 0 undecided, +1 the pro pole:

        x = (mean - 4) / 3

    One definition, in one place. It was computed inline in export_csv.py and
    would have been recomputed in the calibration script, the E1 script and
    three analysis modules -- six chances to get the divisor or the sign
    wrong, and a sign error here does not raise, it reverses every reported
    effect. Importing it is cheap; rederiving it is not.

    Takes the raw mean rather than a ProbeScore so the analysis tier, which
    reads a float out of a CSV column and has no ProbeScore to hand, uses the
    same function the live scripts do.
    """
    return (mean - MIDPOINT) / (max(SCALE_POINTS) - MIDPOINT)
