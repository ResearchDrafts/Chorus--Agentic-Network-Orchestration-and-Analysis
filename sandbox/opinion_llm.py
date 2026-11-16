"""Prompt building and response parsing for the opinion-CSV experiment.

Unlike the bank regime, this pipeline deliberately shows each agent its own
belief as a number, every round, including round 0's assigned starting
point -- the per-round numeric self-report (BELIEF:/CONFIDENCE:) IS the
measurement mechanism here, not a probe read privately from logprobs, so
there is no "a number must never reach the agent" invariant to protect.
That invariant belongs to the bank regime's probe design and does not carry
over.

parse_opinion_response is modeled directly on stance_parser.parse_stance:
a compiled regex tag, an exception on anything unparsable or out of range,
never a clamp.
"""

from __future__ import annotations

import re

from sandbox.opinion_models import OpinionAgentState
from sandbox.prompt_builder import LANGUAGE_DIRECTIVES

_BELIEF_PATTERN = re.compile(r"BELIEF:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_CONFIDENCE_PATTERN = re.compile(r"CONFIDENCE:\s*(\d+)", re.IGNORECASE)

_RESPONSE_FORMAT_INSTRUCTIONS = (
    "First, on their own lines, report:\n"
    "BELIEF: <a number from -2 to 2, where -2 means strongly against and "
    "2 means strongly in favor>\n"
    "CONFIDENCE: <a whole number from 1 to 5, where 1 means you barely "
    "hold this view and 5 means you are completely certain of it>\n"
    "Then write your opinion as a short paragraph explaining your reasoning.\n"
    # Putting BELIEF:/CONFIDENCE: first, not last, is deliberate: a response
    # cut off by max_reply_tokens is always truncated from the end, and a
    # verbose free-text paragraph (worse in languages that tokenize less
    # efficiently than English, e.g. Hindi) can consume the whole budget
    # before ever reaching trailing tags. Tags first means truncation only
    # ever costs the reasoning text, never the measurement.
)


class OpinionParseFailure(Exception):
    """Raised when BELIEF:/CONFIDENCE: is missing or out of range. Caught by
    the orchestrator, never by callers of parse_opinion_response directly."""


def _belief_label(belief: float) -> str:
    """Words, not just the number, the same way the bank regime's lean
    sentence puts a probe-scale position into words -- the number alone is
    given too (see module docstring), but a qualitative anchor helps the
    model reason about what the number means rather than just echoing it."""
    if belief <= -1.5:
        return "strongly against"
    if belief <= -0.5:
        return "leaning against"
    if belief < 0.5:
        return "undecided / neutral"
    if belief < 1.5:
        return "leaning in favor"
    return "strongly in favor"


def build_initial_prompt(
    topic: str, persona_text: str, initial_belief: float, language: str
) -> str:
    belief_label = _belief_label(initial_belief)
    return (
        f"Topic for discussion: {topic}\n\n"
        f"Who you are: {persona_text}\n\n"
        f"Your starting belief about this topic is {initial_belief:.2f} on a "
        f"scale from -2 (strongly against) to +2 (strongly in favor) -- "
        f"that is, you are {belief_label}.\n\n"
        f"Write your opening opinion on this topic, consistent with that "
        f"belief and with who you are.\n\n"
        f"{_RESPONSE_FORMAT_INSTRUCTIONS}\n\n"
        f"{LANGUAGE_DIRECTIVES[language]}"
    )


def build_update_prompt(
    topic: str,
    persona_text: str,
    own_state: OpinionAgentState,
    neighbours: list[OpinionAgentState],
    language: str,
) -> str:
    own_label = _belief_label(own_state.belief)
    neighbour_lines = "\n\n".join(
        f"- {n.opinion_text}\n  (their belief: {n.belief:.2f}, {_belief_label(n.belief)})"
        for n in neighbours
    )
    return (
        f"Topic for discussion: {topic}\n\n"
        f"Who you are: {persona_text}\n\n"
        f"Your current opinion: {own_state.opinion_text}\n"
        f"Your current belief: {own_state.belief:.2f} ({own_label}), "
        f"confidence: {own_state.confidence}/5.\n\n"
        f"Here is what some people you follow have been saying:\n\n"
        f"{neighbour_lines}\n\n"
        f"Write your updated opinion, after considering what you just read. "
        f"You may keep your belief and confidence the same, or change them.\n\n"
        f"{_RESPONSE_FORMAT_INSTRUCTIONS}\n\n"
        f"{LANGUAGE_DIRECTIVES[language]}"
    )


def parse_opinion_response(raw_text: str) -> tuple[str, float, int]:
    """Returns (opinion_text, belief, confidence).

    opinion_text is raw_text with every BELIEF:/CONFIDENCE: occurrence
    stripped and surrounding whitespace trimmed, the same convention
    parse_stance uses for reason_text.
    """
    belief_match = _BELIEF_PATTERN.search(raw_text)
    if belief_match is None:
        raise OpinionParseFailure(f"no BELIEF line found in output: {raw_text[:200]!r}")
    confidence_match = _CONFIDENCE_PATTERN.search(raw_text)
    if confidence_match is None:
        raise OpinionParseFailure(f"no CONFIDENCE line found in output: {raw_text[:200]!r}")

    try:
        belief = float(belief_match.group(1))
    except ValueError as e:
        raise OpinionParseFailure(f"BELIEF value not numeric: {belief_match.group(1)!r}") from e
    if not (-2.0 <= belief <= 2.0):
        raise OpinionParseFailure(f"BELIEF {belief} outside [-2, 2]")

    confidence = int(confidence_match.group(1))
    if not (1 <= confidence <= 5):
        raise OpinionParseFailure(f"CONFIDENCE {confidence} outside [1, 5]")

    opinion_text = _BELIEF_PATTERN.sub("", raw_text)
    opinion_text = _CONFIDENCE_PATTERN.sub("", opinion_text).strip()
    return opinion_text, belief, confidence
