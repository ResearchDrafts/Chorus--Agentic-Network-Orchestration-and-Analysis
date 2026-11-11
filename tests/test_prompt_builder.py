"""Tests for prompt_builder.py.

Organised around phase4.md Section 1's numbered decisions. The RQ1 parity
test and the memory-cap test are the two that protect research validity
rather than merely asserting behaviour.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from sandbox.meme_pool_manager import MemePoolManager
from sandbox.models import MemeInjectionConfig, StanceRecord
from sandbox.prompt_builder import (
    LANGUAGE_DIRECTIVES,
    MEMORY_WINDOW_TURNS,
    PromptBuilder,
)
from sandbox.stance_parser import clamp_and_validate_scale, parse_stance
from tests.factories import STANCE_SCALE, make_agent, make_interaction, make_run


@pytest.fixture
def meme_lookup():
    pool = MemePoolManager(
        MemeInjectionConfig(enabled=True, meme_pool_id="test_pool", injection_rate=1.0),
        random.Random(0),
    )
    return pool.get_meme


def _builder(meme_lookup, *, supports_vision=True, **run_overrides) -> PromptBuilder:
    return PromptBuilder(
        make_run(**run_overrides), supports_vision=supports_vision, meme_lookup=meme_lookup
    )


def _history(n: int) -> list[StanceRecord]:
    return [
        StanceRecord(
            turn=i + 1, stance_value=4.0, reason_text=f"reason {i}", interaction_id=f"i{i}"
        )
        for i in range(n)
    ]


def _text_post(agent_id: str, stance: float = 5.0, reason: str = "a neighbour view"):
    return make_interaction(agent_id, turn=2, stance_after=stance, reason_text=reason)


def _meme_post(agent_id: str, meme_id: str = "meme_test_001", stance: float = 6.5):
    return make_interaction(
        agent_id, turn=2, stance_after=stance, content_type="meme", meme_id=meme_id
    )


# --- Q4.2: language is the only thing that varies (the RQ1 guard) --------


def test_all_three_language_conditions_render(meme_lookup):
    for condition in LANGUAGE_DIRECTIVES:
        built = _builder(meme_lookup, language_condition=condition).build_discussion_prompt(
            make_agent(), [], turn=1
        )
        assert LANGUAGE_DIRECTIVES[condition] in built.text


def test_prompts_differ_across_languages_only_in_the_directive_line(meme_lookup):
    """The RQ1 validity guard.

    Fix A counts memory in turns rather than tokens so no language silently
    receives less context. That protection is worthless if the prompt itself
    differs between conditions, so every line except the directive must be
    byte-identical.
    """
    agent = make_agent(stance_history=_history(3))
    posts = [_text_post("agent_0011"), _text_post("agent_0077", stance=2.0)]

    rendered = {
        condition: _builder(meme_lookup, language_condition=condition)
        .build_discussion_prompt(agent, posts, turn=3)
        .text
        for condition in LANGUAGE_DIRECTIVES
    }

    stripped = {
        condition: [
            line for line in text.split("\n") if line != LANGUAGE_DIRECTIVES[condition]
        ]
        for condition, text in rendered.items()
    }
    baseline = stripped["english"]
    for condition, lines in stripped.items():
        assert lines == baseline, f"{condition} differs from english beyond its directive"


def test_language_directive_length_spread_stays_bounded():
    """The residual asymmetry, measured rather than assumed.

    "Hinglish" cannot be named in as few characters as "English", so exact
    parity is impossible and the prompts do differ in length by a fixed
    amount (42 characters at time of writing). This bounds that constant. If
    it fails, someone has grown a directive into a paragraph and turned a
    rounding error into a real length confound between RQ1's conditions.
    """
    lengths = {c: len(d) for c, d in LANGUAGE_DIRECTIVES.items()}
    spread = max(lengths.values()) - min(lengths.values())
    assert spread <= 60, lengths


def test_language_length_difference_is_negligible_on_a_realistic_prompt(meme_lookup):
    """The same 42 characters, expressed as a share of a full-size prompt.

    Deliberately built at the scale a real run produces (a full 5-turn memory
    window and N=5 neighbour posts) rather than a bare minimum, because the
    fixed offset looks alarming against a toy prompt and negligible against a
    real one, and it is the real one that determines whether RQ1's comparison
    is clean.
    """
    agent = make_agent(stance_history=_history(MEMORY_WINDOW_TURNS))
    posts = [
        _text_post(f"agent_{i:04d}", reason="A reasonably typical neighbour post.")
        for i in range(5)
    ]
    lengths = {
        condition: len(
            _builder(
                meme_lookup,
                language_condition=condition,
                topic="whether remote work should be the default",
            )
            .build_discussion_prompt(agent, posts, turn=6)
            .text
        )
        for condition in LANGUAGE_DIRECTIVES
    }
    spread = max(lengths.values()) - min(lengths.values())
    assert spread / min(lengths.values()) < 0.05, lengths


def test_unknown_language_condition_raises_rather_than_defaulting(meme_lookup):
    run = make_run()
    object.__setattr__(run, "__dict__", {**run.__dict__, "language_condition": "french"})
    with pytest.raises(ValueError, match="no language directive"):
        PromptBuilder(run, supports_vision=True, meme_lookup=meme_lookup)


# --- Q4.5: the output contract stance_parser depends on ------------------


def test_a_reply_following_the_template_round_trips_through_the_parser(meme_lookup):
    """Closes the loop: the format this prompt asks for is the format the
    parser accepts. Asserting the prompt's wording alone would not catch a
    drift between the two."""
    built = _builder(meme_lookup).build_discussion_prompt(make_agent(), [], turn=1)
    assert "STANCE: <a number from 1 to 7>" in built.text

    reply = "STANCE: 6\nThe commuting argument moved me."
    value, reason = parse_stance(reply)
    assert clamp_and_validate_scale(value, STANCE_SCALE) == 6.0
    assert reason == "The commuting argument moved me."


def test_prompt_instructs_the_stance_number_to_appear_only_once(meme_lookup):
    """parse_stance strips EVERY STANCE occurrence from reason_text, so a
    restated stance silently vanishes from RQ2's input."""
    built = _builder(meme_lookup).build_discussion_prompt(make_agent(), [], turn=1)
    assert "that line only" in built.text


def test_scale_bounds_come_from_the_configured_stance_scale(meme_lookup):
    built = _builder(meme_lookup, stance_scale=[0, 1, 2, 3]).build_discussion_prompt(
        make_agent(), [], turn=1
    )
    assert "from 0 to 3" in built.text


def test_stance_anchor_labels_are_rendered(meme_lookup):
    built = _builder(
        meme_lookup,
        stance_low_label="remote work should never be default",
        stance_high_label="remote work should always be default",
    ).build_discussion_prompt(make_agent(), [], turn=1)
    assert "remote work should never be default" in built.text
    assert "remote work should always be default" in built.text


# --- Q4.3: Fix A's memory window -----------------------------------------


def test_memory_renders_at_most_five_turns(meme_lookup):
    agent = make_agent(stance_history=_history(12))
    built = _builder(meme_lookup).build_discussion_prompt(agent, [], turn=13)
    assert built.text.count("  Turn ") == MEMORY_WINDOW_TURNS


def test_memory_slice_holds_even_when_the_cap_was_bypassed(meme_lookup):
    """Fix A's last line of defence.

    validate_assignment cannot see an in-place append (phase4.md D7), so an
    agent can carry more history than the cap allows. The prompt is what
    reaches the model, so the cap must hold here regardless.
    """
    agent = make_agent(stance_history=_history(3))
    for extra in _history(20):
        agent.stance_history.append(extra)
    assert len(agent.stance_history) > MEMORY_WINDOW_TURNS

    built = _builder(meme_lookup).build_discussion_prompt(agent, [], turn=24)
    assert built.text.count("  Turn ") == MEMORY_WINDOW_TURNS


def test_memory_renders_fewer_entries_when_history_is_short(meme_lookup):
    agent = make_agent(stance_history=_history(2))
    built = _builder(meme_lookup).build_discussion_prompt(agent, [], turn=3)
    assert built.text.count("  Turn ") == 2


def test_memory_section_omitted_entirely_when_history_is_empty(meme_lookup):
    built = _builder(meme_lookup).build_discussion_prompt(make_agent(), [], turn=1)
    assert "What you said recently" not in built.text


# --- Q4.9: turn 1 ---------------------------------------------------------


def test_turn_one_omits_the_neighbour_block(meme_lookup):
    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(), [_text_post("agent_0011")], turn=1
    )
    assert "What others in your feed" not in built.text


def test_later_turns_render_neighbour_posts(meme_lookup):
    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(), [_text_post("agent_0011", reason="commuting is wasted life")], turn=2
    )
    assert "What others in your feed" in built.text
    assert "commuting is wasted life" in built.text


# --- Q4.7 / Q4.8: memes, vision, and the single image slot ---------------


def test_meme_neighbour_on_a_vision_backend_attaches_the_image(meme_lookup):
    built = _builder(meme_lookup, supports_vision=True).build_discussion_prompt(
        make_agent(), [_meme_post("agent_0077")], turn=2
    )
    assert built.image is not None
    assert built.used_vision_fallback is False
    assert "When will people learn??" in built.text


def test_meme_neighbour_on_a_text_only_backend_falls_back_to_caption(meme_lookup):
    built = _builder(meme_lookup, supports_vision=False).build_discussion_prompt(
        make_agent(), [_meme_post("agent_0077")], turn=2
    )
    assert built.image is None
    assert built.used_vision_fallback is True
    assert "When will people learn??" in built.text


def test_text_only_backend_never_reads_an_image_from_disk(meme_lookup):
    """vision_fallback documents that the filesystem is untouched on the
    caption branch. A text-only run must not depend on meme images existing."""
    with patch("sandbox.vision_fallback._load_image", side_effect=AssertionError("read disk")):
        built = _builder(meme_lookup, supports_vision=False).build_discussion_prompt(
            make_agent(), [_meme_post("agent_0077")], turn=2
        )
    assert built.image is None


def test_two_meme_neighbours_attach_exactly_one_image_and_flag_the_downgrade(meme_lookup):
    """ModelGateway.generate takes a single image, so the second meme degrades
    to its caption even on a vision backend. used_vision_fallback reports
    that, per Q4.8's widened definition."""
    built = _builder(meme_lookup, supports_vision=True).build_discussion_prompt(
        make_agent(),
        [_meme_post("agent_0011", "meme_test_001"), _meme_post("agent_0077", "meme_test_002")],
        turn=2,
    )
    assert built.image is not None
    assert built.used_vision_fallback is True
    assert "When will people learn??" in built.text
    assert "This is fine." in built.text


def test_no_memes_means_no_image_and_no_fallback(meme_lookup):
    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(), [_text_post("agent_0011")], turn=2
    )
    assert built.image is None
    assert built.used_vision_fallback is False


def test_meme_framing_comes_from_vision_fallback_not_a_local_copy(meme_lookup):
    """vision_fallback claims to be the only place constructing meme-derived
    prompts. If prompt_builder grew its own copy, this drifts silently."""
    from sandbox.vision_fallback import _INSTRUCTION

    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(), [_meme_post("agent_0077")], turn=2
    )
    assert _INSTRUCTION.split(".")[0] in built.text


def test_unknown_meme_id_raises(meme_lookup):
    with pytest.raises(KeyError):
        _builder(meme_lookup).build_discussion_prompt(
            make_agent(), [_meme_post("agent_0077", "meme_does_not_exist")], turn=2
        )


# --- Q4.6: emphasize_format ----------------------------------------------


def test_emphasize_format_changes_only_the_format_block(meme_lookup):
    """A retry must be a response to the same stimulus. If emphasize_format
    altered persona, topic, memory or neighbour content, first attempts and
    retries would be two different conditions in one dataset."""
    builder = _builder(meme_lookup)
    agent = make_agent(stance_history=_history(3))
    posts = [_text_post("agent_0011")]

    plain = builder.build_discussion_prompt(agent, posts, turn=3).text
    emphasised = builder.build_discussion_prompt(
        agent, posts, turn=3, emphasize_format=True
    ).text

    assert emphasised != plain
    # Split on the language directive, the last thing before the format
    # block. Everything ahead of it is the experimental stimulus and must be
    # identical; only what follows may change.
    directive = LANGUAGE_DIRECTIVES["english"]
    assert plain.split(directive)[0] == emphasised.split(directive)[0]
    assert "could not be read" in emphasised
    assert "could not be read" not in plain


def test_emphasize_format_still_satisfies_the_parser_contract(meme_lookup):
    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(), [], turn=1, emphasize_format=True
    )
    assert "STANCE: <a number from 1 to 7>" in built.text


# --- purity ---------------------------------------------------------------


def test_builder_does_not_mutate_its_arguments(meme_lookup):
    agent = make_agent(stance_history=_history(7))
    posts = [_text_post("agent_0011"), _meme_post("agent_0077")]
    before_history = list(agent.stance_history)
    before_memory = list(agent.memory_window)
    before_posts = list(posts)

    _builder(meme_lookup).build_discussion_prompt(agent, posts, turn=8)

    assert agent.stance_history == before_history
    assert agent.memory_window == before_memory
    assert posts == before_posts


def test_stance_values_render_without_trailing_zeroes(meme_lookup):
    """4.0 should read as "4"; 6.5 must stay "6.5"."""
    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(initial_stance=4.0), [_text_post("agent_0011", stance=6.5)], turn=2
    )
    assert "position is 4 on a scale" in built.text
    assert "(position 6.5)" in built.text


# =========================================================================
# RQ1: the bank regime
#
# The generated-regime tests above are untouched and must stay passing: the
# two regimes share build_discussion_prompt and the brief requires the old
# path to be unaffected.
# =========================================================================

from sandbox.prompt_builder import LEAN_SENTENCES, RoundMemory  # noqa: E402

BANK_RUN = dict(
    regime="bank",
    bank_path="data/bank/test_bank.csv",
    probe_question="Should the fare subsidy continue?",
    topic="whether the fare subsidy should continue",
    stance_low_label="the subsidy should end",
    stance_high_label="the subsidy should continue",
)

# Digit-free fixtures. The no-number audit cannot tell a leaked position from
# a persona that happens to contain a numeral, so none of these contain one.
PERSONA = "a bus commuter in Pune who reads the news every morning"
FEED = ["Families already spend a large share of income on this.",
        "Someone has to pay for it, and the bill lands on working people."]
HISTORY = [
    RoundMemory(posts_read=["An earlier cost argument."], reply_text="That matches what I see."),
    RoundMemory(posts_read=["A later access argument."], reply_text="I am less sure now."),
]


def _bank_builder(**run_overrides) -> PromptBuilder:
    overrides = {**BANK_RUN, **run_overrides}
    return PromptBuilder(
        make_run(**overrides), supports_vision=True, meme_lookup=lambda meme_id: None
    )


def _bank_agent(initial_stance: float = 6.0, **overrides):
    return make_agent(persona=PERSONA, initial_stance=initial_stance, **overrides)


def _reply(builder, *, turn=2, posts=None, history=None, agent=None) -> str:
    return builder.build_discussion_prompt(
        agent or _bank_agent(),
        [],
        turn,
        delivered_posts=FEED if posts is None else posts,
        history_rounds=HISTORY if history is None else history,
    ).text


def _stimulus(prompt: str, builder: PromptBuilder) -> str:
    """Everything before the language directive: persona, topic, lean, memory,
    feed. The sections the brief requires to be free of position numbers."""
    directive = LANGUAGE_DIRECTIVES[builder._run.language_condition]
    assert directive in prompt
    return prompt.split(directive)[0]


# --- the fourth language condition ---------------------------------------


def test_the_fourth_directive_exists_and_is_distinct():
    assert set(LANGUAGE_DIRECTIVES) == {"english", "hinglish", "romanized_hindi", "hindi"}
    assert len(set(LANGUAGE_DIRECTIVES.values())) == 4


def test_romanized_hindi_names_roman_script_and_rules_out_devanagari():
    """The condition is Hindi in Roman script. A directive that said only
    "Hindi" would be indistinguishable from the Devanagari arm's."""
    directive = LANGUAGE_DIRECTIVES["romanized_hindi"]
    assert "Roman" in directive
    assert "Devanagari" in directive
    assert "Hinglish" not in directive


def test_all_four_conditions_render_a_bank_prompt():
    for condition in LANGUAGE_DIRECTIVES:
        prompt = _reply(_bank_builder(language_condition=condition))
        assert LANGUAGE_DIRECTIVES[condition] in prompt


# --- no numbers reach an agent (the property the design turns on) --------


@pytest.mark.parametrize("turn", [1, 2, 5])
def test_no_digit_appears_in_the_persona_memory_or_feed_sections(turn):
    """The reason the probe exists. An agent that can see a position number
    does not have to read the arguments to answer: it can move its own number
    towards the numbers it was shown, and the result would look exactly like
    persuasion."""
    builder = _bank_builder()
    stimulus = _stimulus(_reply(builder, turn=turn), builder)

    offenders = [ch for ch in stimulus if ch.isdigit()]
    assert not offenders, f"digit(s) {offenders} in:\n{stimulus}"


def test_no_agent_id_appears_in_a_bank_reply_prompt():
    """An id is a stable handle an agent could track across rounds
    ("agent_0012 always says this"), which turns a feed of arguments into a
    feed of sources."""
    prompt = _reply(_bank_builder())
    assert "agent_" not in prompt
    assert "agent_0000" not in prompt


def test_the_bank_prompt_never_states_a_position():
    prompt = _reply(_bank_builder())
    assert "position" not in prompt.lower()
    assert "on a scale" not in prompt.lower()


def test_the_bank_prompt_asks_for_no_stance_line():
    """A bank reply is asked only to be a reply. Nothing parses it for a
    number, so there is no output contract and no retry-on-parse path."""
    prompt = _reply(_bank_builder())
    assert "STANCE" not in prompt
    assert "Reply to these posts in 1 to 2 sentences." in prompt


# --- section order and content -------------------------------------------


def test_the_sections_appear_in_the_specified_order():
    builder = _bank_builder()
    prompt = _reply(builder, turn=1)
    order = [
        "You are: ",
        "ongoing discussion about",
        "You hold the view that",          # the lean, round 1 only
        "Earlier in this discussion:",
        "Posts in your feed:",
        LANGUAGE_DIRECTIVES["english"],
        "Reply to these posts",
    ]
    positions = [prompt.index(marker) for marker in order]
    assert positions == sorted(positions), prompt


def test_the_feed_renders_each_delivered_post_as_a_user_post():
    prompt = _reply(_bank_builder())
    for text in FEED:
        assert f"A user posted: {text}" in prompt
    assert prompt.count("A user posted:") == len(FEED)


def test_the_feed_preserves_delivery_order():
    """Delivery order is part of the stimulus, and the orchestrator derives it
    from a seeded draw, so the builder must not reorder it."""
    posts = ["first argument", "second argument", "third argument"]
    prompt = _reply(_bank_builder(), posts=posts)
    assert [line.split("A user posted: ")[1] for line in prompt.split("\n")
            if "A user posted:" in line] == posts


def test_an_empty_feed_omits_the_section_entirely():
    """A heading with nothing under it is itself a stimulus, and a different
    one from no feed at all. This is the `delivery: off` case."""
    prompt = _reply(_bank_builder(), posts=[])
    assert "Posts in your feed" not in prompt
    assert "Reply to these posts" in prompt  # the instruction still stands


# --- the lean sentence ---------------------------------------------------


def test_the_lean_sentence_appears_only_in_round_one():
    """After round 1 the agent's own replies are its history; repeating the
    lean would keep re-anchoring it to a starting point it may have moved
    away from."""
    builder = _bank_builder()
    assert "You hold the view that" in _reply(builder, turn=1)
    for turn in (2, 3, 10):
        assert "hold the view that" not in _reply(builder, turn=turn)
        assert "undecided" not in _reply(builder, turn=turn)


@pytest.mark.parametrize("stance", [1, 2, 3, 4, 5, 6, 7])
def test_every_scale_point_has_a_lean_sentence_in_words_only(stance):
    builder = _bank_builder()
    prompt = _reply(builder, turn=1, agent=_bank_agent(initial_stance=float(stance)))
    rendered = LEAN_SENTENCES[stance].format(
        low="the subsidy should end", high="the subsidy should continue"
    )
    assert rendered in prompt
    assert not any(ch.isdigit() for ch in rendered)


def test_the_lean_sentence_uses_the_runs_anchor_labels():
    """So the sentence says something about the topic rather than about the
    scale."""
    builder = _bank_builder(
        stance_low_label="fares should rise", stance_high_label="fares should stay"
    )
    prompt = _reply(builder, turn=1, agent=_bank_agent(initial_stance=7.0))
    assert "fares should stay" in prompt


def test_the_two_ends_lean_opposite_ways():
    builder = _bank_builder()
    low = _reply(builder, turn=1, agent=_bank_agent(initial_stance=1.0))
    high = _reply(builder, turn=1, agent=_bank_agent(initial_stance=7.0))
    assert "the subsidy should end" in low.split("discussion about:")[1].split("Posts")[0]
    assert "the subsidy should continue" in high.split("discussion about:")[1].split("Posts")[0]


def test_the_midpoint_reads_as_undecided():
    prompt = _reply(_bank_builder(), turn=1, agent=_bank_agent(initial_stance=4.0))
    assert "undecided" in prompt


def test_an_off_scale_initial_stance_raises_rather_than_rounding():
    """The bank regime is defined against the seven points the probe reads. A
    silent "you are undecided" would start every agent in a misconfigured run
    from the middle without saying so."""
    builder = _bank_builder(stance_scale=[0.0, 1.5, 3.0])
    with pytest.raises(ValueError, match="no lean sentence"):
        _reply(builder, turn=1, agent=_bank_agent(initial_stance=1.5))
    with pytest.raises(ValueError, match="no lean sentence"):
        _reply(builder, turn=1, agent=_bank_agent(initial_stance=0.0))


# --- memory --------------------------------------------------------------


def test_memory_renders_reads_and_replies_oldest_first():
    prompt = _reply(_bank_builder())
    assert prompt.index("An earlier cost argument.") < prompt.index("A later access argument.")
    assert "You read: An earlier cost argument." in prompt
    assert "You replied: That matches what I see." in prompt


def test_memory_is_capped_at_memory_rounds():
    """Fix A's window, in rounds. This is the cap that reaches the model, so
    it is the one that determines results."""
    history = [
        RoundMemory(posts_read=[f"post from round {w}"], reply_text=f"reply {w}")
        for w in ("alpha", "beta", "gamma", "delta", "epsilon")
    ]
    prompt = _reply(_bank_builder(memory_rounds=2), history=history)

    assert "post from round delta" in prompt and "post from round epsilon" in prompt
    assert "post from round alpha" not in prompt
    assert "post from round beta" not in prompt
    assert prompt.count("You replied:") == 2


def test_memory_rounds_is_read_from_the_run():
    for rounds in (1, 3, 5):
        history = [
            RoundMemory(posts_read=[f"p{i}"], reply_text=f"r{i}") for i in range(8)
        ]
        prompt = _reply(_bank_builder(memory_rounds=rounds), history=history)
        assert prompt.count("You replied:") == rounds


def test_memory_is_omitted_when_there_is_none():
    prompt = _reply(_bank_builder(), history=[])
    assert "Earlier in this discussion" not in prompt


def test_memory_carries_no_round_numbers_and_no_agent_ids():
    """Rounds are separated by a rule rather than numbered: "Round 2" would
    put a digit in front of the model for no benefit."""
    prompt = _reply(_bank_builder())
    memory = prompt.split("Earlier in this discussion:")[1].split("Posts in your feed")[0]
    assert "Round" not in memory
    assert "agent_" not in memory
    assert not any(ch.isdigit() for ch in memory)


def test_a_delivery_off_round_renders_a_reply_with_no_reads():
    """Under `delivery: off` an agent replies to nothing, so its memory has
    replies and no reads."""
    history = [RoundMemory(posts_read=[], reply_text="still thinking about it")]
    prompt = _reply(_bank_builder(), history=history)
    assert "You replied: still thinking about it" in prompt
    assert "You read:" not in prompt


def test_a_failed_round_leaves_no_memory_block():
    """Fix I: a failed turn is never applied, so it spends no memory slot."""
    history = [RoundMemory(posts_read=[], reply_text="")]
    prompt = _reply(_bank_builder(), history=history)
    assert "Earlier in this discussion" not in prompt


# --- the probe prompt ----------------------------------------------------


def test_the_probe_prompt_holds_persona_topic_question_anchors_and_instruction():
    builder = _bank_builder()
    probe = builder.build_probe_prompt(_bank_agent(), HISTORY)

    assert f"You are: {PERSONA}" in probe
    assert "whether the fare subsidy should continue" in probe
    assert "Should the fare subsidy continue?" in probe
    assert "1 means the subsidy should end" in probe
    assert "7 means the subsidy should continue" in probe
    assert probe.endswith("Answer with a single digit from 1 to 7 and nothing else.")


def test_the_probe_prompt_is_plain_text_not_a_built_prompt():
    probe = _bank_builder().build_probe_prompt(_bank_agent(), HISTORY)
    assert isinstance(probe, str)


def test_the_probe_shows_the_same_memory_the_reply_prompt_did():
    builder = _bank_builder()
    probe = builder.build_probe_prompt(_bank_agent(), HISTORY)
    reply = _reply(builder)

    memory = "Earlier in this discussion:" + reply.split("Earlier in this discussion:")[1].split(
        "Posts in your feed"
    )[0].rstrip("\n")
    assert memory.strip() in probe


def test_the_probe_never_asks_for_a_reply_or_a_stance_line():
    probe = _bank_builder().build_probe_prompt(_bank_agent(), HISTORY)
    assert "Reply to these posts" not in probe
    assert "STANCE" not in probe
    assert "sentences" not in probe


def test_the_probe_shows_no_feed():
    """The probe measures; it does not deliver. A feed in the probe context
    would make the measurement part of the stimulus."""
    probe = _bank_builder().build_probe_prompt(_bank_agent(), HISTORY)
    assert "Posts in your feed" not in probe
    assert "A user posted" not in probe


def test_the_probe_carries_no_position_number_before_the_question():
    """The digits in the anchor and instruction lines are the scale's own
    definition -- the model cannot answer "a single digit from 1 to 7" without
    being told that range. Everything ahead of the question must be clean."""
    builder = _bank_builder()
    probe = builder.build_probe_prompt(_bank_agent(), HISTORY)
    stimulus = probe.split("Should the fare subsidy continue?")[0]

    assert not any(ch.isdigit() for ch in stimulus), stimulus
    assert "position" not in probe.lower()
    assert "agent_" not in probe


def test_the_round_zero_probe_is_persona_topic_and_lean_only():
    builder = _bank_builder()
    probe = builder.build_probe_prompt(_bank_agent(initial_stance=2.0), [], include_lean=True)

    assert f"You are: {PERSONA}" in probe
    assert "You hold the view that the subsidy should end." in probe
    assert "Earlier in this discussion" not in probe
    assert "Posts in your feed" not in probe


def test_the_lean_is_absent_from_every_probe_after_round_zero():
    """Including a delivery: off arm, whose history is empty on every round.

    This is why include_lean is a keyword rather than inferred from an empty
    history: inferring would re-anchor that whole arm to its starting lean at
    every probe, reintroducing in the control exactly the confound the probe
    exists to avoid.
    """
    builder = _bank_builder()
    with_history = builder.build_probe_prompt(_bank_agent(), HISTORY)
    delivery_off = builder.build_probe_prompt(_bank_agent(), [])

    for probe in (with_history, delivery_off):
        assert "hold the view that" not in probe
        assert "undecided" not in probe


def test_the_probe_question_is_required_and_never_defaulted():
    """A generic English fallback would silently probe all four arms in
    English, which is the single most destructive thing that could happen to
    this experiment and would leave no trace in the output."""
    run = make_run(**{**BANK_RUN})
    object.__setattr__(run, "__dict__", {**run.__dict__, "probe_question": None})
    builder = PromptBuilder(run, supports_vision=True, meme_lookup=lambda m: None)

    with pytest.raises(ValueError, match="requires probe_question"):
        builder.build_probe_prompt(_bank_agent(), HISTORY)


def test_history_rounds_defaults_to_empty():
    probe = _bank_builder().build_probe_prompt(_bank_agent())
    assert "Earlier in this discussion" not in probe


# --- the RQ1 parity guard, for the bank regime ---------------------------


def test_english_and_hindi_prompts_differ_only_in_the_post_texts_and_directive():
    """The bank-regime restatement of the RQ1 validity guard.

    The orchestrator renders each bank item in the arm's language and hands the
    strings in, so the two prompts differ in exactly the lines carrying bank
    text (feed and memory) plus the one directive line. Everything else --
    persona, topic, lean, headings, the reply instruction -- must be
    byte-identical, or the arms differ in more than the factor under study.
    """
    english_feed = ["The English rendering of the cost argument."]
    hindi_feed = ["लागत तर्क का हिन्दी रूपांतरण।"]
    english_history = [RoundMemory(posts_read=["An English earlier post."], reply_text="Agreed.")]
    hindi_history = [RoundMemory(posts_read=["एक पुरानी हिन्दी पोस्ट।"], reply_text="सहमत।")]

    def render(condition, feed, history):
        builder = _bank_builder(language_condition=condition)
        return builder.build_discussion_prompt(
            _bank_agent(), [], 2, delivered_posts=feed, history_rounds=history
        ).text.split("\n")

    english = render("english", english_feed, english_history)
    hindi = render("hindi", hindi_feed, hindi_history)

    variable = set(english_feed + hindi_feed) | {
        "Agreed.", "सहमत।", "An English earlier post.", "एक पुरानी हिन्दी पोस्ट।",
        LANGUAGE_DIRECTIVES["english"], LANGUAGE_DIRECTIVES["hindi"],
    }

    def skeleton(lines):
        return [
            line for line in lines
            if not any(text and text in line for text in variable)
        ]

    assert skeleton(english) == skeleton(hindi)
    # And the lines that do differ are exactly the ones carrying bank text or
    # the directive -- not, say, a reordered heading.
    assert len(skeleton(english)) < len(english)


def test_the_four_arms_share_every_structural_line():
    builder_lines = {
        condition: _reply(_bank_builder(language_condition=condition)).split("\n")
        for condition in LANGUAGE_DIRECTIVES
    }
    structural = {
        condition: [
            line for line in lines if line != LANGUAGE_DIRECTIVES[condition]
        ]
        for condition, lines in builder_lines.items()
    }
    baseline = structural["english"]
    for condition, lines in structural.items():
        assert lines == baseline, f"{condition} differs beyond its directive"


# --- the generated regime is untouched -----------------------------------


def test_the_generated_regime_still_takes_the_old_path(meme_lookup):
    """Same builder, same method, different regime. The old prompt must be
    byte-for-byte what it was: the generated-regime tests above assert its
    content, and this asserts the regime switch is what selects it."""
    builder = _builder(meme_lookup)  # regime defaults to "generated"
    built = builder.build_discussion_prompt(
        make_agent(stance_history=_history(2)), [_text_post("agent_0011")], turn=2
    )
    assert "Your current position is" in built.text
    assert "STANCE:" in built.text
    assert "A user posted" not in built.text


def test_the_bank_regime_ignores_neighbor_posts_and_emphasize_format(meme_lookup):
    """The bank path has no stance to parse, so there is nothing to re-ask
    for. Passing either must change nothing rather than raise."""
    builder = _bank_builder()
    plain = builder.build_discussion_prompt(
        _bank_agent(), [], 2, delivered_posts=FEED, history_rounds=HISTORY
    ).text
    with_extras = builder.build_discussion_prompt(
        _bank_agent(),
        [_text_post("agent_0011"), _meme_post("agent_0077")],
        2,
        emphasize_format=True,
        delivered_posts=FEED,
        history_rounds=HISTORY,
    ).text
    assert plain == with_extras


def test_a_bank_prompt_never_carries_an_image():
    built = _bank_builder().build_discussion_prompt(
        _bank_agent(), [_meme_post("agent_0077")], 2, delivered_posts=FEED
    )
    assert built.image is None
    assert built.used_vision_fallback is False


def test_the_bank_builder_does_not_mutate_its_arguments():
    builder = _bank_builder()
    agent = _bank_agent()
    feed, history = list(FEED), list(HISTORY)

    builder.build_discussion_prompt(agent, [], 2, delivered_posts=feed, history_rounds=history)
    builder.build_probe_prompt(agent, history)

    assert feed == FEED
    assert history == HISTORY
    assert agent.stance_history == []
