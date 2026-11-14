"""start_mode: how an agent's starting position is set in the bank regime.

A live run showed the original lean sentence is read back by the probe: at
lean 6 every persona answered 7.0 (sd 0.03), so readers started pinned at the
ends of the scale and no argument could move them. These tests cover the two
replacements, "none" and "posts", and that "sentence" is unchanged.
"""

from __future__ import annotations

import json
import random

import pytest
import yaml

from sandbox.bank_manager import (
    STARTER_COUNTS,
    STARTER_MIN_PER_SIDE,
    BankManager,
    starter_items,
)
from sandbox.config_loader import ConfigLoadError, load_run_config
from sandbox.prompt_builder import LEAN_SENTENCES, PromptBuilder
from sandbox.single_exposure import (
    check_one_start_mode,
    start_key_for,
    with_start_mode,
)
from tests.factories import make_agent, make_run

STARTER_PATH = "data/bank/test_starter.csv"
STARTERS = BankManager(STARTER_PATH)

BANK_RUN = dict(
    regime="bank",
    bank_path="data/bank/test_bank.csv",
    probe_question="Should the fare subsidy continue?",
    topic="whether the fare subsidy should continue",
    stance_low_label="strongly against the fare subsidy",
    stance_high_label="strongly in favour of the fare subsidy",
)
PERSONA = "a bus commuter in Pune who reads the news every morning"


def _builder(**overrides) -> PromptBuilder:
    return PromptBuilder(
        make_run(**{**BANK_RUN, **overrides}),
        supports_vision=True,
        meme_lookup=lambda meme_id: None,
    )


def _agent(lean: float = 6.0, **overrides):
    return make_agent(persona=PERSONA, initial_stance=lean, **overrides)


def _reply(builder, agent) -> str:
    return builder.build_discussion_prompt(
        agent, [], 2, delivered_posts=["A post."], history_rounds=[]
    ).text


def _lean_text(lean: int) -> str:
    return LEAN_SENTENCES[lean].format(
        low=BANK_RUN["stance_low_label"], high=BANK_RUN["stance_high_label"]
    )


# --- starter_items ----------------------------------------------------------


@pytest.mark.parametrize("lean", sorted(STARTER_COUNTS))
def test_each_lean_gets_its_composition(lean):
    items = starter_items(STARTERS, 1001, "agent_0003", lean)
    n_con, n_pro = STARTER_COUNTS[lean]
    sides = [STARTERS.side_of(item) for item in items]
    assert sides.count("con") == n_con
    assert sides.count("pro") == n_pro
    assert len(set(items)) == len(items)


def test_the_composition_moves_monotonically_from_con_to_pro():
    """More con at the low end, more pro at the high end: the starter posts
    carry the lean, so they must order the way the scale does."""
    net = [pro - con for con, pro in (STARTER_COUNTS[k] for k in sorted(STARTER_COUNTS))]
    assert net == sorted(net)
    assert STARTER_COUNTS[4][0] == STARTER_COUNTS[4][1]


def test_starter_items_are_a_pure_function_of_seed_key_and_lean():
    first = starter_items(STARTERS, 1001, "agent_0003", 2)
    assert starter_items(STARTERS, 1001, "agent_0003", 2) == first
    draws = {
        tuple(starter_items(STARTERS, seed, f"agent_{i:04d}", 7))
        for seed in (1001, 1002)
        for i in range(20)
    }
    assert len(draws) > 1  # agents differ, rather than all posting the same three


def test_starter_items_refuse_a_fractional_or_out_of_range_lean():
    with pytest.raises(ValueError, match="starter composition"):
        starter_items(STARTERS, 1, "a", 4.5)
    with pytest.raises(ValueError, match="starter composition"):
        starter_items(STARTERS, 1, "a", 8)


# --- schema -----------------------------------------------------------------


def test_the_default_is_the_original_sentence():
    assert make_run(**BANK_RUN).start_mode == "sentence"


def test_posts_needs_a_starter_bank():
    with pytest.raises(ValueError, match="starter_bank_path"):
        make_run(**BANK_RUN, start_mode="posts")


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        make_run(**BANK_RUN, start_mode="lean")


def test_a_non_default_mode_needs_the_bank_regime():
    with pytest.raises(ValueError, match="regime='bank'"):
        make_run(start_mode="none")


# --- the prompts ------------------------------------------------------------


def test_sentence_mode_is_unchanged():
    builder = _builder()
    agent = _agent(6.0)
    assert _lean_text(6) in _reply(builder, agent)
    assert _lean_text(6) in builder.build_probe_prompt(agent, [])


@pytest.mark.parametrize("lean", [1.0, 4.0, 7.0])
def test_none_mode_shows_no_starting_position(lean):
    builder = _builder(start_mode="none")
    agent = _agent(lean)
    for prompt in (_reply(builder, agent), builder.build_probe_prompt(agent, [])):
        assert not any(
            sentence.split("{")[0] in prompt for sentence in LEAN_SENTENCES.values()
        )
        assert "Before this discussion" not in prompt
        assert "\n\n\n" not in prompt  # the section is left out, not left blank


def test_none_mode_prompts_do_not_depend_on_the_lean():
    builder = _builder(start_mode="none")
    probes = {builder.build_probe_prompt(_agent(lean), []) for lean in (1.0, 4.0, 7.0)}
    assert len(probes) == 1


@pytest.mark.parametrize("language", ["english", "hinglish", "romanized_hindi", "hindi"])
def test_posts_mode_shows_the_starter_posts_in_the_run_language(language):
    builder = _builder(
        start_mode="posts", starter_bank_path=STARTER_PATH, language_condition=language
    )
    agent = _agent(2.0, agent_id="agent_0005")
    expected = starter_items(STARTERS, builder._run.seed, "agent_0005", 2.0)
    for prompt in (_reply(builder, agent), builder.build_probe_prompt(agent, [])):
        assert "Before this discussion, you had written these posts:" in prompt
        for item in expected:
            assert f"You posted: {STARTERS.text(item, language)}" in prompt
        assert prompt.count("You posted:") == len(expected)
        assert _lean_text(2) not in prompt


def test_posts_mode_shows_the_same_items_in_every_language():
    """The matched set: four arms, the same starter items, four renderings."""
    shown = set()
    for language in ("english", "hinglish", "romanized_hindi", "hindi"):
        builder = _builder(
            start_mode="posts", starter_bank_path=STARTER_PATH, language_condition=language
        )
        prompt = builder.build_probe_prompt(_agent(7.0, agent_id="agent_0001"), [])
        shown.add(tuple(
            item for item in STARTERS.all_item_ids()
            if STARTERS.text(item, language) in prompt
        ))
    assert len(shown) == 1 and len(next(iter(shown))) == 3


def test_posts_mode_follows_start_key_rather_than_agent_id():
    """A single-exposure trial's agent_id names the trial; its baseline and
    exposed probes must show the same starter posts, so both use the reader's
    start_key."""
    builder = _builder(start_mode="posts", starter_bank_path=STARTER_PATH)
    baseline = builder.build_probe_prompt(_agent(1.0, agent_id="e1|p003|lean1", start_key="p003|lean1"), [])
    exposed = builder.build_probe_prompt(
        _agent(1.0, agent_id="e1|bank_t01|p003|lean1", start_key="p003|lean1"), []
    )
    assert baseline == exposed


def test_posts_mode_prompts_carry_no_digit():
    """The property the bank regime turns on: no position number reaches an
    agent. The starter posts are where a stray one could now enter."""
    builder = _builder(start_mode="posts", starter_bank_path=STARTER_PATH)
    prompt = _reply(builder, _agent(7.0))
    stimulus = prompt.split("Reply to these posts")[0]
    stimulus = stimulus.split("Respond in")[0]
    assert not any(ch.isdigit() for ch in stimulus.split("Before this discussion")[1])


# --- config loading ---------------------------------------------------------


def _config(tmp_path, **overrides):
    raw = yaml.safe_load(open("configs/rq1/smoke.yaml", encoding="utf-8"))
    raw.update(overrides)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_a_posts_config_with_a_valid_starter_bank_loads(tmp_path):
    run = load_run_config(_config(tmp_path, start_mode="posts", starter_bank_path=STARTER_PATH))
    assert run.start_mode == "posts"


def test_a_missing_starter_bank_is_a_config_error(tmp_path):
    with pytest.raises(ConfigLoadError, match="not found"):
        load_run_config(_config(tmp_path, start_mode="posts", starter_bank_path="nope.csv"))


def test_a_starter_bank_sharing_ids_with_the_argument_bank_is_refused(tmp_path):
    raw = yaml.safe_load(open("configs/rq1/smoke.yaml", encoding="utf-8"))
    with pytest.raises(ConfigLoadError, match="shares item_id"):
        load_run_config(_config(tmp_path, start_mode="posts", starter_bank_path=raw["bank_path"]))


def test_a_starter_bank_too_small_for_the_extreme_leans_is_refused(tmp_path):
    rows = open(STARTER_PATH, encoding="utf-8").read().splitlines()
    small = tmp_path / "small.csv"
    small.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")  # drops one con
    with pytest.raises(ConfigLoadError, match=f"needs at least {STARTER_MIN_PER_SIDE}"):
        load_run_config(_config(tmp_path, start_mode="posts", starter_bank_path=str(small)))


# --- single-exposure helpers -----------------------------------------------


def test_with_start_mode_overrides_and_revalidates():
    runs = {"english": make_run(**BANK_RUN)}
    assert with_start_mode(runs, None) is runs
    assert with_start_mode(runs, "none")["english"].start_mode == "none"
    with pytest.raises(ValueError, match="starter_bank_path"):
        with_start_mode(runs, "posts")


def test_check_one_start_mode(tmp_path):
    check_one_start_mode([], "none", tmp_path)
    check_one_start_mode([{"start_mode": "none"}], "none", tmp_path)
    with pytest.raises(ValueError, match="different --runs-dir"):
        check_one_start_mode([{"start_mode": "none"}], "posts", tmp_path)
    with pytest.raises(ValueError, match="different --runs-dir"):
        check_one_start_mode([{"key": "old row"}], "none", tmp_path)  # pre-column rows


def test_start_key_names_the_reader_not_the_trial():
    assert start_key_for(3, 6) == "p003|lean6"
