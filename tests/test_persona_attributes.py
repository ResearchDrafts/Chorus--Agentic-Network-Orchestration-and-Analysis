"""Fix 7: structured persona attributes (off by default via
ExperimentRun.persona_attributes).

Covers, in order: the schema additions, the keyed Big Five draw, the
population builder, config-load validation, rendering, and the two export
paths (agents.csv's new columns, belief_shift_by_trait.csv).
"""

from __future__ import annotations

import json
import random

import pytest
import yaml
from pydantic import ValidationError

import export_csv
from analysis import metrics as m
from sandbox import seed_manager as sm
from sandbox.agent_manager import (
    PERSONA_ATTRIBUTE_FIELDS,
    AgentManager,
    PersonaRow,
    draw_big5,
    load_persona_rows,
)
from sandbox.config_loader import ConfigLoadError, load_run_config
from sandbox.models import Agent
from sandbox.prompt_builder import PromptBuilder
from tests.factories import make_agent, make_run

DEMOGRAPHIC_POOL = "test_pool_demographic"
MISSING_FIELD_POOL = "test_pool_demographic_missing"

BANK_RUN = dict(
    regime="bank",
    bank_path="data/bank/test_bank.csv",
    probe_question="Should the fare subsidy continue?",
    topic="whether the fare subsidy should continue",
    stance_low_label="strongly against the fare subsidy",
    stance_high_label="strongly in favour of the fare subsidy",
)


# --- schema: ExperimentRun.persona_attributes and Agent's new fields -------


def test_persona_attributes_defaults_to_none():
    assert make_run(**BANK_RUN).persona_attributes == "none"


def test_persona_attributes_rejects_an_unknown_value():
    with pytest.raises(ValidationError):
        make_run(**BANK_RUN, persona_attributes="full_psych_profile")


@pytest.mark.parametrize("value", ["demographic", "demographic_big5"])
def test_persona_attributes_accepts_the_two_new_modes(value):
    assert make_run(**BANK_RUN, persona_attributes=value).persona_attributes == value


def test_agent_attribute_fields_default_to_none():
    agent = make_agent()
    assert agent.age is None and agent.gender is None and agent.education is None
    for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        assert getattr(agent, f"trait_{trait}") is None


def test_agent_attribute_fields_round_trip():
    agent = make_agent(
        age=34, gender="a woman", education="a bachelor's degree",
        trait_openness="high", trait_conscientiousness="low",
        trait_extraversion="high", trait_agreeableness="low",
        trait_neuroticism="high",
    )
    restored = Agent.model_validate_json(agent.model_dump_json())
    assert restored.age == 34 and restored.gender == "a woman"
    assert restored.trait_openness == "high" and restored.trait_neuroticism == "high"


# --- the keyed Big Five RNG -------------------------------------------------


def test_persona_trait_rng_is_reproducible():
    first = [sm.persona_trait_rng(1001, 3).random() for _ in range(5)]
    second = [sm.persona_trait_rng(1001, 3).random() for _ in range(5)]
    assert first == second


def test_persona_trait_rng_differs_across_persona_index():
    a = sm.persona_trait_rng(1001, 0).random()
    b = sm.persona_trait_rng(1001, 1).random()
    assert a != b


def test_persona_trait_rng_differs_across_seed():
    a = sm.persona_trait_rng(1001, 3).random()
    b = sm.persona_trait_rng(1002, 3).random()
    assert a != b


def test_persona_trait_rng_rejects_a_negative_index():
    with pytest.raises(ValueError, match="persona_index"):
        sm.persona_trait_rng(1001, -1)


def test_draw_big5_gives_all_five_traits_as_high_or_low():
    traits = draw_big5(1001, 0)
    assert set(traits) == {
        "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
    }
    assert all(level in ("high", "low") for level in traits.values())


def test_draw_big5_matches_a_hand_computation():
    """Pinned against the exact derivation, the same discipline
    test_seed_manager.py uses for call_seed: a drift here would silently
    reassign every persona's traits without a test noticing."""
    rng = random.Random(f"{1001 + sm.PERSONA_ASSIGNMENT_OFFSET}|big5|7")
    expected = {
        trait: ("high" if rng.random() < 0.5 else "low")
        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
    }
    assert draw_big5(1001, 7) == expected


def test_draw_big5_is_independent_of_draw_order_or_lean_assignment():
    """The whole point: unlike a sequential draw off persona_assignment_rng,
    this must not depend on anything else that stream was used for."""
    direct = draw_big5(1001, 5)
    # Simulate having consumed the persona-assignment stream for something
    # else first (e.g. a long sequence of _draw_initial_stance calls) --
    # draw_big5 never touches that stream at all, so this changes nothing.
    decoy_rng = random.Random(1001)
    for _ in range(50):
        decoy_rng.random()
    assert draw_big5(1001, 5) == direct


# --- load_persona_rows -------------------------------------------------------


def test_load_persona_rows_parses_every_field():
    rows = load_persona_rows(DEMOGRAPHIC_POOL)
    assert len(rows) == 4
    assert all(isinstance(row, PersonaRow) for row in rows)
    assert rows[0].age == 29 and rows[0].gender == "a man"
    assert rows[0].education == "a high school diploma"
    assert rows[0].persona.startswith("A skeptical retail worker")


def test_load_persona_rows_raises_naming_the_line_on_plain_text():
    """A plain-text pool (every pre-Fix-7 pool) is not JSON at all."""
    with pytest.raises(ValueError, match="line 1"):
        load_persona_rows("test_pool")


def test_load_persona_rows_raises_on_a_missing_pool():
    with pytest.raises(FileNotFoundError):
        load_persona_rows("no_such_pool_at_all")


def test_load_persona_rows_tolerates_a_missing_demographic_field():
    """The loader itself only requires 'persona' -- config_loader is what
    refuses a row missing age/gender/education."""
    rows = load_persona_rows(MISSING_FIELD_POOL)
    assert rows[1].education is None


# --- AgentManager.initialize_population -------------------------------------


def _bank_run(**overrides):
    base = {**BANK_RUN, "persona_pool_id": "test_pool", "M": 4, "N": 1}
    base.update(overrides)
    return make_run(**base)


def test_persona_attributes_none_is_byte_identical_to_before():
    run = _bank_run()
    agents = AgentManager(run, random.Random(run.seed)).initialize_population()
    assert all(a.age is None and a.gender is None and a.trait_openness is None for a in agents)
    assert agents[0].persona == "A skeptical retail worker who distrusts institutions and values plain talk."


def test_demographic_mode_populates_age_gender_education():
    run = _bank_run(persona_pool_id=DEMOGRAPHIC_POOL, persona_attributes="demographic")
    agents = AgentManager(run, random.Random(run.seed)).initialize_population()
    assert agents[0].age == 29 and agents[0].gender == "a man"
    assert agents[0].education == "a high school diploma"
    assert all(a.trait_openness is None for a in agents)  # demographic, not big5


def test_demographic_big5_mode_also_populates_traits():
    run = _bank_run(persona_pool_id=DEMOGRAPHIC_POOL, persona_attributes="demographic_big5")
    agents = AgentManager(run, random.Random(run.seed)).initialize_population()
    assert agents[0].trait_openness in ("high", "low")
    assert agents[0].age == 29  # demographic fields still populated


def test_shared_persona_index_shares_the_same_traits():
    """M=8 over a 4-row pool: population indices 0 and 4 both draw pool
    index 0, so they must share both the persona text and its traits --
    the same way they already share the free-text description."""
    run = _bank_run(persona_pool_id=DEMOGRAPHIC_POOL, persona_attributes="demographic_big5", M=8)
    agents = AgentManager(run, random.Random(run.seed)).initialize_population()
    assert agents[0].persona == agents[4].persona
    assert agents[0].age == agents[4].age
    assert agents[0].trait_openness == agents[4].trait_openness


def test_traits_do_not_depend_on_lean_assignment():
    """The same seed and persona pool must draw identical traits per pool
    index whether lean_assignment is random or balanced_two_camps -- the
    reason draw_big5 is keyed rather than sequential."""
    random_run = _bank_run(
        persona_pool_id=DEMOGRAPHIC_POOL, persona_attributes="demographic_big5",
        lean_assignment="random", M=4,
    )
    balanced_run = _bank_run(
        persona_pool_id=DEMOGRAPHIC_POOL, persona_attributes="demographic_big5",
        lean_assignment="balanced_two_camps", M=4,
    )
    a = AgentManager(random_run, random.Random(random_run.seed)).initialize_population()
    b = AgentManager(balanced_run, random.Random(balanced_run.seed)).initialize_population()
    for agent_a, agent_b in zip(a, b, strict=True):
        assert agent_a.trait_openness == agent_b.trait_openness
        assert agent_a.trait_neuroticism == agent_b.trait_neuroticism


# --- config_loader validation -----------------------------------------------


def _config(tmp_path, **overrides):
    raw = yaml.safe_load(open("configs/rq1/smoke.yaml", encoding="utf-8"))
    raw.update(overrides)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_a_demographic_config_with_a_complete_pool_loads(tmp_path):
    run = load_run_config(_config(
        tmp_path, persona_attributes="demographic", persona_pool_id=DEMOGRAPHIC_POOL,
    ))
    assert run.persona_attributes == "demographic"


def test_demographic_mode_warns_that_the_templates_are_unverified(tmp_path, capsys):
    """Fix 9: configs/persona_attribute_templates.yaml is a machine
    translation nobody has native-speaker-reviewed, and turning
    persona_attributes on adds a fourth language-varying element (this
    template's own sentence) beyond the bank text, probe question and
    directive line."""
    load_run_config(_config(
        tmp_path, persona_attributes="demographic", persona_pool_id=DEMOGRAPHIC_POOL,
    ))
    out = capsys.readouterr().out
    assert "UNVERIFIED" in out
    assert "native-speaker" in out


def test_none_mode_never_parses_the_pool_as_json(tmp_path):
    """Every existing config points at a plain-text pool; persona_attributes
    defaulting to "none" must not try to JSON-parse it."""
    run = load_run_config(_config(tmp_path))  # smoke.yaml's own plain-text pool_20
    assert run.persona_attributes == "none"


def test_none_mode_prints_no_translation_warning(tmp_path, capsys):
    load_run_config(_config(tmp_path))
    assert "UNVERIFIED" not in capsys.readouterr().out


def test_demographic_mode_against_a_plain_text_pool_is_refused(tmp_path):
    with pytest.raises(ConfigLoadError, match="not valid JSON"):
        load_run_config(_config(
            tmp_path, persona_attributes="demographic", persona_pool_id="pool_20",
        ))


def test_demographic_mode_names_the_row_missing_a_field(tmp_path):
    with pytest.raises(ConfigLoadError, match=r"row 1.*education"):
        load_run_config(_config(
            tmp_path, persona_attributes="demographic", persona_pool_id=MISSING_FIELD_POOL,
        ))


# --- rendering: prompt_builder._persona_section -----------------------------


def _builder(**overrides) -> PromptBuilder:
    return PromptBuilder(
        make_run(**{**BANK_RUN, **overrides}),
        supports_vision=True,
        meme_lookup=lambda meme_id: None,
    )


def _demographic_agent(**overrides) -> Agent:
    return make_agent(
        persona="a bus commuter in Pune", age=34, gender="a woman",
        education="a bachelor's degree", **overrides,
    )


def test_none_mode_prompt_is_unchanged():
    builder = _builder()
    agent = make_agent(persona="a bus commuter in Pune")
    prompt = builder.build_discussion_prompt(
        agent, [], 1, delivered_posts=["a post"], history_rounds=[]
    ).text
    assert "You are: a bus commuter in Pune\n\n" in prompt
    assert "34" not in prompt  # nothing demographic leaks in even if the agent carried it


def test_demographic_mode_appends_one_sentence_in_the_runs_language():
    builder = _builder(persona_attributes="demographic", language_condition="hindi")
    agent = _demographic_agent()
    prompt = builder.build_discussion_prompt(
        agent, [], 1, delivered_posts=["a post"], history_rounds=[]
    ).text
    assert "You are: a bus commuter in Pune" in prompt
    assert "34" in prompt
    assert "है" in prompt  # the Hindi template rendered, not the English one


def test_demographic_big5_mode_appends_the_personality_sentence_too():
    builder = _builder(persona_attributes="demographic_big5", language_condition="english")
    agent = _demographic_agent(
        trait_openness="high", trait_conscientiousness="low",
        trait_extraversion="high", trait_agreeableness="low", trait_neuroticism="high",
    )
    prompt = builder.build_discussion_prompt(
        agent, [], 1, delivered_posts=["a post"], history_rounds=[]
    ).text
    assert "openness to new ideas" in prompt
    assert "high in openness" in prompt
    assert "low in conscientiousness" in prompt


def test_the_persona_section_is_shared_across_reply_and_probe_prompts():
    builder = _builder(persona_attributes="demographic", language_condition="english")
    agent = _demographic_agent()
    reply = builder.build_discussion_prompt(
        agent, [], 1, delivered_posts=["a post"], history_rounds=[]
    ).text
    probe = builder.build_probe_prompt(agent, [])
    for prompt in (reply, probe):
        assert "your highest level of education is a bachelor's degree" in prompt


# --- export_csv.py: agents.csv gains the trait columns ---------------------


def _write_agents_final(run_dir, agents: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "agents_final.jsonl").open("w", encoding="utf-8") as handle:
        for agent in agents:
            handle.write(json.dumps(agent) + "\n")


def _write_interaction(run_dir, **overrides) -> dict:
    base = dict(
        interaction_id="i1", run_id="r", turn=1, speaker_agent_id="agent_0000",
        neighbor_agent_ids=["agent_0001"], stance_before=4.0, stance_after=5.0,
        reason_text="because", content_type="generated_text", prompt_token_count=10,
        completion_token_count=5, model_backend_id="gpt-4o", latency_ms=1,
        api_call_status="success", timestamp_utc="2026-01-01T00:00:00Z",
    )
    base.update(overrides)
    return base


def test_load_run_joins_persona_attributes_from_agents_final(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(json.dumps({"run_id": "run1"}))
    rows = [_write_interaction(run_dir, speaker_agent_id="agent_0000")]
    (run_dir / "interactions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    _write_agents_final(run_dir, [
        {"agent_id": "agent_0000", "age": 34, "gender": "a woman",
         "education": "a bachelor's degree", "trait_openness": "high"},
    ])
    df = export_csv.load_run(run_dir)
    assert df.loc[0, "age"] == 34
    assert df.loc[0, "trait_openness"] == "high"
    assert df.loc[0, "trait_neuroticism"] is None


def test_load_run_fills_attribute_columns_with_none_when_absent(tmp_path):
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(json.dumps({"run_id": "run2"}))
    rows = [_write_interaction(run_dir)]
    (run_dir / "interactions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    # No agents_final.jsonl at all -- an interrupted run.
    df = export_csv.load_run(run_dir)
    for column in export_csv.PERSONA_ATTRIBUTE_COLUMNS:
        assert df.loc[0, column] is None


def test_agents_table_carries_the_trait_columns_through(tmp_path):
    run_dir = tmp_path / "run3"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(json.dumps({"run_id": "run3"}))
    rows = [_write_interaction(run_dir, speaker_agent_id="agent_0000")]
    (run_dir / "interactions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    _write_agents_final(run_dir, [
        {"agent_id": "agent_0000", "gender": "a woman", "trait_openness": "high"},
    ])
    df = export_csv.load_run(run_dir)
    df = export_csv.derive(export_csv.mark_duplicates(df))
    agents = export_csv.agents_table(df)
    assert agents.loc[0, "gender"] == "a woman"
    assert agents.loc[0, "trait_openness"] == "high"


# --- analysis/metrics.py: belief_shift_by_trait.csv -------------------------


def _probe_row(agent_id, turn, x):
    return {"agent_id": agent_id, "turn": turn, "x": x}


def test_agent_belief_shifts_needs_both_round_zero_and_round_k():
    probes = [
        _probe_row("a", 0, 0.0), _probe_row("a", 3, 0.6),
        _probe_row("b", 0, 0.2),  # no round 3 at all
        _probe_row("c", 1, 0.5), _probe_row("c", 3, 0.5),  # no round 0
    ]
    shifts = m.agent_belief_shifts(probes, k=3)
    assert shifts == {"a": pytest.approx(0.6)}


def test_belief_shift_by_trait_computes_mean_and_population_variance():
    rows = [
        {"trait": "gender", "level": "a woman", "language": "english", "shift": 0.2},
        {"trait": "gender", "level": "a woman", "language": "english", "shift": 0.6},
        {"trait": "gender", "level": "a man", "language": "english", "shift": 0.4},
    ]
    summary = m.belief_shift_by_trait(rows)
    women = next(r for r in summary if r["level"] == "a woman")
    # mean = 0.4, population variance = ((0.2-0.4)^2 + (0.6-0.4)^2) / 2 = 0.04
    assert women["n"] == 2
    assert women["mean_abs_shift"] == pytest.approx(0.4)
    assert women["var_abs_shift"] == pytest.approx(0.04)
    men = next(r for r in summary if r["level"] == "a man")
    assert men["n"] == 1 and men["var_abs_shift"] == pytest.approx(0.0)


def test_load_agent_traits_is_none_when_every_row_has_no_traits(tmp_path):
    run_dir = tmp_path / "run"
    _write_agents_final(run_dir, [
        {"agent_id": "agent_0000", "gender": None, "trait_openness": None},
    ])
    assert m.load_agent_traits(run_dir) is None


def test_load_agent_traits_reads_present_columns(tmp_path):
    run_dir = tmp_path / "run"
    _write_agents_final(run_dir, [
        {"agent_id": "agent_0000", "gender": "a woman", "trait_openness": "high"},
    ])
    traits = m.load_agent_traits(run_dir)
    assert traits["agent_0000"]["gender"] == "a woman"
    assert traits["agent_0000"]["trait_openness"] == "high"


def test_load_agent_traits_is_none_without_the_file(tmp_path):
    assert m.load_agent_traits(tmp_path / "no_such_run") is None
