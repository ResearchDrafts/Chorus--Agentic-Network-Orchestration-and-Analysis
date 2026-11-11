"""Tests for sandbox/models.py's own validation rules.

Phase 4 starter. full_design_doc.md Sec 4's constraints were previously
covered only incidentally, through the modules that happen to construct
these models; the cross-field validators and the Fix A memory cap had no
direct coverage at all. phase4.md Section 3 tracks the remaining gaps
(Interaction.check_meme_id_consistency's meme branch,
is_valid_for_analysis, MemeInjectionConfig.check_enabled_requirements).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sandbox.models import Agent


def _make_agent(memory_window: list[str] | None = None) -> Agent:
    return Agent(
        agent_id="agent_0000",
        run_id="test_run",
        persona="a persona",
        language_condition="english",
        model_backend_id="gpt-4o",
        initial_stance=4.0,
        memory_window=memory_window if memory_window is not None else [],
    )


# --- Fix A: the 5-turn memory window (phase4.md D7) ---------------------


def test_construction_rejects_memory_window_over_cap():
    with pytest.raises(ValidationError):
        _make_agent([f"i{i}" for i in range(6)])


def test_assignment_rejects_memory_window_over_cap():
    """The D7 regression.

    Without model_config = ConfigDict(validate_assignment=True), Pydantic v2
    validates only at construction, so this assignment silently succeeded and
    Fix A's cap was enforced solely by agent_manager.py's [-5:] slice -- a
    convention, not the schema-level guarantee full_design_doc.md Sec 4.1
    claims it is.
    """
    agent = _make_agent(["i1", "i2"])
    with pytest.raises(ValidationError):
        agent.memory_window = [f"i{i}" for i in range(8)]


def test_assignment_accepts_memory_window_at_cap():
    """Guards against over-correcting: exactly 5 must remain valid, since
    that is what agent_manager.py assigns on every apply_interaction()."""
    agent = _make_agent()
    agent.memory_window = [f"i{i}" for i in range(5)]
    assert len(agent.memory_window) == 5


def test_inplace_append_bypasses_the_cap():
    """Documents a known limit rather than asserting desired behaviour.

    Pydantic cannot observe in-place list mutation, so validate_assignment
    does not close this path. Fix A is therefore re-enforced at the point of
    use: prompt_builder.py slices [-5:] when rendering memory into a prompt.
    If a future Pydantic version starts catching this, update the test -- do
    not treat the change as a regression.
    """
    agent = _make_agent([f"i{i}" for i in range(5)])
    agent.memory_window.append("overflow")
    assert len(agent.memory_window) == 6


def test_assignment_still_validates_other_fields():
    """validate_assignment applies to every field, not just memory_window."""
    agent = _make_agent()
    with pytest.raises(ValidationError):
        agent.language_condition = "french"


def test_current_stance_falls_back_to_initial_when_history_empty():
    assert _make_agent().current_stance() == 4.0


# --- phase4 audit D10: unknown config keys must not vanish ---------------


def _run_kwargs(**overrides):
    base = dict(
        run_id="r", rq_target="RQ1_RQ2", topic="t", alpha=0.5, M=10, N=3, K=2,
        trial_number=1, language_condition="english", model_backend_id="gpt-4o",
        stance_scale=[1, 2, 3, 4, 5, 6, 7], persona_pool_id="test_pool",
        seed=1, temperature=0.7,
    )
    base.update(overrides)
    return base


def test_unknown_experiment_run_key_is_rejected():
    """The dangerous case: `meme_injections` (plural) used to load fine and
    leave meme injection off, so RQ3's meme-present arm silently ran as a
    second meme-absent arm and returned a clean null result."""
    from sandbox.models import ExperimentRun

    with pytest.raises(ValidationError):
        ExperimentRun(**_run_kwargs(meme_injections={"enabled": True}))


def test_unknown_meme_injection_key_is_rejected():
    from sandbox.models import MemeInjectionConfig

    with pytest.raises(ValidationError):
        MemeInjectionConfig(enabled=False, injection_rat=0.5)


def test_known_optional_keys_still_accepted():
    """Guards against over-correcting: extra="forbid" must not reject the
    legitimate optional fields."""
    from sandbox.models import ExperimentRun

    run = ExperimentRun(**_run_kwargs(
        max_cost_usd=5.0, rate_limits={"groq": 30}, checkpoint_every_n_turns=2,
        stance_low_label="opposed", stance_high_label="in favour",
        api_base="http://localhost:8000/v1",
    ))
    assert run.api_base == "http://localhost:8000/v1"
    assert run.max_cost_usd == 5.0


# --- phase4 audit D11: stance_scale needs distinct values ----------------


def test_stance_scale_with_duplicate_values_is_rejected():
    """[3, 3] passed Field(min_length=2), then min == max so only 3.0 ever
    validated and the entire population ended up failed_logged_null."""
    from sandbox.models import ExperimentRun

    with pytest.raises(ValidationError, match="distinct"):
        ExperimentRun(**_run_kwargs(stance_scale=[3.0, 3.0]))


def test_two_distinct_values_is_enough():
    from sandbox.models import ExperimentRun

    assert ExperimentRun(**_run_kwargs(stance_scale=[1.0, 7.0])).stance_scale == [1.0, 7.0]


# --- RQ1: the four language conditions -----------------------------------


@pytest.mark.parametrize("condition", ["english", "hinglish", "romanized_hindi", "hindi"])
def test_all_four_language_conditions_are_accepted(condition):
    """romanized_hindi is a distinct condition from hinglish: Hindi in Roman
    script is not Hindi-English code-mixing, and RQ1 manipulates language and
    script separately."""
    from sandbox.models import ExperimentRun

    assert ExperimentRun(**_run_kwargs(language_condition=condition)).language_condition == condition
    assert _make_agent_with(language_condition=condition).language_condition == condition


def _make_agent_with(**overrides) -> Agent:
    base = dict(
        agent_id="agent_0000", run_id="test_run", persona="a persona",
        language_condition="english", model_backend_id="gpt-4o", initial_stance=4.0,
    )
    base.update(overrides)
    return Agent(**base)


def test_agent_and_run_share_one_language_pattern():
    """If these drifted, an agent could carry a condition its own run rejects."""
    from sandbox.models import ExperimentRun

    with pytest.raises(ValidationError):
        _make_agent_with(language_condition="romanised_hindi")  # British spelling typo
    with pytest.raises(ValidationError):
        ExperimentRun(**_run_kwargs(language_condition="romanised_hindi"))


# --- RQ1: the bank-regime config fields ----------------------------------


def test_bank_regime_fields_default_to_the_previous_behaviour():
    """Every existing config must keep meaning what it meant, so each new
    field defaults to the pre-bank behaviour rather than to the new one."""
    from sandbox.models import ExperimentRun

    run = ExperimentRun(**_run_kwargs())
    assert run.neighbor_strategy == "alpha"
    assert run.regime == "generated"
    assert run.delivery == "on"
    assert run.bank_path is None
    assert run.probe_question is None
    assert (run.graph_degree, run.graph_rewire_p) == (6, 0.1)
    assert (run.probe_top_logprobs, run.probe_mass_floor) == (20, 0.5)
    assert run.memory_rounds == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"neighbor_strategy": "random"},
        {"regime": "memes"},
        {"delivery": "sometimes"},
        {"graph_rewire_p": 1.5},
        {"graph_rewire_p": -0.1},
        {"probe_top_logprobs": 0},
        {"probe_mass_floor": 0.0},   # exclusive lower bound
        {"probe_mass_floor": 1.5},
        {"memory_rounds": 0},
        {"graph_degree": 0},
    ],
)
def test_out_of_range_bank_fields_are_rejected(overrides):
    from sandbox.models import ExperimentRun

    with pytest.raises(ValidationError):
        ExperimentRun(**_run_kwargs(**overrides))


def test_probe_mass_floor_accepts_exactly_one():
    """The interval is (0, 1]: a floor of 1.0 means "every digit or nothing",
    which is strict but legitimate."""
    from sandbox.models import ExperimentRun

    assert ExperimentRun(**_run_kwargs(probe_mass_floor=1.0)).probe_mass_floor == 1.0


def test_odd_graph_degree_is_rejected_even_under_alpha_sampling():
    """An odd degree is not representable in a Watts-Strogatz ring at all, so
    it is rejected as a property of the number rather than of the run."""
    from sandbox.models import ExperimentRun

    with pytest.raises(ValidationError, match="even"):
        ExperimentRun(**_run_kwargs(graph_degree=5))


def test_graph_degree_is_only_checked_against_m_when_the_graph_is_used():
    """graph_degree defaults to 6 and is meaningless under alpha sampling.
    Validating degree < M unconditionally would reject every M=5 smoke config
    over a field those runs never read."""
    from sandbox.models import ExperimentRun

    assert ExperimentRun(**_run_kwargs(M=5, N=2)).graph_degree == 6  # alpha: fine

    with pytest.raises(ValidationError, match="less than M"):
        ExperimentRun(**_run_kwargs(M=5, N=2, neighbor_strategy="graph", graph_degree=6))


def test_graph_strategy_rejects_n_above_the_degree():
    """Every node starts with exactly graph_degree neighbours and rewiring only
    removes some, so N > degree can never be satisfied for any node. Caught at
    config load rather than per-agent on turn 1."""
    from sandbox.models import ExperimentRun

    with pytest.raises(ValidationError, match="cannot exceed graph_degree"):
        ExperimentRun(**_run_kwargs(M=20, N=5, neighbor_strategy="graph", graph_degree=4))


def test_graph_strategy_accepts_a_valid_combination():
    from sandbox.models import ExperimentRun

    run = ExperimentRun(**_run_kwargs(M=8, N=3, neighbor_strategy="graph", graph_degree=4))
    assert (run.neighbor_strategy, run.graph_degree) == ("graph", 4)


@pytest.mark.parametrize(
    "overrides, missing",
    [
        ({}, "bank_path"),
        ({"bank_path": "data/bank/test_bank.csv"}, "probe_question"),
        ({"probe_question": "q?"}, "bank_path"),
        ({"bank_path": "  ", "probe_question": "q?"}, "bank_path"),
        ({"bank_path": "b.csv", "probe_question": "   "}, "probe_question"),
    ],
)
def test_bank_regime_requires_a_bank_and_a_probe_question(overrides, missing):
    """A missing probe_question is the dangerous one: a generic English
    fallback would probe every language arm in English and destroy the
    comparison the run exists to make."""
    from sandbox.models import ExperimentRun

    with pytest.raises(ValidationError, match=missing):
        ExperimentRun(**_run_kwargs(regime="bank", **overrides))


def test_bank_regime_with_both_fields_is_accepted():
    from sandbox.models import ExperimentRun

    run = ExperimentRun(
        **_run_kwargs(regime="bank", bank_path="data/bank/test_bank.csv", probe_question="q?")
    )
    assert run.bank_path == "data/bank/test_bank.csv"


def test_generated_regime_ignores_the_bank_fields():
    """Guards against over-correcting: the requirement is conditional on the
    regime, so a generated run with neither field must stay valid."""
    from sandbox.models import ExperimentRun

    assert ExperimentRun(**_run_kwargs(regime="generated")).bank_path is None


def test_unknown_bank_field_is_still_rejected():
    """extra="forbid" must keep covering the new block. `probe_mass_floor`
    mistyped as `probe_mass_flor` would otherwise load at the 0.5 default and
    silently accept probes the run meant to reject."""
    from sandbox.models import ExperimentRun

    with pytest.raises(ValidationError):
        ExperimentRun(**_run_kwargs(probe_mass_flor=0.9))


# --- RQ1: the bank-regime Interaction fields -----------------------------


def _interaction_kwargs(**overrides):
    base = dict(
        interaction_id="i1", run_id="r", turn=1, speaker_agent_id="agent_0000",
        neighbor_agent_ids=["agent_0001"], stance_before=4.0, stance_after=5.0,
        reason_text="because", content_type="generated_text", prompt_token_count=10,
        completion_token_count=5, model_backend_id="gpt-4o", latency_ms=1,
        api_call_status="success", timestamp_utc="2026-01-01T00:00:00Z",
    )
    base.update(overrides)
    return base


def test_bank_item_rows_are_accepted_and_default_their_probe_fields():
    from sandbox.models import Interaction

    row = Interaction(**_interaction_kwargs(content_type="bank_item"))
    assert row.items_read == [] and row.posted_item_id is None
    assert row.reply_text == ""
    assert row.probe_mean is None and row.probe_dist is None
    assert row.probe_mass is None and row.side_conf is None


def test_bank_item_row_rejects_a_meme_id():
    """content_type is what every analysis filters on, so the RQ3 meme
    mechanism and the RQ1 bank mechanism must stay distinguishable in the log."""
    from sandbox.models import Interaction

    with pytest.raises(ValidationError, match="bank_item"):
        Interaction(**_interaction_kwargs(content_type="bank_item", meme_id="meme_001"))


def test_turn_zero_is_allowed_for_the_round_zero_probe():
    from sandbox.models import Interaction, StanceRecord

    assert Interaction(**_interaction_kwargs(turn=0, content_type="bank_item")).turn == 0
    assert StanceRecord(turn=0, stance_value=4.0, reason_text="", interaction_id="i1").turn == 0


def test_negative_turn_is_still_rejected():
    from sandbox.models import Interaction

    with pytest.raises(ValidationError):
        Interaction(**_interaction_kwargs(turn=-1, content_type="bank_item"))


def test_round_zero_probe_may_carry_no_neighbours():
    """Nobody has been sampled yet at round 0, so the original
    Field(min_length=1) had to give way to a content-type-aware validator."""
    from sandbox.models import Interaction

    row = Interaction(**_interaction_kwargs(turn=0, content_type="bank_item", neighbor_agent_ids=[]))
    assert row.neighbor_agent_ids == []


@pytest.mark.parametrize("content_type, extra", [("generated_text", {}), ("meme", {"meme_id": "m1"})])
def test_non_bank_rows_still_require_at_least_one_neighbour(content_type, extra):
    """The guarantee the old min_length=1 provided, kept for the rows it
    protected: an empty neighbour list there means sampling silently produced
    nothing."""
    from sandbox.models import Interaction

    with pytest.raises(ValidationError, match="neighbor_agent_id"):
        Interaction(**_interaction_kwargs(content_type=content_type, neighbor_agent_ids=[], **extra))


def test_probe_fields_round_trip():
    from sandbox.models import Interaction

    dist = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
    row = Interaction(**_interaction_kwargs(
        content_type="bank_item", items_read=["i_01", "i_02"], posted_item_id="i_03",
        reply_text="a short reply", probe_mean=4.9, probe_dist=dist,
        probe_mass=0.93, side_conf=0.6,
    ))
    assert row.items_read == ["i_01", "i_02"]
    assert row.posted_item_id == "i_03"
    assert Interaction.model_validate_json(row.model_dump_json()).probe_dist == dist


@pytest.mark.parametrize("length", [0, 6, 8])
def test_probe_dist_must_cover_exactly_the_seven_scale_points(length):
    """A different support means probe_mean is not the mean of this
    distribution, and the two columns would disagree."""
    from sandbox.models import Interaction

    with pytest.raises(ValidationError, match="exactly 7"):
        Interaction(**_interaction_kwargs(content_type="bank_item", probe_dist=[0.1] * length))
