import random
from pathlib import Path

import pytest

from sandbox.agent_manager import AgentManager
from tests.factories import make_interaction as _make_interaction, make_run as _make_run


def _by_id(agents: list, agent_id: str):
    return next(a for a in agents if a.agent_id == agent_id)


def test_initialize_population_cycles_persona_pool():
    run = _make_run(M=100, persona_pool_id="pool_20")
    manager = AgentManager(run, random.Random(run.seed))
    agents = manager.initialize_population()

    assert len(agents) == 100
    pool = [line.strip() for line in Path("data/personas/pool_20.jsonl").read_text().splitlines() if line.strip()]
    assert len(pool) == 20
    for i, agent in enumerate(agents):
        assert agent.persona == pool[i % 20]


def test_identically_seeded_rng_produces_identical_initial_stance_sequences():
    run = _make_run()
    agents1 = AgentManager(run, random.Random(run.seed)).initialize_population()
    agents2 = AgentManager(run, random.Random(run.seed)).initialize_population()

    assert [a.initial_stance for a in agents1] == [a.initial_stance for a in agents2]


def test_get_agent_unknown_id_raises_key_error_uncaught():
    run = _make_run(M=100)
    manager = AgentManager(run, random.Random(run.seed))
    manager.initialize_population()

    with pytest.raises(KeyError):
        manager.get_agent("agent_9999")


def test_missing_persona_pool_raises_file_not_found_at_initialize_time():
    run = _make_run(persona_pool_id="does_not_exist")
    manager = AgentManager(run, random.Random(run.seed))

    with pytest.raises(FileNotFoundError):
        manager.initialize_population()


def test_apply_interaction_appends_history_and_caps_memory_window_at_five():
    run = _make_run(M=2, N=1)
    manager = AgentManager(run, random.Random(run.seed))
    manager.initialize_population()
    agent_id = "agent_0000"

    for turn in range(1, 8):  # 7 turns, more than the cap of 5
        manager.apply_interaction(
            agent_id,
            _make_interaction(agent_id, turn=turn, stance_after=float(turn), interaction_id=f"int_{turn}"),
        )

    agent = manager.get_agent(agent_id)
    assert len(agent.stance_history) == 7
    assert len(agent.memory_window) == 5
    assert agent.memory_window == [f"int_{t}" for t in range(3, 8)]
    assert agent.current_stance() == 7.0


def test_snapshot_is_unaffected_by_later_mutation():
    run = _make_run(M=2, N=1)
    manager = AgentManager(run, random.Random(run.seed))
    manager.initialize_population()

    snap = manager.snapshot()
    manager.apply_interaction("agent_0000", _make_interaction("agent_0000"))

    assert _by_id(snap, "agent_0000").stance_history == []
    assert len(manager.get_agent("agent_0000").stance_history) == 1


def test_restore_from_snapshot_reproduces_agent_state_exactly():
    run = _make_run(M=2, N=1)
    manager = AgentManager(run, random.Random(run.seed))
    manager.initialize_population()

    manager.apply_interaction("agent_0000", _make_interaction("agent_0000", turn=1, interaction_id="int_1"))
    snap = manager.snapshot()

    manager.apply_interaction("agent_0000", _make_interaction("agent_0000", turn=2, interaction_id="int_2"))
    assert len(manager.get_agent("agent_0000").stance_history) == 2

    manager.restore_from_snapshot(snap)
    restored = manager.get_agent("agent_0000")
    assert len(restored.stance_history) == 1
    assert restored.stance_history[0].interaction_id == "int_1"

    # mutating the manager after restore must not affect the snapshot list passed in
    manager.apply_interaction("agent_0000", _make_interaction("agent_0000", turn=3, interaction_id="int_3"))
    assert len(_by_id(snap, "agent_0000").stance_history) == 1


# --- RQ1: the bank regime -------------------------------------------------


def _bank_run(**overrides):
    base = dict(
        M=3, N=1, regime="bank", bank_path="data/bank/test_bank.csv",
        probe_question="Should the fare subsidy continue?",
    )
    base.update(overrides)
    return _make_run(**base)


def _bank_interaction(agent_id="agent_0000", turn=1, **overrides):
    base = dict(
        content_type="bank_item",
        stance_after=4.0,          # must be ignored in favour of probe_mean
        reason_text="the reply",
        reply_text="the reply",
        probe_mean=5.5,
        probe_dist=[0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
        probe_mass=0.9,
        side_conf=1.0,
        items_read=["bank_t01", "bank_t03"],
        posted_item_id="bank_t02",
    )
    base.update(overrides)
    return _make_interaction(agent_id, turn=turn, **base)


def test_bank_regime_records_the_probe_mean_as_the_stance():
    """A bank agent's stance is what the probe measured, not a number it
    wrote: nothing in a bank reply is parsed for a stance at all."""
    manager = AgentManager(_bank_run(), random.Random(1))
    manager.initialize_population()

    manager.apply_interaction("agent_0000", _bank_interaction(probe_mean=6.25))
    agent = manager.get_agent("agent_0000")

    assert agent.stance_history[-1].stance_value == 6.25
    assert agent.current_stance() == 6.25


def test_bank_regime_records_the_reply_as_the_reason_text():
    manager = AgentManager(_bank_run(), random.Random(1))
    manager.initialize_population()

    manager.apply_interaction(
        "agent_0000", _bank_interaction(reply_text="the cost point lands", reason_text="mirrored")
    )
    assert manager.get_agent("agent_0000").stance_history[-1].reason_text == "the cost point lands"


def test_the_round_zero_probe_is_applied_as_a_turn_zero_record():
    """Round 0 measures every agent's starting stance before it reads
    anything, so the dataset carries it rather than leaving it to be inferred
    from initial_stance."""
    manager = AgentManager(_bank_run(), random.Random(1))
    manager.initialize_population()

    manager.apply_interaction(
        "agent_0000",
        _bank_interaction(
            turn=0, probe_mean=3.5, reply_text="", reason_text="",
            items_read=[], posted_item_id=None, neighbor_agent_ids=[],
        ),
    )
    record = manager.get_agent("agent_0000").stance_history[0]
    assert record.turn == 0
    assert record.stance_value == 3.5
    assert record.reason_text == ""


def test_a_bank_row_without_a_probe_mean_raises_rather_than_falling_back():
    """A row with no probe is a row whose probe failed, and Fix I says such a
    row is logged and never applied. Arriving here is an orchestrator bug, and
    a fallback to stance_after would hide it by writing the agent's previous
    stance into history as though it had been measured again."""
    manager = AgentManager(_bank_run(), random.Random(1))
    manager.initialize_population()

    with pytest.raises(ValueError, match="no probe_mean"):
        manager.apply_interaction(
            "agent_0000",
            _bank_interaction(probe_mean=None, api_call_status="failed_logged_null"),
        )


def test_the_generated_regime_still_reads_stance_after():
    """Same method, different regime. A generated row has no probe_mean at
    all, and must keep being applied exactly as before."""
    manager = AgentManager(_make_run(M=3, N=1), random.Random(1))
    manager.initialize_population()

    manager.apply_interaction(
        "agent_0000", _make_interaction("agent_0000", turn=1, stance_after=6.0)
    )
    assert manager.get_agent("agent_0000").current_stance() == 6.0


def test_bank_history_still_feeds_the_memory_window_and_its_cap():
    manager = AgentManager(_bank_run(), random.Random(1))
    manager.initialize_population()

    for turn in range(0, 8):
        manager.apply_interaction(
            "agent_0000",
            _bank_interaction(turn=turn, probe_mean=float(1 + turn % 7),
                              interaction_id=f"probe_{turn}"),
        )

    agent = manager.get_agent("agent_0000")
    assert len(agent.stance_history) == 8
    assert agent.memory_window == [f"probe_{t}" for t in range(3, 8)]


def test_bank_state_survives_a_snapshot_round_trip():
    """Resume reads the checkpoint, so probe means have to survive it."""
    manager = AgentManager(_bank_run(), random.Random(1))
    manager.initialize_population()
    manager.apply_interaction("agent_0000", _bank_interaction(probe_mean=2.75))

    snapshot = manager.snapshot()
    restored = AgentManager(_bank_run(), random.Random(1))
    restored.restore_from_snapshot(snapshot)

    assert restored.get_agent("agent_0000").current_stance() == 2.75


def test_initialize_population_is_unchanged_by_the_regime():
    """initial_stance is drawn the same way; the bank regime only reads it for
    a different purpose (the lean sentence, in words)."""
    generated = AgentManager(_make_run(M=10, N=3), random.Random(7)).initialize_population()
    bank = AgentManager(_bank_run(M=10, N=3), random.Random(7)).initialize_population()

    assert [a.initial_stance for a in generated] == [a.initial_stance for a in bank]
    assert [a.persona for a in generated] == [a.persona for a in bank]
