import pytest

from sandbox.seed_manager import SeedManager


def test_same_seed_produces_identical_neighbor_sampling_draws():
    a = SeedManager(seed=7)
    b = SeedManager(seed=7)
    draws_a = [a.neighbor_sampling_rng.random() for _ in range(20)]
    draws_b = [b.neighbor_sampling_rng.random() for _ in range(20)]
    assert draws_a == draws_b


def test_same_seed_produces_identical_meme_injection_draws():
    a = SeedManager(seed=99)
    b = SeedManager(seed=99)
    draws_a = [a.meme_injection_rng.random() for _ in range(20)]
    draws_b = [b.meme_injection_rng.random() for _ in range(20)]
    assert draws_a == draws_b


def test_three_streams_never_correlate():
    m = SeedManager(seed=1)
    neighbor_draws = [m.neighbor_sampling_rng.random() for _ in range(50)]
    meme_draws = [m.meme_injection_rng.random() for _ in range(50)]
    persona_draws = [m.persona_assignment_rng.random() for _ in range(50)]
    assert neighbor_draws != meme_draws
    assert neighbor_draws != persona_draws
    assert meme_draws != persona_draws


def test_property_returns_same_instance_by_identity():
    m = SeedManager(seed=5)
    assert m.neighbor_sampling_rng is m.neighbor_sampling_rng
    assert m.meme_injection_rng is m.meme_injection_rng
    assert m.persona_assignment_rng is m.persona_assignment_rng


def test_repeated_property_access_advances_stream_not_resets_it():
    m = SeedManager(seed=3)
    first = m.neighbor_sampling_rng.random()
    second = m.neighbor_sampling_rng.random()
    assert first != second


# --- phase4.md D8: per-turn streams survive a resume ---------------------


def test_turn_rng_is_a_pure_function_of_seed_purpose_and_turn():
    """The property that makes resume correct: turn t's draws do not depend
    on whether turns 1..t-1 ran in this process."""
    fresh = SeedManager(42).turn_rng("neighbor", 5)

    consumed = SeedManager(42)
    for turn in range(1, 5):  # burn turns 1-4 as a real run would
        consumed.turn_rng("neighbor", turn).random()
    resumed = consumed.turn_rng("neighbor", 5)

    assert [fresh.random() for _ in range(10)] == [resumed.random() for _ in range(10)]


def test_the_long_lived_streams_do_not_survive_a_resume():
    """Documents why turn_rng exists.

    The long-lived properties advance as turns consume them, and that
    position is not checkpointed, so a resumed run replays earlier draws.
    This asserts the broken behaviour deliberately: if a future change made
    the long-lived streams resume-safe, this test should be revisited rather
    than treated as a regression.
    """
    uninterrupted = SeedManager(42).neighbor_sampling_rng
    turn_one = [uninterrupted.random() for _ in range(5)]
    turn_two = [uninterrupted.random() for _ in range(5)]

    after_resume = SeedManager(42).neighbor_sampling_rng
    replayed = [after_resume.random() for _ in range(5)]

    assert replayed == turn_one       # it replays turn 1...
    assert replayed != turn_two       # ...instead of continuing into turn 2


def test_turn_streams_differ_across_turns_and_purposes():
    seeds = SeedManager(42)
    draws = {
        (purpose, turn): seeds.turn_rng(purpose, turn).random()
        for purpose in ("neighbor", "meme")
        for turn in (1, 2, 3)
    }
    assert len(set(draws.values())) == len(draws)


def test_turn_rng_preserves_fix_m_across_two_runs_sharing_a_seed():
    """Fix M: reusing a seed across language arms must hold neighbour
    sampling and meme injection constant while language varies."""
    english = SeedManager(2026)
    hinglish = SeedManager(2026)
    for turn in (1, 2, 3):
        for purpose in ("neighbor", "meme"):
            a = english.turn_rng(purpose, turn)
            b = hinglish.turn_rng(purpose, turn)
            assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_turn_rng_rejects_unknown_purpose_and_invalid_turn():
    seeds = SeedManager(1)
    with pytest.raises(ValueError, match="unknown rng purpose"):
        seeds.turn_rng("persona", 1)
    with pytest.raises(ValueError, match="turn must be"):
        seeds.turn_rng("neighbor", 0)


# --- RQ1: the argument and probe purposes --------------------------------


def test_the_new_purposes_are_accepted_and_distinct_from_the_old_ones():
    seeds = SeedManager(42)
    draws = {
        (purpose, turn): seeds.turn_rng(purpose, turn).random()
        for purpose in ("neighbor", "meme", "argument", "probe")
        for turn in (1, 2, 3)
    }
    assert len(set(draws.values())) == len(draws)


def test_graph_seed_is_deterministic_and_distinct_from_every_other_stream():
    """A graph built from the same integer a neighbour draw consumes would
    correlate structure with sampling, invisibly."""
    assert SeedManager(42).graph_seed == SeedManager(42).graph_seed
    assert SeedManager(42).graph_seed != SeedManager(43).graph_seed

    seeds = SeedManager(42)
    collisions = {seeds.graph_seed} & {
        42, 43, 44,  # the three long-lived stream offsets
        *(42 + off * turn for off in (1_000_003, 2_000_003, 3_000_003, 4_000_003)
          for turn in range(1, 50)),
    }
    assert not collisions


def test_sender_rng_is_a_pure_function_of_seed_purpose_and_sender():
    """No turn in the derivation: the bank's no-repeat guarantee depends on
    the draw being a function of how many items a sender has used on a side,
    not of which round it is posting in."""
    a = SeedManager(7).sender_rng("argument", "agent_0003")
    b = SeedManager(7).sender_rng("argument", "agent_0003")
    assert [a.random() for _ in range(10)] == [b.random() for _ in range(10)]


def test_sender_rng_differs_across_senders_purposes_and_seeds():
    draws = {
        key: SeedManager(seed).sender_rng(purpose, sender).random()
        for key, (seed, purpose, sender) in {
            "a": (7, "argument", "agent_0000"),
            "b": (7, "argument", "agent_0001"),
            "c": (7, "probe", "agent_0000"),
            "d": (8, "argument", "agent_0000"),
        }.items()
    }
    assert len(set(draws.values())) == 4


def test_sender_rng_rejects_an_unknown_purpose_and_an_empty_sender():
    seeds = SeedManager(1)
    with pytest.raises(ValueError, match="unknown rng purpose"):
        seeds.sender_rng("persona", "agent_0000")
    with pytest.raises(ValueError, match="non-empty"):
        seeds.sender_rng("argument", "")


# --- RQ1: variety must never enter a seed --------------------------------


def test_no_condition_can_reach_a_seed_through_the_constructor():
    """SeedManager takes the base seed and nothing else.

    Stated as a test because it is the structural reason the properties below
    hold: there is no language, topic or trial parameter for a derivation to
    accidentally mix in. If someone adds one, this fails first.
    """
    import inspect

    params = list(inspect.signature(SeedManager.__init__).parameters)
    assert params == ["self", "seed"]


def test_two_runs_differing_only_in_language_draw_identically():
    """RQ1's matched sets rest on this.

    The four language arms of one trial share a seed, so they must share a
    graph and an argument-draw sequence; only the rendering of the text may
    differ. A derivation that mixed language in would give each arm a
    different graph and a different item order, and the four arms would stop
    being the same experiment with one factor changed.
    """
    english, hindi = SeedManager(2026), SeedManager(2026)

    assert english.graph_seed == hindi.graph_seed

    for turn in (1, 2, 3):
        a = english.turn_rng("argument", turn)
        b = hindi.turn_rng("argument", turn)
        assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]

    for sender in ("agent_0000", "agent_0007"):
        a = english.sender_rng("argument", sender)
        b = hindi.sender_rng("argument", sender)
        assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]
