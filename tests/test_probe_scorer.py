"""Tests for probe_scorer.py.

Pure arithmetic, so these are hand-computed rather than property-based: the
numbers below were worked out by hand and are what the module must reproduce,
not what it happens to produce.
"""

from __future__ import annotations

import math

import pytest

from sandbox.probe_scorer import SCALE_POINTS, ProbeMassTooLow, score


def _logprobs(probabilities: dict[str, float]) -> list[tuple[str, float]]:
    """Builds a provider-shaped top-logprobs list from plain probabilities."""
    return [(token, math.log(p)) for token, p in probabilities.items()]


# --- the hand-computed example -------------------------------------------


def test_a_hand_computed_example():
    """Worked by hand:

        raw        5 -> 0.5, 6 -> 0.3, 2 -> 0.1, " the" -> 0.1
        mass       0.5 + 0.3 + 0.1                      = 0.9
        dist       2 -> 0.1/0.9, 5 -> 0.5/0.9, 6 -> 0.3/0.9
        mean       (2*0.1 + 5*0.5 + 6*0.3) / 0.9
                   = (0.2 + 2.5 + 1.8) / 0.9 = 4.5/0.9  = 5.0
        side_conf  mean > 4, so P(s>4)/mass = 0.8/0.9   = 0.888...
    """
    result = score(
        _logprobs({"5": 0.5, "6": 0.3, "2": 0.1, " the": 0.1}), mass_floor=0.5
    )

    assert result.mass == pytest.approx(0.9)
    assert result.mean == pytest.approx(5.0)
    assert result.side_conf == pytest.approx(0.8 / 0.9)
    assert result.dist == pytest.approx(
        [0.0, 0.1 / 0.9, 0.0, 0.0, 0.5 / 0.9, 0.3 / 0.9, 0.0]
    )
    assert sum(result.dist) == pytest.approx(1.0)


def test_dist_is_always_seven_long_and_normalized():
    result = score(_logprobs({"1": 0.4, "7": 0.4, " x": 0.2}), mass_floor=0.5)
    assert len(result.dist) == len(SCALE_POINTS) == 7
    assert sum(result.dist) == pytest.approx(1.0)
    assert result.mean == pytest.approx(4.0)  # 1 and 7 in equal measure


def test_all_mass_on_one_digit_gives_that_digit_exactly():
    for point in SCALE_POINTS:
        result = score(_logprobs({str(point): 1.0}), mass_floor=0.5)
        assert result.mean == pytest.approx(float(point))
        assert result.mass == pytest.approx(1.0)


# --- the space-prefixed token case ---------------------------------------


def test_space_prefixed_tokens_are_read_as_their_digit():
    """Tokenizers differ on leading whitespace. Without this, half the
    providers would appear to put no mass on the scale at all and every probe
    would fail the floor."""
    result = score(_logprobs({" 3": 0.8, " and": 0.2}), mass_floor=0.5)
    assert result.mean == pytest.approx(3.0)
    assert result.mass == pytest.approx(0.8)


def test_duplicate_spellings_of_the_same_digit_are_summed():
    """" 3" and "3" are one answer, not two."""
    result = score(_logprobs({"3": 0.4, " 3": 0.4, "x": 0.2}), mass_floor=0.5)
    assert result.mass == pytest.approx(0.8)
    assert result.mean == pytest.approx(3.0)
    assert result.dist[2] == pytest.approx(1.0)


@pytest.mark.parametrize("token", ["\n5", "5 ", " 5 ", "\t5"])
def test_other_surrounding_whitespace_is_also_stripped(token):
    assert score(_logprobs({token: 0.9, "x": 0.1}), mass_floor=0.5).mean == pytest.approx(5.0)


@pytest.mark.parametrize("token", ["0", "8", "9", "10", "15", "five", "", " ", "3.5", "-3"])
def test_off_scale_tokens_contribute_nothing_but_still_cost_mass(token):
    """A "0" or an "8" is the model answering on a different scale, and a
    multi-digit token is it starting a longer number. Neither is a point on
    this scale, and both are probability spent not answering."""
    result = score(_logprobs({"4": 0.6, token: 0.4}), mass_floor=0.5)
    assert result.mass == pytest.approx(0.6)
    assert result.mean == pytest.approx(4.0)


# --- the low-mass error ---------------------------------------------------


def test_mass_below_the_floor_raises_and_never_clamps():
    """A refusal to measure, not a measurement to paper over. Renormalizing
    0.2 of probability would manufacture a confident stance out of scraps, and
    the fabricated value would look exactly like a real one in the output."""
    with pytest.raises(ProbeMassTooLow) as excinfo:
        score(_logprobs({"5": 0.1, "6": 0.1, " I": 0.5, " cannot": 0.3}), mass_floor=0.5)

    assert excinfo.value.mass == pytest.approx(0.2)
    assert excinfo.value.mass_floor == 0.5
    assert "0.2" in str(excinfo.value)


def test_no_mass_on_the_scale_at_all_raises():
    with pytest.raises(ProbeMassTooLow) as excinfo:
        score(_logprobs({"I": 0.6, " think": 0.4}), mass_floor=0.5)
    assert excinfo.value.mass == 0.0


def test_mass_exactly_at_the_floor_is_accepted():
    """The comparison is `mass < floor`, so the boundary counts as a read.
    Guards against an off-by-one that would reject half the probes in a run
    configured with a floor at the typical mass."""
    result = score(_logprobs({"5": 0.5, " x": 0.5}), mass_floor=0.5)
    assert result.mass == pytest.approx(0.5)
    assert result.mean == pytest.approx(5.0)


def test_a_strict_floor_rejects_what_a_lenient_one_accepts():
    probabilities = {"5": 0.7, " x": 0.3}
    assert score(_logprobs(probabilities), mass_floor=0.5).mean == pytest.approx(5.0)
    with pytest.raises(ProbeMassTooLow):
        score(_logprobs(probabilities), mass_floor=0.9)


@pytest.mark.parametrize("mass_floor", [0.0, -0.1, 1.5])
def test_an_invalid_floor_raises_rather_than_being_coerced(mass_floor):
    with pytest.raises(ValueError, match="mass_floor"):
        score(_logprobs({"4": 1.0}), mass_floor=mass_floor)


def test_an_empty_logprobs_list_is_distinguished_from_low_mass():
    """Nothing to read at all is a gateway or wiring problem, not a model that
    answered off-scale, so it must not arrive as ProbeMassTooLow."""
    with pytest.raises(ValueError, match="nothing to score"):
        score([], mass_floor=0.5)


# --- the exactly-4 case ---------------------------------------------------


def test_mean_exactly_four_reports_the_probability_of_four_itself():
    """At the midpoint there is no side to be confident about, so side_conf
    becomes P(s = 4)."""
    result = score(_logprobs({"4": 0.9, " x": 0.1}), mass_floor=0.5)
    assert result.mean == pytest.approx(4.0)
    assert result.side_conf == pytest.approx(1.0)


def test_the_two_distributions_that_both_mean_four_are_distinguished():
    """The case side_conf exists for: a read split hard between 1 and 7 and
    one piled entirely on 4 have the same mean and mean opposite things."""
    undecided = score(_logprobs({"4": 1.0}), mass_floor=0.5)
    polarized = score(_logprobs({"1": 0.5, "7": 0.5}), mass_floor=0.5)

    assert undecided.mean == polarized.mean == pytest.approx(4.0)
    assert undecided.side_conf == pytest.approx(1.0)
    assert polarized.side_conf == pytest.approx(0.0)


def test_side_conf_uses_the_tail_the_mean_falls_on():
    # mean below 4: P(s < 4).
    low = score(_logprobs({"2": 0.7, "6": 0.3}), mass_floor=0.5)
    assert low.mean == pytest.approx(2 * 0.7 + 6 * 0.3)  # 3.2
    assert low.side_conf == pytest.approx(0.7)

    # mean above 4: P(s > 4).
    high = score(_logprobs({"2": 0.3, "6": 0.7}), mass_floor=0.5)
    assert high.mean == pytest.approx(2 * 0.3 + 6 * 0.7)  # 4.8
    assert high.side_conf == pytest.approx(0.7)


def test_side_conf_excludes_the_midpoint_from_both_tails():
    """P(s>4) and P(s<4) are strict, so mass sitting on 4 belongs to neither
    and a mean just off the midpoint does not report near-certainty."""
    result = score(_logprobs({"4": 0.8, "5": 0.2}), mass_floor=0.5)
    assert result.mean == pytest.approx(4.2)
    assert result.side_conf == pytest.approx(0.2)  # not 1.0


# --- shape ----------------------------------------------------------------


def test_the_result_is_immutable():
    """A probe read is a measurement. Nothing downstream should be able to
    edit one in place and have the edit reach the log."""
    result = score(_logprobs({"5": 1.0}), mass_floor=0.5)
    with pytest.raises(Exception):
        result.mean = 1.0


def test_mass_is_the_raw_share_not_the_renormalized_one():
    """mass is a quality measure, not a stance: a mean of 5.0 read off 0.95
    mass and the same mean read off 0.51 are not equally trustworthy, and only
    this column distinguishes them."""
    confident = score(_logprobs({"5": 0.95, " x": 0.05}), mass_floor=0.5)
    marginal = score(_logprobs({"5": 0.51, " x": 0.49}), mass_floor=0.5)

    assert confident.mean == marginal.mean == pytest.approx(5.0)
    assert confident.mass == pytest.approx(0.95)
    assert marginal.mass == pytest.approx(0.51)


# --- digits int() cannot read --------------------------------------------


def test_a_subscript_or_superscript_digit_in_the_top_k_is_ignored():
    """A live run crashed on this. "₁" and "¹" satisfy str.isdigit but
    int() cannot convert them, and the model does not have to answer one --
    it only has to appear somewhere in the top-k with a sliver of probability.
    They are off-scale tokens like any other, and count against the mass."""
    result = score(
        _logprobs({"5": 0.9, "₁": 0.04, "¹": 0.03, "⑦": 0.03}),
        mass_floor=0.5,
    )

    assert result.mass == pytest.approx(0.9)
    assert result.mean == pytest.approx(5.0)


def test_a_devanagari_digit_counts_as_its_scale_point():
    """Deliberate. A Hindi-arm answer of "४" is an answer of 4; dropping
    it would lower the probe mass in that arm only, which is a difference in
    the instrument between arms."""
    result = score(_logprobs({"४": 0.6, "4": 0.2, "6": 0.2}), mass_floor=0.5)

    assert result.mass == pytest.approx(1.0)
    assert result.dist[SCALE_POINTS.index(4)] == pytest.approx(0.8)


def test_a_devanagari_digit_off_the_scale_is_ignored():
    result = score(_logprobs({"९": 0.3, "2": 0.7}), mass_floor=0.5)  # nine

    assert result.mass == pytest.approx(0.7)
