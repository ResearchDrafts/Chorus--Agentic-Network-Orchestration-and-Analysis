"""Script compliance for the opinion-CSV pipeline.

The behaviour these pin was measured, not imagined: in
opinion_colab_run4's romanized_hindi arm, agent_0 replied in Roman at
rounds 0 and 1, flipped to Devanagari at round 2 with an all-Roman feed,
and never came back. The absorbing half of that -- a wrong-script reply
being stored, re-shown to its author and delivered to its neighbours --
is what these tests exist to prevent.
"""

import pytest

from sandbox.opinion_script import (
    can_repair,
    devanagari_share,
    enforce,
    script_violation,
    to_roman,
)

# agent_0's actual round-2 reply from that run.
REAL_DEVANAGARI = (
    "मुझे उन लोगों के विचारों से ज़्यादा अभिव्यक्ति की स्वतंत्रता का महत्व "
    "समझने में रुचि आई है, लेकिन उन दो लोगों के विचार ने मुझे खुशी दी।"
)
REAL_ROMAN = (
    "Abhivyakti ki swatantrata ko virodhabhas se virodh karne wale ek bhi "
    "nahi chahiye. Yeh ek adhikar hai."
)


@pytest.mark.parametrize("arm", ["english", "hinglish", "romanized_hindi"])
def test_devanagari_is_a_violation_in_every_roman_arm(arm):
    assert script_violation(REAL_DEVANAGARI, arm) is not None
    assert script_violation(REAL_ROMAN, arm) is None


def test_the_hindi_arm_wants_devanagari_and_tolerates_english_loanwords():
    # "free speech" and "social media" are ordinary Hindi text. Judging the
    # hindi arm on the ABSENCE of Latin would reject correct replies, so it
    # is judged on the presence of Devanagari instead.
    assert script_violation("मुझे लगता है कि free speech एक अधिकार है।", "hindi") is None
    assert script_violation(REAL_ROMAN, "hindi") is not None


def test_an_unknown_language_raises_rather_than_being_exempted():
    # Never silently default: an arm this module has not been taught about
    # must not pass the check by not being in a set.
    with pytest.raises(ValueError, match="no script requirement"):
        script_violation(REAL_ROMAN, "tamil")


def test_romanizing_removes_every_devanagari_character():
    out = to_roman(REAL_DEVANAGARI)
    assert script_violation(out, "romanized_hindi") is None
    assert devanagari_share(out) == 0.0


def test_romanizing_leaves_latin_and_the_measurement_lines_untouched():
    mixed = "BELIEF: -1.5\nCONFIDENCE: 4\nMain free speech ko मैंने देखा kaha."
    out = to_roman(mixed)
    assert out.startswith("BELIEF: -1.5\nCONFIDENCE: 4\n")
    assert "free speech" in out
    assert script_violation(out, "romanized_hindi") is None


def test_word_final_schwa_is_deleted_but_single_syllables_survive():
    # "kara sakataa" is not how anyone writes कर सकता -- and not how the
    # model's own complying replies in this experiment write it either.
    assert to_roman("कर") == "kar"
    assert to_roman("विचार") == "vichaar"
    assert to_roman("न") == "na"     # one syllable: deleting leaves nothing
    assert to_roman("है") == "hai"   # matra, not an inherent schwa


def test_romanizing_is_idempotent():
    once = to_roman(REAL_DEVANAGARI)
    assert to_roman(once) == once


def test_enforce_repairs_a_roman_arm_and_reports_it():
    text, repaired = enforce(REAL_DEVANAGARI, "romanized_hindi")
    assert repaired is True
    assert script_violation(text, "romanized_hindi") is None


def test_enforce_is_a_no_op_on_compliant_text():
    text, repaired = enforce(REAL_ROMAN, "hinglish")
    assert repaired is False
    assert text == REAL_ROMAN


def test_a_latin_hindi_reply_is_reported_but_never_guessed_at():
    # Latin -> Devanagari is ambiguous (sakta/shakta) and English is not
    # Hindi at all, so this one is flagged and kept, never invented.
    assert can_repair("hindi") is False
    text, repaired = enforce(REAL_ROMAN, "hindi")
    assert repaired is False
    assert text == REAL_ROMAN
