"""Script compliance for the opinion-CSV experiment: detect a reply written
in the wrong script, and romanize one that is.

Why this exists, from measured behaviour rather than from theory. In the
romanized_hindi arm of opinion_colab_run4 (Qwen2.5-7B, M=100), agent_0's
replies read:

    round 0  Roman        round 1  Roman        round 2  DEVANAGARI
    round 3  Devanagari   round 4  Devanagari   round 5  Devanagari

and agent_1's flipped at round 3 and stayed flipped. Two things about that
shape matter:

1. Round 0 is clean, and round 0 carries the full persona. The persona's own
   "language preference: ..." field is NOT what causes the drift -- the
   hindi arm's agent_0 has "language preference: Romanized Hindi" in its
   persona and wrote correct Devanagari all run. The directive beats the
   persona; both arms show it.
2. agent_0 slipped in round 2 while every post in its feed that round was
   still Roman. The first slip is spontaneous -- one sample out of a
   temperature-0.7 draw against a directive that is ~2% of the prompt --
   not something it caught from a neighbour.

What turns a rare slip into half the population is that **a slip is
absorbing**. build_update_prompt feeds an agent its own previous reply
("Your current opinion: ...") and shows that reply to up to N neighbours. So
Roman -> Devanagari costs one unlucky sample, while Devanagari -> Roman
requires overriding both the model's prior for "Hindi" and the agent's own
visible history. The share can only climb.

Strengthening the directive lowers the per-round slip rate. It cannot make
the state non-absorbing, because nothing in a prompt is a guarantee. This
module is the part that does: a reply in the wrong script is retried, and if
it is still wrong it is romanized before it is ever stored or shown to
anyone. No agent ever reads a script the arm did not ask for, so one slip
stays one slip instead of seeding the population.

The romanization is ITRANS-flavoured and deliberately simple: it is a
transliteration, never a translation, and it only ever rewrites Devanagari
runs, leaving Latin text (English loanwords, the BELIEF line) untouched.
"""

from __future__ import annotations

import re

# U+0900-U+097F. Also catches the Devanagari Extended/vedic blocks' common
# members via the nukta forms below; anything outside is not something a
# chat model produces for Hindi.
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# Which script each arm's replies must be in. romanized_hindi, hinglish and
# english all mean "Latin letters"; only hindi means Devanagari. Keyed by
# the same strings as opinion_models.LANGUAGES.
ROMAN_ARMS = frozenset({"english", "hinglish", "romanized_hindi"})
DEVANAGARI_ARMS = frozenset({"hindi"})

# A hindi reply is judged on whether Devanagari is actually present, not on
# whether Latin is absent: "free speech" and "social media" are ordinary
# Hindi text, and banning Latin outright would reject correct replies.
_MIN_DEVANAGARI_SHARE = 0.20


def devanagari_share(text: str) -> float:
    """Devanagari characters as a share of all characters. 0.0 for empty."""
    if not text:
        return 0.0
    return len(DEVANAGARI_RE.findall(text)) / len(text)


def script_violation(text: str, language: str) -> str | None:
    """None if `text` is in the script `language` requires, else a short
    reason naming what was found. Raises for an unknown language rather than
    passing it: an arm this module has not been taught about must not be
    silently exempted from the check.
    """
    if language in ROMAN_ARMS:
        found = len(DEVANAGARI_RE.findall(text))
        if found:
            return f"{found} Devanagari characters in a {language} reply"
        return None
    if language in DEVANAGARI_ARMS:
        share = devanagari_share(text)
        if share < _MIN_DEVANAGARI_SHARE:
            return (
                f"only {share:.0%} Devanagari characters in a {language} "
                f"reply (expected at least {_MIN_DEVANAGARI_SHARE:.0%})"
            )
        return None
    raise ValueError(f"no script requirement defined for language {language!r}")


def can_repair(language: str) -> bool:
    """True where a violating reply can be mechanically fixed.

    Devanagari -> Latin is a transliteration this module performs. Latin ->
    Devanagari is not: romanized Hindi is ambiguous (`sakta`/`shakta`) and
    English is not Hindi at all, so a hindi-arm reply that came back in
    Latin is reported and kept as-is rather than guessed at.
    """
    return language in ROMAN_ARMS


_INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ॠ": "rii", "ऌ": "li", "ए": "e", "ऐ": "ai", "ओ": "o",
    "औ": "au", "ऍ": "e", "ऑ": "o", "ऎ": "e", "ऒ": "o",
}

_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri",
    "ॄ": "rii", "ॢ": "li", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॅ": "e", "ॉ": "o", "ॆ": "e", "ॊ": "o",
}

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # Precomposed nukta forms. The base+U+093C sequences are folded onto
    # these by _NUKTA below, so both spellings romanize identically.
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}

_NUKTA = {
    "क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y",
}

_SIGNS = {
    "ं": "n", "ँ": "n", "ः": "h", "ॐ": "om",
    "।": ".", "॥": ".", "॰": ".",
    "़": "",  # a stray nukta with no consonant before it
    "ऽ": "",  # avagraha
}

_DIGITS = {d: str(i) for i, d in enumerate("०१२३४५६७८९")}

_VIRAMA = "्"


def to_roman(text: str) -> str:
    """Transliterate every Devanagari run in `text` to Latin, in place.

    Non-Devanagari characters pass through untouched, so punctuation,
    English loanwords and the BELIEF/CONFIDENCE lines survive exactly.
    Each consonant carries its inherent 'a' unless a virama or a matra
    follows -- the one rule that makes the output readable rather than a
    string of bare consonants.

    Word-final schwa deletion is applied on top of that, because Hindi has
    it and a transliteration without it reads as something no speaker
    writes: "kara sakataa" for कर सकता, where the model's own romanized
    replies in this very experiment say "kar sakta". Repaired text is fed
    back into later prompts and read by the analysis, so it should look
    like what the arm produces when it complies, not like a gloss. Only
    word-FINAL, and only where the word keeps at least two vowels, so
    single syllables (न -> "na") stay pronounceable.
    """
    out: list[str] = []
    # Index in `out` of the inherent 'a' most recently emitted, and how many
    # vowels the current Devanagari word has emitted. Both reset at every
    # word boundary; -1 means "nothing to delete".
    last_inherent = -1
    word_vowels = 0

    def _end_word() -> None:
        nonlocal last_inherent, word_vowels
        if last_inherent >= 0 and word_vowels >= 2:
            out[last_inherent] = ""
        last_inherent = -1
        word_vowels = 0

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        # Consonant: base letter, then whatever decides its vowel.
        cons = None
        if ch in _CONSONANTS:
            cons = _CONSONANTS[ch]
            i += 1
            # base + U+093C spelled out, rather than the precomposed form
            if i < n and text[i] == "़" and ch in _NUKTA:
                cons = _NUKTA[ch]
                i += 1
        if cons is not None:
            out.append(cons)
            if i < n and text[i] == _VIRAMA:
                i += 1                      # dead consonant, no vowel
                last_inherent = -1          # a virama is not a deletable schwa
            elif i < n and text[i] in _MATRAS:
                out.append(_MATRAS[text[i]])
                word_vowels += 1
                last_inherent = -1
                i += 1
            else:
                last_inherent = len(out)    # deletable if the word ends here
                out.append("a")             # inherent vowel
                word_vowels += 1
            continue

        if ch in _INDEPENDENT_VOWELS:
            out.append(_INDEPENDENT_VOWELS[ch])
            word_vowels += 1
            last_inherent = -1
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])         # orphan matra, keep the sound
            word_vowels += 1
            last_inherent = -1
        elif ch in _SIGNS:
            # An anusvara/visarga after the schwa is still inside the word,
            # and it is a consonant sound: the schwa before it is not final.
            out.append(_SIGNS[ch])
            last_inherent = -1
        elif ch in _DIGITS:
            _end_word()
            out.append(_DIGITS[ch])
        elif ch == _VIRAMA:
            pass                            # orphan virama
        else:
            _end_word()                     # not Devanagari: word is over
            out.append(ch)                  # and the character is untouched
        i += 1

    _end_word()
    return "".join(out)


def enforce(text: str, language: str) -> tuple[str, bool]:
    """(text, was_repaired). Romanizes a Devanagari reply in a Roman arm;
    returns a hindi-arm violation unchanged, since can_repair is False there.
    """
    if script_violation(text, language) is None:
        return text, False
    if not can_repair(language):
        return text, False
    return to_roman(text), True
