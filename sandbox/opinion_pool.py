"""Loads data/personas/pool_100.jsonl's format for the opinion-CSV experiment.

Despite the .jsonl extension this file is NOT JSON lines -- one free-text
descriptive sentence per agent, the same "plain text pool" convention
sandbox/agent_manager.py's load_persona_pool() already uses for
persona_attributes="none" pools, just with more attributes embedded in the
sentence. The only field this loader extracts structurally is "language
preference: ...", because it is the only one that drives program logic (see
opinion_orchestrator.resolve_languages). Everything else in the line is kept
verbatim as persona_text and handed to the LLM as descriptive context,
un-parsed -- deliberately, to avoid a brittle field-by-field parser for data
that is otherwise just flavour text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sandbox.opinion_models import LANGUAGES

_LANGUAGE_PATTERN = re.compile(r"language preference:\s*([A-Za-z ]+?)\s*(?:;|$)")

# pool_100.jsonl spells these with spaces/capitals; LANGUAGES is lowercase
# with underscores. One direction only -- normalizing the file's spelling
# to the config's, never the reverse.
_NORMALIZE = {
    "english": "english",
    "hindi": "hindi",
    "hinglish": "hinglish",
    "romanized hindi": "romanized_hindi",
}


@dataclass(frozen=True)
class PersonaRecord:
    agent_id: str
    persona_text: str
    language_preference: str  # one of LANGUAGES


def load_opinion_persona_pool(path: str | Path) -> list[PersonaRecord]:
    """Reads one PersonaRecord per non-blank line.

    Raises ValueError naming the 1-indexed line number if a line has no
    "language preference: ..." field, or its value doesn't normalize to one
    of LANGUAGES -- never silently drops a persona or guesses a language,
    per this codebase's "never silently default" convention.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"persona pool not found: {path}")

    records: list[PersonaRecord] = []
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        match = _LANGUAGE_PATTERN.search(line)
        if match is None:
            raise ValueError(
                f"{path}:{line_no}: no 'language preference: ...' field found"
            )

        raw_value = match.group(1).strip().lower()
        language = _NORMALIZE.get(raw_value)
        if language is None or language not in LANGUAGES:
            raise ValueError(
                f"{path}:{line_no}: unrecognized language preference "
                f"{match.group(1).strip()!r}; expected one of "
                f"{sorted(_NORMALIZE)}"
            )

        records.append(
            PersonaRecord(
                agent_id=f"agent_{line_no - 1}",
                persona_text=line,
                language_preference=language,
            )
        )

    if not records:
        raise ValueError(f"{path}: no personas found (file is empty)")

    return records
