"""Abstract-level extraction. Rule-only PICO scaffolding; LLM refines later."""

from __future__ import annotations

import re

from weighted_evidence.models import PICO

_SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "background": re.compile(r"^(background|introduction)\s*[:\-]?\s*", re.I | re.M),
    "methods": re.compile(r"^(methods?|design)\s*[:\-]?\s*", re.I | re.M),
    "results": re.compile(r"^(results?|findings)\s*[:\-]?\s*", re.I | re.M),
    "conclusions": re.compile(r"^(conclusions?|interpretation)\s*[:\-]?\s*", re.I | re.M),
}


def split_structured(abstract: str) -> dict[str, str]:
    """Split a NLM-style structured abstract into labeled sections.

    Falls back to a single `body` section when no structure is detected.
    """

    if not abstract:
        return {}

    lines = abstract.splitlines()
    sections: dict[str, list[str]] = {}
    current = "body"
    sections[current] = []
    for line in lines:
        lowered = line.strip().lower()
        matched = False
        for label in _SECTION_PATTERNS:
            if lowered.startswith(f"{label}:") or lowered.startswith(f"{label} -"):
                current = label
                sections.setdefault(current, [])
                content = line.split(":", 1)[1] if ":" in line else line
                sections[current].append(content.strip())
                matched = True
                break
        if not matched:
            sections.setdefault(current, []).append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


_SAMPLE_SIZE_RE = re.compile(
    r"\b(?:n\s*=\s*|enrolled\s+|randomi[sz]ed\s+|included\s+)(\d{2,6})\b",
    re.I,
)


def extract_sample_size(abstract: str) -> int | None:
    if not abstract:
        return None
    match = _SAMPLE_SIZE_RE.search(abstract)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_pico_naive(abstract: str) -> PICO | None:
    """Coarse PICO from a structured abstract.

    Intentionally minimal: returns whatever section bodies map cleanly to PICO.
    LLM extraction in `llm/parsers.py` handles the hard cases.
    """

    if not abstract:
        return None
    sections = split_structured(abstract)
    if not sections:
        return None
    methods = sections.get("methods", "")
    background = sections.get("background", "")
    return PICO(
        population=_first_sentence(methods) or _first_sentence(background),
        intervention=None,
        comparator=None,
        outcomes=[],
    )


def _first_sentence(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0] if parts else None
