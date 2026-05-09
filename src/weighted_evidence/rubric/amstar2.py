"""AMSTAR-2 skeleton for systematic reviews / meta-analyses.

Rule-based items (PROSPERO presence, reporting completeness markers) plus a
shape that the LLM step in PR 7 can refine. Phase 1 keeps the output
honest: when we don't have enough signal to grade an item, we mark `met=False`
with a clear rationale.
"""

from __future__ import annotations

import re

from weighted_evidence.models import Amstar2Assessment, Amstar2Item, Amstar2Rating, Paper

_ITEMS: tuple[tuple[int, str], ...] = (
    (1, "Did the research questions and inclusion criteria include PICO?"),
    (2, "Did the report reference an explicit a-priori protocol (e.g., PROSPERO)?"),
    (3, "Did the authors explain study design selection?"),
    (4, "Did the authors use a comprehensive literature search?"),
    (5, "Did the authors perform study selection in duplicate?"),
    (6, "Did the authors perform data extraction in duplicate?"),
    (7, "Did the authors provide a list of excluded studies and justify exclusions?"),
    (8, "Did the authors describe the included studies in adequate detail?"),
    (9, "Did the authors use a satisfactory technique for assessing RoB?"),
    (10, "Did the authors report sources of funding for included studies?"),
    (11, "Did the authors use appropriate methods for statistical combination?"),
    (12, "Did the authors assess the potential impact of RoB on combined results?"),
    (13, "Did the authors account for RoB when interpreting results?"),
    (14, "Did the authors provide a satisfactory explanation for heterogeneity?"),
    (15, "Did the authors investigate publication bias?"),
    (16, "Did the authors report any conflict of interest?"),
)

_PROSPERO_RE = re.compile(r"\bPROSPERO\s*(?:CRD\d+)?\b", re.I)
_DUPLICATE_RE = re.compile(r"\b(?:in\s+duplicate|two\s+(?:independent\s+)?reviewers)\b", re.I)
_RISK_OF_BIAS_RE = re.compile(r"\b(?:risk\s+of\s+bias|RoB|Cochrane\s+tool)\b", re.I)
_PUBLICATION_BIAS_RE = re.compile(
    r"\b(?:publication\s+bias|funnel\s+plot|Egger['’]?s\s+test)\b", re.I
)
_HETEROGENEITY_RE = re.compile(r"\b(?:I\^?2|heterogeneity|tau\^?2)\b", re.I)


def _signal(text: str, pattern: re.Pattern[str]) -> bool:
    return bool(pattern.search(text))


def skeleton_amstar2(paper: Paper) -> Amstar2Assessment:
    text = f"{paper.title}\n{paper.abstract or ''}"
    has_prospero = _signal(text, _PROSPERO_RE)
    has_duplicate = _signal(text, _DUPLICATE_RE)
    has_rob = _signal(text, _RISK_OF_BIAS_RE)
    has_pub_bias = _signal(text, _PUBLICATION_BIAS_RE)
    has_heterogeneity = _signal(text, _HETEROGENEITY_RE)

    duplicate_rationale = (
        "Duplicate selection mentioned in abstract."
        if has_duplicate
        else "No duplicate-review language detected."
    )
    items_met: dict[int, tuple[bool, str]] = {
        1: (paper.pico is not None, "PICO present in extracted abstract."),
        2: (
            has_prospero,
            "PROSPERO ID detected in abstract."
            if has_prospero
            else "No PROSPERO / a-priori protocol reference detected.",
        ),
        3: (False, "Study-design selection rationale: requires LLM review."),
        4: (False, "Comprehensive search strategy: requires LLM review."),
        5: (has_duplicate, duplicate_rationale),
        6: (has_duplicate, duplicate_rationale),
        7: (False, "Excluded-studies list: requires full-text or supplement."),
        8: (False, "Included-study detail: requires LLM review."),
        9: (
            has_rob,
            "RoB assessment language detected."
            if has_rob
            else "No RoB-of-included-studies language detected.",
        ),
        10: (False, "Funding-of-included-studies: requires LLM review."),
        11: (False, "Statistical combination methods: requires LLM review."),
        12: (False, "Impact of RoB on combined results: requires LLM review."),
        13: (False, "RoB-aware interpretation: requires LLM review."),
        14: (
            has_heterogeneity,
            "Heterogeneity discussed (I²/tau²)."
            if has_heterogeneity
            else "No heterogeneity discussion detected.",
        ),
        15: (
            has_pub_bias,
            "Publication bias assessment detected."
            if has_pub_bias
            else "No publication-bias assessment language detected.",
        ),
        16: (False, "Conflict-of-interest reporting: requires LLM review."),
    }

    items = [
        Amstar2Item(number=n, description=desc, met=items_met[n][0], rationale=items_met[n][1])
        for n, desc in _ITEMS
    ]

    overall = _overall(items, has_prospero=has_prospero, has_rob=has_rob)
    return Amstar2Assessment(items=items, overall=overall)


def _overall(items: list[Amstar2Item], *, has_prospero: bool, has_rob: bool) -> Amstar2Rating:
    """AMSTAR-2 overall confidence — coarse rule for Phase 1.

    AMSTAR-2 specifies critical items (2, 4, 7, 9, 11, 13, 15). We approximate:
    if multiple critical items fail (rule-only), rate critically_low. Anything
    we can't gauge we leave to the LLM step in PR 7.
    """

    critical_failures = sum(1 for n in (2, 9, 15) if not items[n - 1].met)
    if critical_failures >= 2:
        return Amstar2Rating.critically_low
    if critical_failures == 1:
        return Amstar2Rating.low
    if has_prospero and has_rob:
        return Amstar2Rating.moderate
    return Amstar2Rating.low


__all__ = ["skeleton_amstar2"]
