"""Map PubMed publication types (and MeSH hints) to our `StudyDesign` enum.

Rule-only and intentionally conservative; falls back to `StudyDesign.unknown` when
the signal is ambiguous so the LLM step has a chance to refine it.
"""

from __future__ import annotations

from collections.abc import Iterable

from weighted_evidence.models import StudyDesign

# Order matters: more specific matches first.
_PUBTYPE_RULES: tuple[tuple[str, StudyDesign], ...] = (
    ("randomized controlled trial", StudyDesign.rct),
    ("controlled clinical trial", StudyDesign.rct),
    ("clinical trial, phase iv", StudyDesign.rct),
    ("clinical trial, phase iii", StudyDesign.rct),
    ("meta-analysis", StudyDesign.meta_analysis),
    ("systematic review", StudyDesign.systematic_review),
    ("review", StudyDesign.narrative_review),
    ("observational study", StudyDesign.cohort),
    ("comparative study", StudyDesign.comparative_effectiveness),
    ("case reports", StudyDesign.case_report),
    ("editorial", StudyDesign.editorial),
    ("letter", StudyDesign.editorial),
)

_MESH_HINTS: tuple[tuple[str, StudyDesign], ...] = (
    ("cohort studies", StudyDesign.cohort),
    ("case-control studies", StudyDesign.case_control),
    ("cross-sectional studies", StudyDesign.cross_sectional),
    ("interrupted time series analysis", StudyDesign.interrupted_time_series),
    ("controlled before-after studies", StudyDesign.controlled_before_after),
    ("prospective studies", StudyDesign.cohort),
    ("retrospective studies", StudyDesign.cohort),
)


def classify_design(
    publication_types: Iterable[str],
    mesh_terms: Iterable[str] = (),
    title: str = "",
) -> StudyDesign:
    pts = [pt.strip().lower() for pt in publication_types]
    for needle, design in _PUBTYPE_RULES:
        if any(needle == p or needle in p for p in pts):
            return design

    title_l = title.lower()
    if "randomized" in title_l or "randomised" in title_l:
        return StudyDesign.rct
    if "systematic review" in title_l:
        return StudyDesign.systematic_review
    if "meta-analysis" in title_l or "meta analysis" in title_l:
        return StudyDesign.meta_analysis

    mesh = [m.strip().lower() for m in mesh_terms]
    for needle, design in _MESH_HINTS:
        if any(needle == m for m in mesh):
            return design

    return StudyDesign.unknown
