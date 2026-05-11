"""Map PubMed publication types (and MeSH hints) to our `StudyDesign` enum.

Rule-only and intentionally conservative; falls back to `StudyDesign.unknown` when
the signal is ambiguous so the LLM step has a chance to refine it.
"""

from __future__ import annotations

import re
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


_TITLE_RULES: tuple[tuple[re.Pattern[str], StudyDesign], ...] = (
    # Strongest signals first.
    (re.compile(r"\b(randomi[sz]ed|randomi[sz]ed[-\s]controlled)\b", re.I), StudyDesign.rct),
    (re.compile(r"\bcluster[-\s]randomi[sz]ed\b", re.I), StudyDesign.cluster_rct),
    (re.compile(r"\bcrossover\s+(trial|study|design)\b", re.I), StudyDesign.crossover_rct),
    (re.compile(r"\bsystematic\s+review\b", re.I), StudyDesign.systematic_review),
    (re.compile(r"\bmeta[-\s]?analysis\b", re.I), StudyDesign.meta_analysis),
    # Observational designs — these are what the Surgisphere case missed.
    (
        re.compile(
            r"\b(?:multi(?:national|center|centre)\s+)?registry\s+(?:analysis|study)\b", re.I
        ),
        StudyDesign.cohort,
    ),
    (re.compile(r"\b(prospective|retrospective)\s+cohort\b", re.I), StudyDesign.cohort),
    (re.compile(r"\bcohort\s+study\b", re.I), StudyDesign.cohort),
    (re.compile(r"\bcase[-\s]control\s+study\b", re.I), StudyDesign.case_control),
    (re.compile(r"\bcross[-\s]sectional\s+(study|survey)\b", re.I), StudyDesign.cross_sectional),
    (re.compile(r"\bcomparative\s+effectiveness\b", re.I), StudyDesign.comparative_effectiveness),
    (re.compile(r"\bobservational\s+(study|analysis)\b", re.I), StudyDesign.cohort),
    (re.compile(r"\binterrupted\s+time[-\s]series\b", re.I), StudyDesign.interrupted_time_series),
    (re.compile(r"\bcase\s+series\b", re.I), StudyDesign.case_series),
    (re.compile(r"\bcase\s+report\b", re.I), StudyDesign.case_report),
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

    for pattern, design in _TITLE_RULES:
        if pattern.search(title):
            return design

    mesh = [m.strip().lower() for m in mesh_terms]
    for needle, design in _MESH_HINTS:
        if any(needle == m for m in mesh):
            return design

    return StudyDesign.unknown
