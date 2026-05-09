"""GRADE certainty assessment.

Phase 1 keeps the *starting* certainty rule-based and uses an LLM call (when
available) for the modifier domains (RoB, inconsistency, indirectness,
imprecision, publication bias). The LLM step is wired in `evidence_agent.py`;
this module provides the deterministic skeleton.
"""

from __future__ import annotations

from weighted_evidence.models import (
    GradeAssessment,
    GradeCertainty,
    GradeDomain,
    GradeDomainJudgment,
    StudyDesign,
)

_RCT_LIKE = {
    StudyDesign.rct,
    StudyDesign.cluster_rct,
    StudyDesign.crossover_rct,
}

_HIGH_FROM_SR = {
    StudyDesign.systematic_review,
    StudyDesign.meta_analysis,
}

_OBSERVATIONAL = {
    StudyDesign.cohort,
    StudyDesign.case_control,
    StudyDesign.cross_sectional,
    StudyDesign.controlled_before_after,
    StudyDesign.interrupted_time_series,
    StudyDesign.comparative_effectiveness,
}


def starting_certainty(design: StudyDesign) -> GradeCertainty:
    if design in _RCT_LIKE or design in _HIGH_FROM_SR:
        return GradeCertainty.high
    if design in _OBSERVATIONAL:
        return GradeCertainty.low
    return GradeCertainty.very_low


def apply_modifiers(
    starting: GradeCertainty,
    *,
    downgrades: list[GradeDomain] | None = None,
    upgrades: list[str] | None = None,
) -> GradeCertainty:
    order = [
        GradeCertainty.high,
        GradeCertainty.moderate,
        GradeCertainty.low,
        GradeCertainty.very_low,
    ]
    idx = order.index(starting)
    for d in downgrades or []:
        if d.judgment == GradeDomainJudgment.serious:
            idx = min(idx + 1, len(order) - 1)
        elif d.judgment == GradeDomainJudgment.very_serious:
            idx = min(idx + 2, len(order) - 1)
    for _ in upgrades or []:
        idx = max(idx - 1, 0)
    return order[idx]


def skeleton_grade(design: StudyDesign) -> GradeAssessment:
    start = starting_certainty(design)
    return GradeAssessment(
        starting_certainty=start,
        final_certainty=start,
        downgrades=[],
        upgrades=[],
        rationale=None,
    )
