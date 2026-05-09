"""Final-score aggregation + the agent-facing `reliability_tier` decision table."""

from __future__ import annotations

from dataclasses import dataclass, field

from weighted_evidence.models import (
    Amstar2Assessment,
    Amstar2Rating,
    Citation,
    Disposition,
    FindingsCard,
    FragilityIndex,
    GradeAssessment,
    GradeCertainty,
    GuidelineImpactScore,
    Outcome,
    OutcomeImportance,
    Paper,
    PICOMatch,
    PredatoryFlag,
    ReliabilityTier,
    RetractionStatus,
    RoB2Assessment,
    RoB2Judgment,
    RobinsIAssessment,
    RobinsIJudgment,
    RoBToolAssessment,
    SpinAssessment,
    StudyDesign,
)


@dataclass
class AggregateInput:
    paper: Paper
    grade: GradeAssessment
    rob: RoBToolAssessment | None
    gis: GuidelineImpactScore
    pico_match: PICOMatch | None = None
    fragility: FragilityIndex | None = None
    spin: SpinAssessment | None = None
    retraction: RetractionStatus = field(default_factory=RetractionStatus)
    predatory: PredatoryFlag = field(default_factory=PredatoryFlag)


@dataclass
class Weights:
    grade: float = 0.40
    rob: float = 0.20
    gis: float = 0.18
    pico_match: float = 0.12
    outcome_importance: float = 0.05
    fragility_or_imprecision: float = 0.05


# ---------------------------------------------------------------------------
# Numeric mappings
# ---------------------------------------------------------------------------


_GRADE_SCORE = {
    GradeCertainty.high: 1.0,
    GradeCertainty.moderate: 0.7,
    GradeCertainty.low: 0.4,
    GradeCertainty.very_low: 0.15,
}

_ROB2_SCORE = {
    RoB2Judgment.low: 1.0,
    RoB2Judgment.some_concerns: 0.6,
    RoB2Judgment.high: 0.2,
}

_AMSTAR_SCORE = {
    Amstar2Rating.high: 1.0,
    Amstar2Rating.moderate: 0.75,
    Amstar2Rating.low: 0.4,
    Amstar2Rating.critically_low: 0.1,
}

_ROBINS_SCORE = {
    RobinsIJudgment.low: 1.0,
    RobinsIJudgment.moderate: 0.7,
    RobinsIJudgment.serious: 0.35,
    RobinsIJudgment.critical: 0.1,
    RobinsIJudgment.no_information: 0.4,
}


_OUTCOME_WEIGHT = {
    OutcomeImportance.mortality: 1.0,
    OutcomeImportance.morbidity: 0.85,
    OutcomeImportance.qol: 0.8,
    OutcomeImportance.hospitalization: 0.75,
    OutcomeImportance.symptomatic: 0.6,
    OutcomeImportance.composite: 0.5,
    OutcomeImportance.surrogate_imaging: 0.3,
    OutcomeImportance.surrogate_lab: 0.25,
    OutcomeImportance.other: 0.4,
}


def _rob_score(rob: RoBToolAssessment | None) -> tuple[float, str]:
    if isinstance(rob, RoB2Assessment):
        return _ROB2_SCORE[rob.overall], f"RoB 2 overall: {rob.overall.value}"
    if isinstance(rob, Amstar2Assessment):
        return _AMSTAR_SCORE[rob.overall], f"AMSTAR-2 overall: {rob.overall.value}"
    if isinstance(rob, RobinsIAssessment):
        return _ROBINS_SCORE[rob.overall], f"ROBINS-I overall: {rob.overall.value}"
    return 0.5, "No design-appropriate RoB tool ran; using neutral score."


def _outcome_importance_score(outcomes: list[Outcome]) -> tuple[float, str]:
    if not outcomes:
        return 0.4, "No structured outcomes extracted."
    primaries = [o for o in outcomes if o.is_primary] or outcomes
    weights = [_OUTCOME_WEIGHT.get(o.importance, 0.4) for o in primaries]
    avg = sum(weights) / len(weights)
    top = max(primaries, key=lambda o: _OUTCOME_WEIGHT.get(o.importance, 0.4))
    return avg, f"Top primary outcome: {top.name} ({top.importance.value})"


def _fragility_imprecision_score(
    fragility: FragilityIndex | None,
    grade: GradeAssessment,
) -> tuple[float, str]:
    if fragility is not None:
        if fragility.index <= 4:
            return 0.2, f"Low fragility index ({fragility.index}); finding flips on a few events."
        if fragility.index <= 10:
            return 0.6, f"Modest fragility index ({fragility.index})."
        return 1.0, f"Robust fragility index ({fragility.index})."
    serious_imprecision = any(
        d.name == "imprecision" and d.judgment.value != "not_serious" for d in grade.downgrades
    )
    if serious_imprecision:
        return 0.4, "GRADE flagged serious imprecision."
    return 0.7, "No fragility data; no GRADE imprecision flag."


# ---------------------------------------------------------------------------
# Reliability tier decision table
# ---------------------------------------------------------------------------


def reliability_tier(
    *,
    grade: GradeAssessment,
    rob: RoBToolAssessment | None,
    fragility: FragilityIndex | None,
    spin: SpinAssessment | None,
    retraction: RetractionStatus,
    predatory: PredatoryFlag,
) -> ReliabilityTier:
    if retraction.status == "retracted":
        return ReliabilityTier.retracted

    rob_overall_high = (
        (isinstance(rob, RoB2Assessment) and rob.overall == RoB2Judgment.high)
        or (isinstance(rob, Amstar2Assessment) and rob.overall == Amstar2Rating.critically_low)
        or (isinstance(rob, RobinsIAssessment) and rob.overall == RobinsIJudgment.critical)
    )
    rob_overall_serious = (
        isinstance(rob, RobinsIAssessment) and rob.overall == RobinsIJudgment.serious
    ) or (isinstance(rob, Amstar2Assessment) and rob.overall == Amstar2Rating.low)
    spin_present = bool(spin and spin.present)
    fragility_low = bool(fragility and fragility.index <= 4)

    if grade.final_certainty == GradeCertainty.very_low or rob_overall_high:
        return ReliabilityTier.do_not_rely
    if grade.final_certainty == GradeCertainty.low or rob_overall_serious:
        return ReliabilityTier.weak_signal
    if (
        grade.final_certainty == GradeCertainty.moderate
        or fragility_low
        or spin_present
        or predatory.flagged
    ):
        return ReliabilityTier.use_with_caution
    return ReliabilityTier.rely


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def aggregate_report(
    inp: AggregateInput,
    *,
    weights: Weights | None = None,
    pico_match: PICOMatch | None = None,
) -> FindingsCard:
    w = weights or Weights()
    pico_match = pico_match or inp.pico_match

    grade_score = _GRADE_SCORE[inp.grade.final_certainty]
    rob_score, rob_msg = _rob_score(inp.rob)
    gis_score = inp.gis.score
    pico_score = pico_match.overall if pico_match else 0.5
    outcome_score, outcome_msg = _outcome_importance_score(inp.paper.outcomes)
    fragility_score, fragility_msg = _fragility_imprecision_score(inp.fragility, inp.grade)

    raw = (
        w.grade * grade_score
        + w.rob * rob_score
        + w.gis * gis_score
        + w.pico_match * pico_score
        + w.outcome_importance * outcome_score
        + w.fragility_or_imprecision * fragility_score
    )

    final: float | None = max(0.0, min(1.0, raw))
    disposition = Disposition.ok
    if inp.predatory.flagged and final is not None:
        final = min(final, 0.4)
        disposition = Disposition.flagged
    if inp.retraction.status == "retracted":
        final = None
        disposition = Disposition.retracted

    explanation = [
        Citation(
            signal=f"GRADE {inp.grade.final_certainty.value}",
            contribution=w.grade * grade_score,
            rationale=inp.grade.rationale,
        ),
        Citation(signal="RoB tool", contribution=w.rob * rob_score, rationale=rob_msg),
        Citation(
            signal=f"GIS ({inp.gis.version})",
            contribution=w.gis * gis_score,
            rationale=f"Journal tier {inp.gis.journal_tier:.2f}, recency {inp.gis.recency_decay:.2f}.",
        ),
        Citation(
            signal="PICO directness",
            contribution=w.pico_match * pico_score,
            rationale=pico_match.rationale if pico_match else "No query PICO supplied.",
        ),
        Citation(
            signal="Outcome importance",
            contribution=w.outcome_importance * outcome_score,
            rationale=outcome_msg,
        ),
        Citation(
            signal="Fragility / imprecision",
            contribution=w.fragility_or_imprecision * fragility_score,
            rationale=fragility_msg,
        ),
    ]

    tier = reliability_tier(
        grade=inp.grade,
        rob=inp.rob,
        fragility=inp.fragility,
        spin=inp.spin,
        retraction=inp.retraction,
        predatory=inp.predatory,
    )

    rob_tool_label: str
    if isinstance(inp.rob, RoB2Assessment):
        rob_tool_label = "rob2"
    elif isinstance(inp.rob, Amstar2Assessment):
        rob_tool_label = "amstar2"
    elif isinstance(inp.rob, RobinsIAssessment):
        rob_tool_label = "robins_i"
    else:
        rob_tool_label = "none"

    paper = inp.paper
    return FindingsCard(
        id=paper.identifier.primary() or paper.title,
        title=paper.title,
        journal=paper.journal,
        publication_date=paper.publication_date,
        design=paper.design or StudyDesign.unknown,
        pico=paper.pico,
        outcomes=paper.outcomes,
        grade=inp.grade,
        rob_tool=rob_tool_label,
        rob=inp.rob,
        gis=inp.gis,
        fragility=inp.fragility,
        spin=inp.spin,
        retraction=inp.retraction,
        predatory=inp.predatory,
        pico_match=pico_match,
        final_score=final,
        reliability_tier=tier,
        disposition=disposition,
        explanation=explanation,
        model_version="rule-v0",
    )
