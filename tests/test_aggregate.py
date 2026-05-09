"""Aggregate scorer + reliability_tier decision table."""

from __future__ import annotations

from datetime import datetime

from weighted_evidence.models import (
    Amstar2Assessment,
    Amstar2Rating,
    GradeAssessment,
    GradeCertainty,
    GuidelineImpactScore,
    Identifier,
    Paper,
    PredatoryFlag,
    ReliabilityTier,
    RetractionStatus,
    RoB2Assessment,
    RoB2Domain,
    RoB2Judgment,
    RobinsIAssessment,
    RobinsIDomain,
    RobinsIJudgment,
    SpinAssessment,
    StudyDesign,
)
from weighted_evidence.rubric.aggregate import (
    AggregateInput,
    aggregate_report,
    reliability_tier,
)


def _paper(**overrides: object) -> Paper:
    return Paper(
        identifier=Identifier(pmid="1"),
        title="Example",
        journal="N Engl J Med",
        publication_date=datetime(2020, 1, 1),
        design=StudyDesign.rct,
        sample_size=800,
    ).model_copy(update=dict(overrides))


def _grade(certainty: GradeCertainty) -> GradeAssessment:
    return GradeAssessment(starting_certainty=certainty, final_certainty=certainty)


def _rob2(overall: RoB2Judgment) -> RoB2Assessment:
    return RoB2Assessment(
        domains=[
            RoB2Domain(name="randomization", judgment=overall, rationale=""),
            RoB2Domain(name="deviations", judgment=overall, rationale=""),
            RoB2Domain(name="missing_data", judgment=overall, rationale=""),
            RoB2Domain(name="outcome_measurement", judgment=overall, rationale=""),
            RoB2Domain(name="selective_reporting", judgment=overall, rationale=""),
        ],
        overall=overall,
    )


def _gis(score: float = 0.7) -> GuidelineImpactScore:
    return GuidelineImpactScore(
        score=score,
        version="rule-v0",
        journal_tier=0.9,
        log_sample_size=0.6,
        replication_count=0,
        recency_decay=0.7,
    )


def test_high_quality_rct_is_rely() -> None:
    tier = reliability_tier(
        grade=_grade(GradeCertainty.high),
        rob=_rob2(RoB2Judgment.low),
        fragility=None,
        spin=SpinAssessment(present=False),
        retraction=RetractionStatus(),
        predatory=PredatoryFlag(),
    )
    assert tier == ReliabilityTier.rely


def test_high_rob_downgrades_to_do_not_rely() -> None:
    tier = reliability_tier(
        grade=_grade(GradeCertainty.high),
        rob=_rob2(RoB2Judgment.high),
        fragility=None,
        spin=None,
        retraction=RetractionStatus(),
        predatory=PredatoryFlag(),
    )
    assert tier == ReliabilityTier.do_not_rely


def test_retracted_overrides_everything() -> None:
    tier = reliability_tier(
        grade=_grade(GradeCertainty.high),
        rob=_rob2(RoB2Judgment.low),
        fragility=None,
        spin=None,
        retraction=RetractionStatus(status="retracted"),
        predatory=PredatoryFlag(),
    )
    assert tier == ReliabilityTier.retracted


def test_low_amstar2_is_weak_signal() -> None:
    tier = reliability_tier(
        grade=_grade(GradeCertainty.high),
        rob=Amstar2Assessment(items=[], overall=Amstar2Rating.low),
        fragility=None,
        spin=None,
        retraction=RetractionStatus(),
        predatory=PredatoryFlag(),
    )
    assert tier == ReliabilityTier.weak_signal


def test_serious_robins_i_is_weak_signal() -> None:
    rob = RobinsIAssessment(
        domains=[
            RobinsIDomain(name="confounding", judgment=RobinsIJudgment.serious, rationale=""),
        ],
        overall=RobinsIJudgment.serious,
    )
    tier = reliability_tier(
        grade=_grade(GradeCertainty.moderate),
        rob=rob,
        fragility=None,
        spin=None,
        retraction=RetractionStatus(),
        predatory=PredatoryFlag(),
    )
    assert tier == ReliabilityTier.weak_signal


def test_predatory_caps_score_at_0_4() -> None:
    card = aggregate_report(
        AggregateInput(
            paper=_paper(),
            grade=_grade(GradeCertainty.high),
            rob=_rob2(RoB2Judgment.low),
            gis=_gis(0.95),
            predatory=PredatoryFlag(flagged=True, list_name="example"),
        )
    )
    assert card.final_score is not None
    assert card.final_score <= 0.4


def test_retracted_zeros_final_score() -> None:
    card = aggregate_report(
        AggregateInput(
            paper=_paper(),
            grade=_grade(GradeCertainty.high),
            rob=_rob2(RoB2Judgment.low),
            gis=_gis(0.95),
            retraction=RetractionStatus(status="retracted"),
        )
    )
    assert card.final_score is None
    assert card.reliability_tier == ReliabilityTier.retracted
