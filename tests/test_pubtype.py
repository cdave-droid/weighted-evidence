"""PubMed publication-type → StudyDesign classification."""

from __future__ import annotations

from weighted_evidence.models import StudyDesign
from weighted_evidence.parsing.pubtype import classify_design


def test_rct_from_publication_type() -> None:
    assert classify_design(["Randomized Controlled Trial", "Journal Article"]) == StudyDesign.rct


def test_meta_analysis_beats_review() -> None:
    assert (
        classify_design(["Journal Article", "Meta-Analysis", "Review"]) == StudyDesign.meta_analysis
    )


def test_systematic_review() -> None:
    assert classify_design(["Systematic Review"]) == StudyDesign.systematic_review


def test_title_fallback() -> None:
    assert classify_design([], title="A randomized trial of fluid resuscitation") == StudyDesign.rct


def test_mesh_fallback_for_observational() -> None:
    assert classify_design([], mesh_terms=["Cohort Studies"]) == StudyDesign.cohort


def test_unknown_when_silent() -> None:
    assert classify_design(["Journal Article"]) == StudyDesign.unknown


def test_registry_analysis_in_title_classifies_cohort() -> None:
    """Surgisphere case: pub types didn't include observational, but title did."""

    assert (
        classify_design(
            ["Journal Article", "Retracted Publication"],
            title="HCQ for COVID-19: a multinational registry analysis",
        )
        == StudyDesign.cohort
    )


def test_prospective_cohort_in_title() -> None:
    assert (
        classify_design([], title="A prospective cohort study of sepsis outcomes")
        == StudyDesign.cohort
    )


def test_case_control_in_title() -> None:
    assert (
        classify_design([], title="Vaccination and stroke: a case-control study")
        == StudyDesign.case_control
    )


def test_cross_sectional_in_title() -> None:
    assert (
        classify_design([], title="HPV prevalence: a cross-sectional survey")
        == StudyDesign.cross_sectional
    )


def test_observational_study_in_title() -> None:
    assert (
        classify_design([], title="Statins and dementia: an observational study")
        == StudyDesign.cohort
    )
