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
