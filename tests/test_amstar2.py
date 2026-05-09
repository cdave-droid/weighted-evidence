"""AMSTAR-2 skeleton."""

from __future__ import annotations

from datetime import datetime

from weighted_evidence.models import Amstar2Rating, Identifier, Paper, StudyDesign
from weighted_evidence.rubric.amstar2 import skeleton_amstar2


def _paper(abstract: str) -> Paper:
    return Paper(
        identifier=Identifier(pmid="x"),
        title="Systematic review of...",
        abstract=abstract,
        publication_date=datetime(2023, 1, 1),
        design=StudyDesign.systematic_review,
    )


def test_no_signals_lands_at_critically_low_or_low() -> None:
    assess = skeleton_amstar2(_paper("This systematic review summarizes the literature."))
    assert assess.overall in {Amstar2Rating.critically_low, Amstar2Rating.low}


def test_prospero_plus_rob_plus_pubbias_reaches_moderate() -> None:
    abstract = (
        "This systematic review (PROSPERO CRD42022000123) included RCTs assessed "
        "using the Cochrane risk of bias tool. Two independent reviewers performed "
        "study selection. Publication bias was assessed via funnel plot."
    )
    assess = skeleton_amstar2(_paper(abstract))
    assert assess.overall == Amstar2Rating.moderate
    assert any(item.number == 2 and item.met for item in assess.items)


def test_critical_failures_lower_to_critically_low() -> None:
    # No PROSPERO, no RoB tool, no publication bias.
    assess = skeleton_amstar2(_paper("Brief narrative summary."))
    assert assess.overall == Amstar2Rating.critically_low
