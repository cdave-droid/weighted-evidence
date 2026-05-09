"""GIS uses citation context when supplied; falls back to neutral otherwise."""

from __future__ import annotations

from datetime import datetime

from weighted_evidence.models import CitationContext, Identifier, Paper, StudyDesign
from weighted_evidence.rubric.gis import RuleV0GISModel


def _paper() -> Paper:
    return Paper(
        identifier=Identifier(doi="10.0/x"),
        title="t",
        journal="N Engl J Med",
        publication_date=datetime(2020, 1, 1),
        sample_size=1000,
        design=StudyDesign.rct,
    )


def test_supportive_context_raises_score() -> None:
    base = RuleV0GISModel().score(_paper())
    boosted = RuleV0GISModel().score(
        _paper(), citation_context=CitationContext(total=50, supportive=50)
    )
    assert boosted.score > base.score


def test_disputing_context_lowers_score() -> None:
    boosted = RuleV0GISModel().score(
        _paper(), citation_context=CitationContext(total=50, supportive=50)
    )
    disputed = RuleV0GISModel().score(
        _paper(),
        citation_context=CitationContext(total=50, supportive=10, disputing=40),
    )
    assert disputed.score < boosted.score


def test_no_context_is_neutral() -> None:
    no_ctx = RuleV0GISModel().score(_paper())
    empty_ctx = RuleV0GISModel().score(_paper(), citation_context=CitationContext())
    assert abs(no_ctx.score - empty_ctx.score) < 1e-9
