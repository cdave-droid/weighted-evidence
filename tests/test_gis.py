"""GIS rule-v0 sanity checks."""

from __future__ import annotations

from datetime import datetime

from weighted_evidence.models import Identifier, Paper, StudyDesign
from weighted_evidence.rubric.gis import RuleV0GISModel


def _paper(**overrides: object) -> Paper:
    base = Paper(
        identifier=Identifier(pmid="1"),
        title="Example",
        abstract=None,
        journal="N Engl J Med",
        publication_date=datetime(2020, 1, 1),
        sample_size=1000,
        design=StudyDesign.rct,
    )
    return base.model_copy(update=dict(overrides))


def test_high_tier_journal_dominates() -> None:
    nejm = RuleV0GISModel().score(_paper(journal="N Engl J Med"))
    obscure = RuleV0GISModel().score(_paper(journal="A Random Hospital Bulletin"))
    assert nejm.score > obscure.score
    assert nejm.journal_tier is not None and nejm.journal_tier >= 0.95


def test_log_sample_size_monotone() -> None:
    small = RuleV0GISModel().score(_paper(sample_size=20))
    medium = RuleV0GISModel().score(_paper(sample_size=500))
    huge = RuleV0GISModel().score(_paper(sample_size=20000))
    assert small.score <= medium.score <= huge.score


def test_recency_decays_old_papers() -> None:
    new = RuleV0GISModel().score(_paper(publication_date=datetime(2024, 1, 1)))
    old = RuleV0GISModel().score(_paper(publication_date=datetime(1980, 1, 1)))
    assert new.recency_decay is not None and old.recency_decay is not None
    assert new.recency_decay > old.recency_decay


def test_score_in_range() -> None:
    s = RuleV0GISModel().score(_paper())
    assert 0.0 <= s.score <= 1.0
    assert s.version == "rule-v0"
    assert s.fine_tune_pending is True
