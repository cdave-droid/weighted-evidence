"""Guideline Impact Score (v0).

Phase 1 ships a transparent linear stub:
    score = 0.4*journal_tier + 0.3*log10(sample_size) + 0.2*replication_count + 0.1*recency_decay

Tagged `model_version="rule-v0"`, `fine_tune_pending=True`. Phase 2 swaps in a
trained model behind the same Protocol.

The plan calls for Semantic Scholar `contexts/intents` to replace raw citation
counts, and for guideline-assigned strength to be a weighted feature. Both
plug in here once their retrieval modules land in PR 3/4 — the current stub
keeps the wiring deterministic so Phase 1 remains end-to-end runnable.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Protocol

from weighted_evidence.models import GuidelineImpactScore, Paper

# A coarse, deliberately conservative journal-tier table. Override / expand
# as needed; this is a Phase 1 placeholder, not a definitive ranking.
_JOURNAL_TIERS: dict[str, float] = {
    "n engl j med": 1.0,
    "the new england journal of medicine": 1.0,
    "jama": 0.95,
    "journal of the american medical association": 0.95,
    "lancet": 0.95,
    "the lancet": 0.95,
    "bmj": 0.85,
    "annals of internal medicine": 0.85,
    "intensive care med": 0.8,
    "intensive care medicine": 0.8,
    "am j respir crit care med": 0.8,
    "american journal of respiratory and critical care medicine": 0.8,
    "crit care med": 0.75,
    "critical care medicine": 0.75,
    "chest": 0.7,
}


def _journal_tier(journal: str | None) -> float:
    if not journal:
        return 0.3
    return _JOURNAL_TIERS.get(journal.strip().lower(), 0.4)


def _log_sample_size(paper: Paper) -> float:
    if not paper.sample_size or paper.sample_size <= 0:
        return 0.0
    # Normalize to ~[0,1] across log10(20) .. log10(20000).
    raw = math.log10(paper.sample_size)
    return max(0.0, min(1.0, (raw - math.log10(20)) / (math.log10(20000) - math.log10(20))))


def _recency_decay(paper: Paper, *, half_life_years: float = 8.0) -> float:
    if paper.publication_date is None:
        return 0.5
    years = max(0.0, (datetime.utcnow() - paper.publication_date).days / 365.25)
    return float(0.5 ** (years / half_life_years))


class GISModel(Protocol):
    version: str

    def score(self, paper: Paper) -> GuidelineImpactScore: ...


class RuleV0GISModel:
    version: str = "rule-v0"

    def score(self, paper: Paper) -> GuidelineImpactScore:
        jt = _journal_tier(paper.journal)
        ls = _log_sample_size(paper)
        rc = 0.0  # Replication count placeholder until citation-context is wired (PR 4).
        rd = _recency_decay(paper)

        score = 0.4 * jt + 0.3 * ls + 0.2 * rc + 0.1 * rd
        score = max(0.0, min(1.0, score))
        return GuidelineImpactScore(
            score=score,
            version=self.version,
            fine_tune_pending=True,
            journal_tier=jt,
            log_sample_size=ls,
            replication_count=int(rc),
            recency_decay=rd,
            guideline_citations=[],
        )


def score_gis(paper: Paper, model: GISModel | None = None) -> GuidelineImpactScore:
    return (model or RuleV0GISModel()).score(paper)
