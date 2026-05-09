"""compare()/rank() agent surface."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import pytest

from weighted_evidence.agents import EvidenceAgent
from weighted_evidence.cache import Cache
from weighted_evidence.models import (
    Identifier,
    Paper,
    StudyDesign,
)
from weighted_evidence.retrieval.pubmed import parse_pubmed_xml


def _stub_cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "cache.sqlite")


def _papers(fixtures_dir: Path) -> tuple[Paper, Paper]:
    xml = (fixtures_dir / "ardsnet_arma_pubmed.xml").read_text(encoding="utf-8")
    high_quality = parse_pubmed_xml(xml)
    weak = Paper(
        identifier=Identifier(pmid="999999"),
        title="Single-center retrospective ICU cohort with 50 patients",
        abstract="A retrospective review of 50 patients showed slightly lower lactate.",
        journal="Local Hospital Bulletin",
        publication_date=datetime(2010, 1, 1),
        publication_types=["Journal Article"],
        design=StudyDesign.cohort,
        sample_size=50,
    )
    return high_quality, weak


@pytest.mark.asyncio
async def test_rank_orders_high_quality_first(fixtures_dir: Path, tmp_path: Path) -> None:
    high_quality, weak = _papers(fixtures_dir)

    class _StubAgent(EvidenceAgent):
        async def grade(self, identifier, *, query=None):  # type: ignore[override]
            paper = high_quality if identifier == "high" else weak
            return self.grade_paper(paper, query=query)

    agent = _StubAgent(cache=_stub_cache(tmp_path))
    ranking = await agent.rank(["high", "weak"])
    cards = ranking.cards
    assert cards[0].id != cards[1].id
    # High-quality NEJM RCT should outrank a 50-patient retrospective bulletin.
    assert cards[0].id == "10.1056/nejm200005043421801"


@pytest.mark.asyncio
async def test_compare_emits_pairwise_rationale(fixtures_dir: Path, tmp_path: Path) -> None:
    high_quality, weak = _papers(fixtures_dir)

    class _StubAgent(EvidenceAgent):
        async def grade(self, identifier, *, query=None):  # type: ignore[override]
            paper = high_quality if identifier == "high" else weak
            return self.grade_paper(paper, query=query)

    agent = _StubAgent(cache=_stub_cache(tmp_path))
    comparison = await agent.compare(["high", "weak"])
    assert len(comparison.pairwise) == 1
    pair = comparison.pairwise[0]
    assert pair.winner_id and pair.loser_id and pair.winner_id != pair.loser_id
    assert pair.reasons
    # Either GRADE difference or reliability_tier difference must be cited.
    joined = " | ".join(pair.reasons)
    assert "GRADE" in joined or "reliability_tier" in joined


@pytest.mark.asyncio
async def test_rank_pushes_retracted_to_bottom(fixtures_dir: Path, tmp_path: Path) -> None:
    high_quality, weak = _papers(fixtures_dir)
    retracted = high_quality.model_copy(
        update={
            "publication_types": [*high_quality.publication_types, "Retracted Publication"],
        }
    )

    class _StubAgent(EvidenceAgent):
        async def grade(self, identifier, *, query=None):  # type: ignore[override]
            mapping = {"good": high_quality, "weak": weak, "bad": retracted}
            return self.grade_paper(mapping[cast(str, identifier)], query=query)

    agent = _StubAgent(cache=_stub_cache(tmp_path))
    ranking = await agent.rank(["bad", "good", "weak"])
    assert ranking.cards[-1].reliability_tier.value == "retracted"
