"""End-to-end PubMed XML parsing → grading."""

from __future__ import annotations

from pathlib import Path

from weighted_evidence.agents import EvidenceAgent
from weighted_evidence.cache import Cache
from weighted_evidence.models import GradeCertainty, ReliabilityTier, StudyDesign
from weighted_evidence.retrieval.pubmed import parse_pubmed_xml


def test_parse_ardsnet_pubmed_xml(fixtures_dir: Path) -> None:
    xml = (fixtures_dir / "ardsnet_arma_pubmed.xml").read_text(encoding="utf-8")
    paper = parse_pubmed_xml(xml)

    assert paper.identifier.pmid == "10793162"
    assert paper.identifier.doi == "10.1056/nejm200005043421801"
    assert "lower tidal volumes" in paper.title.lower()
    assert paper.journal is not None and "new england" in paper.journal.lower()
    assert any("Randomized" in pt for pt in paper.publication_types)
    assert paper.publication_date is not None and paper.publication_date.year == 2000


def test_grade_paper_from_xml(fixtures_dir: Path, tmp_path: Path) -> None:
    xml = (fixtures_dir / "ardsnet_arma_pubmed.xml").read_text(encoding="utf-8")
    paper = parse_pubmed_xml(xml)

    cache = Cache(tmp_path / "cache.sqlite")
    agent = EvidenceAgent(cache=cache)
    report = agent.grade_paper(paper)

    assert report.paper.design == StudyDesign.rct
    assert report.card.grade is not None
    assert report.card.grade.final_certainty == GradeCertainty.high
    assert report.card.rob_tool == "rob2"
    assert report.card.final_score is not None
    assert report.card.final_score > 0.5
    assert report.card.reliability_tier in {
        ReliabilityTier.rely,
        ReliabilityTier.use_with_caution,
    }
    # Sample size should have been picked up from "n = 861" in the abstract.
    assert report.paper.sample_size == 861
