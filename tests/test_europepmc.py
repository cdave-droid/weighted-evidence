"""Europe PMC JATS parsing + agent integration with full-text body."""

from __future__ import annotations

from pathlib import Path

from weighted_evidence.agents import EvidenceAgent
from weighted_evidence.cache import Cache
from weighted_evidence.models import Identifier, Paper, StudyDesign
from weighted_evidence.retrieval.europepmc import assemble_body_text, parse_jats


def test_parse_jats_extracts_normalized_sections(fixtures_dir: Path) -> None:
    xml = (fixtures_dir / "europepmc_sample.xml").read_text(encoding="utf-8")
    sections = parse_jats(xml)
    assert "methods" in sections
    assert "results" in sections
    assert "conclusions" in sections
    assert "ARDS" in sections["methods"] or "861" in sections["methods"]


def test_assemble_body_text_orders_sections(fixtures_dir: Path) -> None:
    xml = (fixtures_dir / "europepmc_sample.xml").read_text(encoding="utf-8")
    body = assemble_body_text(parse_jats(xml))
    assert body.index("Methods:") < body.index("Results:") < body.index("Conclusions:")


def test_invalid_xml_returns_empty() -> None:
    assert parse_jats("<not real xml") == {}
    assert parse_jats("") == {}


def test_agent_uses_body_text_for_outcomes_and_effects(fixtures_dir: Path, tmp_path: Path) -> None:
    """When body_text is set, extractors should pull effect sizes from it."""

    xml = (fixtures_dir / "europepmc_sample.xml").read_text(encoding="utf-8")
    sections = parse_jats(xml)
    paper = Paper(
        identifier=Identifier(pmid="10793162", pmcid="PMC1234567"),
        title="Lower tidal volume ventilation",
        abstract="(short abstract)",
        journal="N Engl J Med",
        publication_types=["Randomized Controlled Trial"],
        design=StudyDesign.rct,
        body_sections=sections,
        body_text=assemble_body_text(sections),
    )
    agent = EvidenceAgent(cache=Cache(tmp_path / "cache.sqlite"))
    report = agent.grade_paper(paper)
    # Sample size should be picked up from the body's "861 patients" text.
    assert report.paper.sample_size == 861
    # An outcome with an extracted ARR or HR should be present.
    assert report.paper.outcomes
