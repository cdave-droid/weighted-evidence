"""Retraction guard: PubMed publication-type signal + Retraction Watch."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from weighted_evidence.models import Identifier, Paper, StudyDesign
from weighted_evidence.retrieval.retraction import (
    RetractionWatchSource,
    check_retraction,
    pubmed_retraction_check,
)


def _paper(**overrides: object) -> Paper:
    base = Paper(
        identifier=Identifier(doi="10.1016/s0140-6736(20)31180-6", pmid="32450107"),
        title="HCQ or CQ for COVID-19",
        journal="Lancet",
        publication_date=datetime(2020, 5, 22),
        design=StudyDesign.cohort,
    )
    return base.model_copy(update=dict(overrides))


def test_pubmed_publication_type_marks_retraction() -> None:
    paper = _paper(publication_types=["Journal Article", "Retracted Publication"])
    status = pubmed_retraction_check(paper)
    assert status is not None and status.status == "retracted"
    assert status.source == "pubmed"


def test_pubmed_check_negative_when_not_retracted() -> None:
    paper = _paper(publication_types=["Journal Article"])
    assert pubmed_retraction_check(paper) is None


@pytest.mark.asyncio
async def test_retraction_watch_doi_lookup(fixtures_dir: Path) -> None:
    rw = RetractionWatchSource(path=fixtures_dir / "retraction_watch_sample.csv")
    paper = _paper(publication_types=["Journal Article"])  # no PubMed retraction signal
    status = await check_retraction(paper, rw_source=rw)
    assert status.status == "retracted"
    assert status.source == "retraction_watch"
    await rw.aclose()


@pytest.mark.asyncio
async def test_retraction_watch_negative_lookup(fixtures_dir: Path) -> None:
    rw = RetractionWatchSource(path=fixtures_dir / "retraction_watch_sample.csv")
    paper = _paper(
        identifier=Identifier(doi="10.1056/nejmoa1503326", pmid="25776936"),
        publication_types=["Journal Article"],
    )
    status = await check_retraction(paper, rw_source=rw)
    assert status.status == "none"
    await rw.aclose()
