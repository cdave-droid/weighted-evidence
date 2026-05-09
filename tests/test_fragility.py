"""Fragility Index — Walsh formulation, hand-rolled Fisher's exact."""

from __future__ import annotations

from datetime import datetime

from weighted_evidence.models import Identifier, Paper, StudyDesign
from weighted_evidence.parsing.fragility import (
    _TwoByTwo,
    fragility_for_paper,
    fragility_index,
)


def test_fragility_index_robust_finding() -> None:
    # ARDSNet-like: 31% vs 39.8% in ~430 per arm. Strong, robust.
    table = _TwoByTwo(a=130, b=300, c=170, d=260)  # ~30% vs ~40%
    fi = fragility_index(table)
    assert fi >= 5


def test_fragility_index_already_nonsignificant() -> None:
    table = _TwoByTwo(a=50, b=50, c=52, d=48)
    assert fragility_index(table) == 0


def test_fragility_index_borderline_significant() -> None:
    # Tiny effect, borderline-significant — FI should be small.
    table = _TwoByTwo(a=10, b=90, c=20, d=80)
    fi = fragility_index(table)
    assert 0 <= fi <= 6


def test_fragility_for_paper_skips_when_data_missing() -> None:
    p = Paper(
        identifier=Identifier(pmid="x"),
        title="t",
        publication_date=datetime(2020, 1, 1),
        design=StudyDesign.rct,
    )
    assert fragility_for_paper(p) is None


def test_fragility_for_paper_extracts_from_abstract() -> None:
    p = Paper(
        identifier=Identifier(pmid="x"),
        title="t",
        abstract="Mortality was 31.0% vs 39.8%, P = 0.007",
        publication_date=datetime(2000, 1, 1),
        sample_size=861,
        design=StudyDesign.rct,
    )
    fi = fragility_for_paper(p)
    assert fi is not None
    assert fi.index >= 5
    assert fi.method == "walsh"
