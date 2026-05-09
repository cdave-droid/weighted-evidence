"""PICO directness scoring."""

from __future__ import annotations

from weighted_evidence.models import PICO
from weighted_evidence.rubric.directness import match_pico


def test_no_query_returns_none() -> None:
    assert match_pico(PICO(population="adults"), None) is None


def test_perfect_match_scores_one() -> None:
    pico = PICO(
        population="adults with sepsis",
        intervention="early antibiotics",
        comparator="standard care",
        outcomes=["mortality"],
        setting="ICU",
    )
    res = match_pico(pico, pico)
    assert res is not None
    assert res.overall == 1.0
    assert all(d.mismatch is None for d in res.dimensions)


def test_population_mismatch_surfaced() -> None:
    paper = PICO(population="adults with sepsis", intervention="early antibiotics")
    query = PICO(population="pediatric outpatients", intervention="early antibiotics")
    res = match_pico(paper, query)
    assert res is not None
    pop = next(d for d in res.dimensions if d.dimension == "population")
    assert pop.score < 0.5
    assert pop.mismatch is not None and "population" in pop.mismatch.lower()


def test_paper_with_no_pico_returns_zero() -> None:
    res = match_pico(None, PICO(population="adults"))
    assert res is not None
    assert res.overall == 0.0
