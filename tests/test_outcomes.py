"""Outcome importance classification + extraction."""

from __future__ import annotations

import pytest

from weighted_evidence.models import OutcomeImportance
from weighted_evidence.parsing.outcomes import classify_importance, extract_outcomes


@pytest.mark.parametrize(
    "name, expected",
    [
        ("28-day mortality", OutcomeImportance.mortality),
        ("All-cause death", OutcomeImportance.mortality),
        ("Hospitalization for heart failure", OutcomeImportance.hospitalization),
        ("Quality of life (EQ-5D)", OutcomeImportance.qol),
        ("HbA1c reduction", OutcomeImportance.surrogate_lab),
        ("LV ejection fraction", OutcomeImportance.surrogate_imaging),
        ("Stroke incidence", OutcomeImportance.morbidity),
        ("Pain score", OutcomeImportance.symptomatic),
        ("Composite endpoint", OutcomeImportance.composite),
        ("Customer satisfaction", OutcomeImportance.other),
    ],
)
def test_classify_importance(name: str, expected: OutcomeImportance) -> None:
    assert classify_importance(name) == expected


def test_extract_primary_mortality_outcome() -> None:
    abstract = (
        "Methods: We enrolled 861 patients in a randomized trial. "
        "The primary outcomes were death before discharge and ventilator-free days. "
        "Results: Mortality was 31.0% vs 39.8%, P = 0.007."
    )
    outcomes = extract_outcomes(abstract)
    assert outcomes
    primary = outcomes[0]
    assert primary.is_primary
    assert primary.importance in {OutcomeImportance.mortality, OutcomeImportance.other}


def test_fallback_finds_mortality_when_no_primary_keyword() -> None:
    abstract = "Results: Mortality was 31.0% vs 39.8%, P = 0.007."
    outcomes = extract_outcomes(abstract)
    assert outcomes
    assert outcomes[0].importance == OutcomeImportance.mortality


def test_no_outcomes_for_empty_abstract() -> None:
    assert extract_outcomes(None) == []
    assert extract_outcomes("") == []
