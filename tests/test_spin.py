"""Spin detection."""

from __future__ import annotations

from weighted_evidence.parsing.spin import detect_spin


def test_no_abstract_returns_no_spin() -> None:
    assert detect_spin(None).present is False
    assert detect_spin("").present is False


def test_consistent_positive_no_spin() -> None:
    abstract = (
        "Methods: We enrolled 861 patients.\n"
        "Results: Mortality was 31.0% vs 39.8%, P = 0.007.\n"
        "Conclusions: Lower tidal volumes reduce mortality."
    )
    assert detect_spin(abstract).present is False


def test_severe_spin_flagged() -> None:
    abstract = (
        "Methods: 200 patients.\n"
        "Results: Mortality was 25% vs 26%, P = 0.78. "
        "HR 0.95 (95% CI 0.80 to 1.13).\n"
        "Conclusions: This intervention is effective and warrants further use."
    )
    spin = detect_spin(abstract)
    assert spin.present is True
    assert spin.severity in {"moderate", "severe"}


def test_hedged_positive_with_nonsig_results_is_moderate() -> None:
    abstract = (
        "Methods: 100 patients.\n"
        "Results: HR 0.92 (95% CI 0.78 to 1.09), P = 0.34.\n"
        "Conclusions: The intervention may be effective and warrants further study."
    )
    spin = detect_spin(abstract)
    assert spin.present is True
    assert spin.severity == "moderate"
