"""Effect-size extraction."""

from __future__ import annotations

from weighted_evidence.models import EffectKind, MIDComparison
from weighted_evidence.parsing.effects import extract_effects


def test_hr_with_ci_excludes_null() -> None:
    text = "The hazard ratio for death was 0.78 (95% CI 0.65 to 0.94, P = 0.01)."
    effects = extract_effects(text)
    assert len(effects) == 1
    e = effects[0]
    assert e.kind == EffectKind.hr
    assert e.point == 0.78
    assert e.ci_low == 0.65
    assert e.ci_high == 0.94
    assert e.p == 0.01
    assert e.vs_mid == MIDComparison.exceeds


def test_hr_with_ci_crossing_null() -> None:
    text = "HR 0.95 (95% CI 0.80 to 1.13)"
    effects = extract_effects(text)
    assert effects[0].vs_mid == MIDComparison.crosses_null


def test_hr_with_en_dash_separator() -> None:
    text = "HR 0.85 (95% CI 0.75–0.97)"  # U+2013 between bounds
    effects = extract_effects(text)
    assert effects and effects[0].ci_low == 0.75
    assert effects[0].ci_high == 0.97


def test_rr_pattern() -> None:
    text = "The relative risk was 1.45 [95% CI 1.10-1.92]."
    effects = extract_effects(text)
    assert effects[0].kind == EffectKind.rr
    assert effects[0].ci_low == 1.10


def test_or_pattern() -> None:
    text = "OR 2.3 (1.5 to 3.5)"
    effects = extract_effects(text)
    assert effects[0].kind == EffectKind.or_
    assert effects[0].point == 2.3


def test_two_group_percentages_yield_arr() -> None:
    text = "Mortality was 31.0% vs 39.8%, P = 0.007"
    effects = extract_effects(text)
    arr = next(e for e in effects if e.kind == EffectKind.arr)
    assert arr.point == 0.088
    assert arr.p == 0.007


def test_no_match_returns_empty() -> None:
    assert extract_effects("No numeric results in this sentence.") == []


def test_lancet_middle_dot_decimal() -> None:
    """Lancet, BMJ, Cochrane Library use Unicode middle dot (U+00B7) as decimal."""

    text = "hydroxychloroquine (18·0%; hazard ratio 1·335, 95% CI 1·223-1·457)"
    effects = extract_effects(text)
    assert effects, "Should extract HR despite middle-dot decimals"
    hr = next(e for e in effects if e.kind == EffectKind.hr)
    assert hr.point == 1.335
    assert hr.ci_low == 1.223
    assert hr.ci_high == 1.457
    assert hr.vs_mid == MIDComparison.exceeds


def test_middle_dot_two_group_percentages() -> None:
    text = "Mortality was 18·0% vs 9·3%, P = 0·001"
    effects = extract_effects(text)
    arr = next(e for e in effects if e.kind == EffectKind.arr)
    assert arr.p == 0.001
    assert abs(arr.point - (0.093 - 0.180)) < 1e-9
