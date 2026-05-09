"""Per-outcome clinical_significance verdict synthesis."""

from __future__ import annotations

from weighted_evidence.models import (
    ClinicalSignificanceVerdict,
    EffectKind,
    EffectSize,
    MIDComparison,
    Outcome,
    OutcomeImportance,
)
from weighted_evidence.rubric.clinical_significance import annotate, assess


def _outcome(
    importance: OutcomeImportance,
    *,
    kind: EffectKind = EffectKind.hr,
    point: float = 0.78,
    ci: tuple[float, float] | None = (0.65, 0.94),
    p: float | None = 0.01,
    vs_mid: MIDComparison = MIDComparison.exceeds,
    name: str = "Mortality",
) -> Outcome:
    effect = EffectSize(
        kind=kind,
        point=point,
        ci_low=ci[0] if ci else None,
        ci_high=ci[1] if ci else None,
        p=p,
        vs_mid=vs_mid,
    )
    return Outcome(name=name, importance=importance, is_primary=True, effect=effect)


def test_mortality_with_excluding_ci_is_likely() -> None:
    verdict = assess(_outcome(OutcomeImportance.mortality))
    assert verdict.verdict == ClinicalSignificanceVerdict.likely


def test_mortality_with_ci_crossing_null_is_unlikely() -> None:
    verdict = assess(
        _outcome(
            OutcomeImportance.mortality,
            ci=(0.85, 1.10),
            p=0.20,
            vs_mid=MIDComparison.crosses_null,
        )
    )
    assert verdict.verdict == ClinicalSignificanceVerdict.unlikely


def test_surrogate_with_significant_effect_is_uncertain() -> None:
    verdict = assess(
        _outcome(
            OutcomeImportance.surrogate_lab,
            kind=EffectKind.md,
            point=-0.7,
            ci=(-1.1, -0.3),
            p=0.001,
            vs_mid=MIDComparison.exceeds,
            name="HbA1c reduction",
        )
    )
    assert verdict.verdict == ClinicalSignificanceVerdict.uncertain


def test_no_effect_size_yields_uncertain() -> None:
    o = Outcome(name="something", importance=OutcomeImportance.mortality)
    verdict = assess(o)
    assert verdict.verdict == ClinicalSignificanceVerdict.uncertain


def test_annotate_skips_already_assessed() -> None:
    pre = _outcome(OutcomeImportance.mortality)
    pre = pre.model_copy(
        update={
            "clinical_significance": assess(pre),
        }
    )
    out = annotate([pre])
    assert out[0].clinical_significance is pre.clinical_significance
