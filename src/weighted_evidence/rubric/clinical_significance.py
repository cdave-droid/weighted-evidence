"""Per-outcome `ClinicalSignificance` verdict.

This is a deliberate rule-based synthesis of:
  - effect direction + magnitude
  - confidence interval position (excludes null? exceeds MID?)
  - outcome importance class (mortality > morbidity > qol > surrogate)

The verdict + rationale are what an agent reads instead of re-interpreting
raw stats. When data is missing (no CI, no effect size), we return
`uncertain` with a clear reason so agents know to look further.
"""

from __future__ import annotations

from weighted_evidence.models import (
    ClinicalSignificance,
    ClinicalSignificanceVerdict,
    EffectKind,
    EffectSize,
    MIDComparison,
    Outcome,
    OutcomeImportance,
)

_PATIENT_IMPORTANT = {
    OutcomeImportance.mortality,
    OutcomeImportance.morbidity,
    OutcomeImportance.qol,
    OutcomeImportance.hospitalization,
    OutcomeImportance.symptomatic,
}

_SURROGATE = {
    OutcomeImportance.surrogate_lab,
    OutcomeImportance.surrogate_imaging,
}


def _ci_excludes_null(effect: EffectSize) -> bool:
    if effect.ci_low is None or effect.ci_high is None:
        return False
    if effect.kind in {EffectKind.hr, EffectKind.rr, EffectKind.or_}:
        return effect.ci_low > 1.0 or effect.ci_high < 1.0
    if effect.kind in {EffectKind.arr, EffectKind.md, EffectKind.smd, EffectKind.nnt}:
        return effect.ci_low > 0 or effect.ci_high < 0
    return False


def _statistically_significant(effect: EffectSize) -> bool:
    if effect.p is not None and effect.p < 0.05:
        return True
    return _ci_excludes_null(effect)


def assess(outcome: Outcome) -> ClinicalSignificance:
    importance = outcome.importance
    effect = outcome.effect

    if effect is None:
        return ClinicalSignificance(
            verdict=ClinicalSignificanceVerdict.uncertain,
            rationale=(
                f"No effect size extracted for outcome '{outcome.name}'. "
                "Statistical and clinical significance cannot be assessed."
            ),
        )

    sig = _statistically_significant(effect)
    excludes_null = _ci_excludes_null(effect)

    # Patient-important outcomes — the bar to call this 'likely' is met when
    # the CI excludes the null and either a MID is exceeded or no MID is
    # established but the effect is on a high-importance outcome.
    if importance in _PATIENT_IMPORTANT:
        if excludes_null and effect.vs_mid in {MIDComparison.exceeds, MIDComparison.meets}:
            return ClinicalSignificance(
                verdict=ClinicalSignificanceVerdict.likely,
                rationale=(
                    f"Patient-important outcome ({importance.value}); "
                    "CI excludes null and effect meets/exceeds MID."
                ),
            )
        if excludes_null:
            return ClinicalSignificance(
                verdict=ClinicalSignificanceVerdict.likely,
                rationale=(
                    f"Patient-important outcome ({importance.value}); "
                    "CI fully excludes the null. MID not established for this measure — "
                    "verdict reflects effect on a high-importance endpoint."
                ),
            )
        if sig:
            return ClinicalSignificance(
                verdict=ClinicalSignificanceVerdict.uncertain,
                rationale=(
                    f"Patient-important outcome ({importance.value}); p<0.05 but CI not "
                    "fully reported or includes null — clinical relevance uncertain."
                ),
            )
        return ClinicalSignificance(
            verdict=ClinicalSignificanceVerdict.unlikely,
            rationale=(
                f"Patient-important outcome ({importance.value}); effect is not "
                "statistically significant and CI is consistent with no benefit."
            ),
        )

    # Surrogate outcomes — even when statistically significant, the verdict is
    # capped at 'uncertain' because the link to patient-important outcomes is
    # not established by this paper alone.
    if importance in _SURROGATE:
        if sig and effect.vs_mid == MIDComparison.exceeds:
            return ClinicalSignificance(
                verdict=ClinicalSignificanceVerdict.uncertain,
                rationale=(
                    f"Surrogate outcome ({importance.value}); statistically significant "
                    "and MID exceeded, but downstream patient benefit is not demonstrated."
                ),
            )
        if sig:
            return ClinicalSignificance(
                verdict=ClinicalSignificanceVerdict.uncertain,
                rationale=(
                    f"Surrogate outcome ({importance.value}); statistically significant "
                    "but MID status unclear and patient-important benefit not demonstrated."
                ),
            )
        return ClinicalSignificance(
            verdict=ClinicalSignificanceVerdict.unlikely,
            rationale=(f"Surrogate outcome ({importance.value}); not statistically significant."),
        )

    # Composite or other — defer.
    return ClinicalSignificance(
        verdict=ClinicalSignificanceVerdict.uncertain,
        rationale=(
            f"Outcome class '{importance.value}' does not map cleanly to "
            "patient-important / surrogate; manual review recommended."
        ),
    )


def annotate(outcomes: list[Outcome]) -> list[Outcome]:
    """Return a new list with each outcome's clinical_significance populated."""

    result: list[Outcome] = []
    for o in outcomes:
        if o.clinical_significance is not None:
            result.append(o)
            continue
        verdict = assess(o)
        result.append(o.model_copy(update={"clinical_significance": verdict}))
    return result


__all__ = ["annotate", "assess"]
