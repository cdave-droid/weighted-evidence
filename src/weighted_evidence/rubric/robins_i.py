"""ROBINS-I skeleton for non-randomized intervention studies.

Closes the largest reliability gap from Phase 1's RCT-only RoB tooling:
observational cohorts, case-control, comparative effectiveness, post-marketing
safety. Seven domains:

  1. Confounding (pre-intervention)
  2. Selection of participants (pre-intervention)
  3. Classification of interventions (at-intervention)
  4. Deviations from intended interventions (post-intervention)
  5. Missing data (post-intervention)
  6. Measurement of outcomes (post-intervention)
  7. Selection of the reported result (post-intervention)

Output: per-domain Low / Moderate / Serious / Critical / No Information +
overall (worst domain). Confounding plausibility and missing-data handling
default to "no_information" until the LLM step lands; explicit signals
in the abstract (propensity scores, IV / DiD / target-trial-emulation,
sensitivity analyses) drop those domains to "moderate" or "low".
"""

from __future__ import annotations

import re

from weighted_evidence.models import Paper, RobinsIAssessment, RobinsIDomain, RobinsIJudgment

_ADJUSTMENT_RE = re.compile(
    r"\b(?:propensity\s+score|inverse\s+probability\s+weighting|IPW|"
    r"instrumental\s+variable|difference[-\s]?in[-\s]?differences?|"
    r"target\s+trial(?:\s+emulation)?|adjusted\s+for|multivariable\s+(?:adjustment|"
    r"regression)|stratified\s+analysis)\b",
    re.I,
)
_SENSITIVITY_RE = re.compile(
    r"\b(?:sensitivity\s+analys[ie]s|E-?value|negative\s+control(?:\s+outcome)?)\b", re.I
)
_BLINDED_OUTCOME_RE = re.compile(
    r"\b(?:blinded\s+(?:outcome\s+)?(?:assessment|adjudication)|adjudicated\s+outcomes)\b", re.I
)
_PRESPECIFIED_RE = re.compile(
    r"\b(?:pre[-\s]?specified|pre[-\s]?registered|prospectively\s+registered|"
    r"a[-\s]?priori\s+analysis\s+plan|protocol\s+published)\b",
    re.I,
)
_LOW_LOSS_RE = re.compile(
    r"\b(?:complete[-\s]?case|<\s*5\s*%\s+missing|no\s+loss\s+to\s+follow[-\s]?up)\b", re.I
)


def _domain(name: str, judgment: RobinsIJudgment, rationale: str) -> RobinsIDomain:
    return RobinsIDomain(
        name=name,
        judgment=judgment,
        rationale=rationale,
    )


def skeleton_robins_i(paper: Paper) -> RobinsIAssessment:
    text = f"{paper.title}\n{paper.abstract or ''}"
    has_adjustment = bool(_ADJUSTMENT_RE.search(text))
    has_sensitivity = bool(_SENSITIVITY_RE.search(text))
    has_blinded = bool(_BLINDED_OUTCOME_RE.search(text))
    has_prespecified = bool(_PRESPECIFIED_RE.search(text))
    has_low_loss = bool(_LOW_LOSS_RE.search(text))

    confounding_judgment = (
        RobinsIJudgment.moderate
        if has_adjustment and has_sensitivity
        else RobinsIJudgment.serious
        if has_adjustment
        else RobinsIJudgment.no_information
    )
    confounding_rationale = (
        "Adjustment + sensitivity analyses detected."
        if has_adjustment and has_sensitivity
        else "Adjustment detected but no sensitivity analysis (E-value / IV / negative control)."
        if has_adjustment
        else "No adjustment language detected; residual confounding cannot be ruled out."
    )

    domains = [
        _domain("confounding", confounding_judgment, confounding_rationale),
        _domain(
            "selection",
            RobinsIJudgment.no_information,
            "Selection of participants: requires LLM review of inclusion / exclusion details.",
        ),
        _domain(
            "classification",
            RobinsIJudgment.no_information,
            "Intervention classification: requires LLM review of exposure ascertainment.",
        ),
        _domain(
            "deviations",
            RobinsIJudgment.no_information,
            "Deviations from intended interventions: requires LLM review of cross-over / "
            "co-interventions.",
        ),
        _domain(
            "missing_data",
            RobinsIJudgment.low if has_low_loss else RobinsIJudgment.no_information,
            "Low loss to follow-up reported."
            if has_low_loss
            else "Missing-data handling: requires LLM review.",
        ),
        _domain(
            "measurement",
            RobinsIJudgment.low if has_blinded else RobinsIJudgment.moderate,
            "Blinded / adjudicated outcome assessment detected."
            if has_blinded
            else "No blinded outcome assessment detected; outcome measurement may favor exposure group.",
        ),
        _domain(
            "selective_reporting",
            RobinsIJudgment.low if has_prespecified else RobinsIJudgment.moderate,
            "Pre-specified analysis plan / pre-registration detected."
            if has_prespecified
            else "No pre-specified analysis plan detected; selective reporting cannot be ruled out.",
        ),
    ]

    overall = _overall(domains)
    return RobinsIAssessment(domains=domains, overall=overall)


_RANK = {
    RobinsIJudgment.low: 0,
    RobinsIJudgment.moderate: 1,
    RobinsIJudgment.serious: 2,
    RobinsIJudgment.critical: 3,
    RobinsIJudgment.no_information: 1,  # treat as moderate for overall purposes
}


def _overall(domains: list[RobinsIDomain]) -> RobinsIJudgment:
    worst = max(domains, key=lambda d: _RANK[d.judgment])
    return worst.judgment


__all__ = ["skeleton_robins_i"]
