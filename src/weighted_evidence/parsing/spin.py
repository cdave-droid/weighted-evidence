"""Spin detection — does the abstract conclusion overstate the actual results?

Conservative rule-based heuristics for the most common spin patterns
(Boutron 2010, Yavchitz 2014):

  - Conclusion uses positive language ("effective", "beneficial", "promising",
    "improvement", "significant clinical benefit") while the results section
    reports a non-significant primary outcome (CI crosses null or p >= 0.05).
  - Conclusion focuses on a non-primary or subgroup result rather than the
    prespecified primary outcome.
  - Conclusion overgeneralizes to populations not studied.

LLM refinement is a future hook; rule-based gives us reliable + cheap signal
for clearly spun abstracts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from weighted_evidence.models import SpinAssessment
from weighted_evidence.parsing.abstract import split_structured
from weighted_evidence.parsing.effects import extract_effects

_POSITIVE_LANGUAGE = re.compile(
    r"\b(?:effective(?:ly|ness)?|beneficial|promising|favo[u]?rable|"
    r"improve(?:s|d|ment)?|reduce(?:s|d|ction)?|significant\s+(?:clinical\s+)?benefit|"
    r"superior(?:ity)?|safe\s+and\s+effective)\b",
    re.I,
)

_NEGATIVE_LANGUAGE = re.compile(
    r"\b(?:not\s+(?:effective|significant)|no\s+(?:significant|benefit|"
    r"difference|effect)|failed\s+to\s+(?:show|demonstrate)|"
    r"did\s+not\s+(?:show|reduce|improve)|negative\s+trial)\b",
    re.I,
)

_HEDGE = re.compile(
    r"\b(?:may|might|could|suggest(?:s|ed)?|warrant(?:s)?\s+further|"
    r"hypothesis-generating|exploratory)\b",
    re.I,
)


@dataclass
class _ResultsSignal:
    has_significant_finding: bool
    has_nonsignificant_finding: bool
    rationale: str


def _scan_results(results_text: str) -> _ResultsSignal:
    if not results_text:
        return _ResultsSignal(False, False, "No results section text to scan.")

    effects = extract_effects(results_text)
    sig = False
    nonsig = False

    for e in effects:
        if e.p is not None and e.p < 0.05:
            sig = True
        if e.p is not None and e.p >= 0.05:
            nonsig = True
        if e.ci_low is not None and e.ci_high is not None:
            if e.kind.value in {"hr", "rr", "or"}:
                if e.ci_low > 1.0 or e.ci_high < 1.0:
                    sig = True
                else:
                    nonsig = True
            else:
                if e.ci_low > 0 or e.ci_high < 0:
                    sig = True
                else:
                    nonsig = True

    rationale = (
        f"Detected {len(effects)} effect estimate(s); significant={sig}, non-significant={nonsig}."
    )
    return _ResultsSignal(sig, nonsig, rationale)


def detect_spin(abstract: str | None) -> SpinAssessment:
    if not abstract:
        return SpinAssessment(present=False, severity=None, rationale="No abstract provided.")

    sections = split_structured(abstract)
    conclusion = sections.get("conclusions") or sections.get("conclusion") or ""
    results = sections.get("results", "")

    if not conclusion:
        return SpinAssessment(
            present=False,
            severity=None,
            rationale="No structured conclusion section detected; cannot compare to results.",
        )

    conclusion_positive = bool(_POSITIVE_LANGUAGE.search(conclusion))
    conclusion_negative = bool(_NEGATIVE_LANGUAGE.search(conclusion))
    conclusion_hedged = bool(_HEDGE.search(conclusion))

    signal = _scan_results(results)

    # Classic spin: the conclusion frames the trial as positive when the
    # primary results are non-significant.
    if (
        conclusion_positive
        and not conclusion_negative
        and signal.has_nonsignificant_finding
        and not signal.has_significant_finding
    ):
        severity: Literal["mild", "moderate", "severe"] = (
            "moderate" if conclusion_hedged else "severe"
        )
        return SpinAssessment(
            present=True,
            severity=severity,
            rationale=(
                "Conclusion uses positive language but results section shows only "
                "non-significant findings. " + signal.rationale
            ),
        )

    # Mild spin: conclusion is positive, results contain *both* significant and
    # non-significant findings — possible cherry-picking of a subgroup result.
    if conclusion_positive and signal.has_significant_finding and signal.has_nonsignificant_finding:
        return SpinAssessment(
            present=True,
            severity="mild",
            rationale=(
                "Conclusion is positive; results report a mix of significant and "
                "non-significant findings — verify the primary outcome aligns. " + signal.rationale
            ),
        )

    return SpinAssessment(
        present=False,
        severity=None,
        rationale=("Conclusion language and reported effects are consistent. " + signal.rationale),
    )


__all__ = ["detect_spin"]
