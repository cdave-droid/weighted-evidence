"""Cochrane RoB 2 scaffold.

Provides a rule-based skeleton (registry presence, blinding keyword detection)
and the structured shape an LLM call fills in for the harder domains.
"""

from __future__ import annotations

import re

from weighted_evidence.models import Paper, RoB2Assessment, RoB2Domain, RoB2Judgment

_BLIND_RE = re.compile(r"\b(double[-\s]?blind|triple[-\s]?blind|placebo[-\s]?control)", re.I)
_NCT_RE = re.compile(r"\bNCT\d{8}\b")


def _outcome_measurement_judgment(paper: Paper) -> RoB2Judgment:
    text = f"{paper.title}\n{paper.abstract or ''}"
    return RoB2Judgment.low if _BLIND_RE.search(text) else RoB2Judgment.some_concerns


def _randomization_judgment(paper: Paper) -> RoB2Judgment:
    text = f"{paper.title}\n{paper.abstract or ''}"
    if _NCT_RE.search(text):
        return RoB2Judgment.low
    return RoB2Judgment.some_concerns


def skeleton_rob2(paper: Paper) -> RoB2Assessment:
    domains = [
        RoB2Domain(
            name="randomization",
            judgment=_randomization_judgment(paper),
            rationale=(
                "ClinicalTrials.gov registration ID detected in abstract."
                if _NCT_RE.search(paper.abstract or "")
                else "No registry ID detected; allocation concealment unverified."
            ),
        ),
        RoB2Domain(
            name="deviations",
            judgment=RoB2Judgment.some_concerns,
            rationale="Pending LLM-graded review of intention-to-treat handling.",
        ),
        RoB2Domain(
            name="missing_data",
            judgment=RoB2Judgment.some_concerns,
            rationale="Pending LLM-graded review of attrition and imputation.",
        ),
        RoB2Domain(
            name="outcome_measurement",
            judgment=_outcome_measurement_judgment(paper),
            rationale=(
                "Blinding keyword detected in abstract."
                if _BLIND_RE.search(f"{paper.title}\n{paper.abstract or ''}")
                else "No explicit blinding statement detected."
            ),
        ),
        RoB2Domain(
            name="selective_reporting",
            judgment=RoB2Judgment.some_concerns,
            rationale="Pending LLM-graded review of pre-specified outcomes.",
        ),
    ]

    overall = _overall(domains)
    return RoB2Assessment(domains=domains, overall=overall)


def _overall(domains: list[RoB2Domain]) -> RoB2Judgment:
    if any(d.judgment == RoB2Judgment.high for d in domains):
        return RoB2Judgment.high
    if any(d.judgment == RoB2Judgment.some_concerns for d in domains):
        return RoB2Judgment.some_concerns
    return RoB2Judgment.low
