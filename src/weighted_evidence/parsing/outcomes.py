"""Classify outcome names into the `OutcomeImportance` hierarchy + extract outcome rows.

Patient-important outcomes (mortality, QoL, hospitalization) outrank surrogate
outcomes (lab values, imaging) — this is what lets the framework rank
"mortality reduction" above "biomarker improvement" even at the same GRADE.
"""

from __future__ import annotations

import re

from weighted_evidence.models import EffectSize, Outcome, OutcomeImportance
from weighted_evidence.parsing.effects import extract_effects

_PATTERNS: list[tuple[OutcomeImportance, list[re.Pattern[str]]]] = [
    (
        OutcomeImportance.mortality,
        [
            re.compile(r"\b(all[-\s]?cause\s+)?mortalit(?:y|ies)\b", re.I),
            re.compile(r"\b(?:death|deaths|fatal|case\s+fatality)\b", re.I),
            re.compile(r"\b(?:in[-\s]?hospital|28[-\s]?day|30[-\s]?day|90[-\s]?day)\s+death", re.I),
        ],
    ),
    # hospitalization is checked BEFORE morbidity because phrases like
    # "hospitalization for heart failure" should classify as hospitalization,
    # not morbidity (the heart-failure morbidity pattern would otherwise win).
    (
        OutcomeImportance.hospitalization,
        [
            re.compile(r"\b(?:hospitali[sz]ation|re[-\s]?admission|ICU\s+admission)\b", re.I),
            re.compile(r"\blength\s+of\s+stay\b", re.I),
        ],
    ),
    (
        OutcomeImportance.morbidity,
        [
            re.compile(r"\b(?:stroke|myocardial infarction|MI|heart failure)\b", re.I),
            re.compile(r"\b(?:major adverse cardiovascular|MACE)\b", re.I),
            re.compile(r"\b(?:sepsis|organ failure|SOFA|disability)\b", re.I),
            re.compile(r"\b(?:incidence of [A-Za-z ]+(?:disease|infection|injury))\b", re.I),
        ],
    ),
    (
        OutcomeImportance.qol,
        [
            re.compile(r"\b(?:quality\s+of\s+life|qol|qaly|eq[-\s]?5d|sf[-\s]?36|kccq)\b", re.I),
            re.compile(r"\bpatient[-\s]?reported\s+outcome", re.I),
        ],
    ),
    (
        OutcomeImportance.symptomatic,
        [
            re.compile(r"\b(?:pain|nausea|dyspnea|fatigue|symptom\s+score)\b", re.I),
            re.compile(r"\b(?:NYHA\s+class|MRC\s+dyspnea)\b", re.I),
        ],
    ),
    (
        OutcomeImportance.surrogate_imaging,
        [
            re.compile(r"\b(?:tumor\s+size|lesion\s+volume|infarct\s+volume)\b", re.I),
            re.compile(r"\b(?:LV\s+ejection\s+fraction|LVEF|coronary\s+stenosis)\b", re.I),
        ],
    ),
    (
        OutcomeImportance.surrogate_lab,
        [
            re.compile(r"\b(?:HbA1c|glycated\s+hemoglobin|fasting\s+glucose)\b", re.I),
            re.compile(r"\b(?:LDL|HDL|cholesterol|triglycerides|blood\s+pressure)\b", re.I),
            re.compile(r"\b(?:CRP|C[-\s]?reactive\s+protein|procalcitonin|lactate)\b", re.I),
            re.compile(r"\b(?:viral\s+load|CD4\s+count|biomarker)\b", re.I),
        ],
    ),
    (
        OutcomeImportance.composite,
        [
            re.compile(r"\bcomposite\s+(?:outcome|endpoint)\b", re.I),
            re.compile(r"\bprimary\s+composite\b", re.I),
        ],
    ),
]


def classify_importance(name: str) -> OutcomeImportance:
    if not name:
        return OutcomeImportance.other
    for importance, patterns in _PATTERNS:
        if any(p.search(name) for p in patterns):
            return importance
    return OutcomeImportance.other


_PRIMARY_RE = re.compile(
    r"primary\s+(?:outcome|endpoint)[s]?\s*(?:was|were|of\s+interest|:)?\s*(?P<rest>[^.]{5,200})\.",
    re.I,
)


def extract_outcomes(abstract: str | None) -> list[Outcome]:
    """Best-effort outcome extraction from a structured abstract.

    Pairs primary-outcome phrases with effect sizes found nearby. Anything that
    isn't matched by rule remains for the LLM extractor (PR 4 will wire that in).
    """

    if not abstract:
        return []

    outcomes: list[Outcome] = []
    primary_names: list[str] = []
    for m in _PRIMARY_RE.finditer(abstract):
        rest = m.group("rest").strip()
        primary_names.extend(_split_outcome_phrase(rest))

    for raw in primary_names:
        importance = classify_importance(raw)
        effects = _associate_effect(abstract, raw)
        outcomes.append(
            Outcome(
                name=raw[:160],
                importance=importance,
                is_primary=True,
                effect=effects[0] if effects else None,
            )
        )

    # Fallback: keyword sweep — when no primary outcome was named explicitly
    # but "mortality" appears alongside a percentage difference, emit a single
    # mortality outcome (most common high-value pattern in CC abstracts).
    if not outcomes and re.search(r"\bmortalit", abstract, re.I):
        effects = extract_effects(abstract)
        arr = next((e for e in effects if e.kind.value == "arr"), None)
        outcomes.append(
            Outcome(
                name="Mortality",
                importance=OutcomeImportance.mortality,
                is_primary=True,
                effect=arr,
            )
        )

    return outcomes


def _split_outcome_phrase(phrase: str) -> list[str]:
    parts = re.split(r"\s*(?:,|;|\band\b|\bor\b)\s*", phrase)
    return [p.strip() for p in parts if p.strip()]


def _associate_effect(abstract: str, name: str) -> list[EffectSize]:
    # If the outcome name appears in the abstract, look for effect sizes within
    # the surrounding ±240 char window; otherwise scan the whole abstract.
    pos = abstract.lower().find(name.lower())
    if pos == -1:
        return extract_effects(abstract)
    start = max(0, pos - 240)
    end = min(len(abstract), pos + len(name) + 240)
    local = extract_effects(abstract[start:end])
    return local or extract_effects(abstract)


__all__ = ["classify_importance", "extract_outcomes"]
