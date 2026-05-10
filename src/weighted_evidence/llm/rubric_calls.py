"""Structured-output LLM calls that refine the rule-only rubric skeletons.

Each helper wraps an `LLMProvider.structured()` call with a tool schema and
parses the result back into our Pydantic types. Calls are no-ops when the
agent has no LLM provider configured — rule-only behavior is preserved.
"""

from __future__ import annotations

from typing import Any

from weighted_evidence.llm.base import LLMProvider
from weighted_evidence.models import (
    GradeAssessment,
    GradeCertainty,
    GradeDomain,
    GradeDomainJudgment,
    Outcome,
    Paper,
    RoB2Assessment,
    RoB2Domain,
    RoB2Judgment,
    SpinAssessment,
)
from weighted_evidence.rubric.grade import apply_modifiers

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_GRADE_DOMAIN_NAMES = (
    "risk_of_bias",
    "inconsistency",
    "indirectness",
    "imprecision",
    "publication_bias",
)

_GRADE_MODIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "downgrades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": list(_GRADE_DOMAIN_NAMES)},
                    "judgment": {
                        "type": "string",
                        "enum": ["not_serious", "serious", "very_serious"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["name", "judgment", "rationale"],
            },
        },
        "upgrades": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["large_effect", "dose_response", "plausible_confounding"],
            },
        },
        "rationale": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["downgrades", "rationale"],
}

_ROB2_DOMAIN_NAMES = (
    "randomization",
    "deviations",
    "missing_data",
    "outcome_measurement",
    "selective_reporting",
)

_ROB2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "domains": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": list(_ROB2_DOMAIN_NAMES)},
                    "judgment": {
                        "type": "string",
                        "enum": ["low", "some_concerns", "high"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["name", "judgment", "rationale"],
            },
        },
        "overall": {"type": "string", "enum": ["low", "some_concerns", "high"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["domains", "overall"],
}

_OUTCOME_NORM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "outcomes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "raw": {"type": "string"},
                    "normalized": {"type": "string"},
                    "is_primary": {"type": "boolean"},
                    "importance": {
                        "type": "string",
                        "enum": [
                            "mortality",
                            "morbidity",
                            "qol",
                            "hospitalization",
                            "symptomatic",
                            "surrogate_lab",
                            "surrogate_imaging",
                            "composite",
                            "other",
                        ],
                    },
                    "timepoint": {"type": ["string", "null"]},
                },
                "required": ["raw", "normalized", "is_primary", "importance"],
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["outcomes"],
}

_SPIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "severity": {"type": ["string", "null"], "enum": ["mild", "moderate", "severe", None]},
        "rationale": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["present", "rationale"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _system_for_grade() -> str:
    return (
        "You are a clinical-evidence appraiser applying GRADE to a single "
        "study's contribution to evidence on its primary outcome. Assess each "
        "downgrade domain (risk_of_bias, inconsistency, indirectness, "
        "imprecision, publication_bias). Be conservative — only flag a domain "
        "as 'serious' or 'very_serious' when the abstract supports it. When "
        "uncertain, return 'not_serious' and explain what would have been "
        "needed. Return the structured tool call only."
    )


def _system_for_rob2() -> str:
    return (
        "You are applying Cochrane RoB 2 to a single randomized trial. For "
        "each of the five domains (randomization, deviations from intended "
        "interventions, missing outcome data, measurement of the outcome, "
        "selective reporting), assess 'low', 'some_concerns', or 'high' risk "
        "of bias. Anchor every judgment in text from the abstract that you "
        "can quote. When the abstract is silent on a signaling question, "
        "default to 'some_concerns' rather than 'low'. The overall rating "
        "follows the worst domain. Return the structured tool call only."
    )


def _system_for_outcomes() -> str:
    return (
        "You are extracting the prespecified outcomes from a clinical-trial "
        "abstract. For each, return a short normalized name (e.g., '28-day "
        "mortality'), is_primary, importance class, and timepoint. Importance "
        "classes: mortality > morbidity > qol > hospitalization > symptomatic "
        "> surrogate_imaging > surrogate_lab > composite > other. Return the "
        "structured tool call only."
    )


def _system_for_spin() -> str:
    return (
        "You are detecting spin in a clinical-trial abstract — a discrepancy "
        "between the conclusion's framing and the actual reported results. "
        "Flag spin when (a) the conclusion uses positive language while the "
        "primary outcome was non-significant, (b) the conclusion focuses on a "
        "subgroup rather than the prespecified primary outcome, or (c) the "
        "conclusion overgeneralizes to populations not studied. Severity: "
        "mild, moderate, severe. Return the structured tool call only."
    )


def _user_for_paper(paper: Paper, *, focus: str) -> str:
    parts = [
        f"Title: {paper.title}",
        f"Journal: {paper.journal or 'unknown'}",
        f"Design: {paper.design.value}",
    ]
    if paper.publication_date:
        parts.append(f"Year: {paper.publication_date.year}")
    if paper.sample_size:
        parts.append(f"Sample size: {paper.sample_size}")
    if paper.abstract:
        parts.append(f"\nAbstract:\n{paper.abstract}")
    parts.append(f"\nFocus: {focus}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public refiners
# ---------------------------------------------------------------------------


async def refine_grade(paper: Paper, base: GradeAssessment, *, llm: LLMProvider) -> GradeAssessment:
    response = await llm.structured(
        system=_system_for_grade(),
        user=_user_for_paper(paper, focus="GRADE downgrade and upgrade modifiers."),
        tool_name="emit_grade_modifiers",
        tool_description="Emit GRADE downgrade domains + any upgrade reasons.",
        tool_schema=_GRADE_MODIFIER_SCHEMA,
    )
    payload = response.content or {}
    downgrades_raw = payload.get("downgrades", []) or []
    upgrades_raw = payload.get("upgrades", []) or []

    downgrades = []
    for d in downgrades_raw:
        try:
            downgrades.append(
                GradeDomain(
                    name=d["name"],
                    judgment=GradeDomainJudgment(d["judgment"]),
                    rationale=d.get("rationale", ""),
                )
            )
        except (KeyError, ValueError):
            continue
    upgrades = [
        u for u in upgrades_raw if u in {"large_effect", "dose_response", "plausible_confounding"}
    ]
    final_certainty: GradeCertainty = apply_modifiers(
        base.starting_certainty, downgrades=downgrades, upgrades=upgrades
    )
    return GradeAssessment(
        starting_certainty=base.starting_certainty,
        final_certainty=final_certainty,
        downgrades=downgrades,
        upgrades=upgrades,
        rationale=payload.get("rationale") or base.rationale,
    )


async def refine_rob2(paper: Paper, base: RoB2Assessment, *, llm: LLMProvider) -> RoB2Assessment:
    response = await llm.structured(
        system=_system_for_rob2(),
        user=_user_for_paper(paper, focus="Cochrane RoB 2 per-domain judgment."),
        tool_name="emit_rob2_domains",
        tool_description="Emit RoB 2 per-domain judgments + overall.",
        tool_schema=_ROB2_SCHEMA,
    )
    payload = response.content or {}
    domains_raw = payload.get("domains", []) or []
    domains: list[RoB2Domain] = []
    for d in domains_raw:
        try:
            domains.append(
                RoB2Domain(
                    name=d["name"],
                    judgment=RoB2Judgment(d["judgment"]),
                    rationale=d.get("rationale", ""),
                )
            )
        except (KeyError, ValueError):
            continue
    if not domains:
        return base
    overall_raw = payload.get("overall")
    try:
        overall = RoB2Judgment(overall_raw) if overall_raw else _worst(domains)
    except ValueError:
        overall = _worst(domains)
    return RoB2Assessment(domains=domains, overall=overall)


def _worst(domains: list[RoB2Domain]) -> RoB2Judgment:
    if any(d.judgment == RoB2Judgment.high for d in domains):
        return RoB2Judgment.high
    if any(d.judgment == RoB2Judgment.some_concerns for d in domains):
        return RoB2Judgment.some_concerns
    return RoB2Judgment.low


async def refine_outcomes(paper: Paper, *, llm: LLMProvider) -> list[Outcome] | None:
    """Use the LLM to normalize outcome names and assign importance classes.

    Returns None when the LLM call fails or yields nothing useful so callers
    can fall back to the rule-based extraction.
    """

    if not paper.abstract:
        return None
    response = await llm.structured(
        system=_system_for_outcomes(),
        user=_user_for_paper(paper, focus="Extract prespecified outcomes."),
        tool_name="emit_outcomes",
        tool_description="Emit normalized outcome rows.",
        tool_schema=_OUTCOME_NORM_SCHEMA,
    )
    rows = (response.content or {}).get("outcomes", []) or []
    if not rows:
        return None
    out: list[Outcome] = []
    for r in rows:
        try:
            from weighted_evidence.models import OutcomeImportance

            out.append(
                Outcome(
                    name=r.get("normalized") or r.get("raw") or "",
                    importance=OutcomeImportance(r.get("importance", "other")),
                    is_primary=bool(r.get("is_primary", False)),
                    timepoint=r.get("timepoint"),
                )
            )
        except (KeyError, ValueError):
            continue
    return out or None


async def refine_spin(paper: Paper, *, llm: LLMProvider) -> SpinAssessment | None:
    if not paper.abstract:
        return None
    response = await llm.structured(
        system=_system_for_spin(),
        user=_user_for_paper(paper, focus="Detect spin in the abstract conclusion."),
        tool_name="emit_spin_assessment",
        tool_description="Emit spin presence + severity.",
        tool_schema=_SPIN_SCHEMA,
    )
    payload = response.content or {}
    if "present" not in payload:
        return None
    severity_raw = payload.get("severity")
    severity = severity_raw if severity_raw in {"mild", "moderate", "severe"} else None
    return SpinAssessment(
        present=bool(payload["present"]),
        severity=severity,
        rationale=payload.get("rationale", ""),
    )


__all__ = ["refine_grade", "refine_outcomes", "refine_rob2", "refine_spin"]
