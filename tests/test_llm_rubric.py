"""LLM-graded rubric refinement (mocked provider)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from weighted_evidence.agents import EvidenceAgent
from weighted_evidence.cache import Cache
from weighted_evidence.llm.base import LLMResponse
from weighted_evidence.llm.rubric_calls import (
    refine_grade,
    refine_outcomes,
    refine_rob2,
    refine_spin,
)
from weighted_evidence.models import (
    GradeAssessment,
    GradeCertainty,
    Identifier,
    OutcomeImportance,
    Paper,
    RoB2Assessment,
    RoB2Domain,
    RoB2Judgment,
    StudyDesign,
)


class StubProvider:
    name = "stub"

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def structured(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        tool_schema: dict[str, Any],
        cacheable_system: bool = True,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self.calls.append(tool_name)
        return LLMResponse(content=self.responses.get(tool_name, {}), model="stub")


def _paper() -> Paper:
    return Paper(
        identifier=Identifier(pmid="1"),
        title="A randomized trial of X",
        abstract="Background. Methods. Results. Conclusions.",
        journal="N Engl J Med",
        publication_date=datetime(2020, 1, 1),
        publication_types=["Randomized Controlled Trial"],
        design=StudyDesign.rct,
        sample_size=500,
    )


@pytest.mark.asyncio
async def test_refine_grade_applies_modifiers() -> None:
    base = GradeAssessment(
        starting_certainty=GradeCertainty.high,
        final_certainty=GradeCertainty.high,
    )
    provider = StubProvider(
        {
            "emit_grade_modifiers": {
                "downgrades": [
                    {"name": "imprecision", "judgment": "serious", "rationale": "Wide CI."},
                ],
                "upgrades": [],
                "rationale": "Single moderate downgrade.",
                "confidence": 0.9,
            }
        }
    )
    refined = await refine_grade(_paper(), base, llm=provider)
    assert refined.final_certainty == GradeCertainty.moderate
    assert refined.downgrades and refined.downgrades[0].name == "imprecision"


@pytest.mark.asyncio
async def test_refine_rob2_replaces_skeleton() -> None:
    base = RoB2Assessment(
        domains=[
            RoB2Domain(name="randomization", judgment=RoB2Judgment.some_concerns, rationale=""),
        ],
        overall=RoB2Judgment.some_concerns,
    )
    provider = StubProvider(
        {
            "emit_rob2_domains": {
                "domains": [
                    {
                        "name": "randomization",
                        "judgment": "low",
                        "rationale": "Computer-generated.",
                    },
                    {"name": "deviations", "judgment": "low", "rationale": "ITT analysis."},
                    {"name": "missing_data", "judgment": "low", "rationale": "<5% loss."},
                    {
                        "name": "outcome_measurement",
                        "judgment": "low",
                        "rationale": "Blinded adjudication.",
                    },
                    {
                        "name": "selective_reporting",
                        "judgment": "low",
                        "rationale": "Pre-registered primary outcome.",
                    },
                ],
                "overall": "low",
            }
        }
    )
    refined = await refine_rob2(_paper(), base, llm=provider)
    assert refined.overall == RoB2Judgment.low
    assert len(refined.domains) == 5


@pytest.mark.asyncio
async def test_refine_outcomes_normalizes_names() -> None:
    provider = StubProvider(
        {
            "emit_outcomes": {
                "outcomes": [
                    {
                        "raw": "death before a patient was discharged home and breathing without assistance",
                        "normalized": "In-hospital mortality",
                        "is_primary": True,
                        "importance": "mortality",
                        "timepoint": "in-hospital",
                    },
                    {
                        "raw": "ventilator-free days",
                        "normalized": "Ventilator-free days",
                        "is_primary": True,
                        "importance": "morbidity",
                        "timepoint": "28 days",
                    },
                ],
            }
        }
    )
    out = await refine_outcomes(_paper(), llm=provider)
    assert out is not None
    assert out[0].name == "In-hospital mortality"
    assert out[0].importance == OutcomeImportance.mortality
    assert out[1].importance == OutcomeImportance.morbidity


@pytest.mark.asyncio
async def test_refine_spin_returns_assessment() -> None:
    provider = StubProvider(
        {
            "emit_spin_assessment": {
                "present": True,
                "severity": "moderate",
                "rationale": "Conclusion frames a non-significant primary outcome positively.",
            }
        }
    )
    spin = await refine_spin(_paper(), llm=provider)
    assert spin is not None
    assert spin.present is True
    assert spin.severity == "moderate"


@pytest.mark.asyncio
async def test_agent_uses_llm_when_provided(tmp_path) -> None:
    """End-to-end: agent with stub LLM applies all four refiners."""

    provider = StubProvider(
        {
            "emit_outcomes": {
                "outcomes": [
                    {
                        "raw": "all-cause mortality",
                        "normalized": "All-cause mortality",
                        "is_primary": True,
                        "importance": "mortality",
                        "timepoint": "28 days",
                    }
                ],
            },
            "emit_grade_modifiers": {
                "downgrades": [],
                "upgrades": [],
                "rationale": "No domains warrant downgrading.",
            },
            "emit_rob2_domains": {
                "domains": [
                    {
                        "name": "randomization",
                        "judgment": "low",
                        "rationale": "Concealed allocation.",
                    },
                    {"name": "deviations", "judgment": "low", "rationale": "ITT."},
                    {"name": "missing_data", "judgment": "low", "rationale": "Low attrition."},
                    {"name": "outcome_measurement", "judgment": "low", "rationale": "Blinded."},
                    {
                        "name": "selective_reporting",
                        "judgment": "low",
                        "rationale": "Pre-registered.",
                    },
                ],
                "overall": "low",
            },
            "emit_spin_assessment": {
                "present": False,
                "severity": None,
                "rationale": "Conclusion matches the reported primary outcome.",
            },
        }
    )

    paper = _paper()

    class _OfflineAgent(EvidenceAgent):
        async def grade(self, identifier, *, query=None):  # type: ignore[override]
            # Skip retrieval/retraction/S2 — exercise just the LLM enrichment paths.
            paper_local = paper
            paper_local = await self._llm_enrich_paper(paper_local)
            report = self.grade_paper(paper_local, query=query)
            return await self._llm_enrich_report(report)

    agent = _OfflineAgent(cache=Cache(tmp_path / "cache.sqlite"), llm=provider)
    report = await agent.grade("ignored")
    assert report.card.outcomes
    assert report.card.outcomes[0].name == "All-cause mortality"
    assert report.card.outcomes[0].importance == OutcomeImportance.mortality
    assert report.card.rob and report.card.rob.overall.value == "low"  # type: ignore[union-attr]
    assert "emit_outcomes" in provider.calls
    assert "emit_grade_modifiers" in provider.calls
    assert "emit_rob2_domains" in provider.calls
    assert "emit_spin_assessment" in provider.calls
