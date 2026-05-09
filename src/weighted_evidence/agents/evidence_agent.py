"""Phase 1 orchestrator: retrieve → parse → score → aggregate.

The agent's structure is the seam where each subsequent PR plugs in:
- PR 2 — findings extraction populates `paper.outcomes`.
- PR 3 — quality guards populate `retraction` and `predatory`.
- PR 4 — citation context, fragility, spin, PICO directness flow in here.
- PR 5 — `compare()` and `rank()` reuse the same per-paper pipeline.
"""

from __future__ import annotations

from datetime import datetime

from weighted_evidence.cache import Cache, open_cache
from weighted_evidence.config import Settings
from weighted_evidence.config import settings as load_settings
from weighted_evidence.llm.base import LLMProvider
from weighted_evidence.models import (
    FindingsCard,
    Paper,
    PredatoryFlag,
    RetractionStatus,
    StudyDesign,
    WeightedEvidenceReport,
)
from weighted_evidence.parsing import classify_design, extract_pico_naive
from weighted_evidence.parsing.abstract import extract_sample_size
from weighted_evidence.parsing.outcomes import extract_outcomes
from weighted_evidence.retrieval import RetrievalClient
from weighted_evidence.rubric import score_gis, skeleton_rob2, starting_certainty
from weighted_evidence.rubric.aggregate import AggregateInput, aggregate_report
from weighted_evidence.rubric.clinical_significance import (
    annotate as annotate_clinical_significance,
)
from weighted_evidence.rubric.grade import skeleton_grade


class EvidenceAgent:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: Cache | None = None,
        llm: LLMProvider | None = None,
        retrieval: RetrievalClient | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.cache = cache if cache is not None else open_cache()
        self._llm = llm
        self._retrieval = retrieval
        self._owns_retrieval = retrieval is None

    @property
    def llm(self) -> LLMProvider | None:
        return self._llm

    async def aclose(self) -> None:
        if self._owns_retrieval and self._retrieval is not None:
            await self._retrieval.aclose()

    async def __aenter__(self) -> EvidenceAgent:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _ensure_retrieval(self) -> RetrievalClient:
        if self._retrieval is None:
            self._retrieval = RetrievalClient(settings=self.settings, cache=self.cache)
        return self._retrieval

    async def grade(self, identifier: str) -> WeightedEvidenceReport:
        retrieval = await self._ensure_retrieval()
        paper = await retrieval.fetch(identifier)
        return self.grade_paper(paper)

    def grade_paper(self, paper: Paper) -> WeightedEvidenceReport:
        """Score an already-retrieved Paper. Splitting this off makes tests cheap."""

        enriched = self._enrich_design(paper)
        grade = skeleton_grade(enriched.design)
        rob = skeleton_rob2(enriched) if enriched.design == StudyDesign.rct else None
        gis = score_gis(enriched)

        card = aggregate_report(
            AggregateInput(
                paper=enriched,
                grade=grade,
                rob=rob,
                gis=gis,
                pico_match=None,
                fragility=None,
                spin=None,
                retraction=RetractionStatus(),
                predatory=PredatoryFlag(),
            ),
        )

        return WeightedEvidenceReport(
            paper=enriched,
            card=card,
            generated_at=datetime.utcnow(),
        )

    def _enrich_design(self, paper: Paper) -> Paper:
        if paper.design != StudyDesign.unknown:
            base = paper
        else:
            design = classify_design(paper.publication_types, paper.mesh_terms, title=paper.title)
            base = paper.model_copy(update={"design": design})

        if base.pico is None and base.abstract:
            base = base.model_copy(update={"pico": extract_pico_naive(base.abstract)})
        if base.sample_size is None and base.abstract:
            n = extract_sample_size(base.abstract)
            if n is not None:
                base = base.model_copy(update={"sample_size": n})
        if not base.outcomes and base.abstract:
            extracted = extract_outcomes(base.abstract)
            if extracted:
                base = base.model_copy(
                    update={"outcomes": annotate_clinical_significance(extracted)}
                )

        starting_certainty(base.design)
        return base

    def card(self, paper: Paper) -> FindingsCard:
        """Convenience wrapper: grade then return only the FindingsCard."""

        return self.grade_paper(paper).card
