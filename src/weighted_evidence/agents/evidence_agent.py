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
from weighted_evidence.llm.rubric_calls import (
    refine_grade,
    refine_outcomes,
    refine_rob2,
    refine_spin,
)
from weighted_evidence.models import (
    PICO,
    Amstar2Assessment,
    CitationContext,
    Comparison,
    FindingsCard,
    PairwiseRationale,
    Paper,
    Ranking,
    ReliabilityTier,
    RetractionStatus,
    RoB2Assessment,
    RobinsIAssessment,
    StudyDesign,
    WeightedEvidenceReport,
)
from weighted_evidence.parsing import classify_design, extract_pico_naive
from weighted_evidence.parsing.abstract import extract_sample_size
from weighted_evidence.parsing.fragility import fragility_for_paper
from weighted_evidence.parsing.outcomes import extract_outcomes
from weighted_evidence.parsing.spin import detect_spin
from weighted_evidence.retrieval import RetrievalClient
from weighted_evidence.retrieval.predatory import PredatorySource
from weighted_evidence.retrieval.retraction import (
    RetractionWatchSource,
    check_retraction,
    pubmed_retraction_check,
)
from weighted_evidence.retrieval.semantic_scholar import SemanticScholarClient
from weighted_evidence.rubric import (
    score_gis,
    skeleton_amstar2,
    skeleton_rob2,
    skeleton_robins_i,
    starting_certainty,
)
from weighted_evidence.rubric.aggregate import AggregateInput, aggregate_report
from weighted_evidence.rubric.clinical_significance import (
    annotate as annotate_clinical_significance,
)
from weighted_evidence.rubric.directness import match_pico
from weighted_evidence.rubric.grade import skeleton_grade


class EvidenceAgent:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: Cache | None = None,
        llm: LLMProvider | None = None,
        retrieval: RetrievalClient | None = None,
        retraction_watch: RetractionWatchSource | None = None,
        predatory: PredatorySource | None = None,
        semantic_scholar: SemanticScholarClient | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.cache = cache if cache is not None else open_cache()
        self._llm = llm
        self._retrieval = retrieval
        self._owns_retrieval = retrieval is None
        self._retraction_watch = retraction_watch
        self._predatory = predatory or PredatorySource()
        self._semantic_scholar = semantic_scholar

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

    async def grade(
        self,
        identifier: str,
        *,
        query: PICO | None = None,
    ) -> WeightedEvidenceReport:
        retrieval = await self._ensure_retrieval()
        paper = await retrieval.fetch(identifier)
        retraction = await check_retraction(paper, rw_source=self._retraction_watch)
        citation_context = await self._fetch_citation_context(paper)
        if self._llm is not None:
            paper = await self._llm_enrich_paper(paper)
        report = self._grade(
            paper,
            retraction=retraction,
            citation_context=citation_context,
            query=query,
        )
        if self._llm is not None:
            report = await self._llm_enrich_report(report)
        return report

    async def _llm_enrich_paper(self, paper: Paper) -> Paper:
        """Use the LLM to normalize outcomes before scoring."""

        if self._llm is None:
            return paper
        try:
            outcomes = await refine_outcomes(paper, llm=self._llm)
        except Exception:
            return paper
        if outcomes:
            from weighted_evidence.rubric.clinical_significance import (
                annotate as annotate_clinical_significance,
            )

            return paper.model_copy(update={"outcomes": annotate_clinical_significance(outcomes)})
        return paper

    async def _llm_enrich_report(self, report: WeightedEvidenceReport) -> WeightedEvidenceReport:
        """Refine GRADE modifiers, RoB 2 domains, and spin via the LLM, then re-aggregate."""

        if self._llm is None:
            return report
        paper = report.paper
        card = report.card

        try:
            new_grade = await refine_grade(paper, card.grade, llm=self._llm) if card.grade else None
        except Exception:
            new_grade = card.grade

        new_rob: RoB2Assessment | Amstar2Assessment | RobinsIAssessment | None = card.rob
        if isinstance(card.rob, RoB2Assessment):
            try:
                new_rob = await refine_rob2(paper, card.rob, llm=self._llm)
            except Exception:
                new_rob = card.rob

        try:
            new_spin = await refine_spin(paper, llm=self._llm)
        except Exception:
            new_spin = card.spin
        new_spin = new_spin or card.spin

        if new_grade is None:
            return report

        # Re-aggregate so reliability_tier + final_score reflect the refined inputs.
        from weighted_evidence.rubric.aggregate import AggregateInput, aggregate_report

        new_card = aggregate_report(
            AggregateInput(
                paper=paper,
                grade=new_grade,
                rob=new_rob,
                gis=card.gis,  # type: ignore[arg-type]
                pico_match=card.pico_match,
                fragility=card.fragility,
                spin=new_spin,
                retraction=card.retraction,
                predatory=card.predatory,
            ),
        )
        if card.citation_context is not None:
            new_card = new_card.model_copy(update={"citation_context": card.citation_context})
        return report.model_copy(update={"card": new_card})

    async def _fetch_citation_context(self, paper: Paper) -> CitationContext | None:
        if self._semantic_scholar is None:
            return None
        try:
            return await self._semantic_scholar.fetch_citation_contexts(
                doi=paper.identifier.doi, pmid=paper.identifier.pmid
            )
        except Exception:
            return None

    def grade_paper(
        self,
        paper: Paper,
        *,
        retraction: RetractionStatus | None = None,
        citation_context: CitationContext | None = None,
        query: PICO | None = None,
    ) -> WeightedEvidenceReport:
        """Score an already-retrieved Paper. Sync; pass external signals explicitly."""

        if retraction is None:
            retraction = pubmed_retraction_check(paper) or RetractionStatus()
        return self._grade(
            paper,
            retraction=retraction,
            citation_context=citation_context,
            query=query,
        )

    def _grade(
        self,
        paper: Paper,
        *,
        retraction: RetractionStatus,
        citation_context: CitationContext | None = None,
        query: PICO | None = None,
    ) -> WeightedEvidenceReport:
        enriched = self._enrich_design(paper)
        grade = skeleton_grade(enriched.design)
        rob = _select_rob_tool(enriched)
        gis = score_gis(enriched, citation_context=citation_context)
        predatory = self._predatory.check(enriched)
        fragility = fragility_for_paper(enriched) if enriched.design == StudyDesign.rct else None
        spin = detect_spin(enriched.abstract)
        pico_match = match_pico(enriched.pico, query)

        card = aggregate_report(
            AggregateInput(
                paper=enriched,
                grade=grade,
                rob=rob,
                gis=gis,
                pico_match=pico_match,
                fragility=fragility,
                spin=spin,
                retraction=retraction,
                predatory=predatory,
            ),
        )
        if citation_context is not None:
            card = card.model_copy(update={"citation_context": citation_context})

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

    # ------------------------------------------------------------------
    # Agent ranking surface
    # ------------------------------------------------------------------

    async def grade_many(
        self,
        identifiers: list[str],
        *,
        query: PICO | None = None,
    ) -> list[WeightedEvidenceReport]:
        """Grade multiple papers concurrently, sharing one retrieval client."""

        import asyncio

        coros = [self.grade(i, query=query) for i in identifiers]
        return list(await asyncio.gather(*coros))

    async def rank(
        self,
        identifiers: list[str],
        *,
        query: PICO | None = None,
    ) -> Ranking:
        """Query-conditioned ranking. When `query` is None this is just a sort by
        final_score; supplying a query routes the directness signal into the
        aggregate so per-paper scores reflect the query.
        """

        reports = await self.grade_many(identifiers, query=query)
        ordered = sorted(
            reports,
            key=lambda r: (
                _tier_rank(r.card.reliability_tier),
                r.card.final_score if r.card.final_score is not None else -1.0,
            ),
            reverse=True,
        )
        rationale = (
            f"Ranked {len(ordered)} papers"
            + (
                f" against query population='{query.population}'"
                if query and query.population
                else ""
            )
            + ". Order reflects reliability_tier first, then final_score; retracted "
            + "papers sort to the bottom."
        )
        return Ranking(
            query=query,
            cards=[r.card for r in ordered],
            rationale=rationale,
        )

    async def compare(
        self,
        identifiers: list[str],
        *,
        query: PICO | None = None,
    ) -> Comparison:
        """Pairwise comparison. Each (winner, loser) pair carries an explicit list
        of reasons drawn from the difference in their FindingsCards.
        """

        ranking = await self.rank(identifiers, query=query)
        cards = ranking.cards
        pairwise: list[PairwiseRationale] = []
        for i, winner in enumerate(cards):
            for loser in cards[i + 1 :]:
                pairwise.append(_pairwise(winner, loser))
        return Comparison(ordered=cards, pairwise=pairwise)


# ---------------------------------------------------------------------------
# Helpers for compare/rank
# ---------------------------------------------------------------------------


_TIER_ORDER = {
    ReliabilityTier.rely: 4,
    ReliabilityTier.use_with_caution: 3,
    ReliabilityTier.weak_signal: 2,
    ReliabilityTier.do_not_rely: 1,
    ReliabilityTier.retracted: 0,
}


def _tier_rank(tier: ReliabilityTier) -> int:
    return _TIER_ORDER.get(tier, 0)


def _pairwise(winner: FindingsCard, loser: FindingsCard) -> PairwiseRationale:
    reasons: list[str] = []
    tier_diff = winner.reliability_tier != loser.reliability_tier
    if tier_diff:
        reasons.append(
            f"reliability_tier: {winner.reliability_tier.value} > {loser.reliability_tier.value}"
        )
    if winner.grade and loser.grade and winner.grade.final_certainty != loser.grade.final_certainty:
        reasons.append(
            f"GRADE: {winner.grade.final_certainty.value} > {loser.grade.final_certainty.value}"
        )
    w_outcome = _top_outcome_importance(winner)
    l_outcome = _top_outcome_importance(loser)
    if w_outcome and l_outcome and w_outcome != l_outcome:
        reasons.append(f"primary outcome importance: {w_outcome} > {l_outcome}")
    if winner.gis and loser.gis and (winner.gis.score - loser.gis.score) > 0.05:
        reasons.append(
            f"GIS: {winner.gis.score:.2f} > {loser.gis.score:.2f} "
            f"(supportive citations / journal tier)"
        )
    if winner.fragility and loser.fragility and winner.fragility.index > loser.fragility.index + 3:
        reasons.append(
            f"fragility: {winner.fragility.index} events vs "
            f"{loser.fragility.index} events to flip significance"
        )
    if (
        winner.pico_match
        and loser.pico_match
        and winner.pico_match.overall - loser.pico_match.overall > 0.1
    ):
        reasons.append(
            f"PICO directness vs query: {winner.pico_match.overall:.2f} "
            f"> {loser.pico_match.overall:.2f}"
        )
    delta = None
    if winner.final_score is not None and loser.final_score is not None:
        delta = winner.final_score - loser.final_score
    if not reasons:
        reasons.append(
            "no qualitative differentiator surfaced; ranking driven by raw final_score difference"
        )
    return PairwiseRationale(
        winner_id=winner.id,
        loser_id=loser.id,
        reasons=reasons,
        score_delta=delta,
        tier_difference=tier_diff,
    )


def _top_outcome_importance(card: FindingsCard) -> str | None:
    primaries = [o for o in card.outcomes if o.is_primary]
    target = primaries or card.outcomes
    if not target:
        return None
    return target[0].importance.value


_RCT_LIKE = {StudyDesign.rct, StudyDesign.cluster_rct, StudyDesign.crossover_rct}
_SR_LIKE = {StudyDesign.systematic_review, StudyDesign.meta_analysis}
_NRSI_LIKE = {
    StudyDesign.cohort,
    StudyDesign.case_control,
    StudyDesign.controlled_before_after,
    StudyDesign.interrupted_time_series,
    StudyDesign.comparative_effectiveness,
}


def _select_rob_tool(
    paper: Paper,
) -> Amstar2Assessment | RoB2Assessment | RobinsIAssessment | None:
    """Pick the design-appropriate RoB tool: RoB 2, AMSTAR-2, or ROBINS-I.

    Returns None when none applies (e.g., editorials, narrative reviews).
    """

    if paper.design in _RCT_LIKE:
        return skeleton_rob2(paper)
    if paper.design in _SR_LIKE:
        return skeleton_amstar2(paper)
    if paper.design in _NRSI_LIKE:
        return skeleton_robins_i(paper)
    return None
