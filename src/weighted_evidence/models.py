"""Core type contracts for weighted-evidence.

Every public surface of the SDK speaks these models. They are versioned with the
package; downstream agents pin a major version of the schema.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StudyDesign(StrEnum):
    rct = "rct"
    cluster_rct = "cluster_rct"
    crossover_rct = "crossover_rct"
    systematic_review = "systematic_review"
    meta_analysis = "meta_analysis"
    cohort = "cohort"
    case_control = "case_control"
    cross_sectional = "cross_sectional"
    case_series = "case_series"
    case_report = "case_report"
    controlled_before_after = "controlled_before_after"
    interrupted_time_series = "interrupted_time_series"
    comparative_effectiveness = "comparative_effectiveness"
    diagnostic_accuracy = "diagnostic_accuracy"
    prediction_model = "prediction_model"
    qualitative = "qualitative"
    narrative_review = "narrative_review"
    editorial = "editorial"
    other = "other"
    unknown = "unknown"


class OutcomeImportance(StrEnum):
    mortality = "mortality"
    morbidity = "morbidity"
    qol = "qol"
    hospitalization = "hospitalization"
    symptomatic = "symptomatic"
    surrogate_lab = "surrogate_lab"
    surrogate_imaging = "surrogate_imaging"
    composite = "composite"
    other = "other"


class ClinicalSignificanceVerdict(StrEnum):
    likely = "likely"
    uncertain = "uncertain"
    unlikely = "unlikely"
    not_applicable = "not_applicable"


class GradeCertainty(StrEnum):
    high = "high"
    moderate = "moderate"
    low = "low"
    very_low = "very_low"


class GradeDomainJudgment(StrEnum):
    not_serious = "not_serious"
    serious = "serious"
    very_serious = "very_serious"


class RoB2Judgment(StrEnum):
    low = "low"
    some_concerns = "some_concerns"
    high = "high"


class Amstar2Rating(StrEnum):
    high = "high"
    moderate = "moderate"
    low = "low"
    critically_low = "critically_low"


class RobinsIJudgment(StrEnum):
    low = "low"
    moderate = "moderate"
    serious = "serious"
    critical = "critical"
    no_information = "no_information"


class ReliabilityTier(StrEnum):
    rely = "rely"
    use_with_caution = "use_with_caution"
    weak_signal = "weak_signal"
    do_not_rely = "do_not_rely"
    retracted = "retracted"


class EffectKind(StrEnum):
    nnt = "nnt"
    arr = "arr"
    rr = "rr"
    hr = "hr"
    or_ = "or"
    md = "md"
    smd = "smd"
    proportion = "proportion"
    other = "other"


class MIDComparison(StrEnum):
    exceeds = "exceeds"
    meets = "meets"
    below = "below"
    crosses_null = "crosses_null"
    not_established = "not_established"
    unknown = "unknown"


class CitationIntent(StrEnum):
    supportive = "supportive"
    disputing = "disputing"
    mentioning = "mentioning"
    methodological = "methodological"


class Disposition(StrEnum):
    ok = "ok"
    retracted = "retracted"
    flagged = "flagged"


# ---------------------------------------------------------------------------
# Provenance + Tracked wrapper
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Where in the source document a value was found."""

    source: Literal["abstract", "fulltext", "table", "metadata", "external"] = "abstract"
    section: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    sha256: str | None = None
    url: str | None = None


class Tracked(BaseModel, Generic[T]):
    """Wraps an extracted value with confidence and provenance.

    Generic over the value type. Used pervasively for LLM-extracted fields so
    downstream agents can decide when to fall back to human review.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: T
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: Provenance | None = None
    extractor: Literal["rule", "llm", "external"] = "rule"


# ---------------------------------------------------------------------------
# PICO + Outcomes + Effect sizes
# ---------------------------------------------------------------------------


class PICO(BaseModel):
    population: str | None = None
    intervention: str | None = None
    comparator: str | None = None
    outcomes: list[str] = Field(default_factory=list)
    setting: str | None = None
    timeframe: str | None = None


class EffectSize(BaseModel):
    kind: EffectKind
    point: float
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float = 0.95
    p: float | None = None
    mid: float | None = Field(default=None, description="Minimal important difference, if known.")
    vs_mid: MIDComparison = MIDComparison.unknown


class ClinicalSignificance(BaseModel):
    verdict: ClinicalSignificanceVerdict
    rationale: str


class Outcome(BaseModel):
    name: str
    importance: OutcomeImportance = OutcomeImportance.other
    is_primary: bool = False
    timepoint: str | None = None
    effect: EffectSize | None = None
    clinical_significance: ClinicalSignificance | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------


class Identifier(BaseModel):
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None

    def primary(self) -> str:
        return self.doi or self.pmid or self.pmcid or ""


class Author(BaseModel):
    name: str
    affiliation: str | None = None


class Paper(BaseModel):
    """Normalized representation of a single paper, post-retrieval."""

    identifier: Identifier
    title: str
    abstract: str | None = None
    authors: list[Author] = Field(default_factory=list)
    journal: str | None = None
    publication_date: datetime | None = None
    publication_types: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    fulltext_xml: str | None = None
    pico: PICO | None = None
    design: StudyDesign = StudyDesign.unknown
    outcomes: list[Outcome] = Field(default_factory=list)
    sample_size: int | None = None
    follow_up: str | None = None
    funding: str | None = None
    conflicts_of_interest: str | None = None


# ---------------------------------------------------------------------------
# Rubric assessments
# ---------------------------------------------------------------------------


class GradeDomain(BaseModel):
    name: Literal[
        "risk_of_bias",
        "inconsistency",
        "indirectness",
        "imprecision",
        "publication_bias",
    ]
    judgment: GradeDomainJudgment
    rationale: str


class GradeAssessment(BaseModel):
    starting_certainty: GradeCertainty
    final_certainty: GradeCertainty
    downgrades: list[GradeDomain] = Field(default_factory=list)
    upgrades: list[Literal["large_effect", "dose_response", "plausible_confounding"]] = Field(
        default_factory=list
    )
    rationale: str | None = None


class RoB2Domain(BaseModel):
    name: Literal[
        "randomization",
        "deviations",
        "missing_data",
        "outcome_measurement",
        "selective_reporting",
    ]
    judgment: RoB2Judgment
    rationale: str


class RoB2Assessment(BaseModel):
    domains: list[RoB2Domain] = Field(default_factory=list)
    overall: RoB2Judgment


class Amstar2Item(BaseModel):
    number: int
    description: str
    met: bool
    rationale: str | None = None


class Amstar2Assessment(BaseModel):
    items: list[Amstar2Item] = Field(default_factory=list)
    overall: Amstar2Rating


class RobinsIDomain(BaseModel):
    name: Literal[
        "confounding",
        "selection",
        "classification",
        "deviations",
        "missing_data",
        "measurement",
        "selective_reporting",
    ]
    judgment: RobinsIJudgment
    rationale: str


class RobinsIAssessment(BaseModel):
    domains: list[RobinsIDomain] = Field(default_factory=list)
    overall: RobinsIJudgment


# Type alias for the chosen RoB tool's output (one of three).
RoBToolAssessment = RoB2Assessment | Amstar2Assessment | RobinsIAssessment


# ---------------------------------------------------------------------------
# Quality guards + supplementary signals
# ---------------------------------------------------------------------------


class FragilityIndex(BaseModel):
    index: int
    quotient: float | None = None
    method: Literal["walsh", "approximate", "unavailable"] = "walsh"


class SpinAssessment(BaseModel):
    present: bool
    severity: Literal["mild", "moderate", "severe"] | None = None
    rationale: str | None = None


class RetractionStatus(BaseModel):
    status: Literal["none", "retracted", "concern", "correction"] = "none"
    source: Literal["pubmed", "retraction_watch", "manual"] | None = None
    notice_url: str | None = None
    notice_date: datetime | None = None


class PredatoryFlag(BaseModel):
    flagged: bool = False
    list_name: str | None = None


class CitationContext(BaseModel):
    total: int = 0
    supportive: int = 0
    disputing: int = 0
    mentioning: int = 0
    methodological: int = 0
    examples: list[str] = Field(default_factory=list)


class PICODimensionMatch(BaseModel):
    dimension: Literal["population", "intervention", "comparator", "outcome", "setting"]
    score: float = Field(ge=0.0, le=1.0)
    mismatch: str | None = None


class PICOMatch(BaseModel):
    overall: float = Field(ge=0.0, le=1.0)
    dimensions: list[PICODimensionMatch] = Field(default_factory=list)
    rationale: str | None = None


class GuidelineCitation(BaseModel):
    guideline: str
    society: str | None = None
    year: int | None = None
    rec_strength: Literal["strong", "conditional", "ungraded"] | None = None
    evidence_certainty: Literal["high", "moderate", "low", "very_low"] | None = None
    grading_system: str | None = None
    normalized: float = Field(ge=0.0, le=1.0)


class GuidelineImpactScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    version: str = "rule-v0"
    fine_tune_pending: bool = True
    journal_tier: float | None = None
    log_sample_size: float | None = None
    replication_count: int | None = None
    recency_decay: float | None = None
    guideline_citations: list[GuidelineCitation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Explanation + Findings Card + Report
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A single explanation entry tying a signal to a source span and its score contribution."""

    signal: str
    contribution: float
    rationale: str | None = None
    provenance: Provenance | None = None


class FindingsCard(BaseModel):
    """Stable agent-facing JSON. Pinned shape across model versions."""

    schema_version: Literal["1"] = "1"
    id: str
    title: str
    journal: str | None = None
    publication_date: datetime | None = None
    design: StudyDesign
    pico: PICO | None = None
    outcomes: list[Outcome] = Field(default_factory=list)
    grade: GradeAssessment | None = None
    rob_tool: Literal["rob2", "amstar2", "robins_i", "none"] = "none"
    rob: RoBToolAssessment | None = None
    gis: GuidelineImpactScore | None = None
    fragility: FragilityIndex | None = None
    spin: SpinAssessment | None = None
    retraction: RetractionStatus = Field(default_factory=RetractionStatus)
    predatory: PredatoryFlag = Field(default_factory=PredatoryFlag)
    citation_context: CitationContext | None = None
    pico_match: PICOMatch | None = None
    final_score: float | None = None
    reliability_tier: ReliabilityTier
    disposition: Disposition = Disposition.ok
    explanation: list[Citation] = Field(default_factory=list)
    model_version: str = "rule-v0"


class WeightedEvidenceReport(BaseModel):
    """Full internal record. Includes the FindingsCard plus the underlying Paper."""

    paper: Paper
    card: FindingsCard
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def final_score(self) -> float | None:
        return self.card.final_score

    @property
    def reliability_tier(self) -> ReliabilityTier:
        return self.card.reliability_tier


class PairwiseRationale(BaseModel):
    """Why one paper ranks above another. Returned by `EvidenceAgent.compare()`."""

    winner_id: str | None
    """None when the comparison is inconclusive (effective tie)."""
    loser_id: str | None
    reasons: list[str] = Field(default_factory=list)
    score_delta: float | None = None
    tier_difference: bool = False


class Comparison(BaseModel):
    """Pairwise comparison + a final ordering."""

    ordered: list[FindingsCard]
    pairwise: list[PairwiseRationale] = Field(default_factory=list)


class Ranking(BaseModel):
    """Query-conditioned ranking. Each entry carries its weighted score for the query."""

    query: PICO | None = None
    cards: list[FindingsCard]
    rationale: str
