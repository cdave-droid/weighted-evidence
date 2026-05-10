# Changelog

## [0.1.0] — 2026-05-10

First public alpha. Phase 1 of the framework — grades and ranks biomedical
papers so AI agents weight evidence the way the literature itself does.

### Added
- Pydantic v2 model layer: `Paper`, `PICO`, `Outcome`, `EffectSize`,
  `ClinicalSignificance`, `GradeAssessment`, `RoB2Assessment`,
  `Amstar2Assessment`, `RobinsIAssessment`, `FragilityIndex`,
  `SpinAssessment`, `RetractionStatus`, `PredatoryFlag`, `CitationContext`,
  `PICOMatch`, `GuidelineImpactScore`, `ReliabilityTier`, `Tracked[T]`
  generic with confidence + provenance, `FindingsCard` (stable agent-facing
  schema), `WeightedEvidenceReport`, `Comparison`, `Ranking`,
  `PairwiseRationale`.
- Retrieval: PubMed E-utilities (esearch + efetch + lxml parser),
  Retraction Watch CSV loader, Semantic Scholar citations
  (contexts/intents), predatory-journal blocklist with override path.
- Parsing: PubMed publication-type → `StudyDesign`; abstract PICO regex;
  effect-size extraction (HR/RR/OR/ARR with CIs and P-values); outcome
  name + importance classification (mortality > morbidity > qol >
  surrogate); fragility index via hand-rolled Fisher's exact (no scipy);
  spin detection.
- Rubric: rule-based GRADE skeleton with LLM-refined downgrade modifiers;
  RoB 2 (RCTs) with rule-based randomization + outcome-measurement
  detection plus LLM domain refinement; AMSTAR-2 (SRs/MAs); ROBINS-I
  (non-randomized intervention studies); transparent v0 GIS that uses
  Semantic Scholar citation contexts when available; aggregate scorer
  with hard vetoes (retraction → score=None; predatory cap at 0.4) and a
  deterministic `reliability_tier` decision table.
- LLM: Anthropic provider with prompt caching, OpenAI optional via extra,
  structured-output rubric refinement helpers in `llm/rubric_calls.py`.
- Agent: `EvidenceAgent.grade()`, `card()`, `compare()`, `rank()`,
  `grade_many()`. Pairwise rationale cites GRADE / reliability_tier /
  outcome importance / GIS / fragility / PICO directness deltas.
- CLI (Typer): `weighted-evidence grade | card | rank | compare | cache`.
- MCP server (FastMCP, optional `mcp` extra): `grade_paper`,
  `findings_card`, `rank_papers`, `compare_papers`.
- 92 unit + integration tests; CI matrix (3.11 + 3.12) with ruff +
  ruff-format + mypy --strict + pytest. Trusted-publisher PyPI release
  workflow on `v*` tag.
