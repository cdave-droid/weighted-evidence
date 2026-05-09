# Changelog

## [Unreleased]

### Added
- Initial Phase 1 scaffold: SDK + CLI vertical slice for `grade <identifier>`.
- Pydantic v2 models: `Paper`, `PICO`, `Outcome`, `EffectSize`, `GradeAssessment`, `RoB2Assessment`, `Amstar2Assessment`, `RobinsIAssessment`, `FragilityIndex`, `SpinAssessment`, `RetractionStatus`, `PredatoryFlag`, `CitationContext`, `PICOMatch`, `GuidelineImpactScore`, `ReliabilityTier`, `Tracked[T]`, `FindingsCard`, `WeightedEvidenceReport`.
- Retrieval: PubMed E-utilities (esummary + efetch).
- Parsing: PubMed publication-type → `StudyDesign`; abstract PICO regex.
- LLM provider: Anthropic (Claude) with prompt caching, OpenAI optional.
- Rubric: rule-based GRADE starting certainty + LLM modifier prompts; rule-based RoB 2 hooks; v0 `GuidelineImpactScore` (transparent linear); aggregate scorer with hard vetoes and `reliability_tier` decision table.
- Agent: `EvidenceAgent.grade(identifier) -> WeightedEvidenceReport`.
- CLI: `weighted-evidence grade`.
