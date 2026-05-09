"""weighted-evidence: grade and rank biomedical papers for AI agents."""

from weighted_evidence.models import (
    ClinicalSignificance,
    EffectSize,
    FindingsCard,
    Outcome,
    OutcomeImportance,
    Paper,
    ReliabilityTier,
    StudyDesign,
    WeightedEvidenceReport,
)

__all__ = [
    "ClinicalSignificance",
    "EffectSize",
    "FindingsCard",
    "Outcome",
    "OutcomeImportance",
    "Paper",
    "ReliabilityTier",
    "StudyDesign",
    "WeightedEvidenceReport",
]

__version__ = "0.1.0"
