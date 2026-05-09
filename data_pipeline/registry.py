"""Specialty registry. Each entry maps a society to its grading vocabulary +
a normalization function that emits `evidence_strength_score in [0, 1]`.

Phase 2 will populate this; the file ships now so adding a specialty is a
one-PR job rather than a refactor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GradingSystem:
    name: str
    """Human-readable label (e.g. 'GRADE', 'AHA-Class+LOE', 'USPSTF', 'NICE')."""
    normalize: Callable[[str], float]
    """Maps a native grade label onto `evidence_strength_score in [0, 1]`."""


@dataclass(frozen=True)
class SocietySpec:
    society: str
    grading_system: GradingSystem
    aliases: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Specialty:
    name: str
    societies: tuple[SocietySpec, ...]


REGISTRY: tuple[Specialty, ...] = ()
"""Filled in by data_pipeline/ingest/<specialty>/__init__.py at runtime."""
