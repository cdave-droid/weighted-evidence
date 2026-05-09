"""Predatory-journal blocklist guard.

Loads a curated text file (one journal per line, # comments allowed) and
matches a paper's journal title against it. Conservative on purpose:
predatory-journal lists are imperfect and can be weaponized, so we ship a
small seed list and document the override path clearly.

Users can override / extend by:
  - setting `WEIGHTED_EVIDENCE_PREDATORY_LIST=/path/to/list.txt` in env, or
  - passing a path to `PredatorySource(path=...)`.
"""

from __future__ import annotations

import os
from pathlib import Path

from weighted_evidence.models import Paper, PredatoryFlag

# Conservative seed list. Real deployments should override via env / file.
# Citations: scholarship around predatory publishing has historically used
# Beall's list and successors; we intentionally do not embed any specific list.
_DEFAULT_BLOCKLIST: tuple[str, ...] = ()


class PredatorySource:
    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            env = os.environ.get("WEIGHTED_EVIDENCE_PREDATORY_LIST")
            path = Path(env) if env else None
        self.path = path
        self._blocklist: set[str] | None = None
        self._list_name: str | None = None

    def _load(self) -> None:
        if self._blocklist is not None:
            return
        entries: set[str] = set(_DEFAULT_BLOCKLIST)
        list_name = "default"
        if self.path is not None and self.path.exists():
            list_name = self.path.name
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                entries.add(line.lower())
        self._blocklist = entries
        self._list_name = list_name

    def check(self, paper: Paper) -> PredatoryFlag:
        self._load()
        assert self._blocklist is not None
        if not paper.journal:
            return PredatoryFlag(flagged=False)
        if paper.journal.strip().lower() in self._blocklist:
            return PredatoryFlag(flagged=True, list_name=self._list_name)
        return PredatoryFlag(flagged=False)


__all__ = ["PredatorySource"]
