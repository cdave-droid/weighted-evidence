"""Predatory journal guard."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from weighted_evidence.models import Identifier, Paper
from weighted_evidence.retrieval.predatory import PredatorySource


def _paper(journal: str | None) -> Paper:
    return Paper(
        identifier=Identifier(doi="10.0000/x"),
        title="t",
        journal=journal,
        publication_date=datetime(2024, 1, 1),
    )


def test_default_blocklist_is_conservative() -> None:
    src = PredatorySource()
    flag = src.check(_paper("New England Journal of Medicine"))
    assert flag.flagged is False


def test_custom_blocklist_flags_match(tmp_path: Path) -> None:
    list_path = tmp_path / "blocklist.txt"
    list_path.write_text(
        "# weighted-evidence predatory list (test)\n"
        "Journal of Suspicious Publishing\n"
        "Open Mega-Journal of Everything\n",
        encoding="utf-8",
    )
    src = PredatorySource(path=list_path)
    flag = src.check(_paper("Journal of Suspicious Publishing"))
    assert flag.flagged is True
    assert flag.list_name == "blocklist.txt"


def test_no_journal_returns_unflagged(tmp_path: Path) -> None:
    list_path = tmp_path / "list.txt"
    list_path.write_text("X\n", encoding="utf-8")
    src = PredatorySource(path=list_path)
    flag = src.check(_paper(None))
    assert flag.flagged is False
