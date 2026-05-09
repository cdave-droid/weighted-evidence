"""Identifier classification + normalization."""

from __future__ import annotations

import pytest

from weighted_evidence.retrieval.resolver import IdentifierKind, classify, normalize, resolve


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("10.1056/NEJMoa1503326", IdentifierKind.doi),
        ("https://doi.org/10.1056/NEJMoa1503326", IdentifierKind.doi),
        ("doi:10.1056/NEJMoa1503326", IdentifierKind.doi),
        ("10793162", IdentifierKind.pmid),
        ("PMC123456", IdentifierKind.pmcid),
        ("pmc123456", IdentifierKind.pmcid),
        ("not-an-id", IdentifierKind.unknown),
    ],
)
def test_classify(raw: str, expected: IdentifierKind) -> None:
    assert classify(raw) == expected


def test_normalize_strips_doi_prefixes() -> None:
    assert normalize("https://doi.org/10.1056/NEJMoa1503326") == "10.1056/nejmoa1503326"
    assert normalize("doi:10.1056/NEJMoa1503326") == "10.1056/nejmoa1503326"


def test_normalize_uppercases_pmcid() -> None:
    assert normalize("pmc123") == "PMC123"


def test_resolve_populates_correct_field() -> None:
    kind, ident = resolve("10793162")
    assert kind == IdentifierKind.pmid
    assert ident.pmid == "10793162"
    assert ident.doi is None

    kind, ident = resolve("10.1056/NEJMoa1503326")
    assert kind == IdentifierKind.doi
    assert ident.doi == "10.1056/nejmoa1503326"
