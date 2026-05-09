"""Normalize free-form identifiers (DOI / PMID / PMCID) into structured form."""

from __future__ import annotations

import re
from enum import StrEnum

from weighted_evidence.models import Identifier


class IdentifierKind(StrEnum):
    doi = "doi"
    pmid = "pmid"
    pmcid = "pmcid"
    unknown = "unknown"


_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
_PMID_RE = re.compile(r"^\d{1,9}$")
_PMCID_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)


def classify(raw: str) -> IdentifierKind:
    s = raw.strip()
    s = s.removeprefix("https://doi.org/")
    s = s.removeprefix("http://doi.org/")
    s = s.removeprefix("doi:")
    if _DOI_RE.match(s):
        return IdentifierKind.doi
    if _PMID_RE.match(s):
        return IdentifierKind.pmid
    if _PMCID_RE.match(s):
        return IdentifierKind.pmcid
    return IdentifierKind.unknown


def normalize(raw: str) -> str:
    s = raw.strip()
    s = s.removeprefix("https://doi.org/")
    s = s.removeprefix("http://doi.org/")
    s = s.removeprefix("doi:")
    kind = classify(s)
    if kind == IdentifierKind.pmcid:
        return s.upper()
    if kind == IdentifierKind.doi:
        return s.lower()
    return s


def resolve(raw: str) -> tuple[IdentifierKind, Identifier]:
    """Best-effort resolution. Cross-resolution (DOI <-> PMID) happens later via PubMed."""

    s = normalize(raw)
    kind = classify(s)
    ident = Identifier()
    if kind == IdentifierKind.doi:
        ident.doi = s
    elif kind == IdentifierKind.pmid:
        ident.pmid = s
    elif kind == IdentifierKind.pmcid:
        ident.pmcid = s
    return kind, ident
