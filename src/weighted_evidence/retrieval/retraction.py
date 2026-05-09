"""Retraction guard.

Two evidence sources:
  1. PubMed publication types — `Retracted Publication`, `Retraction of Publication`.
     Already in our cached PubMed XML, so this is free.
  2. Retraction Watch — public CSV (https://retractiondatabase.org). Loaded via
     a lightweight wrapper that the user can point at a local copy or a URL.

A retraction is a hard veto: aggregate.py emits final_score=None and
disposition=retracted when this returns status="retracted".
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential_jitter

from weighted_evidence.cache import Cache, cache_key
from weighted_evidence.models import Paper, RetractionStatus

_PUBMED_RETRACTED_TYPES = {
    "retracted publication",
    "retraction of publication",
    "retracted article",
}


def pubmed_retraction_check(paper: Paper) -> RetractionStatus | None:
    """Inspect PubMed publication types for a retraction marker. Sync, free, fast."""

    pts = {pt.lower() for pt in paper.publication_types}
    if pts & _PUBMED_RETRACTED_TYPES:
        return RetractionStatus(
            status="retracted",
            source="pubmed",
            notice_url=f"https://pubmed.ncbi.nlm.nih.gov/{paper.identifier.pmid}/"
            if paper.identifier.pmid
            else None,
        )
    return None


@dataclass(frozen=True)
class _RWEntry:
    doi: str | None
    pmid: str | None
    notice_url: str | None
    notice_date: datetime | None


class RetractionWatchSource:
    """Loads a Retraction Watch CSV export and indexes by DOI / PMID.

    Pass a local path (preferred for tests + offline use) or a URL. Cached so
    repeated lookups reuse the parsed index.
    """

    def __init__(
        self,
        *,
        path: Path | None = None,
        url: str | None = None,
        cache: Cache | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.path = path
        self.url = url
        self.cache = cache
        self._client = client
        self._owns_client = client is None
        self._index_doi: dict[str, _RWEntry] | None = None
        self._index_pmid: dict[str, _RWEntry] | None = None

    async def _load_csv(self) -> str:
        if self.path is not None:
            return self.path.read_text(encoding="utf-8")
        if self.url is None:
            raise ValueError("RetractionWatchSource needs either path= or url=.")

        if self.cache:
            cached = self.cache.get("retraction", cache_key("rw", self.url))
            if cached is not None:
                return cached

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_exponential_jitter(initial=0.5, max=8.0),
                reraise=True,
            ):
                with attempt:
                    resp = await self._client.get(self.url)
                    resp.raise_for_status()
                    text = resp.text
        except RetryError as exc:  # pragma: no cover - defensive
            raise RuntimeError("Retraction Watch fetch failed after retries") from exc

        if self.cache:
            self.cache.set("retraction", cache_key("rw", self.url), text)
        return text

    async def _ensure_index(self) -> None:
        if self._index_doi is not None:
            return
        text = await self._load_csv()
        self._index_doi = {}
        self._index_pmid = {}
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            doi = (row.get("OriginalPaperDOI") or row.get("DOI") or "").strip().lower() or None
            pmid = (row.get("OriginalPaperPubMedID") or row.get("PMID") or "").strip() or None
            notice_url = (
                (
                    row.get("RetractionPubMedID")
                    and f"https://pubmed.ncbi.nlm.nih.gov/{row['RetractionPubMedID']}/"
                )
                or row.get("URLs")
                or None
            )
            notice_date = _parse_date(row.get("RetractionDate") or row.get("Date") or "")
            entry = _RWEntry(doi=doi, pmid=pmid, notice_url=notice_url, notice_date=notice_date)
            if doi:
                self._index_doi[doi] = entry
            if pmid:
                self._index_pmid[pmid] = entry

    async def lookup(self, paper: Paper) -> RetractionStatus | None:
        await self._ensure_index()
        assert self._index_doi is not None and self._index_pmid is not None
        ident = paper.identifier
        entry: _RWEntry | None = None
        if ident.doi:
            entry = self._index_doi.get(ident.doi.lower())
        if entry is None and ident.pmid:
            entry = self._index_pmid.get(ident.pmid)
        if entry is None:
            return None
        return RetractionStatus(
            status="retracted",
            source="retraction_watch",
            notice_url=entry.notice_url,
            notice_date=entry.notice_date,
        )

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()


def _parse_date(raw: str) -> datetime | None:
    raw = raw.strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


async def check_retraction(
    paper: Paper, *, rw_source: RetractionWatchSource | None = None
) -> RetractionStatus:
    """Combined check: PubMed publication-type signal + optional Retraction Watch."""

    pm = pubmed_retraction_check(paper)
    if pm is not None:
        return pm
    if rw_source is not None:
        rw = await rw_source.lookup(paper)
        if rw is not None:
            return rw
    return RetractionStatus(status="none")


__all__ = ["RetractionWatchSource", "check_retraction", "pubmed_retraction_check"]
