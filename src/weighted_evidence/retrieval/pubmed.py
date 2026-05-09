"""PubMed E-utilities client (esearch, esummary, efetch) + XML parser."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from lxml import etree
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential_jitter,
)

from weighted_evidence.cache import Cache, cache_key
from weighted_evidence.config import Settings
from weighted_evidence.models import Author, Identifier, Paper, StudyDesign

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedError(RuntimeError):
    pass


class PubMedClient:
    def __init__(
        self,
        *,
        settings: Settings,
        cache: Cache | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> PubMedClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _get(self, path: str, params: dict[str, str]) -> str:
        url = f"{EUTILS}/{path}"
        params = {**params, "tool": self.settings.pubmed_tool}
        if self.settings.pubmed_email:
            params["email"] = self.settings.pubmed_email
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key

        ck = cache_key("pubmed", path, sorted(params.items()))
        if self.cache:
            cached = self.cache.get("http", ck)
            if cached is not None:
                return cached

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_exponential_jitter(initial=0.5, max=8.0),
                reraise=True,
            ):
                with attempt:
                    resp = await self._client.get(url, params=params)
                    resp.raise_for_status()
                    text = resp.text
        except RetryError as exc:  # pragma: no cover - defensive
            raise PubMedError(f"PubMed {path} failed after retries") from exc

        if self.cache:
            self.cache.set("http", ck, text)
        return text

    async def doi_to_pmid(self, doi: str) -> str | None:
        xml = await self._get(
            "esearch.fcgi",
            {"db": "pubmed", "term": f"{doi}[doi]", "retmode": "xml"},
        )
        root = etree.fromstring(xml.encode("utf-8"))
        ids = root.xpath("//IdList/Id/text()")
        return ids[0] if ids else None

    async def efetch_pubmed(self, pmid: str) -> str:
        return await self._get(
            "efetch.fcgi",
            {"db": "pubmed", "id": pmid, "retmode": "xml"},
        )

    async def fetch(self, ident: Identifier) -> Paper:
        pmid = ident.pmid
        if pmid is None and ident.doi:
            pmid = await self.doi_to_pmid(ident.doi)
        if pmid is None:
            raise PubMedError(f"Could not resolve identifier to a PubMed ID: {ident}")
        xml = await self.efetch_pubmed(pmid)
        return parse_pubmed_xml(xml, fallback_identifier=ident)


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def _text(node: Any) -> str | None:
    if node is None:
        return None
    text = "".join(node.itertext()).strip()
    return text or None


def _join_abstract(article: Any) -> str | None:
    parts: list[str] = []
    for ab in article.xpath(".//Abstract/AbstractText"):
        label = ab.get("Label")
        body = "".join(ab.itertext()).strip()
        if not body:
            continue
        parts.append(f"{label}: {body}" if label else body)
    return "\n\n".join(parts) if parts else None


def _publication_date(article: Any) -> datetime | None:
    for path in (
        ".//PubDate",
        ".//ArticleDate",
    ):
        node = article.find(path)
        if node is None:
            continue
        year = _text(node.find("Year"))
        month = _text(node.find("Month")) or "1"
        day = _text(node.find("Day")) or "1"
        if not year:
            continue
        try:
            month_num = (
                datetime.strptime(month[:3], "%b").month if not month.isdigit() else int(month)
            )
        except ValueError:
            month_num = 1
        try:
            return datetime(int(year), month_num, int(day))
        except ValueError:
            try:
                return datetime(int(year), month_num, 1)
            except ValueError:
                return datetime(int(year), 1, 1)
    return None


def parse_pubmed_xml(xml: str, *, fallback_identifier: Identifier | None = None) -> Paper:
    root = etree.fromstring(xml.encode("utf-8"))
    article = root.find(".//PubmedArticle")
    if article is None:
        raise PubMedError("No <PubmedArticle> element in response")

    pmid = _text(article.find(".//MedlineCitation/PMID"))

    doi = None
    pmcid = None
    for aid in article.xpath(".//ArticleIdList/ArticleId"):
        kind = aid.get("IdType", "").lower()
        val = (aid.text or "").strip()
        if kind == "doi":
            doi = val.lower()
        elif kind == "pmc":
            pmcid = val.upper() if val.startswith("PMC") else f"PMC{val}"

    ident = Identifier(
        doi=doi or (fallback_identifier.doi if fallback_identifier else None),
        pmid=pmid or (fallback_identifier.pmid if fallback_identifier else None),
        pmcid=pmcid or (fallback_identifier.pmcid if fallback_identifier else None),
    )

    title = _text(article.find(".//ArticleTitle")) or "(no title)"
    abstract = _join_abstract(article)

    journal = _text(article.find(".//Journal/Title")) or _text(
        article.find(".//Journal/ISOAbbreviation")
    )

    authors: list[Author] = []
    for au in article.xpath(".//AuthorList/Author"):
        last = _text(au.find("LastName"))
        first = _text(au.find("ForeName")) or _text(au.find("Initials"))
        collective = _text(au.find("CollectiveName"))
        if collective:
            authors.append(Author(name=collective))
            continue
        if last:
            name = f"{first} {last}".strip() if first else last
            aff = _text(au.find(".//Affiliation"))
            authors.append(Author(name=name, affiliation=aff))

    pub_types = [
        (pt.text or "").strip()
        for pt in article.xpath(".//PublicationTypeList/PublicationType")
        if pt.text
    ]

    keywords = [(k.text or "").strip() for k in article.xpath(".//KeywordList/Keyword") if k.text]
    mesh = [
        (m.text or "").strip()
        for m in article.xpath(".//MeshHeadingList/MeshHeading/DescriptorName")
        if m.text
    ]

    return Paper(
        identifier=ident,
        title=title,
        abstract=abstract,
        authors=authors,
        journal=journal,
        publication_date=_publication_date(article),
        publication_types=pub_types,
        keywords=keywords,
        mesh_terms=mesh,
        design=StudyDesign.unknown,
    )
