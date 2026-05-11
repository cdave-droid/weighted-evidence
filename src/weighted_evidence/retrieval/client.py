"""Unified retrieval entrypoint.

Pipeline:
  1. Resolve identifier kind (DOI / PMID / PMCID).
  2. Fetch PubMed metadata + structured abstract.
  3. If the paper has a PMCID, try Europe PMC full-text and attach
     `body_sections` + `body_text` to the Paper. Paywalled / not-OA papers
     fall back to abstract-only.
"""

from __future__ import annotations

import httpx

from weighted_evidence.cache import Cache, open_cache
from weighted_evidence.config import Settings
from weighted_evidence.config import settings as load_settings
from weighted_evidence.models import Paper
from weighted_evidence.retrieval.europepmc import (
    EuropePMCClient,
    assemble_body_text,
    parse_jats,
)
from weighted_evidence.retrieval.pubmed import PubMedClient
from weighted_evidence.retrieval.resolver import IdentifierKind, resolve


class RetrievalClient:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: Cache | None = None,
        http_client: httpx.AsyncClient | None = None,
        fetch_fulltext: bool = True,
    ) -> None:
        self.settings = settings or load_settings()
        self.cache = cache if cache is not None else open_cache()
        self.fetch_fulltext = fetch_fulltext
        self._http = http_client
        self._owns_http = http_client is None
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    async def aclose(self) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()

    async def __aenter__(self) -> RetrievalClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def fetch(self, identifier: str) -> Paper:
        kind, ident = resolve(identifier)
        if kind is IdentifierKind.unknown:
            raise ValueError(f"Could not classify identifier: {identifier!r}")
        async with PubMedClient(
            settings=self.settings,
            cache=self.cache,
            client=self._http,
        ) as pubmed:
            paper = await pubmed.fetch(ident)

        if self.fetch_fulltext and paper.identifier.pmcid:
            paper = await self._attach_fulltext(paper)
        return paper

    async def _attach_fulltext(self, paper: Paper) -> Paper:
        pmcid = paper.identifier.pmcid
        if not pmcid:
            return paper
        epmc = EuropePMCClient(settings=self.settings, cache=self.cache, client=self._http)
        try:
            xml = await epmc.fetch_fulltext_xml(pmcid)
        finally:
            await epmc.aclose()
        if not xml:
            return paper
        sections = parse_jats(xml)
        if not sections:
            return paper
        return paper.model_copy(
            update={
                "fulltext_xml": xml,
                "body_sections": sections,
                "body_text": assemble_body_text(sections),
            }
        )
