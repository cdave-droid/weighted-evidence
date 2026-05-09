"""Unified retrieval entrypoint. Resolves identifier kind, dispatches to PubMed."""

from __future__ import annotations

import httpx

from weighted_evidence.cache import Cache, open_cache
from weighted_evidence.config import Settings
from weighted_evidence.config import settings as load_settings
from weighted_evidence.models import Paper
from weighted_evidence.retrieval.pubmed import PubMedClient
from weighted_evidence.retrieval.resolver import IdentifierKind, resolve


class RetrievalClient:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: Cache | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.cache = cache if cache is not None else open_cache()
        self._http = http_client
        self._owns_http = http_client is None
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

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
            return await pubmed.fetch(ident)
