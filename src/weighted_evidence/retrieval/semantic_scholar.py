"""Semantic Scholar client. Fetches paper metadata + citation contexts/intents.

Citation contexts let the GIS reflect *how* the field treats a paper —
supportive, disputing, mentioning, or methodological — instead of raw
citation count. A paper cited 100x supportively outranks one cited 100x
as a counterexample.

Free public endpoint (no API key needed for low-volume reads); pass
SEMANTIC_SCHOLAR_API_KEY to raise rate limits.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential_jitter

from weighted_evidence.cache import Cache, cache_key
from weighted_evidence.config import Settings
from weighted_evidence.models import CitationContext

S2_BASE = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarError(RuntimeError):
    pass


class SemanticScholarClient:
    def __init__(
        self,
        *,
        settings: Settings,
        cache: Cache | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self._client = client
        self._owns_client = client is None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        if self.settings.semantic_scholar_api_key:
            return {"x-api-key": self.settings.semantic_scholar_api_key}
        return {}

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        client = await self._ensure_client()
        url = f"{S2_BASE}/{path.lstrip('/')}"
        ck = cache_key("s2", path, sorted((params or {}).items()))
        if self.cache:
            cached = self.cache.get("http", ck)
            if cached is not None:
                cached_result = json.loads(cached)
                if not isinstance(cached_result, dict):
                    raise SemanticScholarError(f"Unexpected cached non-dict for {path}")
                return cached_result
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_exponential_jitter(initial=0.5, max=8.0),
                reraise=True,
            ):
                with attempt:
                    resp = await client.get(url, headers=self._headers(), params=params)
                    resp.raise_for_status()
                    text = resp.text
        except RetryError as exc:  # pragma: no cover - defensive
            raise SemanticScholarError(f"S2 {path} failed after retries") from exc
        if self.cache:
            self.cache.set("http", ck, text)
        result = json.loads(text)
        if not isinstance(result, dict):
            raise SemanticScholarError(f"Unexpected non-dict response from {path}")
        return result

    async def fetch_citation_contexts(
        self, *, doi: str | None = None, pmid: str | None = None, limit: int = 100
    ) -> CitationContext:
        """Aggregate citation contexts/intents for a paper into our `CitationContext`.

        Returns an empty (zeroed) context when neither identifier resolves;
        the GIS treats that the same as 'no citation data', falling back to
        sample-size + journal tier signals.
        """

        ident = _s2_id(doi=doi, pmid=pmid)
        if ident is None:
            return CitationContext()
        try:
            data = await self._get(
                f"paper/{ident}/citations",
                {
                    "limit": str(min(max(limit, 1), 1000)),
                    "fields": "intents,isInfluential,contexts",
                },
            )
        except SemanticScholarError:
            return CitationContext()

        rows = data.get("data", [])
        supportive = disputing = mentioning = methodological = 0
        examples: list[str] = []
        for row in rows:
            intents = [str(x).lower() for x in (row.get("intents") or [])]
            contexts = [str(x) for x in (row.get("contexts") or [])]
            if "result" in intents or row.get("isInfluential"):
                supportive += 1
            elif "background" in intents:
                mentioning += 1
            elif "methodology" in intents:
                methodological += 1
            else:
                mentioning += 1
            if contexts and len(examples) < 5:
                examples.append(contexts[0][:240])

            # Heuristic: snippets that mention "contradict", "fail to replicate",
            # "could not reproduce", "in contrast" mark as disputing.
            joined = " ".join(contexts).lower()
            if any(
                marker in joined
                for marker in (
                    "contradict",
                    "fail to replicate",
                    "could not reproduce",
                    "in contrast to",
                    "did not confirm",
                )
            ):
                disputing += 1
                # If we counted it as supportive above, decrement.
                supportive = max(0, supportive - 1)

        total = supportive + disputing + mentioning + methodological
        return CitationContext(
            total=total,
            supportive=supportive,
            disputing=disputing,
            mentioning=mentioning,
            methodological=methodological,
            examples=examples,
        )


def _s2_id(*, doi: str | None, pmid: str | None) -> str | None:
    if doi:
        return f"DOI:{doi}"
    if pmid:
        return f"PMID:{pmid}"
    return None


__all__ = ["SemanticScholarClient", "SemanticScholarError"]
