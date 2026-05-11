"""Europe PMC full-text retrieval (open-access papers only).

Endpoint: https://www.ebi.ac.uk/europepmc/webservices/rest/article/PMC/{PMCID}/fullTextXML

Returns JATS XML which we parse into a section dict (methods / results /
discussion / conclusions / etc.) plus a flat plain-text body. When a paper
is paywalled or not in PMC, the request 4xx's and we return None — the
agent gracefully falls back to abstract-only behavior.

This is what unlocks effect-size and spin extraction from the body text,
not just the structured abstract.
"""

from __future__ import annotations

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

EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class EuropePMCError(RuntimeError):
    pass


class EuropePMCClient:
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
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def fetch_fulltext_xml(self, pmcid: str) -> str | None:
        """Fetch JATS full-text XML for a PMCID.

        Returns None when the paper isn't open-access in PMC (404) or the
        endpoint refuses the request (4xx). Callers fall back to abstract-only.
        """

        pmcid = pmcid.upper()
        pmcid_num = pmcid[3:] if pmcid.startswith("PMC") else pmcid
        url = f"{EUROPEPMC}/article/PMC/{pmcid_num}/fullTextXML"
        ck = cache_key("europepmc", "fullTextXML", pmcid_num)

        if self.cache:
            cached = self.cache.get("http", ck)
            if cached is not None:
                return cached or None  # empty string = known-missing

        client = await self._ensure_client()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_exponential_jitter(initial=0.5, max=8.0),
                reraise=True,
            ):
                with attempt:
                    resp = await client.get(url)
                    if resp.status_code == 404 or resp.status_code == 401:
                        if self.cache:
                            self.cache.set("http", ck, "")  # cache the miss
                        return None
                    resp.raise_for_status()
                    text = resp.text
        except RetryError:  # pragma: no cover - defensive
            return None
        if self.cache:
            self.cache.set("http", ck, text)
        return text


# ---------------------------------------------------------------------------
# JATS XML → section dict
# ---------------------------------------------------------------------------


_SECTION_NORMALIZE = {
    "background": "background",
    "introduction": "background",
    "methods": "methods",
    "method": "methods",
    "materials and methods": "methods",
    "patients and methods": "methods",
    "study design and methods": "methods",
    "results": "results",
    "findings": "results",
    "discussion": "discussion",
    "conclusion": "conclusions",
    "conclusions": "conclusions",
    "interpretation": "conclusions",
    "limitations": "limitations",
}


def _node_text(node: Any) -> str:
    """Flatten an lxml node to plain text, preserving paragraph breaks."""

    parts: list[str] = []
    for child in node.iter():
        if child.tag is etree.Comment:
            continue
        if child.text:
            parts.append(child.text)
        if child.tag in {"p", "sec", "title", "list-item"} and child.tail is None:
            parts.append("\n")
        if child.tail:
            parts.append(child.tail)
    return " ".join(" ".join(parts).split())


def parse_jats(xml: str) -> dict[str, str]:
    """Return a `{normalized_section_name: text}` dict from JATS XML.

    Only top-level body sections are extracted. Unknown section titles are
    kept under their original lowercase title.
    """

    if not xml:
        return {}
    try:
        root = etree.fromstring(xml.encode("utf-8"))
    except etree.XMLSyntaxError:
        return {}
    sections: dict[str, str] = {}
    for sec in root.xpath(".//body//sec"):
        title_node = sec.find("title")
        if title_node is None or not title_node.text:
            continue
        raw_title = " ".join(title_node.text.split()).lower().rstrip(":.")
        norm = _SECTION_NORMALIZE.get(raw_title, raw_title)
        text = _node_text(sec)
        if not text:
            continue
        if norm in sections:
            sections[norm] = sections[norm] + "\n\n" + text
        else:
            sections[norm] = text
    return sections


def assemble_body_text(sections: dict[str, str]) -> str:
    """Stitch a section dict back into a single plain-text body.

    Preserves a stable order so downstream regex matching is deterministic.
    """

    order = ["background", "methods", "results", "discussion", "conclusions", "limitations"]
    parts: list[str] = []
    for name in order:
        if name in sections:
            parts.append(f"{name.capitalize()}:\n{sections[name]}")
    for name, text in sections.items():
        if name not in order:
            parts.append(f"{name.capitalize()}:\n{text}")
    return "\n\n".join(parts)


__all__ = ["EuropePMCClient", "EuropePMCError", "assemble_body_text", "parse_jats"]
