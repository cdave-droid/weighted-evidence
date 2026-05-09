"""MCP server for weighted-evidence.

Exposes:
  - grade_paper(identifier)
  - findings_card(identifier)
  - rank_papers(identifiers, query_pico?)
  - compare_papers(identifiers, query_pico?)

Install with the `mcp` extra: `uv pip install -e '.[mcp]'`. Run via the
`weighted-evidence-mcp` console script (stdio transport).
"""

from __future__ import annotations

from typing import Any

from weighted_evidence.agents import EvidenceAgent
from weighted_evidence.models import PICO


async def _grade(identifier: str) -> str:
    async with EvidenceAgent() as agent:
        report = await agent.grade(identifier)
    return report.model_dump_json(indent=2)


async def _card(identifier: str) -> str:
    async with EvidenceAgent() as agent:
        report = await agent.grade(identifier)
    return report.card.model_dump_json(indent=2)


async def _rank(identifiers: list[str], query_dict: dict[str, Any] | None = None) -> str:
    query = PICO.model_validate(query_dict) if query_dict else None
    async with EvidenceAgent() as agent:
        ranking = await agent.rank(identifiers, query=query)
    return ranking.model_dump_json(indent=2)


async def _compare(identifiers: list[str], query_dict: dict[str, Any] | None = None) -> str:
    query = PICO.model_validate(query_dict) if query_dict else None
    async with EvidenceAgent() as agent:
        comparison = await agent.compare(identifiers, query=query)
    return comparison.model_dump_json(indent=2)


def _build_server() -> Any:
    """Lazy import so the package installs cleanly without the `mcp` extra."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "weighted-evidence-mcp requires the `mcp` extra. "
            "Install with: uv pip install 'weighted-evidence[mcp]'"
        ) from exc

    mcp = FastMCP("weighted-evidence")

    @mcp.tool()
    async def grade_paper(identifier: str) -> str:
        """Grade a single paper. Returns the full WeightedEvidenceReport JSON."""

        return await _grade(identifier)

    @mcp.tool()
    async def findings_card(identifier: str) -> str:
        """Emit only the agent-facing FindingsCard JSON (stable schema)."""

        return await _card(identifier)

    @mcp.tool()
    async def rank_papers(identifiers: list[str], query_pico: dict[str, Any] | None = None) -> str:
        """Rank papers, optionally conditioned on a query PICO."""

        return await _rank(identifiers, query_pico)

    @mcp.tool()
    async def compare_papers(
        identifiers: list[str], query_pico: dict[str, Any] | None = None
    ) -> str:
        """Pairwise comparison with explicit per-pair rationale."""

        return await _compare(identifiers, query_pico)

    return mcp


def run() -> None:
    """Console-script entry point: stdio MCP server."""

    server = _build_server()
    server.run()  # type: ignore[no-untyped-call]


__all__ = ["run"]


if __name__ == "__main__":  # pragma: no cover
    run()
