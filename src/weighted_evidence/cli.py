"""Typer CLI surface. Phase 1 ships `grade` only; later PRs add card/compare/rank."""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from weighted_evidence.agents import EvidenceAgent
from weighted_evidence.cache import open_cache
from weighted_evidence.models import PICO

app = typer.Typer(
    name="weighted-evidence",
    help="Grade and rank biomedical papers for AI agents.",
    add_completion=False,
)


@app.command()
def grade(
    identifier: str = typer.Argument(..., help="DOI, PMID, or PMCID."),
    pretty: bool = typer.Option(True, "--pretty/--raw", help="Pretty-print or raw JSON."),
) -> None:
    """Fetch and grade a single paper, emitting its full WeightedEvidenceReport JSON."""

    console = Console()

    async def run() -> str:
        async with EvidenceAgent() as agent:
            report = await agent.grade(identifier)
        return report.model_dump_json(indent=2 if pretty else None)

    payload = asyncio.run(run())
    if pretty:
        console.print(Panel.fit("Weighted evidence report", style="bold cyan"))
        console.print(JSON(payload))
    else:
        typer.echo(payload)


@app.command()
def card(
    identifier: str = typer.Argument(..., help="DOI, PMID, or PMCID."),
    pretty: bool = typer.Option(True, "--pretty/--raw", help="Pretty-print or raw JSON."),
) -> None:
    """Emit only the agent-facing FindingsCard JSON (stable schema)."""

    console = Console()

    async def run() -> str:
        async with EvidenceAgent() as agent:
            report = await agent.grade(identifier)
        return report.card.model_dump_json(indent=2 if pretty else None)

    payload = asyncio.run(run())
    if pretty:
        console.print(Panel.fit("Findings card", style="bold cyan"))
        console.print(JSON(payload))
    else:
        typer.echo(payload)


def _build_query(
    population: str | None,
    intervention: str | None,
    comparator: str | None,
    outcome: str | None,
    setting: str | None,
) -> PICO | None:
    if not any([population, intervention, comparator, outcome, setting]):
        return None
    return PICO(
        population=population,
        intervention=intervention,
        comparator=comparator,
        outcomes=[outcome] if outcome else [],
        setting=setting,
    )


@app.command()
def rank(
    identifiers: list[str] = typer.Argument(..., help="Two or more DOI/PMID/PMCID values."),
    population: str | None = typer.Option(None, help="Query PICO: population."),
    intervention: str | None = typer.Option(None, help="Query PICO: intervention."),
    comparator: str | None = typer.Option(None, help="Query PICO: comparator."),
    outcome: str | None = typer.Option(None, help="Query PICO: primary outcome."),
    setting: str | None = typer.Option(None, help="Query PICO: setting."),
) -> None:
    """Rank multiple papers, optionally conditioned on a query PICO."""

    query = _build_query(population, intervention, comparator, outcome, setting)
    console = Console()

    async def run() -> str:
        async with EvidenceAgent() as agent:
            ranking = await agent.rank(list(identifiers), query=query)
        return ranking.model_dump_json(indent=2)

    payload = asyncio.run(run())
    console.print(Panel.fit("Ranking", style="bold cyan"))
    console.print(JSON(payload))


@app.command()
def compare(
    identifiers: list[str] = typer.Argument(..., help="Two or more DOI/PMID/PMCID values."),
    population: str | None = typer.Option(None, help="Query PICO: population."),
    intervention: str | None = typer.Option(None, help="Query PICO: intervention."),
    comparator: str | None = typer.Option(None, help="Query PICO: comparator."),
    outcome: str | None = typer.Option(None, help="Query PICO: primary outcome."),
    setting: str | None = typer.Option(None, help="Query PICO: setting."),
) -> None:
    """Pairwise comparison with explicit rationale per pair."""

    query = _build_query(population, intervention, comparator, outcome, setting)
    console = Console()

    async def run() -> str:
        async with EvidenceAgent() as agent:
            comparison = await agent.compare(list(identifiers), query=query)
        return comparison.model_dump_json(indent=2)

    payload = asyncio.run(run())
    console.print(Panel.fit("Comparison", style="bold cyan"))
    console.print(JSON(payload))


@app.command()
def cache(
    action: str = typer.Argument(..., help="One of: clear"),
    namespace: str | None = typer.Option(None, "--namespace", help="Limit to a namespace."),
) -> None:
    """Inspect or clear the on-disk cache."""

    if action != "clear":
        raise typer.BadParameter(f"Unknown action {action!r}. Try: clear")
    c = open_cache()
    cleared = c.clear(namespace)
    typer.echo(json.dumps({"cleared": cleared, "namespace": namespace or "all"}))


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
