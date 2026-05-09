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
    """Fetch and grade a single paper, emitting its FindingsCard JSON."""

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
