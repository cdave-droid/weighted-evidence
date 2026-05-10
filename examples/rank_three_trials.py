"""Rank three sepsis trials against a query PICO and emit pairwise rationale.

    uv run python examples/rank_three_trials.py

Demonstrates the agent ranking surface: `compare()` returns ordered
FindingsCards plus per-pair rationale citing the qualitative differentiators.
"""

from __future__ import annotations

import asyncio

from weighted_evidence.agents import EvidenceAgent
from weighted_evidence.models import PICO

# ProCESS / ARISE / PROMISE — three landmark sepsis early-goal-directed-therapy
# RCTs. Substitute your own DOIs/PMIDs for any clinical question.
TRIALS = [
    "10.1056/NEJMoa1401602",  # ProCESS (NEJM 2014)
    "10.1056/NEJMoa1404380",  # ARISE  (NEJM 2014)
    "10.1056/NEJMoa1500896",  # PROMISE (NEJM 2015)
]

QUERY = PICO(
    population="adult ICU patients with septic shock and fluid-refractory hypotension",
    intervention="protocolized early goal-directed therapy",
    comparator="usual care",
    outcomes=["90-day all-cause mortality"],
    setting="ICU",
)


async def main() -> int:
    async with EvidenceAgent() as agent:
        comparison = await agent.compare(TRIALS, query=QUERY)

    print("Ranked order (best -> worst):")
    for i, card in enumerate(comparison.ordered, 1):
        print(
            f"  {i}. {card.id:30}  "
            f"tier={card.reliability_tier.value:18}  "
            f"score={card.final_score and round(card.final_score, 3)}"
        )

    print("\nPairwise rationale:")
    for pair in comparison.pairwise:
        print(f"  {pair.winner_id} > {pair.loser_id}")
        for r in pair.reasons:
            print(f"      - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
