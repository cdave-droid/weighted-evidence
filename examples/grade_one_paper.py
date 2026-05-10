"""Grade a single paper end-to-end.

    uv run python examples/grade_one_paper.py 10.1056/NEJMoa1503326

If `ANTHROPIC_API_KEY` is set, the LLM-graded rubric refinement runs
automatically; otherwise the framework falls back to its rule-only path.
"""

from __future__ import annotations

import asyncio
import sys

from weighted_evidence.agents import EvidenceAgent
from weighted_evidence.config import settings
from weighted_evidence.llm.anthropic import AnthropicProvider


async def main(identifier: str) -> int:
    cfg = settings()
    llm = AnthropicProvider(settings=cfg) if cfg.anthropic_api_key else None

    async with EvidenceAgent(llm=llm) as agent:
        report = await agent.grade(identifier)

    card = report.card
    print(f"Paper:           {card.title}")
    print(
        f"Journal / Year:  {card.journal} / {card.publication_date and card.publication_date.year}"
    )
    print(f"Design:          {card.design.value}")
    print(f"GRADE certainty: {card.grade and card.grade.final_certainty.value}")
    print(f"RoB tool:        {card.rob_tool}")
    print(f"GIS:             {card.gis and round(card.gis.score, 2)}")
    print(f"Reliability:     {card.reliability_tier.value}")
    print(f"Final score:     {card.final_score and round(card.final_score, 3)}")
    print()
    if card.outcomes:
        print("Outcomes:")
        for o in card.outcomes:
            cs = o.clinical_significance.verdict.value if o.clinical_significance else "n/a"
            print(f"  - {o.name[:80]} [{o.importance.value}] -> {cs}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: grade_one_paper.py <DOI|PMID|PMCID>", file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1])))
