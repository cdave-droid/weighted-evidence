# weighted-evidence

[![CI](https://github.com/cdave-droid/weighted-evidence/actions/workflows/ci.yml/badge.svg)](https://github.com/cdave-droid/weighted-evidence/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Open-source Python framework + MCP server that grades and ranks biomedical
papers so AI agents weight evidence the way the literature itself does.

## Why

AI agents that source biomedical research today treat every indexed paper as
roughly equal — a 50-patient retrospective cohort and a multi-center RCT
carry the same weight when an agent retrieves them. weighted-evidence
pre-computes, for each paper:

- whether the **finding is clinically significant** — effect size + CI vs
  MID + outcome importance, surfaced as a per-outcome
  `clinical_significance` verdict (`likely | uncertain | unlikely`); not
  just statistically significant; and
- whether the **finding is reliable** given how the study was designed and
  conducted — GRADE + RoB 2 / AMSTAR-2 / ROBINS-I + fragility + spin +
  retraction guard + predatory-journal guard + guideline-weighted citation
  context, compressed into a top-level
  `reliability_tier ∈ {rely | use_with_caution | weak_signal | do_not_rely | retracted}`.

Both judgments are surfaced on a stable `FindingsCard` JSON. Agents read
the verdict directly instead of re-interpreting raw stats.

## Status

`v0.1.0` — first public alpha. Phase 1 is functionally complete.

## Install

```bash
uv venv
uv pip install -e ".[dev]"             # development
# or
uv pip install -e ".[mcp]"             # MCP server extra
# or, once published:
uv pip install weighted-evidence
```

Optional environment:

```bash
export ANTHROPIC_API_KEY=...   # enables LLM-refined rubric (PR 7)
export NCBI_API_KEY=...        # raises PubMed rate limits
export SEMANTIC_SCHOLAR_API_KEY=...   # raises Semantic Scholar rate limits
```

## CLI

```bash
weighted-evidence grade 10.1056/NEJMoa1503326     # full report
weighted-evidence card  10.1056/NEJMoa1503326     # FindingsCard only
weighted-evidence rank  10793162 30772908 29347874 \
    --population "adult ICU septic shock" \
    --outcome "28-day mortality"
weighted-evidence compare PMID1 PMID2             # pairwise rationale
```

## Python

```python
from weighted_evidence.agents import EvidenceAgent
from weighted_evidence.models import PICO
from weighted_evidence.llm.anthropic import AnthropicProvider

async def main():
    async with EvidenceAgent(llm=AnthropicProvider()) as agent:
        report = await agent.grade("10.1056/NEJMoa1503326")
        print(report.card.reliability_tier, report.card.final_score)

        ranking = await agent.rank(
            ["10793162", "30772908", "29347874"],
            query=PICO(population="adult ICU septic shock"),
        )
        for card in ranking.cards:
            print(card.id, card.reliability_tier.value)
```

## MCP server

```bash
weighted-evidence-mcp     # stdio
```

Tools exposed:

- `grade_paper(identifier)` — full report
- `findings_card(identifier)` — agent-facing JSON
- `rank_papers(identifiers, query_pico?)` — query-conditioned ranking
- `compare_papers(identifiers, query_pico?)` — pairwise rationale

See [`examples/mcp_quickstart.md`](examples/mcp_quickstart.md) for Claude
Desktop / Cursor config snippets.

## Rubric at a glance

| Tool       | Applies to                               | Output                                           |
|------------|------------------------------------------|--------------------------------------------------|
| GRADE      | Body of evidence on a question           | High / Moderate / Low / Very Low                 |
| RoB 2      | Randomized trials                        | Per-domain Low / Some concerns / High            |
| AMSTAR-2   | Systematic reviews                       | Critically low / low / moderate / high           |
| ROBINS-I   | Non-randomized intervention studies      | Per-domain Low / Moderate / Serious / Critical   |
| GIS (v0)   | All papers                               | 0–1 score; uses Semantic Scholar contexts when available |

Plus per-outcome `clinical_significance` verdict, top-level
`reliability_tier` decision table, fragility index, spin detection,
retraction guard (PubMed publication types + Retraction Watch), and a
predatory-journal blocklist with a clear override path.

## Aggregate formula

```
final_score = 0.40*GRADE + 0.20*RoB_tool + 0.18*GIS
            + 0.12*PICOMatch + 0.05*outcome_importance
            + 0.05*fragility_or_imprecision
```

Hard vetoes: retraction → `final_score=None`; predatory-journal hit caps
at `0.4`. Weights are configurable.

## Phase 2 (next)

- Critical-Care guideline corpus (Surviving Sepsis, SCCM, ESICM, NICE) —
  every cited reference extracted with the *guideline's own
  evidence-strength label* — fed into the GIS as the dominant feature.
- Fine-tuned `GISModel` (Llama-3.1-8B + LoRA) replaces the rule-v0 GIS
  behind the same Protocol — call sites unchanged.
- Specialty-pluggable from day one: `data_pipeline/ingest/<specialty>/<society>.py`.

## License

MIT.
