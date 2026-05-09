# weighted-evidence

Open-source Python framework + MCP server that grades and ranks biomedical papers so AI agents weight evidence the way the literature itself does.

## Why

AI agents that source biomedical research today treat every indexed paper as roughly equal — a 50-patient retrospective cohort and a multi-center RCT carry the same weight when an agent retrieves them. weighted-evidence pre-computes, for each paper:

- whether the **finding is clinically significant** (effect size + CI vs MID + outcome importance) — not just statistically significant, and
- whether the **finding is reliable** given how the study was designed and conducted (GRADE + RoB 2 / AMSTAR-2 / ROBINS-I + fragility + spin + retraction + predatory-journal guard + guideline-weighted citation context).

Both judgments are surfaced as agent-readable verdicts on a stable `FindingsCard` JSON.

## Status

Phase 1, alpha. The first vertical slice grades a single paper end-to-end:

```bash
weighted-evidence grade 10.1056/NEJMoa1503326
```

## Install

```bash
uv venv
uv pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...
export NCBI_API_KEY=...   # optional, raises rate limits
```

## Quick start

```python
from weighted_evidence.agents import EvidenceAgent

agent = EvidenceAgent()
report = await agent.grade("10.1056/NEJMoa1503326")
print(report.final_score, report.reliability_tier)
```

## MCP server

```bash
weighted-evidence-mcp   # stdio
```

Wire into Claude Desktop or Cursor — see `examples/mcp_quickstart.md`.

## Rubric at a glance

| Tool       | Applies to                              | Output                                |
|------------|-----------------------------------------|---------------------------------------|
| GRADE      | Body of evidence on a question          | High / Moderate / Low / Very Low      |
| RoB 2      | Randomized trials                       | Per-domain Low / Some / High          |
| AMSTAR-2   | Systematic reviews                      | Critically low / low / moderate / high|
| ROBINS-I   | Non-randomized intervention studies     | Per-domain Low / Moderate / Serious / Critical |
| GIS (v0)   | All papers                              | 0–1 score, weighted by guideline-assigned strength |

Plus: per-outcome `clinical_significance` verdict, top-level `reliability_tier`, fragility index, spin detection, citation context, retraction guard, predatory-journal guard.

## License

MIT.
