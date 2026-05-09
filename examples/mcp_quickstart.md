# MCP server quickstart

`weighted-evidence` ships an MCP server that exposes paper grading, ranking,
and comparison to any MCP-compatible client (Claude Desktop, Cursor, etc.).

## Install

```bash
uv pip install -e ".[mcp]"
export ANTHROPIC_API_KEY=...
export NCBI_API_KEY=...   # optional, raises rate limits
```

## Run

```bash
weighted-evidence-mcp
```

This launches a stdio MCP server. Tools exposed:

- `grade_paper(identifier)` — full report
- `findings_card(identifier)` — agent-facing JSON only
- `rank_papers(identifiers, query_pico?)` — query-conditioned ranking
- `compare_papers(identifiers, query_pico?)` — pairwise rationale

## Claude Desktop config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%/Claude/claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "weighted-evidence": {
      "command": "weighted-evidence-mcp",
      "env": {
        "ANTHROPIC_API_KEY": "sk-...",
        "NCBI_API_KEY": "..."
      }
    }
  }
}
```

## Cursor config

In `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "weighted-evidence": {
      "command": "weighted-evidence-mcp"
    }
  }
}
```

## Example agent prompt

> Use weighted-evidence to compare these three papers on early antibiotics
> for sepsis: 10.1056/NEJMoa1703058, 10.1056/NEJMoa1701380, 10.1056/NEJMoa1903305.
> Tell me which to rely on for an adult ICU patient with fluid-refractory
> hypotension.

The agent calls `compare_papers([...], query_pico={"population": "adult ICU
fluid-refractory hypotension", "intervention": "early antibiotics"})` and
gets back ordered FindingsCards with explicit per-pair rationale citing
GRADE certainty, reliability_tier, fragility, and PICO directness.
