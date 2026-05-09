"""Structured PICO directness — how well does this paper match the agent's query?

Returns a `PICOMatch` with per-dimension scores and surfaced mismatches so an
agent can see *why* a paper is (or isn't) a good fit. Used by `rank()` for
query-conditioned reranking.
"""

from __future__ import annotations

import re

from weighted_evidence.models import PICO, PICODimensionMatch, PICOMatch


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t for t in re.findall(r"[A-Za-z][A-Za-z0-9-]+", text.lower()) if len(t) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _dimension_score(
    dim: str, paper_value: str | None, query_value: str | None
) -> PICODimensionMatch:
    if not query_value:
        return PICODimensionMatch(
            dimension=dim,
            score=1.0,
            mismatch=None,
        )
    pa, qa = _tokens(paper_value), _tokens(query_value)
    score = _jaccard(pa, qa)
    if score >= 0.5:
        mismatch = None
    elif paper_value is None:
        mismatch = (
            f"Query specified {dim}='{query_value}' but paper does not report this dimension."
        )
    else:
        mismatch = (
            f"{dim} mismatch: paper reports '{paper_value[:80]}', "
            f"query asked about '{query_value[:80]}'."
        )
    return PICODimensionMatch(
        dimension=dim,
        score=score,
        mismatch=mismatch,
    )


def match_pico(paper_pico: PICO | None, query: PICO | None) -> PICOMatch | None:
    """Score how directly `paper_pico` matches the caller's `query` PICO.

    Returns None when there's no query (no reranking signal to apply); the
    aggregate falls back to its neutral 0.5 directness term.
    """

    if query is None:
        return None
    if paper_pico is None:
        return PICOMatch(
            overall=0.0,
            dimensions=[],
            rationale="Paper has no extracted PICO; cannot directly match against query.",
        )

    paper_outcome_str = "; ".join(paper_pico.outcomes) if paper_pico.outcomes else None
    query_outcome_str = "; ".join(query.outcomes) if query.outcomes else None

    dims = [
        _dimension_score("population", paper_pico.population, query.population),
        _dimension_score("intervention", paper_pico.intervention, query.intervention),
        _dimension_score("comparator", paper_pico.comparator, query.comparator),
        _dimension_score("outcome", paper_outcome_str, query_outcome_str),
        _dimension_score("setting", paper_pico.setting, query.setting),
    ]

    overall = sum(d.score for d in dims) / len(dims)
    mismatches = [d.mismatch for d in dims if d.mismatch]
    rationale = (
        " | ".join(mismatches)
        if mismatches
        else "Paper PICO is well aligned with the query across all dimensions."
    )
    return PICOMatch(overall=overall, dimensions=dims, rationale=rationale)


__all__ = ["match_pico"]
