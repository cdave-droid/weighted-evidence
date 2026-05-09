"""Fragility Index for binary-outcome RCTs.

Walsh et al. 2014: the Fragility Index is the minimum number of patients who
would need to switch from non-event to event in the treatment group (the
smaller of the two) to render a statistically significant result
non-significant. A FI <= 4 indicates a finding that would flip on a handful
of events — important context an agent should weight.

We compute it when a 2x2 contingency is recoverable from the abstract
("31.0% vs 39.8%" alongside a sample size suffices); otherwise return None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import comb

from weighted_evidence.models import FragilityIndex, Paper


def _fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p-value for a 2x2 table — hypergeometric."""

    n = a + b + c + d
    if n == 0:
        return 1.0
    row1 = a + b
    col1 = a + c
    if row1 == 0 or col1 == 0 or row1 == n or col1 == n:
        return 1.0
    denom = comb(n, col1)
    observed_p = comb(row1, a) * comb(n - row1, col1 - a) / denom

    total = 0.0
    k_lo = max(0, col1 - (n - row1))
    k_hi = min(row1, col1)
    for k in range(k_lo, k_hi + 1):
        p_k = comb(row1, k) * comb(n - row1, col1 - k) / denom
        # 'as or more extreme' uses probabilities <= observed (with tolerance).
        if p_k <= observed_p + 1e-12:
            total += p_k
    return min(total, 1.0)


@dataclass(frozen=True)
class _TwoByTwo:
    a: int  # treatment events
    b: int  # treatment non-events
    c: int  # control events
    d: int  # control non-events


_PCT_PAIR = re.compile(
    r"(?P<a_pct>\d+\.?\d*)\s*%\s*(?:vs\.?|versus|compared\s+with)\s*(?P<b_pct>\d+\.?\d*)\s*%",
    re.I,
)


def _build_table(abstract: str, total_n: int) -> _TwoByTwo | None:
    m = _PCT_PAIR.search(abstract)
    if m is None:
        return None
    pct_t = float(m.group("a_pct")) / 100.0
    pct_c = float(m.group("b_pct")) / 100.0
    # Best-effort: assume balanced groups. Real implementations can refine when
    # the abstract reports group-specific Ns.
    n_t = total_n // 2
    n_c = total_n - n_t
    a = round(pct_t * n_t)
    c = round(pct_c * n_c)
    b = n_t - a
    d = n_c - c
    if min(a, b, c, d) < 0:
        return None
    return _TwoByTwo(a=a, b=b, c=c, d=d)


def _fisher_p(table: _TwoByTwo) -> float:
    return _fisher_two_sided(table.a, table.b, table.c, table.d)


def fragility_index(table: _TwoByTwo, *, alpha: float = 0.05) -> int:
    """Walsh-style FI: increment events in the lower-event arm until p >= alpha."""

    p0 = _fisher_p(table)
    if p0 >= alpha:
        return 0  # already non-significant
    a, b, c, d = table.a, table.b, table.c, table.d
    # Choose the arm with fewer events; flip non-event to event there.
    if a + c == 0:
        return 0
    flip_treatment = a <= c
    fi = 0
    while True:
        if flip_treatment and b > 0:
            a += 1
            b -= 1
        elif not flip_treatment and d > 0:
            c += 1
            d -= 1
        else:
            break
        fi += 1
        if _fisher_p(_TwoByTwo(a=a, b=b, c=c, d=d)) >= alpha:
            return fi
        if fi > 200:  # pragma: no cover - safety cap
            return fi
    return fi


def fragility_for_paper(paper: Paper) -> FragilityIndex | None:
    """Best-effort FI for an RCT with a binary primary outcome.

    Returns None when we can't recover a 2x2 contingency from the abstract.
    """

    if paper.sample_size is None or paper.sample_size < 10:
        return None
    if not paper.abstract:
        return None
    table = _build_table(paper.abstract, paper.sample_size)
    if table is None:
        return None
    fi = fragility_index(table)
    quotient = fi / paper.sample_size if paper.sample_size > 0 else None
    return FragilityIndex(index=fi, quotient=quotient, method="walsh")


__all__ = ["FragilityIndex", "fragility_for_paper", "fragility_index"]
