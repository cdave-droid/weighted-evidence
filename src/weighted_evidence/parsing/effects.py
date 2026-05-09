"""Extract `EffectSize` rows from a paper's results text.

Rule-based first (cheap, deterministic, runs in CI without an API key). The
patterns below cover the most common ways effect estimates are reported in
abstracts and results sections; an LLM fallback hooks in via
`parse_effects_with_llm` for cases the patterns miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from weighted_evidence.models import EffectKind, EffectSize, MIDComparison

# Range separators in medical text include hyphen, en-dash (U+2013), and "to".
_EN_DASH = "–"
_RANGE_SEP = f"(?:[-{_EN_DASH}]|to|,)"


# Capture an estimate + 95% CI in any of the common forms:
#   HR 0.78 (95% CI 0.65 to 0.94)
#   HR, 0.78; 95% CI, 0.65-0.94
#   hazard ratio of 0.78 [0.65 to 0.94]
def _ratio_regex(label_alt: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b(?:{label_alt})\b"  # measure name
        r"[^0-9(\[\n]{0,40}"  # up to 40 chars of connector text
        r"(?P<point>-?\d+\.?\d*)"  # point estimate
        r"\s*[\(\[]\s*"  # opening bracket
        r"(?:95\s*%?\s*CI[\s,:.]*)?"  # optional "95% CI" prefix
        r"(?P<lo>-?\d+\.?\d*)"  # lower bound
        rf"\s*{_RANGE_SEP}\s*"  # separator
        r"(?P<hi>-?\d+\.?\d*)"  # upper bound
        r"[^\)\]]*"  # rest until close (e.g., ", P=0.01")
        r"[\)\]]",  # closing bracket
        re.I,
    )


_RATIO_PATTERNS: tuple[tuple[re.Pattern[str], EffectKind], ...] = (
    (_ratio_regex(r"HR|hazard\s+ratio"), EffectKind.hr),
    (_ratio_regex(r"RR|risk\s+ratio|relative\s+risk"), EffectKind.rr),
    (_ratio_regex(r"OR|odds\s+ratio"), EffectKind.or_),
)

# 31.0% vs 39.8%, P = 0.007  → ARR
_TWO_GROUP_PCT = re.compile(
    r"(?P<a>\d+\.?\d*)\s*%\s*(?:vs\.?|versus|compared\s+with)\s*(?P<b>\d+\.?\d*)\s*%"
    r"(?:[^.]*?(?:P|p)\s*[=<]\s*(?P<p>\d+\.?\d*))?",
)

_P_VALUE = re.compile(r"\b(?:P|p)\s*[=<]\s*(\d+\.?\d*)")


@dataclass(frozen=True)
class _Hit:
    kind: EffectKind
    point: float
    ci_low: float | None
    ci_high: float | None
    p: float | None
    span: tuple[int, int]
    surrounding: str


def _ratio_vs_null(kind: EffectKind, ci_low: float | None, ci_high: float | None) -> MIDComparison:
    """Quick 'does the CI cross 1?' for ratio measures."""

    if ci_low is None or ci_high is None:
        return MIDComparison.unknown
    if kind in {EffectKind.hr, EffectKind.rr, EffectKind.or_}:
        if ci_low > 1.0 or ci_high < 1.0:
            return MIDComparison.exceeds
        return MIDComparison.crosses_null
    return MIDComparison.unknown


def _arr_vs_null(ci_low: float | None, ci_high: float | None) -> MIDComparison:
    """For absolute risk reduction expressed as a proportion difference."""

    if ci_low is None or ci_high is None:
        return MIDComparison.unknown
    if ci_low > 0 or ci_high < 0:
        return MIDComparison.exceeds
    return MIDComparison.crosses_null


def extract_effects(text: str) -> list[EffectSize]:
    """Pull every effect estimate the rules can recognize.

    Returns an empty list when nothing matches; the caller can fall back to
    the LLM extractor in `parse_effects_with_llm`.
    """

    if not text:
        return []

    hits: list[_Hit] = []
    for pattern, kind in _RATIO_PATTERNS:
        for m in pattern.finditer(text):
            try:
                point = float(m.group("point"))
                lo = float(m.group("lo"))
                hi = float(m.group("hi"))
            except ValueError:  # pragma: no cover - regex guarantees floats
                continue
            window = text[m.start() : m.end() + 80]
            p_match = _P_VALUE.search(window)
            p = float(p_match.group(1)) if p_match else None
            hits.append(
                _Hit(
                    kind=kind,
                    point=point,
                    ci_low=min(lo, hi),
                    ci_high=max(lo, hi),
                    p=p,
                    span=m.span(),
                    surrounding=text[max(0, m.start() - 40) : m.end() + 40],
                )
            )

    for m in _TWO_GROUP_PCT.finditer(text):
        a = float(m.group("a")) / 100.0
        b = float(m.group("b")) / 100.0
        arr = b - a
        p = float(m.group("p")) if m.group("p") else None
        hits.append(
            _Hit(
                kind=EffectKind.arr,
                point=round(arr, 4),
                ci_low=None,
                ci_high=None,
                p=p,
                span=m.span(),
                surrounding=text[max(0, m.start() - 40) : m.end() + 40],
            )
        )

    seen: set[tuple[EffectKind, float]] = set()
    effects: list[EffectSize] = []
    for h in hits:
        key = (h.kind, h.point)
        if key in seen:
            continue
        seen.add(key)
        if h.kind == EffectKind.arr:
            vs_mid = _arr_vs_null(h.ci_low, h.ci_high)
        else:
            vs_mid = _ratio_vs_null(h.kind, h.ci_low, h.ci_high)
        effects.append(
            EffectSize(
                kind=h.kind,
                point=h.point,
                ci_low=h.ci_low,
                ci_high=h.ci_high,
                p=h.p,
                vs_mid=vs_mid,
            )
        )
    return effects


__all__ = ["extract_effects"]
