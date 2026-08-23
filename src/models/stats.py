"""Small statistics helpers for honest accuracy reporting (Task 1).

The single most misleading thing in a trading-model report is an accuracy number
without an error bar.  With a few hundred test rows, a point estimate of "52%" is
compatible with anything from "no skill" to "modest edge" — so every accuracy in
this project should be reported with a confidence interval and an explicit test
of whether it *significantly* beats the base rate.

These functions use the **Wilson score interval** (better than the normal
approximation for proportions near 0/1 and small n) and a one-proportion
significance check against a fixed base rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProportionCI:
    """A proportion estimate with a Wilson score confidence interval."""

    k: int              # successes (e.g. correct predictions)
    n: int              # trials (e.g. total predictions)
    point: float        # k / n
    low: float          # lower CI bound
    high: float         # upper CI bound
    z: float            # z used (1.96 ≈ 95%)


def wilson_interval(k: int, n: int, *, z: float = 1.96) -> ProportionCI:
    """Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Standard-normal quantile (1.96 → 95%, 2.576 → 99%).

    Returns:
        A :class:`ProportionCI`.  For ``n == 0`` the interval is ``[0, 1]``.

    Raises:
        ValueError: If ``k < 0``, ``n < 0``, or ``k > n``.
    """
    if n < 0 or k < 0:
        raise ValueError(f"k and n must be non-negative, got k={k}, n={n}")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if n == 0:
        return ProportionCI(k=0, n=0, point=float("nan"), low=0.0, high=1.0, z=z)

    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return ProportionCI(
        k=k, n=n, point=phat,
        low=max(0.0, centre - margin),
        high=min(1.0, centre + margin),
        z=z,
    )


@dataclass(frozen=True)
class SignificanceResult:
    """Result of testing whether an accuracy beats a fixed base rate."""

    accuracy: float
    base_rate: float
    edge_pp: float          # (accuracy - base_rate) * 100
    ci_low: float
    ci_high: float
    beats_base_rate: bool   # base_rate strictly below the CI (significant edge)
    worse_than_base: bool   # base_rate strictly above the CI
    verdict: str


def accuracy_vs_base_rate(
    k: int,
    n: int,
    base_rate: float,
    *,
    z: float = 1.96,
) -> SignificanceResult:
    """Test whether an observed accuracy significantly differs from a base rate.

    "Significant" here means the base rate lies outside the Wilson CI for the
    accuracy — a simple, transparent check that avoids over-claiming from a
    point estimate.

    Args:
        k: Correct predictions.
        n: Total predictions.
        base_rate: The majority-class share to beat (in [0, 1]).
        z: Standard-normal quantile.

    Returns:
        A :class:`SignificanceResult` with the CI and a plain-language verdict.
    """
    ci = wilson_interval(k, n, z=z)
    acc = ci.point
    beats = base_rate < ci.low
    worse = base_rate > ci.high
    if n == 0:
        verdict = "no data"
    elif beats:
        verdict = "significantly beats base rate"
    elif worse:
        verdict = "significantly worse than base rate"
    else:
        verdict = "indistinguishable from base rate (within noise)"
    return SignificanceResult(
        accuracy=acc,
        base_rate=base_rate,
        edge_pp=(acc - base_rate) * 100.0 if n else float("nan"),
        ci_low=ci.low,
        ci_high=ci.high,
        beats_base_rate=beats,
        worse_than_base=worse,
        verdict=verdict,
    )


def format_ci(ci: ProportionCI, *, pct: bool = True) -> str:
    """Render a proportion CI as ``52.0% [48.1%, 55.8%]`` (or fractions)."""
    if ci.n == 0:
        return "n/a (no data)"
    scale = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    return (
        f"{ci.point * scale:.1f}{unit} "
        f"[{ci.low * scale:.1f}{unit}, {ci.high * scale:.1f}{unit}]"
    )
