"""Multi-strategy ensemble — classic named techniques, voted together.

Each strategy below is a well-known method from the technical-trading canon,
implemented plainly and transparently:

* **EMA crossover** — trend (fast vs slow exponential moving average).
* **MACD** — Gerald Appel's momentum oscillator (MACD line vs signal line).
* **Donchian breakout** — the Turtle-trader channel breakout (Dennis/Eckhardt).
* **RSI reversion** — Wilder's Relative Strength Index, overbought/oversold.
* **Bollinger Bands** — John Bollinger's volatility bands, mean-reversion.

They are combined by a **weighted vote** into one BUY / SELL / HOLD decision — a
standard ensemble/voting approach.  This mixes *trend* methods (which win in
trends) with *mean-reversion* methods (which win in ranges) so no single regime
dominates.

Honest boundary: BTC short-horizon direction is ~50% (measured elsewhere in this
project).  Combining several ~50% signals does **not** manufacture an edge — an
ensemble lowers variance and enforces discipline, nothing more.  A paper-trading
run of this will hover near break-even and **lose to fees** over time; that is
the honest lesson, not a strategy to fund.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ── Individual strategies: each returns +1 (buy) / -1 (sell) / 0 (flat) ─────


def strat_ema_cross(candles: pd.DataFrame, *, fast: int = 12, slow: int = 26) -> int:
    """Trend: long when the fast EMA is above the slow EMA, short below."""
    c = candles["close"].astype("float64")
    f, s = _ema(c, fast).iloc[-1], _ema(c, slow).iloc[-1]
    return 1 if f > s else (-1 if f < s else 0)


def strat_macd(
    candles: pd.DataFrame, *, fast: int = 12, slow: int = 26, signal: int = 9
) -> int:
    """Momentum: MACD line above its signal line ⇒ buy, below ⇒ sell."""
    c = candles["close"].astype("float64")
    macd = _ema(c, fast) - _ema(c, slow)
    sig = _ema(macd, signal)
    return 1 if macd.iloc[-1] > sig.iloc[-1] else (-1 if macd.iloc[-1] < sig.iloc[-1] else 0)


def strat_donchian(candles: pd.DataFrame, *, n: int = 20) -> int:
    """Breakout (Turtle): close above the prior N-high ⇒ buy, below N-low ⇒ sell."""
    c = candles["close"].astype("float64")
    if len(c) < n + 1:
        return 0
    prior = c.iloc[-(n + 1):-1]
    # Strict breakout: merely equalling the prior extreme (e.g. a flat line) is
    # not a breakout.
    if c.iloc[-1] > prior.max():
        return 1
    if c.iloc[-1] < prior.min():
        return -1
    return 0


def strat_rsi_reversion(
    candles: pd.DataFrame, *, n: int = 14, low: float = 30.0, high: float = 70.0
) -> int:
    """Mean-reversion: RSI oversold ⇒ buy, overbought ⇒ sell (Wilder)."""
    r = float(_rsi(candles["close"].astype("float64"), n).iloc[-1])
    if np.isnan(r):
        return 0
    return 1 if r < low else (-1 if r > high else 0)


def strat_bollinger(candles: pd.DataFrame, *, n: int = 20, k: float = 2.0) -> int:
    """Mean-reversion: close below the lower band ⇒ buy, above the upper ⇒ sell."""
    c = candles["close"].astype("float64")
    if len(c) < n:
        return 0
    mid = c.rolling(n).mean().iloc[-1]
    sd = c.rolling(n).std(ddof=0).iloc[-1]
    if np.isnan(sd) or sd == 0:
        return 0
    price = c.iloc[-1]
    if price < mid - k * sd:
        return 1
    if price > mid + k * sd:
        return -1
    return 0


@dataclass(frozen=True)
class Strategy:
    """A named strategy, its family, its vote weight, and its function."""

    name: str
    kind: str  # "trend" or "reversion"
    weight: float
    fn: Callable[[pd.DataFrame], int]


STRATEGIES: tuple[Strategy, ...] = (
    Strategy("EMA cross", "trend", 1.0, strat_ema_cross),
    Strategy("MACD", "trend", 1.0, strat_macd),
    Strategy("Donchian breakout", "trend", 1.0, strat_donchian),
    Strategy("RSI reversion", "reversion", 1.0, strat_rsi_reversion),
    Strategy("Bollinger", "reversion", 1.0, strat_bollinger),
)

# |net vote| at/above which the ensemble issues a directional decision.
DECISION_THRESHOLD = 0.20


def ensemble_signal(
    candles: pd.DataFrame,
    *,
    strategies: tuple[Strategy, ...] = STRATEGIES,
) -> dict[str, Any]:
    """Combine the strategies into one BUY / SELL / HOLD decision by weighted vote.

    Args:
        candles: Recent OHLC candles (``close``; ``high``/``low`` optional),
            chronological.
        strategies: The strategy set to vote (defaults to :data:`STRATEGIES`).

    Returns:
        Dict with ``decision`` ("BUY"/"SELL"/"HOLD"), ``net`` (weighted vote in
        −1..1), ``conviction`` (``|net|``), ``votes`` (per-strategy ±1/0) and
        ``n_buy`` / ``n_sell`` counts.
    """
    votes: dict[str, int] = {}
    wsum = 0.0
    num = 0.0
    for strat in strategies:
        try:
            v = int(strat.fn(candles))
        except Exception:  # noqa: BLE001 - a degenerate strategy must not break the vote
            v = 0
        votes[strat.name] = v
        num += strat.weight * v
        wsum += strat.weight
    net = num / wsum if wsum else 0.0
    decision = (
        "BUY" if net >= DECISION_THRESHOLD
        else "SELL" if net <= -DECISION_THRESHOLD
        else "HOLD"
    )
    return {
        "decision": decision,
        "net": float(net),
        "conviction": abs(float(net)),
        "votes": votes,
        "n_buy": sum(1 for v in votes.values() if v > 0),
        "n_sell": sum(1 for v in votes.values() if v < 0),
    }
