"""Multi-timeframe confluence + volatility-regime gating (expert discipline).

Grounded in classic multi-timeframe analysis — most directly Alexander Elder's
**Triple Screen** system (*Trading for a Living*): read the *tide* on a higher
timeframe, time on a lower one, and only act when they **agree**.  A professional
trades selectively; the default state is **no setup — sit out**.

Two ideas are combined:

* **Confluence** — run the same next-candle signal on several timeframes
  (1m/5m/15m/1h) and take a *weighted* directional vote, with **higher
  timeframes weighted more** (they set the trend).  Broad agreement = a
  higher-quality moment; disagreement = stand aside.
* **Volatility-regime gate** — the vol-direction model (Task 3, ~69% CV) says
  whether volatility is likely to **expand**.  A directional setup is only
  "grade A" when timeframes align *and* volatility is expanding (a move needs
  fuel); in a contraction, even aligned timeframes usually chop.

Honest boundary: this does **not** beat the measured ~50% direction ceiling.
It is a *filter* that flags the best moments and tells you to skip the rest —
selectivity and discipline, not a price oracle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.dashboard.live_predictor import next_candle_signal

# Low → high.  The higher timeframe is the "tide" (Elder), so it carries more
# weight in the directional vote.
CONFLUENCE_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "1h")
_TF_WEIGHT: dict[str, float] = {"1m": 1.0, "5m": 1.5, "15m": 2.0, "1h": 3.0}

# |weighted net| below this reads as MIXED → no directional setup (sit out).
DIR_DEADZONE = 0.15
# Weighted-net magnitude at/above which timeframes are "strongly aligned".
STRONG_ALIGN = 0.60


@dataclass(frozen=True)
class TFCall:
    """One timeframe's next-candle call."""

    timeframe: str
    predicted: str      # "UP" / "DOWN" / "NEUTRAL"
    score: float        # signed conviction, −1..1
    regime: str         # "trend" / "range"
    weight: float       # timeframe weight in the confluence vote


def gather_timeframe_calls(
    binance_symbol: str,
    timeframes: tuple[str, ...] = CONFLUENCE_TIMEFRAMES,
    *,
    n_candles: int = 200,
    poll: Callable[[str, str], pd.DataFrame] | None = None,
) -> list[TFCall]:
    """Run the next-candle signal across several timeframes.

    Args:
        binance_symbol: e.g. ``"BTCUSDT"``.
        timeframes: Binance intervals, low → high.
        n_candles: Candles to fetch per timeframe.
        poll: Injectable ``(symbol, interval) -> candles`` (for tests); defaults
            to the live Binance fetch.  A timeframe that fails to fetch or lacks
            enough candles is skipped, not fatal.

    Returns:
        A list of :class:`TFCall` (may be shorter than ``timeframes``).
    """
    if poll is None:
        from src.dashboard.signals import fetch_live_candles

        def poll(sym: str, tf: str) -> pd.DataFrame:  # noqa: D401 - thin default
            return fetch_live_candles(sym, tf, n_candles=n_candles)

    calls: list[TFCall] = []
    for tf in timeframes:
        try:
            candles = poll(binance_symbol, tf)
            sig = next_candle_signal(candles)
        except Exception:  # noqa: BLE001 - one flaky timeframe must not break the panel
            continue
        calls.append(TFCall(
            timeframe=tf, predicted=sig["predicted"], score=float(sig["score"]),
            regime=sig["regime"], weight=_TF_WEIGHT.get(tf, 1.0),
        ))
    return calls


def confluence(calls: list[TFCall]) -> dict[str, Any]:
    """Weighted directional agreement across timeframes.

    Args:
        calls: Output of :func:`gather_timeframe_calls`.

    Returns:
        Dict with ``direction`` ("UP"/"DOWN"/"MIXED"), ``net`` (weighted vote in
        −1..1), ``strength`` (``|net|``), ``n_up`` / ``n_down`` / ``n_neutral``,
        ``agree`` (largest same-direction count) and ``n_tf``.
    """
    empty = {
        "direction": "MIXED", "net": 0.0, "strength": 0.0,
        "n_up": 0, "n_down": 0, "n_neutral": 0, "agree": 0, "n_tf": 0,
    }
    if not calls:
        return empty
    wsum = sum(c.weight for c in calls)
    if wsum <= 0:
        return empty
    num = sum(
        c.weight * (1 if c.predicted == "UP" else -1 if c.predicted == "DOWN" else 0)
        for c in calls
    )
    net = num / wsum
    n_up = sum(1 for c in calls if c.predicted == "UP")
    n_down = sum(1 for c in calls if c.predicted == "DOWN")
    n_neutral = sum(1 for c in calls if c.predicted == "NEUTRAL")
    direction = "UP" if net > DIR_DEADZONE else ("DOWN" if net < -DIR_DEADZONE else "MIXED")
    return {
        "direction": direction, "net": float(net), "strength": abs(float(net)),
        "n_up": n_up, "n_down": n_down, "n_neutral": n_neutral,
        "agree": max(n_up, n_down), "n_tf": len(calls),
    }


def setup_verdict(
    conf: dict[str, Any],
    vol_regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Grade the moment by combining confluence with the volatility regime.

    Grades (a disciplined trader acts on **A**, considers **B**, skips the rest):

    * **A — high quality**: timeframes strongly aligned *and* volatility expanding.
    * **B — moderate**: strong alignment *or* (a lean with expanding volatility).
    * **C — weak**: a directional lean without support.
    * **No setup**: timeframes disagree → sit out.

    Args:
        conf: Output of :func:`confluence`.
        vol_regime: Output of
            :func:`src.dashboard.outlook.predict_volatility_regime` (or ``None``).

    Returns:
        Dict with ``grade``, ``action``, ``direction``, ``expanding`` and
        ``strong_align``.
    """
    direction = conf["direction"]
    expanding = bool(vol_regime and vol_regime.get("regime") == "EXPAND")
    strong_align = conf["strength"] >= STRONG_ALIGN and direction != "MIXED"

    if direction == "MIXED":
        grade, action = "No setup", "Timeframes disagree — sit out."
    elif strong_align and expanding:
        grade = "A — high quality"
        action = f"Timeframes align {direction} and volatility is expanding — best setup."
    elif strong_align or expanding:
        grade = "B — moderate"
        why = "strong timeframe alignment" if strong_align else "expanding volatility"
        action = f"{direction} lean with {why} — half-conviction at most."
    else:
        grade = "C — weak"
        action = f"Weak {direction} lean, no support — usually skip."

    return {
        "grade": grade, "action": action, "direction": direction,
        "expanding": expanding, "strong_align": strong_align,
    }
