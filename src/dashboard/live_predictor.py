"""Live self-scoring next-candle predictor (updates every refresh).

Unlike the daily model (one signal a day), this predicts the **next candle** of a
short interval (1m/5m/…) *before it closes*, using the live technical rating, then
scores itself the moment that candle closes — a running, honest forward-test that
grows every refresh.

Flow, per refresh:

1. Fetch the recent **closed** candles (the forming one is excluded).
2. Call the next (forming) candle's direction from the technical rating — one call
   per candle (deduplicated by the target candle's open time).
3. Score any earlier prediction whose target candle has since closed: green candle
   (close > open) = "up", red = "down"; ``✅`` if the call matched, else ``❌``.

It is a rule-based momentum indicator, **not** a proven edge — on noisy 1m bars
the hit rate settles near ~50%.  The table shows that honestly, in real time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_CALL_GLYPH = {"UP": "▲ UP", "DOWN": "▼ DOWN", "NEUTRAL": "■ NEUTRAL"}
_ACTUAL_GLYPH = {"up": "▲ up", "down": "▼ down", "flat": "– flat"}


def candle_direction(open_p: float, close_p: float) -> str:
    """Direction of a candle: ``"up"`` (green), ``"down"`` (red) or ``"flat"``."""
    if close_p > open_p:
        return "up"
    if close_p < open_p:
        return "down"
    return "flat"


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def next_candle_signal(candles: pd.DataFrame) -> dict[str, Any]:
    """Two-sided next-candle call from short-term **mean reversion**.

    The Live-tab gauge is trend-following, which in an uptrend only ever says
    "up" — useless for a single next candle.  This is deliberately symmetric: it
    leans **DOWN when price is stretched above** its short EMA / overbought, and
    **UP when stretched below** / oversold, plus a bid-ask-bounce term that
    reverts the last candle.  So it can (and does) call both directions.

    It is still a weak, ~50% short-horizon indicator — mean reversion just makes
    the calls balanced rather than one-sided.

    Args:
        candles: Recent closed OHLC candles (``open`` / ``close``), chronological.

    Returns:
        Dict with ``predicted`` ("UP"/"DOWN"/"NEUTRAL"), ``score`` and ``votes``.
    """
    close = candles["close"].astype("float64").reset_index(drop=True)
    open_ = candles["open"].astype("float64").reset_index(drop=True)
    if len(close) < 50:
        raise ValueError(f"next-candle signal needs ≥50 candles, got {len(close)}")

    resid = close - _ema(close, 20)
    resid_std = resid.tail(50).std(ddof=1)
    z = resid.iloc[-1] / resid_std if resid_std and not np.isnan(resid_std) else 0.0
    rsi = float(_rsi(close, 7).iloc[-1])
    last_dir = candle_direction(open_.iloc[-1], close.iloc[-1])

    votes = {
        # Stretched above its short EMA → expect a pull back (down), and vice-versa.
        "stretch vs EMA20": -1 if z > 0.5 else (1 if z < -0.5 else 0),
        # Fast RSI overbought/oversold → mean-revert.
        "RSI(7)": -1 if rsi > 60 else (1 if rsi < 40 else 0),
        # Bid-ask bounce: lean opposite the last candle.
        "prev-candle reversion": {"up": -1, "down": 1, "flat": 0}[last_dir],
    }
    score = sum(votes.values()) / len(votes)
    predicted = "UP" if score > 0.1 else ("DOWN" if score < -0.1 else "NEUTRAL")
    return {"predicted": predicted, "score": score, "votes": votes}


def predict_next_candle(candles: pd.DataFrame) -> dict[str, Any]:
    """Call the next candle's direction (two-sided mean-reversion signal).

    Returns a dict with ``predicted`` ("UP"/"DOWN"/"NEUTRAL") and ``score``.
    """
    return next_candle_signal(candles)


def update_predictions(
    preds: dict[str, dict[str, Any]],
    candles: pd.DataFrame,
    *,
    live_price: float,
    now: pd.Timestamp,
    max_keep: int = 60,
) -> dict[str, dict[str, Any]]:
    """Add a prediction for the forming candle and score any that have matured.

    Args:
        preds: State keyed by the target candle's ISO open time.
        candles: Recent **closed** OHLC candles (``open_time`` / ``open`` /
            ``close``), chronological.
        live_price: Current price, recorded as the prediction's reference.
        now: Timestamp of this prediction.
        max_keep: Trim to the most recent this-many predictions.

    Returns:
        The updated ``preds`` dict.
    """
    if candles is None or len(candles) < 50:
        return preds
    candles = candles.sort_values("open_time").reset_index(drop=True)
    ot = candles["open_time"]
    step = ot.diff().median()
    if pd.isna(step):
        return preds

    # Predict the forming (not-yet-closed) candle, once.
    target_open = pd.Timestamp(ot.iloc[-1]) + step
    key = target_open.isoformat()
    if key not in preds:
        p = predict_next_candle(candles)
        preds[key] = {
            "predicted_at": pd.Timestamp(now),
            "predicted": p["predicted"],
            "score": float(p["score"]),
            "price": float(live_price),
            "target_open": target_open,
            "actual": None,
            "result": None,
        }

    # Score matured predictions: their target candle is now a closed candle.
    by_open = {pd.Timestamp(o).isoformat(): i for i, o in enumerate(ot)}
    for k, pr in preds.items():
        if pr["result"] is not None:
            continue
        idx = by_open.get(k)
        if idx is None:
            continue  # target candle hasn't closed yet
        row = candles.iloc[idx]
        actual = candle_direction(float(row["open"]), float(row["close"]))
        pr["actual"] = actual
        if pr["predicted"] == "NEUTRAL" or actual == "flat":
            pr["result"] = "—"
        else:
            ok = (pr["predicted"] == "UP" and actual == "up") or (
                pr["predicted"] == "DOWN" and actual == "down"
            )
            pr["result"] = "✅" if ok else "❌"

    if len(preds) > max_keep:
        keep = sorted(preds.items(), key=lambda kv: kv[1]["predicted_at"])[-max_keep:]
        preds = dict(keep)
    return preds


def _strength(score: float) -> str:
    """Confidence label from the vote score magnitude (1/2/3 agreeing votes)."""
    a = abs(score)
    if a >= 0.99:
        return "●●● strong"
    if a >= 0.6:
        return "●●○ medium"
    if a > 0:
        return "●○○ weak"
    return "—"


def _direction_stats(
    preds: dict[str, dict[str, Any]], call: str,
) -> dict[str, Any]:
    """Scored tally for a single call direction ("UP" or "DOWN")."""
    scored = [
        p for p in preds.values()
        if p["predicted"] == call and p["result"] in ("✅", "❌")
    ]
    n = len(scored)
    ok = sum(1 for p in scored if p["result"] == "✅")
    pending = sum(
        1 for p in preds.values()
        if p["predicted"] == call and p["result"] is None
    )
    return {
        "n_calls": n,
        "n_correct": ok,
        "n_pending": pending,
        "hit_rate": (100.0 * ok / n) if n else None,
    }


def predictions_table(
    preds: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Render the predictions as a table (most recent first) + a running tally.

    Returns:
        ``(table, summary)`` where ``summary`` has ``n_scored`` / ``n_correct`` /
        ``n_pending`` / ``hit_rate`` and a ``by_call`` sub-dict with a separate
        buy-side (``UP``) and sell-side (``DOWN``) breakdown.
    """
    cols = ["Predicted at", "Call", "Conf", "Price", "Target candle",
            "Actual", "Result"]
    rows = sorted(preds.values(), key=lambda p: p["predicted_at"], reverse=True)
    table = pd.DataFrame([
        {
            "Predicted at": pd.Timestamp(p["predicted_at"]).strftime("%H:%M:%S"),
            "Call": _CALL_GLYPH.get(p["predicted"], p["predicted"]),
            "Conf": _strength(p.get("score", 0.0)),
            "Price": f"${p['price']:,.0f}",
            "Target candle": pd.Timestamp(p["target_open"]).strftime("%H:%M"),
            "Actual": _ACTUAL_GLYPH.get(p["actual"], "⏳ pending"),
            "Result": p["result"] or "⏳",
        }
        for p in rows
    ], columns=cols)

    scored = [p for p in preds.values() if p["result"] in ("✅", "❌")]
    n_correct = sum(1 for p in scored if p["result"] == "✅")
    summary = {
        "n_scored": len(scored),
        "n_correct": n_correct,
        "n_pending": sum(1 for p in preds.values() if p["result"] is None),
        "hit_rate": (100.0 * n_correct / len(scored)) if scored else None,
        "by_call": {
            "UP": _direction_stats(preds, "UP"),
            "DOWN": _direction_stats(preds, "DOWN"),
        },
    }
    return table, summary


def poll_predictor(
    binance_symbol: str,
    binance_interval: str,
    preds: dict[str, dict[str, Any]],
    *,
    n_candles: int = 200,
) -> dict[str, dict[str, Any]]:
    """Fetch live candles + price and advance the predictor state one step.

    Thin I/O wrapper around :func:`update_predictions` (kept out of the pure
    functions so those stay unit-testable).
    """
    from src.dashboard.signals import fetch_live_candles, fetch_live_price

    candles = fetch_live_candles(binance_symbol, binance_interval, n_candles=n_candles)
    live_price = fetch_live_price(binance_symbol)
    return update_predictions(
        preds, candles, live_price=live_price, now=pd.Timestamp.now(tz="UTC"),
    )
