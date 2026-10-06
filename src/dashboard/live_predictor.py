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
   (close > open) = "up", red = "down"; a correct match is marked as correct,
   otherwise wrong.

It is a rule-based momentum indicator, **not** a proven edge — on noisy 1m bars
the hit rate settles near ~50%.  The table shows that honestly, in real time.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Live predictions are persisted here (one file per symbol+interval) so the
# scoreboard survives a browser refresh, logout, or reopen — it is not kept only
# in the ephemeral Streamlit session.
PRED_LOG_DIR = "data/signal_log"
_PRED_COLUMNS = [
    "target_open", "predicted_at", "predicted", "score", "price", "actual", "result",
]

_CALL_GLYPH = {"UP": "UP", "DOWN": "DOWN", "NEUTRAL": "NEUTRAL"}
_ACTUAL_GLYPH = {"up": "up", "down": "down", "flat": "flat"}

# A hit rate on a handful of calls is noise: 2/3 = 67% is one lucky flip, and
# flipping a fair coin 6 times gives 2 heads all the time.  Below this many
# *scored* calls we refuse to present a percentage as if it were skill — the UI
# shows "sample too small" instead, so no one bets on a 3-sample 67%.
MIN_SCORED_FOR_HIT_RATE = 30

# |trend strength| (EMA10–EMA30 gap in units of per-candle volatility) at which
# the signal *fully* trusts the trend.  Below it, trend is blended smoothly with
# mean-reversion (weight = |trend| / this), so only a genuinely strong, sustained
# trend makes the call one-sided; ordinary chop stays two-sided mean-reversion.
# This is still a ~50% indicator over the long run — it does not create an edge.
TREND_SATURATION = 2.5


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


def _trend_strength(close: pd.Series) -> float:
    """EMA10–EMA30 gap, normalised by recent per-candle volatility.

    Positive → up-trend, negative → down-trend; the magnitude is in units of
    typical per-candle move, so it is comparable across price levels and
    volatility regimes.
    """
    ema_fast = _ema(close, 10).iloc[-1]
    ema_slow = _ema(close, 30).iloc[-1]
    vol = float(close.pct_change().tail(30).std(ddof=1))
    if not vol or np.isnan(vol):
        return 0.0
    return float((ema_fast - ema_slow) / close.iloc[-1] / vol)


def next_candle_signal(candles: pd.DataFrame) -> dict[str, Any]:
    """Blended next-candle call: mean-revert the chop, follow only strong trends.

    Two textbook behaviours, blended smoothly so neither dominates the way a
    brittle hard switch does (which made the call one-sided "UP" in any drift):

    * **Mean-reversion** (``mr``): lean **DOWN when stretched above** the short
      EMA / overbought, **UP when stretched below** / oversold, plus a
      bid-ask-bounce term.  Two-sided — this rules in ordinary chop.
    * **Trend follow**: a signed tilt from :func:`_trend_strength`, trusted in
      proportion to how strong the trend is (``w = |trend| / TREND_SATURATION``,
      capped at 1).  Only a *genuinely strong* trend makes the call one-sided.

    The final score is ``w · trend + (1 − w) · mr`` — chop ⇒ pure mean-reversion
    (balanced), a strong sustained trend ⇒ follow it.  It stays a weak, ~50%
    short-horizon indicator; the blend only keeps it balanced and stops the
    systematic trend-fighting.  It does **not** create a proven edge.

    Args:
        candles: Recent closed OHLC candles (``open`` / ``close``), chronological.

    Returns:
        Dict with ``predicted`` ("UP"/"DOWN"/"NEUTRAL"), ``score`` (−1..1),
        ``votes``, ``regime`` ("trend"/"range"), ``trend_strength`` and
        ``mr_score``.
    """
    close = candles["close"].astype("float64").reset_index(drop=True)
    open_ = candles["open"].astype("float64").reset_index(drop=True)
    if len(close) < 50:
        raise ValueError(f"next-candle signal needs ≥50 candles, got {len(close)}")

    # Mean-reversion vote (two-sided), the default in the common ranging case.
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
    mr = sum(votes.values()) / len(votes)

    # Trend tilt, trusted in proportion to its strength (0 in chop, 1 when strong).
    trend = _trend_strength(close)
    trend_component = float(np.clip(trend / TREND_SATURATION, -1.0, 1.0))
    trust = abs(trend_component)
    # Cap trend-trust so mean-reversion ALWAYS keeps a say: when a strong trend
    # and an overbought/oversold reading conflict, the call is genuinely
    # uncertain, so confidence must fall — an expert never rides a stretched
    # trend at full conviction.
    w = min(0.75, trust)
    score = w * trend_component + (1.0 - w) * mr

    predicted = "UP" if score > 0.1 else ("DOWN" if score < -0.1 else "NEUTRAL")
    return {
        "predicted": predicted, "score": float(score), "votes": votes,
        "regime": "trend" if trust >= 0.5 else "range",
        "trend_strength": trend, "mr_score": mr,
    }


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
            pr["result"] = "correct" if ok else "wrong"

    if len(preds) > max_keep:
        keep = sorted(preds.items(), key=lambda kv: kv[1]["predicted_at"])[-max_keep:]
        preds = dict(keep)
    return preds


def _strength(score: float) -> str:
    """Signal-alignment label from the score magnitude.

    Deliberately humble words — this is how *aligned* the sub-signals are, **not**
    a probability of being right.  A 1-minute direction call is ~50% however the
    dots read, so the top tier is "firm", never "strong".
    """
    a = abs(score)
    if a >= 0.66:
        return "●●● firm"
    if a >= 0.33:
        return "●●○ mild"
    if a > 0:
        return "●○○ faint"
    return "—"


def _direction_stats(
    preds: dict[str, dict[str, Any]], call: str,
) -> dict[str, Any]:
    """Scored tally for a single call direction ("UP" or "DOWN")."""
    scored = [
        p for p in preds.values()
        if p["predicted"] == call and p["result"] in ("correct", "wrong")
    ]
    n = len(scored)
    ok = sum(1 for p in scored if p["result"] == "correct")
    pending = sum(
        1 for p in preds.values()
        if p["predicted"] == call and p["result"] is None
    )
    return {
        "n_calls": n,
        "n_correct": ok,
        "n_pending": pending,
        "hit_rate": (100.0 * ok / n) if n else None,
        "reliable": n >= MIN_SCORED_FOR_HIT_RATE,
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
            "Actual": _ACTUAL_GLYPH.get(p["actual"], "pending"),
            "Result": p["result"] or "pending",
        }
        for p in rows
    ], columns=cols)

    scored = [p for p in preds.values() if p["result"] in ("correct", "wrong")]
    n_correct = sum(1 for p in scored if p["result"] == "correct")
    summary = {
        "n_scored": len(scored),
        "n_correct": n_correct,
        "n_pending": sum(1 for p in preds.values() if p["result"] is None),
        "hit_rate": (100.0 * n_correct / len(scored)) if scored else None,
        "reliable": len(scored) >= MIN_SCORED_FOR_HIT_RATE,
        "min_scored": MIN_SCORED_FOR_HIT_RATE,
        "by_call": {
            "UP": _direction_stats(preds, "UP"),
            "DOWN": _direction_stats(preds, "DOWN"),
        },
    }
    return table, summary


def predictions_path(binance_symbol: str, binance_interval: str) -> Path:
    """On-disk CSV path for a symbol+interval's persisted predictions."""
    return Path(PRED_LOG_DIR) / f"live_preds_{binance_symbol}_{binance_interval}.csv"


def save_predictions(preds: dict[str, dict[str, Any]], path: Path | str) -> None:
    """Persist predictions to CSV atomically (survives refresh / logout / reopen).

    Args:
        preds: The predictor state dict.
        path: Destination CSV path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(preds.values(), key=lambda p: p["predicted_at"])
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PRED_COLUMNS)
        writer.writeheader()
        for p in rows:
            writer.writerow({
                "target_open": pd.Timestamp(p["target_open"]).isoformat(),
                "predicted_at": pd.Timestamp(p["predicted_at"]).isoformat(),
                "predicted": p["predicted"],
                "score": f"{float(p.get('score', 0.0)):.6f}",
                "price": f"{float(p['price']):.2f}",
                "actual": p["actual"] or "",
                "result": p["result"] or "",
            })
    tmp.replace(path)  # atomic swap — a crash mid-write never corrupts the log


def load_predictions(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load persisted predictions back into the predictor state dict.

    Args:
        path: CSV path written by :func:`save_predictions`.

    Returns:
        The state dict keyed by the target candle's ISO open time (empty if the
        file is missing or unreadable — persistence must never break the panel).
    """
    path = Path(path)
    if not path.is_file():
        return {}
    preds: dict[str, dict[str, Any]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                target_open = pd.Timestamp(row["target_open"])
                preds[target_open.isoformat()] = {
                    "predicted_at": pd.Timestamp(row["predicted_at"]),
                    "predicted": row["predicted"],
                    "score": float(row["score"]) if row["score"] else 0.0,
                    "price": float(row["price"]) if row["price"] else 0.0,
                    "target_open": target_open,
                    "actual": row["actual"] or None,
                    "result": row["result"] or None,
                }
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("could not load persisted predictions from %s: %s", path, exc)
        return {}
    return preds


def accuracy_over_time(preds: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Cumulative hit-rate of scored predictions over time (for the compare chart).

    Args:
        preds: The predictor state dict.

    Returns:
        DataFrame with ``time`` (when each call was made), ``hit_rate`` (running
        % correct up to and including that call) and ``n`` (scored count so far),
        chronological.  Empty if nothing has been scored yet.
    """
    scored = sorted(
        (p for p in preds.values() if p["result"] in ("correct", "wrong")),
        key=lambda p: p["predicted_at"],
    )
    rows: list[dict[str, Any]] = []
    correct = 0
    for i, p in enumerate(scored, start=1):
        if p["result"] == "correct":
            correct += 1
        rows.append({
            "time": pd.Timestamp(p["predicted_at"]),
            "hit_rate": 100.0 * correct / i,
            "n": i,
        })
    return pd.DataFrame(rows, columns=["time", "hit_rate", "n"])


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
