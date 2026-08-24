"""Live technical-rating signal call for the 'Live' tab (BUY / SELL / NEUTRAL).

The Live tab embeds TradingView's own canvas, so the project's model markers
can't be drawn on it.  This module instead produces a **signal call** beside the
chart — the same idea as TradingView's "Technicals" Buy/Sell gauge — aggregating
a handful of classic indicators computed from **live Binance candles** for the
selected market and interval:

* EMA(10) vs EMA(30) trend
* price vs SMA(50)
* RSI(14) oversold / overbought
* MACD vs signal
* 10-bar momentum

Each votes ``+1`` (up) / ``-1`` (down) / ``0`` (neutral); the average maps to a
STRONG BUY → STRONG SELL call.  This is a transparent, rule-based indicator —
**not** a proven-profitable prediction.  On this project's own measurements
short-horizon direction has no demonstrated edge, so the call is honest context,
not a forecast (the panel says so).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.dashboard.signals import fetch_live_candles

# Rating band → (label, colour).  Greens for buy, reds for sell, grey neutral.
RATING_STYLE: dict[str, str] = {
    "STRONG BUY": "#0ecb81",
    "BUY": "#2ebd85",
    "NEUTRAL": "#848e9c",
    "SELL": "#f6787f",
    "STRONG SELL": "#f6465d",
}

# Live-tab interval labels → Binance interval strings.
_INTERVAL_TO_BINANCE = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1D": "1d", "1W": "1w",
}


def to_binance(symbol_tv: str, interval_label: str) -> tuple[str, str]:
    """Map a Live-tab TradingView symbol + interval to Binance equivalents.

    ``BINANCE:BTCUSDT`` → ``BTCUSDT``; ``BITSTAMP:BTCUSD`` / ``COINBASE:BTCUSD``
    fall back to the Binance ``…USDT`` proxy so the rating always has a live
    Binance feed.

    Args:
        symbol_tv: TradingView symbol, e.g. ``"BINANCE:BTCUSDT"``.
        interval_label: Live-tab interval label, e.g. ``"1m"`` or ``"1D"``.

    Returns:
        ``(binance_symbol, binance_interval)``.
    """
    base = symbol_tv.split(":")[-1].upper()
    if base.endswith("USD") and not base.endswith("USDT"):
        base = base + "T"  # BTCUSD → BTCUSDT
    return base, _INTERVAL_TO_BINANCE.get(interval_label, "1m")


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def classify(score: float) -> str:
    """Map an average vote in ``[-1, 1]`` to a rating label."""
    if score >= 0.5:
        return "STRONG BUY"
    if score >= 0.1:
        return "BUY"
    if score > -0.1:
        return "NEUTRAL"
    if score > -0.5:
        return "SELL"
    return "STRONG SELL"


def compute_technical_rating(candles: pd.DataFrame) -> dict[str, Any]:
    """Aggregate classic indicators on the latest candle into a signal call.

    Args:
        candles: OHLC DataFrame with a ``close`` column (chronological), at least
            ~50 rows so SMA(50) is defined.

    Returns:
        Dict with ``call`` (label), ``score`` (avg vote in [-1,1]), ``votes``
        (per-indicator ``+1/0/-1``), ``n_up`` / ``n_down`` counts, ``rsi`` and
        ``price``.

    Raises:
        ValueError: If there are too few candles.
    """
    close = candles["close"].astype("float64").reset_index(drop=True)
    if len(close) < 50:
        raise ValueError(f"technical rating needs ≥50 candles, got {len(close)}")

    ema_fast = _ema(close, 10).iloc[-1]
    ema_slow = _ema(close, 30).iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    rsi = float(_rsi(close, 14).iloc[-1])
    macd = _ema(close, 12) - _ema(close, 26)
    macd_line, signal_line = macd.iloc[-1], _ema(macd, 9).iloc[-1]
    momentum = close.iloc[-1] - close.iloc[-11]
    price = float(close.iloc[-1])

    votes: dict[str, int] = {
        "EMA 10/30": 1 if ema_fast > ema_slow else -1,
        "Price vs SMA50": 1 if price > sma50 else -1,
        "RSI(14)": 1 if rsi < 30 else (-1 if rsi > 70 else 0),
        "MACD": 1 if macd_line > signal_line else -1,
        "Momentum(10)": 1 if momentum > 0 else (-1 if momentum < 0 else 0),
    }
    score = sum(votes.values()) / len(votes)
    return {
        "call": classify(score),
        "score": score,
        "votes": votes,
        "n_up": sum(1 for v in votes.values() if v > 0),
        "n_down": sum(1 for v in votes.values() if v < 0),
        "rsi": rsi,
        "price": price,
    }


def rate_symbol(
    binance_symbol: str,
    binance_interval: str,
    *,
    n_candles: int = 200,
    base_url: str = "https://api.binance.com",
) -> dict[str, Any]:
    """Fetch live candles for a symbol/interval and compute its rating.

    Args:
        binance_symbol: Binance pair, e.g. ``"BTCUSDT"``.
        binance_interval: Binance interval, e.g. ``"1m"`` / ``"1d"``.
        n_candles: How many candles to fetch (≥50 needed for SMA50).
        base_url: Override for testing.

    Returns:
        The rating dict from :func:`compute_technical_rating`, plus ``symbol``
        and ``interval``.
    """
    candles = fetch_live_candles(
        binance_symbol, binance_interval, n_candles=n_candles, base_url=base_url,
    )
    rating = compute_technical_rating(candles)
    rating["symbol"] = binance_symbol
    rating["interval"] = binance_interval
    return rating
