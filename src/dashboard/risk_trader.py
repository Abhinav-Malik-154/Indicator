"""Risk-managed, volatility-gated trading — where profit actually comes from.

Tasks so far proved short-horizon *direction* is ~50%, so chasing a higher hit
rate is a dead end.  Profit comes from **expectancy**, not accuracy:

    expectancy = win% · avg_win − loss% · avg_loss

You can be right well under half the time and still profit if wins are bigger
than losses.  This engine adds the three professional pieces that make that
happen, on top of the strategy ensemble:

1. **Volatility gate** — only *enter* when volatility is expanding
   (:func:`vol_expanding`).  Volatility clusters (the same principle behind the
   project's 69% vol-regime edge), so a move is more likely to *go* somewhere;
   in a contraction the engine sits out and pays no fees.
2. **ATR stop-loss + take-profit** — every trade gets a predefined stop and a
   larger target (default **2:1** reward:risk).  Losers are cut small, winners
   run — so the average win outweighs the average loss even below a 50% hit
   rate (2:1 breaks even at just 33% wins).
3. **Risk-based position sizing** — risk a fixed fraction of equity per trade
   (:func:`position_size`), so one bad trade can't blow up the account.

It reuses the portfolio state, analytics and persistence from
:mod:`src.dashboard.paper_trader`; only the *decision/execution* step differs.
Still fake money — but now the equity curve is a fair test of a *real* trading
method, not a coin-flip that only pays fees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.dashboard.paper_trader import (
    _MAX_CURVE,
    _MAX_TRADES,
    FEE_RATE,
    PAPER_LOG_DIR,
)
from src.dashboard.strategies import ensemble_signal

# Defaults (tuned for reward:risk asymmetry, not for a magic win rate).
RISK_FRAC = 0.02        # risk 2% of equity per trade
STOP_ATR = 1.5          # stop distance = 1.5 × ATR
REWARD_RISK = 2.0       # target = 2 × the stop distance → 2:1 R:R
ATR_WINDOW = 14
VOL_SHORT, VOL_LONG = 10, 40


def risk_path(binance_symbol: str, binance_interval: str) -> Path:
    """On-disk JSON path for a symbol+interval's risk-managed run."""
    return Path(PAPER_LOG_DIR) / f"risk_trade_{binance_symbol}_{binance_interval}.json"


# ── Indicators / gates ─────────────────────────────────────────────────────


def atr(candles: pd.DataFrame, n: int = ATR_WINDOW) -> float:
    """Average True Range (Wilder) — the volatility unit for stops/targets."""
    high = candles["high"].astype("float64")
    low = candles["low"].astype("float64")
    close = candles["close"].astype("float64")
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    val = tr.rolling(n).mean().iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def vol_expanding(
    candles: pd.DataFrame, *, short: int = VOL_SHORT, long: int = VOL_LONG
) -> bool:
    """True when recent volatility is at least its longer-window average.

    A cheap, self-contained proxy for the volatility-regime edge on the trading
    timeframe: only take directional risk when the market is *waking up*.
    """
    ret = candles["close"].astype("float64").pct_change().dropna()
    if len(ret) < long:
        return True  # not enough history to judge → don't block entries
    recent = float(ret.tail(short).std(ddof=1))
    baseline = float(ret.tail(long).std(ddof=1))
    if baseline <= 0:
        return True
    return recent >= baseline


def compute_bracket(
    entry: float, atr_val: float, *, stop_atr: float = STOP_ATR, rr: float = REWARD_RISK
) -> tuple[float, float]:
    """Stop-loss and take-profit prices from ATR, with an asymmetric R:R."""
    risk = stop_atr * atr_val
    return entry - risk, entry + risk * rr


def position_size(
    equity: float, entry: float, stop: float, cash: float, price: float,
    *, risk_frac: float = RISK_FRAC, fee_rate: float = FEE_RATE,
) -> float:
    """BTC quantity so a stop-out loses ~``risk_frac`` of equity, capped by cash."""
    risk_per_unit = entry - stop
    if risk_per_unit <= 0 or price <= 0:
        return 0.0
    qty_by_risk = (equity * risk_frac) / risk_per_unit
    qty_by_cash = cash / (price * (1.0 + fee_rate))
    return max(0.0, min(qty_by_risk, qty_by_cash))


# ── Execution ──────────────────────────────────────────────────────────────


def _mark_equity(state: dict[str, Any], price: float, now: str) -> None:
    equity = state["cash"] + state["btc"] * price
    curve = state["equity_curve"]
    curve.append({"time": now, "equity": round(equity, 2), "price": round(price, 2)})
    del curve[:-_MAX_CURVE]


def _enter(
    state: dict[str, Any], price: float, qty: float, stop: float, target: float,
    *, now: str, fee_rate: float,
) -> None:
    cost = qty * price
    fee = cost * fee_rate
    state["cash"] -= cost + fee
    state["btc"] = qty
    state["entry_price"] = price
    state["stop_price"] = stop
    state["target_price"] = target
    state["fees_paid"] += fee
    state["trades"].append(
        {"time": now, "side": "BUY", "price": price, "qty": qty, "fee": fee,
         "stop": stop, "target": target}
    )
    del state["trades"][:-_MAX_TRADES]


def _exit(state: dict[str, Any], price: float, reason: str, *, now: str, fee_rate: float) -> None:
    qty = state["btc"]
    gross = qty * price
    fee = gross * fee_rate
    proceeds = gross - fee
    realized = proceeds - qty * float(state["entry_price"] or price)
    state["cash"] += proceeds
    state["btc"] = 0.0
    state["entry_price"] = None
    state["stop_price"] = None
    state["target_price"] = None
    state["fees_paid"] += fee
    state["trades"].append(
        {"time": now, "side": "SELL", "price": price, "qty": qty, "fee": fee,
         "realized": realized, "reason": reason}
    )
    del state["trades"][:-_MAX_TRADES]


def risk_step(
    state: dict[str, Any],
    candles: pd.DataFrame,
    price: float,
    decision: str,
    *,
    now: str,
    candle_open: str,
    fee_rate: float = FEE_RATE,
    risk_frac: float = RISK_FRAC,
    stop_atr: float = STOP_ATR,
    rr: float = REWARD_RISK,
) -> dict[str, Any]:
    """Advance the risk-managed portfolio one step.

    Exits (stop / target) are checked **every** call against the live price;
    entries happen at most **once per candle** and only when the ensemble says
    BUY *and* volatility is expanding.
    """
    if price <= 0:
        return state

    # 1) Manage an open position — exits fire any time, not just on a new candle.
    if state["btc"] > 0:
        stop = state.get("stop_price")
        target = state.get("target_price")
        if stop is not None and price <= stop:
            _exit(state, price, "stop", now=now, fee_rate=fee_rate)
        elif target is not None and price >= target:
            _exit(state, price, "target", now=now, fee_rate=fee_rate)

    # 2) Consider a new entry, once per candle.
    if state.get("last_candle") != candle_open:
        state["last_candle"] = candle_open
        if state["btc"] == 0 and decision == "BUY" and vol_expanding(candles):
            atr_val = atr(candles)
            if atr_val > 0:
                stop, target = compute_bracket(price, atr_val, stop_atr=stop_atr, rr=rr)
                equity = state["cash"] + state["btc"] * price
                qty = position_size(
                    equity, price, stop, state["cash"], price,
                    risk_frac=risk_frac, fee_rate=fee_rate,
                )
                if qty > 0:
                    _enter(state, price, qty, stop, target, now=now, fee_rate=fee_rate)

    _mark_equity(state, price, now)
    return state


def poll_risk_trader(
    binance_symbol: str,
    binance_interval: str,
    state: dict[str, Any],
    *,
    n_candles: int = 200,
    fee_rate: float = FEE_RATE,
) -> dict[str, Any]:
    """Fetch live candles + price, run the ensemble, and step the risk engine."""
    from src.dashboard.signals import fetch_live_candles, fetch_live_price

    candles = fetch_live_candles(binance_symbol, binance_interval, n_candles=n_candles)
    price = float(fetch_live_price(binance_symbol))
    if not state.get("first_price"):
        state["first_price"] = price
    sig = ensemble_signal(candles)
    # Trend-following long/flat: go long when the market is in an **uptrend**
    # (price above its 50-EMA) and the ensemble is not bearish — and (checked in
    # risk_step) volatility is expanding. This keeps the system *participating*
    # in up-moves instead of sitting idle whenever the 5 strategies cancel to
    # HOLD. It still exits on the ATR stop or target.
    close = candles["close"].astype("float64")
    uptrend = bool(close.iloc[-1] > close.ewm(span=50, adjust=False).mean().iloc[-1])
    entry = "BUY" if (uptrend and sig["net"] >= 0) else "HOLD"
    candle_open = pd.Timestamp(candles["open_time"].iloc[-1]).isoformat()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    state = risk_step(
        state, candles, price, entry,
        now=now, candle_open=candle_open, fee_rate=fee_rate,
    )
    state["last_signal"] = sig
    state["last_entry"] = entry
    state["uptrend"] = uptrend
    state["last_price"] = price
    state["vol_expanding"] = vol_expanding(candles)
    return state
