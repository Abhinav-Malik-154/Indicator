"""Paper-trading simulator — fake money, honest P&L (fees included).

Starts with a virtual ₹10,000 and lets the multi-strategy ensemble
(:mod:`src.dashboard.strategies`) decide **BUY / SELL / HOLD** each new candle,
long-only on spot: a BUY puts all cash into BTC, a SELL returns all BTC to cash.
Every trade pays a fee, so the equity curve is *honest* — a real measuring stick
of whether the strategy actually makes money.  With the default public indicators
(no measured edge) it mostly pays fees; if a component with a genuine, measured
edge is added, the curve reflects that truthfully.  That honesty is what makes it
useful — it proves an edge instead of assuming one.

State is a plain JSON-serialisable dict, persisted to disk so the run survives a
browser refresh, logout, or reopen — just like the live predictor.

**Fake money only. Not financial advice.** Real Indian exchanges charge more than
the default fee and add 1% TDS, so a live result would be *worse* than this.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.dashboard.strategies import ensemble_signal

logger = logging.getLogger(__name__)

STARTING_CAPITAL = 10_000.0        # ₹, virtual
FEE_RATE = 0.001                   # 0.1% per trade (optimistic vs Indian exchanges)
PAPER_LOG_DIR = "data/signal_log"
_MAX_TRADES = 500
_MAX_CURVE = 3000  # ~16h of 20s ticks — a longer investment window on the graph


def new_portfolio(capital: float = STARTING_CAPITAL) -> dict[str, Any]:
    """A fresh flat portfolio holding only cash."""
    return {
        "starting_capital": float(capital),
        "cash": float(capital),
        "btc": 0.0,
        "entry_price": None,
        "stop_price": None,     # risk-managed exit floor (set on entry)
        "target_price": None,   # risk-managed profit target (set on entry)
        "first_price": None,   # first price seen → buy-and-hold benchmark anchor
        "fees_paid": 0.0,
        "last_candle": None,
        "trades": [],
        "equity_curve": [],
    }


def paper_path(binance_symbol: str, binance_interval: str) -> Path:
    """On-disk JSON path for a symbol+interval's paper-trading run."""
    return Path(PAPER_LOG_DIR) / f"paper_trade_{binance_symbol}_{binance_interval}.json"


def _mark_equity(state: dict[str, Any], price: float, now: str) -> None:
    equity = state["cash"] + state["btc"] * price
    curve = state["equity_curve"]
    # Store the price too so a buy-and-hold benchmark curve can be reconstructed.
    curve.append({"time": now, "equity": round(equity, 2), "price": round(price, 2)})
    del curve[:-_MAX_CURVE]


def apply_decision(
    state: dict[str, Any],
    price: float,
    decision: str,
    *,
    now: str,
    candle_open: str,
    fee_rate: float = FEE_RATE,
) -> dict[str, Any]:
    """Execute one long-only decision, at most once per candle.

    Args:
        state: Portfolio state (mutated and returned).
        price: Current price.
        decision: ``"BUY"`` / ``"SELL"`` / ``"HOLD"``.
        now: ISO timestamp of this step.
        candle_open: ISO open time of the latest candle (dedupe key — the trade
            fires at most once per candle).
        fee_rate: Per-trade fee fraction.

    Returns:
        The updated ``state``.
    """
    if price <= 0:
        return state
    # Mark-to-market every step, but only trade once per fresh candle.
    if state.get("last_candle") == candle_open:
        _mark_equity(state, price, now)
        return state
    state["last_candle"] = candle_open

    btc, cash = state["btc"], state["cash"]
    if decision == "BUY" and cash > 0 and btc == 0:
        fee = cash * fee_rate
        qty = (cash - fee) / price
        state.update(btc=qty, cash=0.0, entry_price=price)
        state["fees_paid"] += fee
        state["trades"].append(
            {"time": now, "side": "BUY", "price": price, "qty": qty, "fee": fee}
        )
    elif decision == "SELL" and btc > 0:
        gross = btc * price
        fee = gross * fee_rate
        proceeds = gross - fee
        realized = proceeds - btc * float(state["entry_price"] or price)
        state.update(cash=proceeds, btc=0.0, entry_price=None)
        state["fees_paid"] += fee
        state["trades"].append({
            "time": now, "side": "SELL", "price": price, "qty": btc,
            "fee": fee, "realized": realized,
        })

    del state["trades"][:-_MAX_TRADES]
    _mark_equity(state, price, now)
    return state


def portfolio_summary(state: dict[str, Any], price: float) -> dict[str, Any]:
    """Current equity, P&L, position and activity for display.

    Args:
        state: Portfolio state.
        price: Current price for mark-to-market.

    Returns:
        Dict with ``equity``, ``pnl``, ``pnl_pct``, ``position``, ``n_trades``,
        ``fees_paid``, ``unrealized`` and ``holding``.
    """
    equity = state["cash"] + state["btc"] * price
    start = state["starting_capital"]
    pnl = equity - start
    holding = state["btc"] > 0
    unrealized = (
        state["btc"] * (price - float(state["entry_price"]))
        if holding and state["entry_price"] else 0.0
    )
    return {
        "equity": equity,
        "pnl": pnl,
        "pnl_pct": (100.0 * pnl / start) if start else 0.0,
        "position": "BTC (long)" if holding else "cash (flat)",
        "holding": holding,
        "entry_price": state["entry_price"],
        "n_trades": len(state["trades"]),
        "fees_paid": state["fees_paid"],
        "unrealized": unrealized,
    }


def buy_and_hold(
    state: dict[str, Any], price: float, *, fee_rate: float = FEE_RATE
) -> dict[str, Any]:
    """Benchmark: what one buy-and-hold of the whole ₹ at the first price is worth.

    The honest yardstick every trader checks first — a strategy only earns its
    fees if it beats simply holding.
    """
    start = state["starting_capital"]
    fp = state.get("first_price")
    if not fp or fp <= 0:
        return {"value": start, "pnl": 0.0, "pnl_pct": 0.0, "first_price": None}
    btc = start * (1 - fee_rate) / fp
    value = btc * price
    pnl = value - start
    return {"value": value, "pnl": pnl, "pnl_pct": 100.0 * pnl / start, "first_price": fp}


def trade_stats(state: dict[str, Any]) -> dict[str, Any]:
    """Win/loss statistics from closed (SELL) round-trips."""
    realized = [t["realized"] for t in state["trades"] if "realized" in t]
    wins = [r for r in realized if r > 0]
    losses = [r for r in realized if r <= 0]
    n = len(realized)
    return {
        "n_closed": n,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": (100.0 * len(wins) / n) if n else None,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "best": max(realized) if realized else 0.0,
        "worst": min(realized) if realized else 0.0,
        "total_realized": sum(realized) if realized else 0.0,
    }


def max_drawdown(state: dict[str, Any]) -> float:
    """Largest peak-to-trough drop of the equity curve, as a negative percent."""
    curve = [p["equity"] for p in state.get("equity_curve", [])]
    if len(curve) < 2:
        return 0.0
    peak = curve[0]
    mdd = 0.0
    for equity in curve:
        peak = max(peak, equity)
        if peak > 0:
            mdd = min(mdd, (equity - peak) / peak)
    return 100.0 * mdd


# ── Persistence (survives refresh / logout / reopen) ───────────────────────


def save_portfolio(state: dict[str, Any], path: Path | str) -> None:
    """Persist the portfolio to JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh)
    tmp.replace(path)


def load_portfolio(
    path: Path | str, *, capital: float = STARTING_CAPITAL
) -> dict[str, Any]:
    """Load a persisted portfolio, or a fresh one if none/unreadable."""
    path = Path(path)
    if not path.is_file():
        return new_portfolio(capital)
    try:
        with path.open(encoding="utf-8") as fh:
            state = json.load(fh)
        # Ensure all expected keys exist (forward-compatible with older files).
        base = new_portfolio(capital)
        base.update({k: state[k] for k in base if k in state})
        return base
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("could not load paper portfolio from %s: %s", path, exc)
        return new_portfolio(capital)


def reset_portfolio(path: Path | str, *, capital: float = STARTING_CAPITAL) -> dict[str, Any]:
    """Start a fresh run and persist it (used by the dashboard's Reset button)."""
    state = new_portfolio(capital)
    save_portfolio(state, path)
    return state


def poll_paper_trader(
    binance_symbol: str,
    binance_interval: str,
    state: dict[str, Any],
    *,
    n_candles: int = 200,
    fee_rate: float = FEE_RATE,
) -> dict[str, Any]:
    """Fetch live candles + price, run the ensemble, and step the portfolio once.

    Thin I/O wrapper (kept out of the pure functions so those stay testable).
    """
    from src.dashboard.signals import fetch_live_candles, fetch_live_price

    candles = fetch_live_candles(binance_symbol, binance_interval, n_candles=n_candles)
    price = float(fetch_live_price(binance_symbol))
    if not state.get("first_price"):
        state["first_price"] = price  # anchor the buy-and-hold benchmark
    sig = ensemble_signal(candles)
    candle_open = pd.Timestamp(candles["open_time"].iloc[-1]).isoformat()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    state = apply_decision(
        state, price, sig["decision"], now=now, candle_open=candle_open, fee_rate=fee_rate
    )
    state["last_signal"] = sig
    state["last_price"] = price
    return state
