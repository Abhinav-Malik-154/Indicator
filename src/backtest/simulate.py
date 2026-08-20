"""Day-by-day strategy simulation with realistic trade costs (Phase 5).

Iron rules enforced here:

* A trade at T uses only the close price at T (already known) and the signal
  derived from features at T.  The return from T to T+1 is applied AFTER the
  signal is read — never before.
* Fee + slippage are charged on EVERY side of EVERY trade: once on entry and
  once on exit.  Forgetting to charge exit costs is a silent bias source.
* Zero signals fired ⇒ equity stays exactly at starting_notional (tested).

Run from the repo root::

    python -m src.backtest.report
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DIRECTION = {1: "long", -1: "short", 0: "cash"}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """All outputs of one backtest run.

    Equity curve is indexed by the same dates as the input price series.
    Positions series is indexed by the signal dates (one per day where a
    position was held or cash; same length as signals input).
    """

    equity_curve: pd.Series
    positions: pd.Series
    trades: list[dict[str, Any]]
    starting_notional: float
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    n_trades: int
    win_rate_pct: float
    avg_trade_pct: float
    n_days: int
    label: str = ""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    equity_curve: pd.Series,
    trades: list[dict[str, Any]],
    *,
    starting_notional: float,
) -> dict[str, Any]:
    """Derive summary statistics from the equity curve and trade log."""
    final_equity = float(equity_curve.iloc[-1])
    total_return_pct = (final_equity / starting_notional - 1.0) * 100.0

    n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if n_days > 0:
        n_years = n_days / 365.25
        cagr_pct = ((final_equity / starting_notional) ** (1.0 / n_years) - 1.0) * 100.0
    else:
        cagr_pct = float("nan")

    daily_rets = equity_curve.pct_change().dropna()
    if len(daily_rets) > 1 and daily_rets.std() > 1e-12:
        sharpe = float(daily_rets.mean() / daily_rets.std() * math.sqrt(252))
    else:
        sharpe = float("nan")

    running_max = equity_curve.cummax()
    drawdowns = (equity_curve / running_max - 1.0) * 100.0
    max_drawdown_pct = float(drawdowns.min())

    n_trades = len(trades)
    if n_trades > 0:
        wins = sum(1 for t in trades if t["win"])
        win_rate_pct = wins / n_trades * 100.0
        avg_trade_pct = sum(t["pnl_pct"] for t in trades) / n_trades
    else:
        win_rate_pct = float("nan")
        avg_trade_pct = float("nan")

    return {
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "n_trades": n_trades,
        "win_rate_pct": win_rate_pct,
        "avg_trade_pct": avg_trade_pct,
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------


def simulate_strategy(
    prices: pd.Series,
    signals: pd.Series,
    *,
    fee_rate: float,
    slippage_rate: float,
    starting_notional: float,
    label: str = "",
) -> BacktestResult:
    """Simulate a signal-driven strategy with per-side transaction costs.

    Contract:
    - ``prices`` has exactly ``len(signals) + 1`` entries.  ``prices[i]`` is
      the close at time T_i; ``prices[i+1]`` is the close at T_{i+1}.
    - ``signals[i]`` tells the model's view at the END of day T_i, which
      determines the position held during [T_i, T_{i+1}].  Only ``prices[i]``
      (and earlier) are used to produce ``signals[i]`` — no lookahead.
    - The daily return for that period is ``prices[i+1] / prices[i] - 1``.
    - A position change at T_i incurs ``(fee_rate + slippage_rate)`` of equity
      on EACH side (exit cost for the old position, entry cost for the new).
    - An open position at the end of the series is forcibly closed at the last
      price (costs applied).

    Args:
        prices: Close prices, DatetimeIndex, length N+1.
        signals: Integer signals {1=long, 0=cash, -1=short}, length N.
        fee_rate: Fraction of equity lost to exchange fees per trade side.
        slippage_rate: Fraction of equity lost to slippage per trade side.
        starting_notional: Starting equity in currency units.
        label: Human-readable name for this result (e.g. "LR pruned").

    Returns:
        :class:`BacktestResult` with equity curve, trade log, and metrics.

    Raises:
        ValueError: If ``len(prices) != len(signals) + 1``.
    """
    if len(prices) != len(signals) + 1:
        raise ValueError(
            f"prices must have len(signals)+1 entries; "
            f"got {len(prices)} prices and {len(signals)} signals"
        )
    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("fee_rate and slippage_rate must be non-negative")

    cost_per_side = fee_rate + slippage_rate
    equity = starting_notional
    position = 0  # 0=cash, 1=long, -1=short

    equity_vals: list[float] = [equity]
    equity_dates: list[Any] = [prices.index[0]]
    position_vals: list[int] = []

    trades: list[dict[str, Any]] = []
    trade_entry_equity: float = 0.0
    trade_entry_date: Any = None
    trade_direction: int = 0

    prices_arr = prices.to_numpy(dtype="float64")
    signals_arr = signals.to_numpy(dtype="int64")

    def _close_trade(exit_date: Any) -> None:
        nonlocal equity
        equity *= 1.0 - cost_per_side
        pnl_pct = (equity - trade_entry_equity) / trade_entry_equity * 100.0
        trades.append(
            {
                "entry_date": trade_entry_date,
                "exit_date": exit_date,
                "direction": _DIRECTION[trade_direction],
                "entry_equity": round(trade_entry_equity, 4),
                "exit_equity": round(equity, 4),
                "pnl_pct": round(pnl_pct, 4),
                "win": pnl_pct > 0.0,
            }
        )

    for i, sig in enumerate(signals_arr):
        sig = int(sig)
        current_date = prices.index[i]
        next_date = prices.index[i + 1]

        if sig != position:
            # Exit current position
            if position != 0:
                _close_trade(current_date)
            # Enter new position
            if sig != 0:
                equity *= 1.0 - cost_per_side
                trade_entry_equity = equity
                trade_entry_date = current_date
                trade_direction = sig
            position = sig

        position_vals.append(position)

        # Apply the daily return — no lookahead: only prices[i+1] used here,
        # and signals[i] was already committed before this line.
        daily_ret = prices_arr[i + 1] / prices_arr[i] - 1.0
        equity *= 1.0 + position * daily_ret

        equity_vals.append(equity)
        equity_dates.append(next_date)

    # Close any position still open at the end of the window
    if position != 0:
        _close_trade(prices.index[-1])
        equity_vals[-1] = equity  # update final curve point to post-exit equity

    equity_curve = pd.Series(equity_vals, index=equity_dates, name="equity")
    positions = pd.Series(
        [_DIRECTION[p] for p in position_vals],
        index=prices.index[:-1],
        name="position",
    )

    logger.info(
        "%s: %d trades, final equity %.2f (start %.2f)",
        label or "strategy", len(trades), equity_curve.iloc[-1], starting_notional,
    )

    metrics = _compute_metrics(equity_curve, trades, starting_notional=starting_notional)
    return BacktestResult(
        equity_curve=equity_curve,
        positions=positions,
        trades=trades,
        starting_notional=starting_notional,
        label=label,
        **metrics,
    )


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def signals_from_proba(
    prob_up: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Convert raw P(up) probabilities to integer signals {1, 0, -1}.

    Mirrors the confidence-gating logic in Phase 4's ``evaluate.py``:
    long if P(up) > threshold, short if P(up) < 1-threshold, else cash.

    Args:
        prob_up: Per-row predicted P(up) from a classifier.
        threshold: Confidence threshold in [0.5, 1).

    Returns:
        Integer array of 1 (long), 0 (cash), -1 (short), same length.
    """
    signals = np.zeros(len(prob_up), dtype="int64")
    signals[prob_up > threshold] = 1
    signals[prob_up < 1.0 - threshold] = -1
    return signals
