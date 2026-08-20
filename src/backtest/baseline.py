"""Buy-and-hold baseline for Phase 5 backtesting.

The baseline buys at the first close in the test window and holds to the last
close.  Costs: one entry fee + one exit fee (the same per-side cost as every
other trade in the strategy).  No fee is charged during the holding period
because there are no intermediate trades — documented here so the comparison
is transparent.

This module is a thin wrapper over :func:`~src.backtest.simulate.simulate_strategy`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtest.simulate import BacktestResult, simulate_strategy

logger = logging.getLogger(__name__)


def simulate_buyhold(
    prices: pd.Series,
    *,
    fee_rate: float,
    slippage_rate: float,
    starting_notional: float,
) -> BacktestResult:
    """Simulate buying BTC at prices[0] and holding until prices[-1].

    Cost structure: one entry fee + one exit fee (2 sides total), identical
    per-side rate to the strategy.  No intermediate fees because there are no
    intermediate trades.  This is conservative relative to the strategy, which
    pays fees on every round-trip signal.

    Args:
        prices: Close prices covering the backtest window (length >= 2).
        fee_rate: Fraction of equity lost to exchange fees per trade side.
        slippage_rate: Fraction of equity lost to slippage per trade side.
        starting_notional: Starting equity in currency units.

    Returns:
        :class:`BacktestResult` with a single long trade.
    """
    if len(prices) < 2:
        raise ValueError(f"prices must have at least 2 entries, got {len(prices)}")

    # All-long signal for every bar except the last price point.
    # simulate_strategy sees len(prices)-1 signals (all = 1), enters on day 0,
    # exits at the end — giving exactly 1 entry fee + 1 exit fee.
    n_signals = len(prices) - 1
    signals = pd.Series(
        np.ones(n_signals, dtype="int64"),
        index=prices.index[:-1],
        name="signal",
    )
    result = simulate_strategy(
        prices,
        signals,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        starting_notional=starting_notional,
        label="buy-and-hold",
    )
    logger.info(
        "buy-and-hold: entry %.2f -> exit %.2f, return %.2f%%",
        prices.iloc[0],
        prices.iloc[-1],
        result.total_return_pct,
    )
    return result
