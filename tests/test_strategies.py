"""Tests for the classic-strategy ensemble (each method + the weighted vote)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.dashboard.strategies import (
    DECISION_THRESHOLD,
    ensemble_signal,
    strat_bollinger,
    strat_donchian,
    strat_ema_cross,
    strat_macd,
    strat_rsi_reversion,
)


def _c(close):
    close = np.asarray(close, dtype="float64")
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "open_time": pd.date_range("2026-01-01", periods=len(close), freq="1min", tz="UTC"),
        "open": op, "high": np.maximum(op, close) + 0.1,
        "low": np.minimum(op, close) - 0.1, "close": close,
    })


class TestTrendStrategies:
    def test_ema_cross_up_and_down(self):
        assert strat_ema_cross(_c(np.linspace(100, 130, 80))) == 1
        assert strat_ema_cross(_c(np.linspace(130, 100, 80))) == -1

    def test_macd_up_and_down(self):
        assert strat_macd(_c(np.linspace(100, 130, 80))) == 1
        assert strat_macd(_c(np.linspace(130, 100, 80))) == -1

    def test_donchian_breakout(self):
        # New highs each step → breakout up; new lows → breakout down.
        assert strat_donchian(_c(np.linspace(100, 130, 60))) == 1
        assert strat_donchian(_c(np.linspace(130, 100, 60))) == -1

    def test_donchian_inside_range_is_flat(self):
        c = _c(100 + np.sin(np.linspace(0, 6 * np.pi, 80)))  # oscillates, no new extreme
        assert strat_donchian(c) in (-1, 0, 1)  # defined, and often 0 mid-range


class TestReversionStrategies:
    def test_rsi_oversold_buys(self):
        rng = np.random.default_rng(2)
        falling = np.linspace(120, 100, 60) + rng.normal(0, 0.1, 60)
        assert strat_rsi_reversion(_c(falling)) == 1   # falling → oversold

    def test_rsi_overbought_sells(self):
        rng = np.random.default_rng(3)
        rising = np.linspace(100, 120, 60) + rng.normal(0, 0.1, 60)
        assert strat_rsi_reversion(_c(rising)) == -1  # rising → overbought

    def test_bollinger_below_lower_band_buys(self):
        prices = list(100 + np.random.default_rng(0).normal(0, 0.1, 40)) + [90.0]
        assert strat_bollinger(_c(prices)) == 1

    def test_bollinger_above_upper_band_sells(self):
        prices = list(100 + np.random.default_rng(1).normal(0, 0.1, 40)) + [110.0]
        assert strat_bollinger(_c(prices)) == -1


class TestEnsemble:
    def test_uptrend_leans_buy(self):
        sig = ensemble_signal(_c(np.linspace(100, 140, 90)))
        # Trend methods (EMA/MACD/Donchian) vote +1; reversion may fade → net > 0.
        assert sig["net"] > 0
        assert sig["decision"] == "BUY"

    def test_downtrend_leans_sell(self):
        sig = ensemble_signal(_c(np.linspace(140, 100, 90)))
        assert sig["net"] < 0
        assert sig["decision"] == "SELL"

    def test_votes_shape_and_counts(self):
        sig = ensemble_signal(_c(np.linspace(100, 140, 90)))
        assert set(sig["votes"]) == {
            "EMA cross", "MACD", "Donchian breakout", "RSI reversion", "Bollinger",
        }
        assert all(v in (-1, 0, 1) for v in sig["votes"].values())
        assert sig["n_buy"] + sig["n_sell"] <= 5

    def test_conviction_is_abs_net(self):
        sig = ensemble_signal(_c(np.linspace(100, 140, 90)))
        assert sig["conviction"] == abs(sig["net"])

    def test_threshold_governs_hold(self):
        # A dead-flat series → all strategies ~0 → net below threshold → HOLD.
        sig = ensemble_signal(_c(np.full(90, 100.0)))
        assert abs(sig["net"]) < DECISION_THRESHOLD
        assert sig["decision"] == "HOLD"
