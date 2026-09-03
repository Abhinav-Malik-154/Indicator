"""Tests for the risk-managed, volatility-gated trading engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dashboard.paper_trader import new_portfolio
from src.dashboard.risk_trader import (
    atr,
    compute_bracket,
    position_size,
    risk_path,
    risk_step,
    vol_expanding,
)


def _candles(close, *, high=None, low=None, start="2026-01-01"):
    close = np.asarray(close, dtype="float64")
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "open_time": pd.date_range(start, periods=len(close), freq="5min", tz="UTC"),
        "open": op,
        "high": close + 1.0 if high is None else np.asarray(high, dtype="float64"),
        "low": close - 1.0 if low is None else np.asarray(low, dtype="float64"),
        "close": close,
    })


class TestIndicators:
    def test_atr_positive_for_ranging_market(self):
        rng = np.random.default_rng(0)
        c = _candles(100 + np.cumsum(rng.normal(0, 1, 60)))
        assert atr(c) > 0

    def test_vol_expanding_true_when_recent_wilder(self):
        # Calm then wild → recent vol > baseline → expanding.
        calm = np.full(40, 0.0)
        wild = np.array([0.03, -0.03] * 10)
        close = 100 * np.exp(np.cumsum(np.r_[calm, wild]))
        assert vol_expanding(_candles(close)) is True

    def test_vol_expanding_false_when_recent_calm(self):
        wild = np.array([0.03, -0.03] * 20)
        calm = np.full(20, 0.0005)
        close = 100 * np.exp(np.cumsum(np.r_[wild, calm]))
        assert vol_expanding(_candles(close)) is False

    def test_vol_expanding_defaults_true_without_history(self):
        assert vol_expanding(_candles(np.linspace(100, 101, 10))) is True


class TestBracketAndSizing:
    def test_bracket_is_asymmetric_2to1(self):
        stop, target = compute_bracket(100.0, 2.0, stop_atr=1.5, rr=2.0)
        assert stop == pytest.approx(97.0)          # 100 - 1.5*2
        assert target == pytest.approx(106.0)       # 100 + 1.5*2*2
        # reward is exactly rr× the risk
        assert (target - 100) == pytest.approx(2.0 * (100 - stop))

    def test_position_size_risks_fixed_fraction(self):
        # risk 2% of ₹10,000 = ₹200; stop ₹5 away → 40 units (if cash allows).
        qty = position_size(10_000, 100.0, 95.0, cash=1_000_000, price=100.0,
                            risk_frac=0.02, fee_rate=0.0)
        assert qty == pytest.approx(40.0)

    def test_position_size_capped_by_cash(self):
        qty = position_size(10_000, 100.0, 95.0, cash=500.0, price=100.0,
                            risk_frac=0.02, fee_rate=0.0)
        assert qty == pytest.approx(5.0)  # can't spend more than ₹500

    def test_position_size_zero_on_bad_stop(self):
        assert position_size(10_000, 100.0, 100.0, 10_000, 100.0) == 0.0


class TestRiskStep:
    @staticmethod
    def _wild_candles():
        # Volatility-expanding series so the gate opens.
        calm = np.full(40, 0.0)
        wild = np.array([0.02, -0.02] * 10)
        close = 100 * np.exp(np.cumsum(np.r_[calm, wild]))
        return _candles(close)

    def test_buy_signal_enters_when_vol_expanding(self):
        s = new_portfolio()
        c = self._wild_candles()
        s = risk_step(s, c, 100.0, "BUY", now="t1", candle_open="c1")
        assert s["btc"] > 0
        assert s["stop_price"] is not None and s["target_price"] is not None
        assert s["trades"][-1]["side"] == "BUY"

    def test_no_entry_when_vol_contracting(self):
        wild = np.array([0.03, -0.03] * 20)
        calm = np.full(20, 0.0002)
        close = 100 * np.exp(np.cumsum(np.r_[wild, calm]))
        s = risk_step(new_portfolio(), _candles(close), 100.0, "BUY",
                      now="t1", candle_open="c1")
        assert s["btc"] == 0.0  # gate kept it flat

    def test_no_entry_on_hold_or_sell(self):
        c = self._wild_candles()
        s = risk_step(new_portfolio(), c, 100.0, "HOLD", now="t", candle_open="c1")
        assert s["btc"] == 0.0

    def test_stop_loss_exits(self):
        s = new_portfolio()
        c = self._wild_candles()
        s = risk_step(s, c, 100.0, "BUY", now="t1", candle_open="c1")
        stop = s["stop_price"]
        # Price drops below the stop on a later tick → forced exit.
        s = risk_step(s, c, stop - 1.0, "HOLD", now="t2", candle_open="c1")
        assert s["btc"] == 0.0
        assert s["trades"][-1]["reason"] == "stop"

    def test_take_profit_exits(self):
        s = new_portfolio()
        c = self._wild_candles()
        s = risk_step(s, c, 100.0, "BUY", now="t1", candle_open="c1")
        target = s["target_price"]
        s = risk_step(s, c, target + 1.0, "HOLD", now="t2", candle_open="c1")
        assert s["btc"] == 0.0
        assert s["trades"][-1]["reason"] == "target"
        assert s["trades"][-1]["realized"] > 0  # a winning trade

    def test_one_entry_per_candle(self):
        s = new_portfolio()
        c = self._wild_candles()
        s = risk_step(s, c, 100.0, "BUY", now="t1", candle_open="c1")
        # Exit, then a second BUY on the SAME candle must not re-enter.
        s = risk_step(s, c, s["stop_price"] - 1, "HOLD", now="t2", candle_open="c1")
        s = risk_step(s, c, 100.0, "BUY", now="t3", candle_open="c1")
        assert s["btc"] == 0.0  # same candle → no new entry

    def test_path_per_symbol_interval(self):
        assert risk_path("BTCUSDT", "5m") != risk_path("BTCUSDT", "15m")
        assert "risk_trade" in str(risk_path("BTCUSDT", "5m"))
