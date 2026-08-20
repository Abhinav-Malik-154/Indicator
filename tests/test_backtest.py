"""Unit tests for the Phase 5 backtest engine.

Priority order (a silent bug here corrupts the reported P&L directly):

1. Fees are applied on EVERY side of EVERY trade — verified by comparing
   simulated equity with and without fees on the same fixture.
2. No future price leaks into position sizing at time T — the return from
   T → T+1 must be applied AFTER the signal at T is read.
3. Zero signals ⇒ equity stays exactly at starting_notional.
4. Hand-calculated equity curve matches the simulator on a 5-day fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.baseline import simulate_buyhold
from src.backtest.simulate import BacktestResult, signals_from_proba, simulate_strategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_DATE = pd.Timestamp("2024-01-01", tz="UTC")


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range(_BASE_DATE, periods=n, freq="D")


def _prices(values: list[float]) -> pd.Series:
    return pd.Series(values, index=_dates(len(values)), name="close", dtype="float64")


def _signals(values: list[int], n_prices: int) -> pd.Series:
    """signals has one fewer entry than prices."""
    assert len(values) == n_prices - 1
    return pd.Series(values, index=_dates(len(values)), dtype="int64", name="signal")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_wrong_length_raises(self):
        p = _prices([100.0, 110.0, 120.0])
        # make a signals series that is one entry too long
        s_bad = pd.Series([1, 1, 1], index=_dates(3), dtype="int64")
        with pytest.raises(ValueError, match="len\\(signals\\)\\+1"):
            simulate_strategy(p, s_bad, fee_rate=0.0, slippage_rate=0.0, starting_notional=10_000.0)

    def test_negative_fee_raises(self):
        p = _prices([100.0, 110.0])
        s = _signals([0], 2)
        with pytest.raises(ValueError, match="non-negative"):
            simulate_strategy(p, s, fee_rate=-0.001, slippage_rate=0.0, starting_notional=10_000.0)

    def test_zero_signals_equity_unchanged(self):
        """All-cash strategy: equity must equal starting_notional exactly."""
        prices = _prices([100.0, 110.0, 90.0, 130.0, 80.0])
        signals = _signals([0, 0, 0, 0], 5)
        result = simulate_strategy(
            prices, signals, fee_rate=0.01, slippage_rate=0.005, starting_notional=10_000.0
        )
        assert result.total_return_pct == pytest.approx(0.0, abs=1e-10)
        assert result.n_trades == 0
        assert np.isnan(result.win_rate_pct)
        assert all(v == pytest.approx(10_000.0, abs=1e-10) for v in result.equity_curve)

    def test_equity_curve_length_equals_prices(self):
        prices = _prices([100.0, 110.0, 105.0])
        signals = _signals([1, 0], 3)
        result = simulate_strategy(
            prices, signals, fee_rate=0.0, slippage_rate=0.0, starting_notional=10_000.0
        )
        assert len(result.equity_curve) == len(prices)

    def test_positions_length_equals_signals(self):
        prices = _prices([100.0, 110.0, 105.0])
        signals = _signals([1, 0], 3)
        result = simulate_strategy(
            prices, signals, fee_rate=0.0, slippage_rate=0.0, starting_notional=10_000.0
        )
        assert len(result.positions) == len(signals)


# ---------------------------------------------------------------------------
# No-fee return logic (validates position math before mixing in fee logic)
# ---------------------------------------------------------------------------


class TestReturnMathNoFees:
    def test_single_long_trade_correct_return(self):
        """Long 100→110: return must be exactly +10%."""
        prices = _prices([100.0, 110.0])
        signals = _signals([1], 2)
        result = simulate_strategy(
            prices, signals, fee_rate=0.0, slippage_rate=0.0, starting_notional=10_000.0
        )
        assert result.total_return_pct == pytest.approx(10.0, rel=1e-9)
        assert result.equity_curve.iloc[-1] == pytest.approx(11_000.0, rel=1e-9)

    def test_single_short_trade_correct_return(self):
        """Short 100→110: price rose, short loses 10%."""
        prices = _prices([100.0, 110.0])
        signals = _signals([-1], 2)
        result = simulate_strategy(
            prices, signals, fee_rate=0.0, slippage_rate=0.0, starting_notional=10_000.0
        )
        assert result.total_return_pct == pytest.approx(-10.0, rel=1e-9)

    def test_cash_day_contributes_zero_return(self):
        """Price moves +10% on a cash day must leave equity unchanged."""
        prices = _prices([100.0, 110.0, 110.0])
        signals = _signals([0, 1], 3)  # cash day 0, then long day 1 (100→110 applies)
        result = simulate_strategy(
            prices, signals, fee_rate=0.0, slippage_rate=0.0, starting_notional=10_000.0
        )
        # Day 0: cash — no gain from 100→110
        # Day 1: long — 110 stays 110 → 0% return
        assert result.equity_curve.iloc[1] == pytest.approx(10_000.0, rel=1e-9)
        assert result.equity_curve.iloc[2] == pytest.approx(10_000.0, rel=1e-9)

    def test_no_lookahead_signal_at_t_drives_return_t_to_t1(self):
        """signal[0] applies to return from prices[0] to prices[1], not [1] to [2]."""
        # If signal[0]=cash and signal[1]=long, and prices go up only in the
        # first step, a system with lookahead would show a gain; correct one shows 0.
        prices = _prices([100.0, 200.0, 200.0])  # big move only in first step
        signals = _signals([0, 1], 3)             # cash on day 0, long on day 1
        result = simulate_strategy(
            prices, signals, fee_rate=0.0, slippage_rate=0.0, starting_notional=10_000.0
        )
        # Day 0 position=cash: misses the 100%+ move
        # Day 1 position=long: 200→200 = 0%
        assert result.total_return_pct == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Fee correctness
# ---------------------------------------------------------------------------


class TestFeeApplication:
    def test_fees_reduce_equity_vs_zero_fee(self):
        """With fees, final equity must be strictly less than without fees."""
        prices = _prices([100.0, 110.0, 100.0, 115.0])
        signals = _signals([1, 0, 1], 4)  # two round-trips

        no_fee = simulate_strategy(
            prices, signals, fee_rate=0.0, slippage_rate=0.0, starting_notional=10_000.0
        )
        with_fee = simulate_strategy(
            prices, signals, fee_rate=0.01, slippage_rate=0.0, starting_notional=10_000.0
        )
        assert with_fee.equity_curve.iloc[-1] < no_fee.equity_curve.iloc[-1]

    def test_fee_charged_on_every_trade_side(self):
        """Hand-verify that entry AND exit fees are both deducted on one trade."""
        # One long trade: 100→110, fee=1% per side.
        # Entry: 10000 × (1-0.01) = 9900
        # Return: 9900 × (110/100) = 10890
        # Exit (forced at end): 10890 × (1-0.01) = 10781.10
        prices = _prices([100.0, 110.0])
        signals = _signals([1], 2)
        result = simulate_strategy(
            prices, signals, fee_rate=0.01, slippage_rate=0.0, starting_notional=10_000.0
        )
        expected_final = 10_000.0 * 0.99 * (110.0 / 100.0) * 0.99
        assert result.equity_curve.iloc[-1] == pytest.approx(expected_final, rel=1e-9)

    def test_two_round_trips_four_fee_deductions(self):
        """Two separate long trades incur 4 total fee deductions (2 entries + 2 exits)."""
        # Trade 1: 100→110, fee=2% per side
        #   enter: ×0.98, return: ×1.10, exit: ×0.98
        # Trade 2: 110→121, fee=2% per side
        #   enter: ×0.98, return: ×1.10, exit: ×0.98
        prices = _prices([100.0, 110.0, 121.0])
        # signal[0]=1 (long), signal[1]=1 (stay long) — wait, staying long
        # doesn't re-charge fees. Let me route through cash to force two trades.
        prices = _prices([100.0, 110.0, 110.0, 121.0])
        signals = _signals([1, 0, 1], 4)  # long, cash, long → 2 entry + 2 exit fees
        result = simulate_strategy(
            prices, signals, fee_rate=0.02, slippage_rate=0.0, starting_notional=10_000.0
        )
        c = 0.98  # 1 - fee per side
        expected = (
            10_000.0 * c  # enter trade 1
            * (110.0 / 100.0)  # return
            * c  # exit trade 1
            * 1.0  # cash day: 110→110 no return
            * c  # enter trade 2
            * (121.0 / 110.0)  # return
            * c  # exit trade 2
        )
        assert result.equity_curve.iloc[-1] == pytest.approx(expected, rel=1e-9)
        assert result.n_trades == 2

    def test_holding_position_does_not_recharge_fees(self):
        """Staying in the same position across consecutive days incurs no extra fee."""
        # All signals=1 (always long): only 1 entry + 1 exit
        prices = _prices([100.0, 110.0, 121.0, 133.1])
        signals = _signals([1, 1, 1], 4)
        result = simulate_strategy(
            prices, signals, fee_rate=0.01, slippage_rate=0.0, starting_notional=10_000.0
        )
        expected = 10_000.0 * 0.99 * (133.1 / 100.0) * 0.99
        assert result.equity_curve.iloc[-1] == pytest.approx(expected, rel=1e-9)
        assert result.n_trades == 1  # one round-trip


# ---------------------------------------------------------------------------
# Five-day hand-calculated fixture
# ---------------------------------------------------------------------------


class TestHandCalculatedFixture:
    """End-to-end check against a manually verified equity trace.

    Prices: [100, 110, 105, 115, 120]  (5 prices, 4 signal days)
    Signals: [1, 0, -1, 1]
    fee=1%, slippage=0%, notional=10 000

    Trace:
      day 0: signal=1, was cash → entry cost: 10000×0.99=9900
              return 110/100=+10%: 9900×1.10=10890
      day 1: signal=0, was long → exit: 10890×0.99=10781.10
              cash, 105/110=-4.545%: 10781.10×1.0=10781.10
      day 2: signal=-1, was cash → entry cost: 10781.10×0.99=10673.289
              short, 115/105=+9.524%: 10673.289×(1-0.09524)=9657.069
      day 3: signal=1, was short → exit: 9657.069×0.99=9560.498
              entry: 9560.498×0.99=9464.893
              long, 120/115=+4.348%: 9464.893×1.04348=9876.330
      end: forced exit: 9876.330×0.99=9777.567
    """

    prices: pd.Series
    signals: pd.Series
    result: BacktestResult

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.prices = _prices([100.0, 110.0, 105.0, 115.0, 120.0])
        self.signals = _signals([1, 0, -1, 1], 5)
        self.result = simulate_strategy(
            self.prices, self.signals,
            fee_rate=0.01, slippage_rate=0.0, starting_notional=10_000.0,
        )

    def _trace(self) -> list[float]:
        """Hand-computed equity after each price step."""
        c = 0.99
        e0 = 10_000.0
        e1 = e0 * c * (110 / 100)          # enter long, day 0 return
        e2 = e1 * c * 1.0                   # exit long, cash day
        e3 = e2 * c * (1 - (115 - 105) / 105)  # enter short, day 2 return
        e4 = e3 * c * c * (120 / 115)      # exit short + enter long, day 3 return
        e_final = e4 * c                    # forced exit
        return [10_000.0, e1, e2, e3, e_final]

    def test_final_equity_matches_hand_calc(self):
        trace = self._trace()
        assert self.result.equity_curve.iloc[-1] == pytest.approx(trace[-1], rel=1e-6)

    def test_three_trades_recorded(self):
        # trade 1: long (day0→day1), trade 2: short (day2→day3),
        # trade 3: long (day3→end)
        assert self.result.n_trades == 3

    def test_positions_correct(self):
        expected = ["long", "cash", "short", "long"]
        assert list(self.result.positions) == expected


# ---------------------------------------------------------------------------
# Buy-and-hold baseline
# ---------------------------------------------------------------------------


class TestBuyAndHold:
    def test_single_trade_in_trade_log(self):
        prices = _prices([100.0, 110.0, 121.0])
        result = simulate_buyhold(
            prices, fee_rate=0.01, slippage_rate=0.0, starting_notional=10_000.0
        )
        assert result.n_trades == 1

    def test_same_as_all_long_strategy(self):
        """simulate_buyhold must equal simulate_strategy with all-long signals."""
        prices = _prices([100.0, 105.0, 110.0, 108.0])
        signals = _signals([1, 1, 1], 4)
        strategy = simulate_strategy(
            prices, signals, fee_rate=0.005, slippage_rate=0.001, starting_notional=5_000.0
        )
        buyhold = simulate_buyhold(
            prices, fee_rate=0.005, slippage_rate=0.001, starting_notional=5_000.0
        )
        assert strategy.total_return_pct == pytest.approx(buyhold.total_return_pct, rel=1e-9)

    def test_single_entry_plus_exit_fee(self):
        """Buy-and-hold: entry+exit fee only, no intermediate fees."""
        # 100→120 with fee=1%: 10000×0.99×(120/100)×0.99
        prices = _prices([100.0, 120.0])
        result = simulate_buyhold(
            prices, fee_rate=0.01, slippage_rate=0.0, starting_notional=10_000.0
        )
        expected = 10_000.0 * 0.99 * (120.0 / 100.0) * 0.99
        assert result.equity_curve.iloc[-1] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# signals_from_proba
# ---------------------------------------------------------------------------


class TestSignalsFromProba:
    def test_above_threshold_is_long(self):
        proba = np.array([0.65, 0.70, 0.55])
        sigs = signals_from_proba(proba, threshold=0.60)
        assert list(sigs) == [1, 1, 0]

    def test_below_complement_is_short(self):
        proba = np.array([0.35, 0.30, 0.45])
        sigs = signals_from_proba(proba, threshold=0.60)
        assert list(sigs) == [-1, -1, 0]

    def test_in_band_is_cash(self):
        proba = np.array([0.50, 0.55, 0.45])
        sigs = signals_from_proba(proba, threshold=0.60)
        assert list(sigs) == [0, 0, 0]
