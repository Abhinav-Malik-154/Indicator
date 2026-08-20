"""Tests for Phase 7 multi-regime backtesting.

Coverage:
- Regime window definitions are correct and complete.
- In-sample / out-of-sample flags are set mechanically, not manually.
- No regime window overlaps the val/test gap (2024-08-17).
- OOS window is genuinely outside the training period.
- score_regime produces valid BacktestResult objects on synthetic data.
- Edge case: window with fewer than 5 usable rows raises ValueError.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.baseline import simulate_buyhold
from src.backtest.multi_regime import (
    _TRAIN_PERIOD_END,
    _TRAIN_PERIOD_START,
    _VAL_PERIOD_END,
    REGIMES,
    RegimeSummary,
    RegimeWindow,
    format_regime_block,
    format_summary_table,
    score_regime,
)
from src.backtest.simulate import BacktestResult, simulate_strategy

# ── Constants from the training split ─────────────────────────────────────────

_TRAIN_START = pd.Timestamp("2020-01-01", tz="UTC")
_TRAIN_END   = pd.Timestamp("2024-08-16", tz="UTC")
_VAL_END     = pd.Timestamp("2025-08-14", tz="UTC")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_merged(n: int = 40, start: str = "2020-01-15") -> pd.DataFrame:
    """Synthetic merged feature+label table."""
    dates = pd.date_range(start, periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "open_time": dates,
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
            "label_up_5d": rng.integers(0, 2, size=n).astype(float),
        }
    )


def _make_raw_close(n: int = 60, start: str = "2020-01-15") -> pd.Series:
    """Synthetic close price Series."""
    dates = pd.date_range(start, periods=n, freq="D", tz="UTC")
    prices = 10_000.0 * np.cumprod(
        1.0 + np.random.default_rng(7).uniform(-0.05, 0.05, n)
    )
    return pd.Series(prices, index=dates, name="close", dtype="float64")


def _make_artifacts(feature_cols: list[str]) -> dict:
    """Minimal artifact dict with mocked sklearn models (all predict P=0.6 up)."""
    scaler = MagicMock()
    scaler.transform = lambda X: X.values

    logreg = MagicMock()
    logreg.predict_proba = lambda X: np.column_stack(
        [np.full(len(X), 0.4), np.full(len(X), 0.6)]
    )

    lgb_model = MagicMock()
    lgb_model.predict_proba = lambda X: np.column_stack(
        [np.full(len(X), 0.4), np.full(len(X), 0.6)]
    )

    return {
        "scaler": scaler,
        "logistic_regression": logreg,
        "lightgbm": lgb_model,
        "manifest": {"feature_cols": feature_cols},
    }


def _make_cfg(threshold: float = 0.60) -> dict:
    return {"modeling": {"confidence_threshold": threshold}}


def _make_bt_cfg() -> dict:
    return {"fee_rate": 0.001, "slippage_rate": 0.001, "starting_notional": 10_000.0}


def _make_regime_summary(regime: RegimeWindow) -> RegimeSummary:
    """Build a RegimeSummary with trivial (all-cash) backtest results."""
    n = 20
    dates = pd.date_range("2020-02-01", periods=n + 1, freq="D", tz="UTC")
    prices = pd.Series(np.linspace(10_000, 11_000, n + 1), index=dates, name="close")
    signals = pd.Series(np.zeros(n, dtype="int64"), index=dates[:-1], name="signal")
    kw = {"fee_rate": 0.001, "slippage_rate": 0.001, "starting_notional": 10_000.0}
    lr  = simulate_strategy(prices, signals, **kw)
    lgb = simulate_strategy(prices, signals, **kw)
    bah = simulate_buyhold(prices, **kw)
    return RegimeSummary(
        regime=regime, lr=lr, lgb=lgb, buyhold=bah,
        n_window_rows=n, n_usable_rows=n, n_nan_dropped=0,
    )


# ── Regime definition tests ───────────────────────────────────────────────────

class TestRegimeDefinitions:
    def test_exactly_five_regimes(self):
        assert len(REGIMES) == 5

    def test_all_fields_non_empty(self):
        for r in REGIMES:
            assert r.name,      f"{r.name}: name empty"
            assert r.label,     f"{r.name}: label empty"
            assert r.start,     f"{r.name}: start empty"
            assert r.end,       f"{r.name}: end empty"
            assert r.rationale, f"{r.name}: rationale empty"

    def test_start_before_end(self):
        for r in REGIMES:
            start = pd.Timestamp(r.start)
            end   = pd.Timestamp(r.end)
            assert start < end, f"{r.name}: start >= end"

    def test_four_in_sample_one_oos(self):
        is_count  = sum(1 for r in REGIMES if r.in_sample)
        oos_count = sum(1 for r in REGIMES if not r.in_sample)
        assert is_count  == 4, f"expected 4 IS regimes, got {is_count}"
        assert oos_count == 1, f"expected 1 OOS regime, got {oos_count}"

    def test_in_sample_flag_consistent_with_training_period(self):
        """Any window overlapping [train_start, train_end] must be IS."""
        for r in REGIMES:
            start = pd.Timestamp(r.start, tz="UTC")
            end   = pd.Timestamp(r.end,   tz="UTC")
            overlaps_train = start <= _TRAIN_PERIOD_END and end >= _TRAIN_PERIOD_START
            assert r.in_sample == overlaps_train, (
                f"{r.name}: in_sample={r.in_sample} but overlaps_train={overlaps_train}"
            )

    def test_oos_window_starts_after_val_period(self):
        oos = next(r for r in REGIMES if not r.in_sample)
        oos_start = pd.Timestamp(oos.start, tz="UTC")
        assert oos_start > _VAL_PERIOD_END, (
            f"OOS window starts {oos_start} but val ends {_VAL_PERIOD_END}"
        )

    def test_oos_window_matches_phase5_test_split(self):
        oos = next(r for r in REGIMES if not r.in_sample)
        assert oos.start == "2025-08-16", f"OOS start mismatch: {oos.start}"
        assert oos.end   == "2026-08-11", f"OOS end mismatch: {oos.end}"

    def test_no_oos_window_straddles_train_val_gap(self):
        """No OOS window should span the gap between training and val."""
        gap = pd.Timestamp("2024-08-17", tz="UTC")
        for r in REGIMES:
            if r.in_sample:
                continue
            start = pd.Timestamp(r.start, tz="UTC")
            end   = pd.Timestamp(r.end,   tz="UTC")
            assert not (start <= gap <= end), (
                f"{r.name}: OOS window straddles the train/val gap"
            )

    def test_unique_names(self):
        names = [r.name for r in REGIMES]
        assert len(names) == len(set(names)), "duplicate regime names"

    def test_regime_window_is_frozen(self):
        r = REGIMES[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.name = "hacked"  # type: ignore[misc]

    def test_covid_crash_is_in_sample(self):
        covid = next(r for r in REGIMES if r.name == "covid_crash")
        assert covid.in_sample

    def test_phase5_test_is_oos(self):
        phase5 = next(r for r in REGIMES if r.name == "phase5_test")
        assert not phase5.in_sample

    def test_regime_names_known(self):
        names = {r.name for r in REGIMES}
        expected = {
            "covid_crash", "bull_2020_2021", "bear_2022",
            "recovery_2023", "phase5_test",
        }
        assert names == expected


# ── score_regime unit tests ───────────────────────────────────────────────────

class TestScoreRegime:
    @pytest.fixture
    def base_kwargs(self):
        feature_cols = ["feat_a", "feat_b"]
        regime = RegimeWindow(
            name="test_regime",
            label="Test regime",
            start="2020-02-01",
            end="2020-02-28",
            rationale="unit test",
            in_sample=True,
        )
        return dict(
            regime=regime,
            merged=_make_merged(n=40, start="2020-01-15"),
            feature_cols=feature_cols,
            artifacts=_make_artifacts(feature_cols),
            raw_close=_make_raw_close(n=60, start="2020-01-15"),
            cfg=_make_cfg(),
            bt_cfg=_make_bt_cfg(),
        )

    def test_returns_regime_summary(self, base_kwargs):
        s = score_regime(**base_kwargs)
        assert isinstance(s, RegimeSummary)

    def test_lr_lgb_buyhold_are_backtest_results(self, base_kwargs):
        s = score_regime(**base_kwargs)
        assert isinstance(s.lr,      BacktestResult)
        assert isinstance(s.lgb,     BacktestResult)
        assert isinstance(s.buyhold, BacktestResult)

    def test_n_usable_rows_positive(self, base_kwargs):
        s = score_regime(**base_kwargs)
        assert s.n_usable_rows > 0

    def test_n_window_rows_ge_n_usable_rows(self, base_kwargs):
        s = score_regime(**base_kwargs)
        assert s.n_window_rows >= s.n_usable_rows

    def test_nan_drop_count_consistent(self, base_kwargs):
        s = score_regime(**base_kwargs)
        assert s.n_nan_dropped == s.n_window_rows - s.n_usable_rows

    def test_buyhold_has_exactly_one_trade(self, base_kwargs):
        s = score_regime(**base_kwargs)
        assert s.buyhold.n_trades == 1

    def test_in_sample_flag_propagated(self, base_kwargs):
        s = score_regime(**base_kwargs)
        assert s.regime.in_sample == base_kwargs["regime"].in_sample

    def test_equity_start_equals_starting_notional(self, base_kwargs):
        s = score_regime(**base_kwargs)
        assert s.lr.equity_curve.iloc[0] == pytest.approx(
            base_kwargs["bt_cfg"]["starting_notional"], rel=1e-6
        )

    def test_too_few_rows_raises_value_error(self, base_kwargs):
        tiny_regime = RegimeWindow(
            name="tiny",
            label="Tiny window",
            start="2020-03-10",
            end="2020-03-11",
            rationale="edge case",
            in_sample=True,
        )
        with pytest.raises(ValueError, match="usable rows"):
            score_regime(
                regime=tiny_regime,
                merged=base_kwargs["merged"],
                feature_cols=base_kwargs["feature_cols"],
                artifacts=base_kwargs["artifacts"],
                raw_close=base_kwargs["raw_close"],
                cfg=base_kwargs["cfg"],
                bt_cfg=base_kwargs["bt_cfg"],
            )

    def test_nan_features_are_dropped_not_propagated(self, base_kwargs):
        """NaN rows should be dropped; window should still compute cleanly."""
        merged = base_kwargs["merged"].copy()
        mask = merged["open_time"] >= pd.Timestamp("2020-02-01", tz="UTC")
        first_five = merged[mask].index[:5]
        merged.loc[first_five, "feat_a"] = float("nan")
        base_kwargs["merged"] = merged

        s = score_regime(**base_kwargs)
        assert s.n_nan_dropped >= 5

    def test_zero_probability_yields_cash_signals(self, base_kwargs):
        """All P=0.4 < 0.6 threshold → cash signals → 0 trades for LR/LGB."""
        artifacts = base_kwargs["artifacts"]
        artifacts["logistic_regression"].predict_proba = lambda X: np.column_stack(
            [np.full(len(X), 0.6), np.full(len(X), 0.4)]
        )
        artifacts["lightgbm"].predict_proba = lambda X: np.column_stack(
            [np.full(len(X), 0.6), np.full(len(X), 0.4)]
        )
        s = score_regime(**base_kwargs)
        assert s.lr.n_trades  == 0
        assert s.lgb.n_trades == 0


# ── Formatting tests ──────────────────────────────────────────────────────────

class TestFormatting:
    def test_format_block_in_sample_contains_warning(self):
        regime = REGIMES[0]  # covid_crash, IS
        s = _make_regime_summary(regime)
        block = format_regime_block(s)
        assert "IN-SAMPLE" in block
        assert "not evidence of skill" in block.lower() or "NOT evidence" in block

    def test_format_block_oos_contains_meaningful_label(self):
        oos_regime = next(r for r in REGIMES if not r.in_sample)
        s = _make_regime_summary(oos_regime)
        block = format_regime_block(s)
        assert "OUT-OF-SAMPLE" in block
        assert "only meaningful" in block

    def test_format_block_has_total_return_row(self):
        s = _make_regime_summary(REGIMES[0])
        block = format_regime_block(s)
        assert "Total return" in block

    def test_format_block_has_sharpe_row(self):
        s = _make_regime_summary(REGIMES[0])
        block = format_regime_block(s)
        assert "Sharpe" in block

    def test_format_summary_table_contains_is_and_oos(self):
        summaries = [_make_regime_summary(r) for r in REGIMES]
        table = format_summary_table(summaries)
        assert " IS " in table or "IS\n" in table or table.count("IS") >= 4
        assert "OOS" in table

    def test_format_summary_table_contains_all_regime_labels(self):
        summaries = [_make_regime_summary(r) for r in REGIMES]
        table = format_summary_table(summaries)
        for r in REGIMES:
            assert r.label in table, f"'{r.label}' missing from summary table"

    def test_format_summary_table_warns_about_is(self):
        summaries = [_make_regime_summary(r) for r in REGIMES]
        table = format_summary_table(summaries)
        assert "NOT evidence of skill" in table
