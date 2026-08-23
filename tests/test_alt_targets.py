"""Tests for Task 3: reframed targets (volatility-direction, triple-barrier,
meta-labelling).

These exercise the pure label builders on small synthetic candle frames with
known structure, plus the leakage boundary (last-N rows unlabelled) and the
meta-label correctness grading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels.alt_targets import (
    compute_meta_labels,
    compute_triple_barrier_labels,
    compute_volatility_direction_labels,
    realized_volatility,
)


def _candles(closes, *, highs=None, lows=None):
    n = len(closes)
    closes = np.asarray(closes, dtype="float64")
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "high": closes if highs is None else np.asarray(highs, dtype="float64"),
        "low": closes if lows is None else np.asarray(lows, dtype="float64"),
        "close": closes,
    })


# ── Realized volatility ────────────────────────────────────────────────────


class TestRealizedVolatility:
    def test_warmup_is_nan(self):
        rv = realized_volatility(pd.Series(np.linspace(100, 110, 20)), window=5)
        assert rv.iloc[:5].isna().all()
        assert rv.iloc[5:].notna().all()

    def test_constant_returns_zero_vol(self):
        # Constant *growth rate* → constant log return → zero rolling std.
        close = pd.Series(100 * (1.01 ** np.arange(30)))
        rv = realized_volatility(close, window=5)
        assert rv.dropna().abs().max() < 1e-9


# ── Volatility-direction ───────────────────────────────────────────────────


class TestVolatilityDirection:
    def test_columns_and_binary(self):
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 100)))
        out = compute_volatility_direction_labels(_candles(close), vol_window=7)
        assert set(out.columns) == {
            "open_time", "current_vol", "forward_vol", "label_voldir",
        }
        vals = out["label_voldir"].dropna().unique()
        assert set(vals).issubset({0.0, 1.0})

    def test_last_window_unlabelled(self):
        rng = np.random.default_rng(1)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
        out = compute_volatility_direction_labels(_candles(close), vol_window=7)
        # Last vol_window rows have no forward window → NaN.
        assert out["label_voldir"].iloc[-7:].isna().all()

    def test_expansion_detected(self):
        # Calm then volatile: mid-series row should be labelled expansion (1).
        calm = np.full(20, 0.0)
        vol = np.array([0.05, -0.05] * 10)
        rets = np.concatenate([calm, vol])
        close = 100 * np.exp(np.cumsum(rets))
        out = compute_volatility_direction_labels(_candles(close), vol_window=5)
        # A row just before the regime shift looks forward into high vol → 1.
        assert out["label_voldir"].iloc[16] == 1.0


# ── Triple-barrier ─────────────────────────────────────────────────────────


class TestTripleBarrier:
    # A gently oscillating warm-up so trailing σ is non-zero (barriers exist).
    _WARMUP = np.array([100.0, 100.2] * 13)[:26]  # indices 0..25, ends at 100.0

    def test_upper_barrier_hit(self):
        # Calm oscillation, then a sharp jump up → upper barrier hit first.
        closes = np.concatenate([self._WARMUP, [100.0, 100.0, 130.0, 130.0]])
        highs = closes.copy()
        highs[28] = 130.0
        out = compute_triple_barrier_labels(
            _candles(closes, highs=highs, lows=closes),
            vol_window=20, upper_mult=1.0, lower_mult=1.0, max_horizon=5,
        )
        row = out.iloc[25]
        assert row["tb_label"] == 1.0
        assert row["tb_barrier"] == "up"

    def test_lower_barrier_hit(self):
        closes = np.concatenate([self._WARMUP, [100.0, 70.0, 70.0, 70.0]])
        lows = closes.copy()
        lows[27] = 70.0
        out = compute_triple_barrier_labels(
            _candles(closes, highs=closes, lows=lows),
            vol_window=20, upper_mult=1.0, lower_mult=1.0, max_horizon=5,
        )
        row = out.iloc[25]
        assert row["tb_label"] == 0.0
        assert row["tb_barrier"] == "down"

    def test_warmup_unlabelled(self):
        rng = np.random.default_rng(2)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
        out = compute_triple_barrier_labels(_candles(close), vol_window=20)
        # First vol_window rows have no σ → unlabelled.
        assert out["tb_label"].iloc[:20].isna().all()


# ── Meta-labelling ─────────────────────────────────────────────────────────


class TestMetaLabels:
    def test_correct_and_wrong_grading(self):
        # close goes up every step → forward return positive everywhere.
        close = np.array([100, 101, 102, 103, 104], dtype="float64")
        df = _candles(close)
        side = np.array([1, 1, -1, 1, 0])  # long,long,short,long,flat
        out = compute_meta_labels(df, side, horizon=1)
        # T0 long, price up → correct (1). T2 short, price up → wrong (0).
        assert out["meta_label"].iloc[0] == 1.0
        assert out["meta_label"].iloc[2] == 0.0
        # T4 flat side (0) → ungraded NaN; also last row has no forward return.
        assert np.isnan(out["meta_label"].iloc[4])

    def test_length_mismatch_raises(self):
        df = _candles([100, 101, 102])
        with pytest.raises(ValueError):
            compute_meta_labels(df, np.array([1, 1]), horizon=1)

    def test_dead_zone_ungraded(self):
        close = np.array([100, 100.05, 100.10], dtype="float64")  # ~0.05% moves
        df = _candles(close)
        out = compute_meta_labels(df, np.array([1, 1, 1]), horizon=1, dead_zone_pct=0.2)
        # Moves inside ±0.2% → ungraded.
        assert out["meta_label"].iloc[0:2].isna().all()
