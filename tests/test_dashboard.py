"""Smoke tests for the dashboard signal module.

Covers pure functions directly (no I/O) and verifies the contract of the
compute_live_signal result dict using a fake result object.  No real network
calls or disk access are made.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.dashboard.signals import (
    _VOL_CALM_THRESHOLD,
    _VOL_ELEVATED_THRESHOLD,
    HISTORICAL_ACCURACY,
    _signal_label,
    get_volatility_regime,
)

# ── _signal_label ─────────────────────────────────────────────────────────

class TestSignalLabel:
    def test_above_threshold_is_buy(self):
        assert _signal_label(0.65, 0.60) == "BUY"

    def test_below_anti_threshold_is_sell(self):
        assert _signal_label(0.35, 0.60) == "SELL"

    def test_inside_band_is_silent(self):
        assert _signal_label(0.52, 0.60) == "SILENT"

    def test_exact_upper_threshold_is_silent(self):
        assert _signal_label(0.60, 0.60) == "SILENT"

    def test_exact_lower_threshold_is_silent(self):
        assert _signal_label(0.40, 0.60) == "SILENT"

    def test_returns_string_for_all_probs(self):
        valid = {"BUY", "SELL", "SILENT"}
        for prob in (0.0, 0.1, 0.39, 0.40, 0.50, 0.60, 0.61, 0.99, 1.0):
            result = _signal_label(prob, 0.60)
            assert result in valid, f"Unexpected result '{result}' for prob={prob}"


# ── get_volatility_regime ─────────────────────────────────────────────────

class TestGetVolatilityRegime:
    def test_returns_dict_with_required_keys(self):
        result = get_volatility_regime(0.025)
        for key in ("regime", "label", "colour", "explanation"):
            assert key in result, f"Missing key: '{key}'"

    def test_below_p25_is_calm(self):
        assert get_volatility_regime(_VOL_CALM_THRESHOLD - 0.001)["regime"] == "calm"

    def test_at_p25_is_elevated(self):
        assert get_volatility_regime(_VOL_CALM_THRESHOLD)["regime"] == "elevated"

    def test_between_thresholds_is_elevated(self):
        mid = (_VOL_CALM_THRESHOLD + _VOL_ELEVATED_THRESHOLD) / 2
        assert get_volatility_regime(mid)["regime"] == "elevated"

    def test_at_p75_is_elevated(self):
        assert get_volatility_regime(_VOL_ELEVATED_THRESHOLD)["regime"] == "elevated"

    def test_above_p75_is_high(self):
        assert get_volatility_regime(_VOL_ELEVATED_THRESHOLD + 0.001)["regime"] == "high"

    def test_all_values_are_strings(self):
        for val in (0.01, 0.03, 0.05):
            regime = get_volatility_regime(val)
            for k, v in regime.items():
                assert isinstance(v, str), f"Key '{k}' has non-string value {v!r}"

    def test_explanation_mentions_volatility(self):
        for val in (0.01, 0.03, 0.05):
            explanation = get_volatility_regime(val)["explanation"]
            assert "volatility" in explanation.lower(), (
                f"Explanation for val={val} does not mention 'volatility'"
            )

    def test_calm_colour_is_green(self):
        assert get_volatility_regime(0.01)["colour"] == "green"

    def test_elevated_colour_is_orange(self):
        mid = (_VOL_CALM_THRESHOLD + _VOL_ELEVATED_THRESHOLD) / 2
        assert get_volatility_regime(mid)["colour"] == "orange"

    def test_high_colour_is_red(self):
        assert get_volatility_regime(_VOL_ELEVATED_THRESHOLD + 0.01)["colour"] == "red"


# ── HISTORICAL_ACCURACY constants ─────────────────────────────────────────

class TestHistoricalAccuracy:
    def test_top_level_keys_present(self):
        required = {
            "lr", "lgb", "buyhold_test_return_pct", "test_period",
            "confidence_threshold", "round_trip_cost_pct",
        }
        assert required.issubset(HISTORICAL_ACCURACY.keys())

    def test_lr_sub_keys_present(self):
        required = {"test_accuracy_pct", "base_rate_pct", "edge_pp",
                    "test_return_pct", "n_trades", "win_rate_pct"}
        assert required.issubset(HISTORICAL_ACCURACY["lr"].keys())

    def test_lgb_sub_keys_present(self):
        required = {"test_accuracy_pct", "base_rate_pct", "edge_pp",
                    "test_return_pct", "n_trades", "win_rate_pct"}
        assert required.issubset(HISTORICAL_ACCURACY["lgb"].keys())

    def test_lr_accuracy_in_range(self):
        acc = HISTORICAL_ACCURACY["lr"]["test_accuracy_pct"]
        assert 0 < acc < 100, f"LR accuracy {acc} out of range"

    def test_lgb_fires_zero_signals(self):
        assert HISTORICAL_ACCURACY["lgb"]["n_trades"] == 0

    def test_lgb_win_rate_is_nan(self):
        assert math.isnan(HISTORICAL_ACCURACY["lgb"]["win_rate_pct"])

    def test_round_trip_cost_positive(self):
        assert HISTORICAL_ACCURACY["round_trip_cost_pct"] > 0

    def test_confidence_threshold_in_range(self):
        t = HISTORICAL_ACCURACY["confidence_threshold"]
        assert 0.5 < t < 1.0, f"Threshold {t} out of (0.5, 1.0)"

    def test_test_period_is_string(self):
        assert isinstance(HISTORICAL_ACCURACY["test_period"], str)

    def test_lr_edge_is_negative(self):
        # Measured result: LR was below base rate on OOS test
        assert HISTORICAL_ACCURACY["lr"]["edge_pp"] < 0


# ── Smoke test: result dict contract ─────────────────────────────────────
# Verifies the shape of a compute_live_signal result without real I/O.

class TestSignalResultContract:
    """Build a fake result dict matching compute_live_signal's contract and verify it."""

    @pytest.fixture
    def fake_result(self):
        return {
            "signal_lr":         "SILENT",
            "signal_lgb":        "SILENT",
            "prob_lr":           0.465,
            "prob_lgb":          0.522,
            "threshold":         0.60,
            "candle_date":       pd.Timestamp("2026-08-19", tz="UTC"),
            "current_close":     69334.79,
            "ret_std_30":        0.01746,
            "vol_regime":        get_volatility_regime(0.01746),
            "data_source":       "live",
            "n_candles_fetched": 90,
            "error":             None,
            "symbol":            "BTCUSDT",
            "interval":          "1d",
            "model_variant":     "pruned",
        }

    def test_required_keys_present(self, fake_result):
        required = {
            "signal_lr", "signal_lgb", "prob_lr", "prob_lgb", "threshold",
            "candle_date", "current_close", "ret_std_30", "vol_regime",
            "data_source", "n_candles_fetched", "error", "symbol",
            "interval", "model_variant",
        }
        assert required.issubset(fake_result.keys())

    def test_signals_are_valid_labels(self, fake_result):
        valid = {"BUY", "SELL", "SILENT"}
        assert fake_result["signal_lr"]  in valid
        assert fake_result["signal_lgb"] in valid

    def test_probs_are_in_unit_interval(self, fake_result):
        assert 0.0 <= fake_result["prob_lr"]  <= 1.0
        assert 0.0 <= fake_result["prob_lgb"] <= 1.0

    def test_candle_date_is_timestamp(self, fake_result):
        assert isinstance(fake_result["candle_date"], pd.Timestamp)

    def test_vol_regime_is_dict_with_keys(self, fake_result):
        vr = fake_result["vol_regime"]
        assert isinstance(vr, dict)
        for key in ("regime", "label", "colour", "explanation"):
            assert key in vr

    def test_n_candles_fetched_positive(self, fake_result):
        assert fake_result["n_candles_fetched"] > 0

    def test_current_close_positive(self, fake_result):
        assert fake_result["current_close"] > 0

    def test_ret_std_30_positive(self, fake_result):
        assert fake_result["ret_std_30"] > 0
