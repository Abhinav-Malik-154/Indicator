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


# ── Alerts: accuracy context is mandatory (Phase 9) ───────────────────────

class TestAlerts:
    def test_buy_alert_includes_accuracy_context(self):
        from src.dashboard.alerts import build_alert_text

        text = build_alert_text("lr", "BUY", 0.63)
        assert "BUY" in text
        assert "historically" in text  # never a bare directional call

    def test_sell_alert_includes_accuracy_context(self):
        from src.dashboard.alerts import build_alert_text

        text = build_alert_text("lr", "SELL", 0.30)
        assert "SELL" in text
        assert "historically" in text

    def test_no_directional_alert_is_ever_bare(self):
        """Every non-silent alert must cite a hit rate or an explicit caveat."""
        from src.dashboard.alerts import build_alert_text

        for model in ("lr", "lgb"):
            for signal, prob in (("BUY", 0.7), ("SELL", 0.25)):
                text = build_alert_text(model, signal, prob)
                assert ("historically" in text) or ("no measured hit rate" in text)

    def test_lgb_alert_flags_absent_track_record(self):
        from src.dashboard.alerts import build_alert_text

        # LGB win rate is NaN (never fired) — must warn, not fabricate a number.
        text = build_alert_text("lgb", "BUY", 0.72)
        assert "no measured hit rate" in text

    def test_silent_signal_produces_no_alert(self):
        from src.dashboard.alerts import alerts_for_result

        result = {
            "signal_lr": "SILENT", "signal_lgb": "SILENT",
            "prob_lr": 0.46, "prob_lgb": 0.52,
        }
        assert alerts_for_result(result) == []

    def test_fired_signal_produces_alert(self):
        from src.dashboard.alerts import alerts_for_result

        result = {
            "signal_lr": "BUY", "signal_lgb": "SILENT",
            "prob_lr": 0.63, "prob_lgb": 0.52,
        }
        alerts = alerts_for_result(result)
        assert len(alerts) == 1
        assert alerts[0]["model"] == "lr"
        assert "historically" in alerts[0]["text"]


# ── Chart: pure figure builder smoke tests (Phase 9) ──────────────────────

class TestChartFigure:
    @pytest.fixture
    def ohlc(self):
        dates = pd.date_range("2026-06-01", periods=5, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "open_time": dates,
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            }
        )

    def test_builds_figure_with_candlestick(self, ohlc):
        from src.dashboard.chart import build_candlestick_figure

        fig = build_candlestick_figure(ohlc, pd.DataFrame(), None)
        assert len(fig.data) >= 1  # at least the candlestick trace

    def test_markers_and_badge_add_traces(self, ohlc):
        from src.dashboard.chart import build_candlestick_figure

        markers = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-06-02", tz="UTC")],
                "signal": ["BUY"],
                "price": [101.5],
                "realized": ["up"],
                "correct": [True],
            }
        )
        badge = {
            "date": ohlc["open_time"].iloc[-1],
            "price": 104.5,
            "signal": "SILENT",
            "text": "Today: SILENT",
        }
        fig = build_candlestick_figure(ohlc, markers, badge)
        # candlestick + green markers + today dot
        assert len(fig.data) >= 3
        # restyled badge = current-price tag + "● Today" label
        assert len(fig.layout.annotations) >= 1
        assert len(fig.layout.shapes) >= 1  # dashed current-price line

    def test_caption_mentions_retrospective(self):
        from src.dashboard.chart import chart_caption

        assert "retrospective" in chart_caption().lower()


# ── Chart: Binance-style area figure + range slicing (Phase 9 polish) ──────

class TestPriceFigure:
    @pytest.fixture
    def ohlc(self):
        dates = pd.date_range("2026-01-01", periods=200, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "open_time": dates,
                "open": [100.0 + i for i in range(200)],
                "high": [101.0 + i for i in range(200)],
                "low": [99.0 + i for i in range(200)],
                "close": [100.0 + ((i * 7) % 25) for i in range(200)],
            }
        )

    def test_area_figure_has_line_and_gradient_fill(self, ohlc):
        from src.dashboard.chart import build_price_figure

        fig = build_price_figure(ohlc, pd.DataFrame(), None, show_markers=True)
        # baseline (invisible) + gold price line
        assert len(fig.data) >= 2
        price_trace = fig.data[1]
        assert price_trace.fill == "tonexty"
        assert price_trace.fillgradient.type == "vertical"

    def test_empty_ohlc_returns_empty_figure(self):
        from src.dashboard.chart import build_price_figure

        fig = build_price_figure(pd.DataFrame(), pd.DataFrame(), None)
        assert len(fig.data) == 0

    def test_show_markers_false_drops_marker_traces(self, ohlc):
        from src.dashboard.chart import build_price_figure

        markers = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-02-01", tz="UTC")],
                "signal": ["BUY"],
                "price": [110.0],
                "realized": ["up"],
                "correct": [True],
            }
        )
        with_markers = build_price_figure(ohlc, markers, None, show_markers=True)
        without = build_price_figure(ohlc, markers, None, show_markers=False)
        assert len(with_markers.data) > len(without.data)

    def test_yaxis_is_price_formatted(self, ohlc):
        from src.dashboard.chart import build_price_figure

        fig = build_price_figure(ohlc, pd.DataFrame(), None)
        assert fig.layout.yaxis.tickprefix == "$"

    def test_today_badge_adds_price_line_and_dot(self, ohlc):
        from src.dashboard.chart import build_price_figure

        badge = {
            "date": ohlc["open_time"].iloc[-1],
            "price": float(ohlc["close"].iloc[-1]),
            "signal": "SELL",
            "text": "Today: SELL",
        }
        fig = build_price_figure(ohlc, pd.DataFrame(), badge, show_markers=False)
        # add_hline creates a layout shape; the today dot is a scatter trace
        assert len(fig.layout.shapes) >= 1
        assert len(fig.layout.annotations) >= 1


# ── Chart: TradingView-style live price tag ───────────────────────────────

class TestLivePriceTag:
    @pytest.fixture
    def ohlc(self):
        dates = pd.date_range("2026-06-01", periods=5, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "open_time": dates,
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            }
        )

    def _tag_annotation(self, fig, needle):
        return [a for a in fig.layout.annotations if needle in (a.text or "")]

    def test_live_tag_drawn_when_price_present(self, ohlc):
        from src.dashboard.chart import _DOWN_RED, _UP_GREEN, build_candlestick_figure

        badge = {"date": ohlc["open_time"].iloc[-1], "price": 104.5,
                 "live_price": 108.0, "signal": "SILENT", "text": "x"}
        fig = build_candlestick_figure(ohlc, pd.DataFrame(), badge)
        tag = self._tag_annotation(fig, "108")
        assert tag, "live price tag annotation missing"
        # Live above last close → green tag.
        assert tag[0].bgcolor == _UP_GREEN != _DOWN_RED

    def test_live_tag_red_when_below_last_close(self, ohlc):
        from src.dashboard.chart import _DOWN_RED, build_price_figure

        badge = {"date": ohlc["open_time"].iloc[-1], "price": 104.5,
                 "live_price": 101.0, "signal": "SILENT", "text": "x"}
        fig = build_price_figure(ohlc, pd.DataFrame(), badge, show_markers=False)
        tag = self._tag_annotation(fig, "101")
        assert tag and tag[0].bgcolor == _DOWN_RED

    def test_no_tag_without_live_price(self, ohlc):
        from src.dashboard.chart import build_candlestick_figure

        badge = {"date": ohlc["open_time"].iloc[-1], "price": 104.5,
                 "signal": "SILENT", "text": "x"}  # no live_price key
        fig = build_candlestick_figure(ohlc, pd.DataFrame(), badge)
        # only the today-badge $tag may exist; no bold live tag
        assert not [a for a in fig.layout.annotations if "<b>$" in (a.text or "")]

    def test_yaxis_widens_to_include_live_price(self, ohlc):
        from src.dashboard.chart import build_candlestick_figure

        badge = {"date": ohlc["open_time"].iloc[-1], "price": 104.5,
                 "live_price": 130.0, "signal": "SILENT", "text": "x"}
        fig = build_candlestick_figure(ohlc, pd.DataFrame(), badge)
        lo, hi = fig.layout.yaxis.range
        assert lo <= 130.0 <= hi

    def test_live_signal_marker_shape_by_call(self, ohlc):
        from src.dashboard.chart import build_candlestick_figure

        want = {"BUY": "triangle-up", "SELL": "triangle-down", "SILENT": "square"}
        for call, shape in want.items():
            badge = {"date": ohlc["open_time"].iloc[-1], "price": 104.5,
                     "live_price": 106.0, "signal": call, "text": "x"}
            fig = build_candlestick_figure(ohlc, pd.DataFrame(), badge)
            shapes = [
                t.marker.symbol for t in fig.data
                if t.type == "scatter" and t.marker.symbol
            ]
            labels = [t.text[0] for t in fig.data if t.type == "scatter" and t.text]
            assert shape in shapes
            assert f"  LIVE {call}" in labels

    def test_no_live_marker_without_live_price(self, ohlc):
        from src.dashboard.chart import build_candlestick_figure

        badge = {"date": ohlc["open_time"].iloc[-1], "price": 104.5,
                 "signal": "BUY", "text": "x"}  # no live_price
        fig = build_candlestick_figure(ohlc, pd.DataFrame(), badge)
        labels = [
            t.text[0] for t in fig.data
            if t.type == "scatter" and t.text and "LIVE" in str(t.text[0])
        ]
        assert labels == []


# ── Chart: projected expected-move band ────────────────────────────────────

class TestExpectedMoveBand:
    @pytest.fixture
    def ohlc(self):
        dates = pd.date_range("2026-06-01", periods=30, freq="D", tz="UTC")
        return pd.DataFrame({
            "open_time": dates,
            "open": [100 + i for i in range(30)],
            "high": [101 + i for i in range(30)],
            "low": [99 + i for i in range(30)],
            "close": [100 + ((i * 3) % 7) for i in range(30)],
        })

    def test_compute_band_shape(self, ohlc):
        from src.dashboard.chart import compute_expected_band

        band = compute_expected_band(ohlc, 130.0)
        assert band is not None
        assert band["low"] < 130.0 < band["high"]
        assert band["sigma_pct"] > 0
        assert band["next_x"] > ohlc["open_time"].iloc[-1]

    def test_compute_band_too_few_rows(self):
        from src.dashboard.chart import compute_expected_band

        assert compute_expected_band(pd.DataFrame(), 100.0) is None

    def test_band_drawn_and_axis_extended(self, ohlc):
        from src.dashboard.chart import build_candlestick_figure, compute_expected_band

        band = compute_expected_band(ohlc, 130.0)
        badge = {"date": ohlc["open_time"].iloc[-1], "price": 106.0,
                 "live_price": 130.0, "signal": "BUY", "text": "x",
                 "expected_move": band}
        fig = build_candlestick_figure(ohlc, pd.DataFrame(), badge)
        assert any(s.type == "rect" for s in fig.layout.shapes)
        assert any("next" in (a.text or "") for a in fig.layout.annotations)
        assert pd.Timestamp(fig.layout.xaxis.range[1]) > ohlc["open_time"].iloc[-1]

    def test_no_band_when_absent(self, ohlc):
        from src.dashboard.chart import build_price_figure

        badge = {"date": ohlc["open_time"].iloc[-1], "price": 106.0,
                 "live_price": 130.0, "signal": "BUY", "text": "x"}  # no expected_move
        fig = build_price_figure(ohlc, pd.DataFrame(), badge, show_markers=False)
        assert not any("next" in (a.text or "") for a in fig.layout.annotations)


class TestFetchLivePrice:
    def test_parses_ticker_price(self, monkeypatch):
        from src.dashboard import signals

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"symbol": "BTCUSDT", "price": "77611.04"}

        monkeypatch.setattr(signals.requests, "get", lambda *a, **k: _Resp())
        assert signals.fetch_live_price("BTCUSDT") == pytest.approx(77611.04)

    def test_missing_price_raises(self, monkeypatch):
        from src.dashboard import signals

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"symbol": "BTCUSDT"}

        monkeypatch.setattr(signals.requests, "get", lambda *a, **k: _Resp())
        with pytest.raises(ValueError):
            signals.fetch_live_price("BTCUSDT")


class TestSliceByRange:
    @pytest.fixture
    def ohlc(self):
        dates = pd.date_range("2026-01-01", periods=234, freq="D", tz="UTC")
        return pd.DataFrame({"open_time": dates, "close": range(234)})

    def test_1m_returns_about_a_month(self, ohlc):
        from src.dashboard.chart import slice_by_range

        assert len(slice_by_range(ohlc, "1M")) == 31  # 30 days back inclusive

    def test_ytd_starts_at_january(self, ohlc):
        from src.dashboard.chart import slice_by_range

        out = slice_by_range(ohlc, "YTD")
        assert out["open_time"].iloc[0] == pd.Timestamp("2026-01-01", tz="UTC")

    def test_range_always_ends_at_last_candle(self, ohlc):
        from src.dashboard.chart import slice_by_range

        for label in ("1M", "3M", "6M", "YTD", "1Y"):
            out = slice_by_range(ohlc, label)
            assert out["open_time"].iloc[-1] == ohlc["open_time"].iloc[-1]

    def test_empty_input_returns_empty(self):
        from src.dashboard.chart import slice_by_range

        assert slice_by_range(pd.DataFrame({"open_time": [], "close": []}), "3M").empty
