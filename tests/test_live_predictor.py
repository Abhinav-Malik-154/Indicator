"""Tests for the live self-scoring next-candle predictor (pure logic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dashboard.live_predictor import (
    candle_direction,
    next_candle_signal,
    predictions_table,
    update_predictions,
)


def _candles(n, drift=0.5, start="2026-08-22 10:00"):
    base = np.linspace(100, 100 + drift * n, n)
    return pd.DataFrame({
        "open_time": pd.date_range(start, periods=n, freq="1min", tz="UTC"),
        "open": base, "high": base + 0.5, "low": base - 0.5, "close": base + 0.2,
    })


def _append_candle(c, tgt_open, direction):
    """Append one target candle that closes 'up' (green) or 'down' (red)."""
    o = float(c["close"].iloc[-1])
    cl = o + 1.0 if direction == "up" else o - 1.0
    row = pd.DataFrame({
        "open_time": [tgt_open], "open": [o],
        "high": [max(o, cl) + 0.5], "low": [min(o, cl) - 0.5], "close": [cl],
    })
    return pd.concat([c, row], ignore_index=True)


class TestCandleDirection:
    def test_up_down_flat(self):
        assert candle_direction(100, 101) == "up"
        assert candle_direction(101, 100) == "down"
        assert candle_direction(100, 100) == "flat"


class TestUpdatePredictions:
    def test_adds_one_prediction_for_forming_candle(self):
        c = _candles(60)
        now = pd.Timestamp("2026-08-22 11:00", tz="UTC")
        preds = update_predictions({}, c, live_price=130.0, now=now)
        assert len(preds) == 1
        (p,) = preds.values()
        assert p["predicted"] in {"UP", "DOWN", "NEUTRAL"}
        assert p["result"] is None  # forming candle not closed yet

    def test_dedupes_same_forming_candle(self):
        c = _candles(60)
        now = pd.Timestamp("2026-08-22 11:00", tz="UTC")
        preds = update_predictions({}, c, live_price=130.0, now=now)
        preds = update_predictions(preds, c, live_price=130.1, now=now)
        assert len(preds) == 1  # same target candle → not duplicated

    def test_scores_correct_when_outcome_matches_call(self):
        c = _candles(60, drift=0.5)
        now = pd.Timestamp("2026-08-22 11:00", tz="UTC")
        preds = update_predictions({}, c, live_price=130.0, now=now)
        (key,) = preds
        call = preds[key]["predicted"]
        if call == "NEUTRAL":  # neutral is never scored
            return
        match = "up" if call == "UP" else "down"
        c2 = _append_candle(c, preds[key]["target_open"], match)
        preds = update_predictions(preds, c2, live_price=float(c2["close"].iloc[-1]),
                                   now=now + pd.Timedelta(minutes=1))
        assert preds[key]["result"] == "✅"

    def test_wrong_when_outcome_opposes_call(self):
        c = _candles(60, drift=0.5)
        now = pd.Timestamp("2026-08-22 11:00", tz="UTC")
        preds = update_predictions({}, c, live_price=130.0, now=now)
        (key,) = preds
        call = preds[key]["predicted"]
        if call == "NEUTRAL":
            return
        opposite = "down" if call == "UP" else "up"
        c2 = _append_candle(c, preds[key]["target_open"], opposite)
        preds = update_predictions(preds, c2, live_price=float(c2["close"].iloc[-1]),
                                   now=now + pd.Timedelta(minutes=1))
        assert preds[key]["result"] == "❌"

    def test_signal_is_two_sided(self):
        # An oscillating series must produce BOTH up and down calls — the whole
        # point of the fix (the old trend rating only ever said UP).
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(0, 1.0, 300))
        c = pd.DataFrame({
            "open_time": pd.date_range("2026-08-22 10:00", periods=300, freq="1min", tz="UTC"),
            "open": prices, "high": prices + 0.5, "low": prices - 0.5,
            "close": np.r_[prices[1:], prices[-1]],
        })
        calls = {next_candle_signal(c.iloc[:i])["predicted"] for i in range(60, len(c))}
        assert "UP" in calls and "DOWN" in calls

    def test_too_few_candles_noop(self):
        assert update_predictions({}, _candles(10), live_price=100.0,
                                  now=pd.Timestamp.now(tz="UTC")) == {}

    def test_trims_to_max_keep(self):
        preds = {
            f"2026-08-22T{h:02d}:00:00+00:00": {
                "predicted_at": pd.Timestamp(f"2026-08-22 {h:02d}:00", tz="UTC"),
                "predicted": "UP", "score": 1.0, "price": 100.0,
                "target_open": pd.Timestamp(f"2026-08-22 {h:02d}:01", tz="UTC"),
                "actual": None, "result": "✅",
            }
            for h in range(10)
        }
        out = update_predictions(preds, _candles(60), live_price=130.0,
                                 now=pd.Timestamp("2026-08-22 11:00", tz="UTC"),
                                 max_keep=5)
        assert len(out) == 5


class TestPredictionsTable:
    def test_table_and_summary(self):
        c = _candles(60, drift=0.5)
        now = pd.Timestamp("2026-08-22 11:00", tz="UTC")
        preds = update_predictions({}, c, live_price=130.0, now=now)
        (key,) = preds
        call = preds[key]["predicted"]
        if call == "NEUTRAL":
            return
        match = "up" if call == "UP" else "down"
        c2 = _append_candle(c, preds[key]["target_open"], match)
        preds = update_predictions(preds, c2, live_price=float(c2["close"].iloc[-1]),
                                   now=now + pd.Timedelta(minutes=1))
        table, summ = predictions_table(preds)
        assert list(table.columns) == [
            "Predicted at", "Call", "Conf", "Price", "Target candle",
            "Actual", "Result",
        ]
        assert summ["n_scored"] == 1 and summ["n_correct"] == 1
        assert summ["hit_rate"] == 100.0
        assert summ["n_pending"] >= 1  # the new forming-candle call

    def test_empty(self):
        table, summ = predictions_table({})
        assert table.empty
        assert summ["hit_rate"] is None and summ["n_scored"] == 0
        assert summ["by_call"]["UP"]["hit_rate"] is None
        assert summ["by_call"]["DOWN"]["hit_rate"] is None


class TestDirectionBreakdown:
    @staticmethod
    def _mk(call, result, score, hour):
        return {
            "predicted_at": pd.Timestamp(f"2026-08-22 {hour:02d}:00", tz="UTC"),
            "predicted": call, "score": score, "price": 100.0,
            "target_open": pd.Timestamp(f"2026-08-22 {hour:02d}:01", tz="UTC"),
            "actual": None, "result": result,
        }

    def test_separate_buy_and_sell_hit_rates(self):
        preds = {
            "a": self._mk("UP", "✅", 1.0, 10),
            "b": self._mk("UP", "❌", 0.33, 11),
            "c": self._mk("DOWN", "✅", 0.67, 12),
            "d": self._mk("DOWN", "✅", 1.0, 13),
            "e": self._mk("DOWN", "❌", 0.33, 14),
            "f": self._mk("UP", None, 0.67, 15),  # pending
        }
        _, summ = predictions_table(preds)
        up, dn = summ["by_call"]["UP"], summ["by_call"]["DOWN"]
        assert up["n_calls"] == 2 and up["n_correct"] == 1 and up["hit_rate"] == 50.0
        assert up["n_pending"] == 1
        assert dn["n_calls"] == 3 and dn["n_correct"] == 2
        assert dn["hit_rate"] == pytest.approx(200 / 3)

    def test_confidence_column_labels(self):
        table, _ = predictions_table({
            "a": self._mk("UP", "✅", 1.0, 10),
            "b": self._mk("DOWN", "❌", 1 / 3, 11),
        })
        assert "Conf" in table.columns
        confs = set(table["Conf"])
        assert any("strong" in c for c in confs)
        assert any("weak" in c for c in confs)
