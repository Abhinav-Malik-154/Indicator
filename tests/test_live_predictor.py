"""Tests for the live self-scoring next-candle predictor (pure logic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dashboard.live_predictor import (
    MIN_SCORED_FOR_HIT_RATE,
    _strength,
    accuracy_over_time,
    candle_direction,
    load_predictions,
    next_candle_signal,
    predictions_path,
    predictions_table,
    save_predictions,
    update_predictions,
)


def _candles(n, drift=0.5, start="2026-08-22 10:00"):
    base = np.linspace(100, 100 + drift * n, n)
    return pd.DataFrame({
        "open_time": pd.date_range(start, periods=n, freq="1min", tz="UTC"),
        "open": base, "high": base + 0.5, "low": base - 0.5, "close": base + 0.2,
    })


def _series(close, start="2026-01-01 00:00"):
    """Candles driven by a close-price array; open = previous close (realistic)."""
    close = np.asarray(close, dtype="float64")
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "open_time": pd.date_range(start, periods=len(close), freq="1min", tz="UTC"),
        "open": op,
        "high": np.maximum(op, close) + 0.2,
        "low": np.minimum(op, close) - 0.2,
        "close": close,
    })


def _sine(n, *, period=20, amp=2.0, base=100.0):
    """A clean oscillation — the canonical ranging market."""
    return base + amp * np.sin(2 * np.pi * np.arange(n) / period)


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

    def test_follows_strong_uptrend(self):
        # A steady up-trend must be RIDDEN (UP), not faded — the whole point of
        # the regime split (old mean-reversion called DOWN into the trend).
        sig = next_candle_signal(_candles(80, drift=0.6))
        assert sig["predicted"] == "UP"
        assert sig["regime"] == "trend"
        assert sig["trend_strength"] > 0

    def test_follows_strong_downtrend(self):
        sig = next_candle_signal(_candles(80, drift=-0.6))
        assert sig["predicted"] == "DOWN"
        assert sig["regime"] == "trend"
        assert sig["trend_strength"] < 0

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
            "a": self._mk("UP", "✅", 1.0, 10),   # |score|=1.0 → firm
            "b": self._mk("DOWN", "❌", 0.2, 11),  # |score|=0.2 → faint
        })
        assert "Conf" in table.columns
        confs = set(table["Conf"])
        assert any("firm" in c for c in confs)
        assert any("faint" in c for c in confs)


class TestExpertChartBehaviour:
    """Canonical chart scenarios with a known 'right answer' a technical trader
    would give, encoded as regression guards for the signal's behaviour.

    Verified empirically against the implementation; each asserts the *behaviour*
    (follow strong trends, fade extremes, be humble at turning points, stay
    balanced in a range), never a claim of real predictive edge.
    """

    # ── Trend regime: follow a strong, sustained move ──────────────────────
    def test_strong_uptrend_is_followed(self):
        sig = next_candle_signal(_series(np.linspace(100, 130, 90)))
        assert sig["predicted"] == "UP"
        assert sig["regime"] == "trend"

    def test_strong_downtrend_is_followed(self):
        sig = next_candle_signal(_series(np.linspace(130, 100, 90)))
        assert sig["predicted"] == "DOWN"
        assert sig["regime"] == "trend"

    # ── Range regime: mean-revert the extremes ─────────────────────────────
    def test_overbought_peak_reverts_down(self):
        # A clean oscillation ending exactly at a peak → stretched above → DOWN.
        sig = next_candle_signal(_series(_sine(66)))
        assert sig["predicted"] == "DOWN"
        assert sig["mr_score"] < 0  # mean-reversion is voting down

    def test_oversold_trough_reverts_up(self):
        sig = next_candle_signal(_series(_sine(76)))
        assert sig["predicted"] == "UP"
        assert sig["mr_score"] > 0

    def test_oscillating_range_is_two_sided_and_balanced(self):
        c = _series(_sine(160))
        calls = [next_candle_signal(c.iloc[:i])["predicted"] for i in range(60, len(c))]
        n_up = calls.count("UP")
        n_dn = calls.count("DOWN")
        assert n_up > 0 and n_dn > 0                     # both directions fire
        assert min(n_up, n_dn) / max(n_up, n_dn) > 0.5   # roughly balanced, not one-sided

    # ── Confidence discipline: humble, and driven by agreement ─────────────
    def test_never_claims_strong(self):
        # No 1-minute direction call should ever read as high certainty.
        for s in np.linspace(-1, 1, 41):
            assert "strong" not in _strength(float(s))

    def test_conflict_lowers_conviction(self):
        # Strong trend BUT overbought (price stretched) = genuine conflict →
        # the call must NOT be the top "firm" tier.
        sig = next_candle_signal(_series(np.linspace(100, 130, 90)))
        assert "firm" not in _strength(sig["score"])

    def test_aligned_trend_and_pullback_earns_firm(self):
        # Strong up-trend with a small pullback (price back near its EMA, so
        # mean-reversion no longer opposes) is the one high-conviction setup.
        base = list(np.linspace(100, 120, 80)) + [119.4, 118.9, 118.6]
        sig = next_candle_signal(_series(base))
        assert sig["predicted"] == "UP"
        assert "firm" in _strength(sig["score"])

    def test_score_is_bounded(self):
        for close in (np.linspace(100, 200, 90), _sine(90), _sine(90, amp=8)):
            sig = next_candle_signal(_series(close))
            assert -1.0 <= sig["score"] <= 1.0


class TestAccuracyOverTime:
    @staticmethod
    def _mk(result, hour):
        return {
            "predicted_at": pd.Timestamp("2026-08-22", tz="UTC") + pd.Timedelta(hours=hour),
            "predicted": "UP", "score": 1.0, "price": 100.0,
            "target_open": pd.Timestamp("2026-08-22", tz="UTC") + pd.Timedelta(hours=hour),
            "actual": None, "result": result,
        }

    def test_cumulative_hit_rate(self):
        preds = {
            "a": self._mk("✅", 1), "b": self._mk("✅", 2),
            "c": self._mk("❌", 3), "d": self._mk("✅", 4),
        }
        df = accuracy_over_time(preds)
        assert list(df["n"]) == [1, 2, 3, 4]
        # running %: 100, 100, 66.7, 75
        assert df["hit_rate"].iloc[0] == 100.0
        assert df["hit_rate"].iloc[2] == pytest.approx(200 / 3)
        assert df["hit_rate"].iloc[-1] == 75.0
        assert df["time"].is_monotonic_increasing

    def test_ignores_pending_and_neutral(self):
        preds = {"a": self._mk("✅", 1), "b": self._mk(None, 2), "c": self._mk("—", 3)}
        assert len(accuracy_over_time(preds)) == 1

    def test_empty(self):
        assert accuracy_over_time({}).empty


class TestPersistence:
    """Predictions must survive a refresh/logout — i.e. round-trip through disk."""

    @staticmethod
    def _preds():
        c = _candles(60)
        preds = update_predictions(
            {}, c, live_price=130.0, now=pd.Timestamp("2026-08-22 11:00", tz="UTC")
        )
        # Add a scored one (distinct target candle) so we also cover
        # actual/result round-tripping. Keys are always target_open.isoformat().
        extra_open = pd.Timestamp("2026-08-22 10:58", tz="UTC")
        preds[extra_open.isoformat()] = {
            "predicted_at": pd.Timestamp("2026-08-22 10:57", tz="UTC"),
            "predicted": "DOWN", "score": -0.42, "price": 129.5,
            "target_open": extra_open,
            "actual": "down", "result": "✅",
        }
        return preds

    def test_round_trip_preserves_state(self, tmp_path):
        path = tmp_path / "live_preds.csv"
        preds = self._preds()
        save_predictions(preds, path)
        loaded = load_predictions(path)
        assert set(loaded) == set(preds)
        for k in preds:
            for field in ("predicted", "actual", "result"):
                assert loaded[k][field] == preds[k][field]
            assert loaded[k]["price"] == pytest.approx(preds[k]["price"])
            assert loaded[k]["predicted_at"] == preds[k]["predicted_at"]

    def test_loaded_keys_match_target_open_isoformat(self):
        # Keys must equal target_open.isoformat() so scoring still matches candles.
        preds = self._preds()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            save_predictions(preds, f.name)
            loaded = load_predictions(f.name)
        for k, p in loaded.items():
            assert k == pd.Timestamp(p["target_open"]).isoformat()

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_predictions(tmp_path / "nope.csv") == {}

    def test_path_is_per_symbol_interval(self):
        p1 = predictions_path("BTCUSDT", "1m")
        p2 = predictions_path("BTCUSDT", "5m")
        assert p1 != p2
        assert "BTCUSDT" in str(p1) and p1.suffix == ".csv"


class TestSampleSizeGuard:
    @staticmethod
    def _mk(call, result, hour):
        return {
            "predicted_at": pd.Timestamp("2026-08-22", tz="UTC") + pd.Timedelta(hours=hour),
            "predicted": call, "score": 1.0, "price": 100.0,
            "target_open": pd.Timestamp("2026-08-22", tz="UTC") + pd.Timedelta(hours=hour),
            "actual": None, "result": result,
        }

    def test_small_sample_not_reliable(self):
        preds = {str(i): self._mk("UP", "✅", i) for i in range(3)}  # 3/3 = 100%
        _, summ = predictions_table(preds)
        assert summ["hit_rate"] == 100.0        # value still computed…
        assert summ["reliable"] is False        # …but flagged not trustworthy
        assert summ["by_call"]["UP"]["reliable"] is False
        assert summ["min_scored"] == MIN_SCORED_FOR_HIT_RATE

    def test_big_sample_reliable(self):
        n = MIN_SCORED_FOR_HIT_RATE + 5
        preds = {
            str(i): self._mk("UP", "✅" if i % 2 else "❌", i) for i in range(n)
        }
        _, summ = predictions_table(preds)
        assert summ["reliable"] is True
        assert summ["by_call"]["UP"]["reliable"] is True
