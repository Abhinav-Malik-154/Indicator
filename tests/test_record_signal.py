"""Tests for the live signal log and its leakage-immune forward test (Phase 9).

Coverage:
- Idempotency: recording twice for the same candle date appends nothing the
  second time — running the daily job twice can never duplicate a row.
- Format stability: appends never rewrite, reorder, or corrupt existing rows
  (verified byte-for-byte), and the column layout is fixed.
- Forward test: the "accumulating" gate below 20 days, and a correct accuracy
  computation from a hand-built fixture above 20 days, including SILENT
  exclusion, dead-zone exclusion, and "outcome not known yet" exclusion.

No network calls or real model artifacts are used — everything runs on
synthetic result dicts and hand-built price lookups.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard.live_track_record import (
    MIN_DAYS_FOR_ACCURACY,
    accumulating_message,
    build_close_lookup,
    evaluate_forward_test,
)
from src.monitor.record_signal import (
    LOG_COLUMNS,
    append_signal,
    read_signal_log,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_result(
    date: str,
    *,
    close: float = 100.0,
    prob_lr: float = 0.52,
    sig_lr: str = "SILENT",
    prob_lgb: float = 0.52,
    sig_lgb: str = "SILENT",
    threshold: float = 0.60,
) -> dict:
    """Build a minimal compute_live_signal-shaped result dict."""
    return {
        "candle_date": pd.Timestamp(date, tz="UTC"),
        "current_close": close,
        "threshold": threshold,
        "prob_lr": prob_lr,
        "signal_lr": sig_lr,
        "prob_lgb": prob_lgb,
        "signal_lgb": sig_lgb,
    }


def _make_log(rows: list[dict]) -> pd.DataFrame:
    """Build a signal-log DataFrame with the canonical columns."""
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def _now(ts: str = "2026-08-20T00:05:00Z") -> pd.Timestamp:
    return pd.Timestamp(ts)


# ── Idempotency ────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_first_append_writes_two_rows(self, tmp_path):
        log = tmp_path / "live_signals.csv"
        out = append_signal(_make_result("2026-08-19"), log_path=log, now=_now())
        assert out["appended"] is True
        assert out["rows_written"] == 2  # one per model
        assert len(read_signal_log(log)) == 2

    def test_second_append_same_date_is_noop(self, tmp_path):
        log = tmp_path / "live_signals.csv"
        res = _make_result("2026-08-19")
        append_signal(res, log_path=log, now=_now("2026-08-20T00:05:00Z"))
        out2 = append_signal(res, log_path=log, now=_now("2026-08-20T23:00:00Z"))
        assert out2["appended"] is False
        assert out2["reason"] == "already_logged"
        assert out2["rows_written"] == 0
        # Still exactly two rows — no duplicate.
        assert len(read_signal_log(log)) == 2

    def test_running_three_times_same_day_stays_two_rows(self, tmp_path):
        log = tmp_path / "live_signals.csv"
        res = _make_result("2026-08-19")
        for _ in range(3):
            append_signal(res, log_path=log, now=_now())
        assert len(read_signal_log(log)) == 2

    def test_different_dates_accumulate(self, tmp_path):
        log = tmp_path / "live_signals.csv"
        append_signal(_make_result("2026-08-19"), log_path=log, now=_now())
        append_signal(_make_result("2026-08-20"), log_path=log, now=_now())
        df = read_signal_log(log)
        assert len(df) == 4
        assert set(df["candle_date"]) == {"2026-08-19", "2026-08-20"}

    def test_idempotency_keyed_on_candle_not_wallclock(self, tmp_path):
        """Two runs on the same candle at different wall-clock times = one entry."""
        log = tmp_path / "live_signals.csv"
        res = _make_result("2026-08-19")
        append_signal(res, log_path=log, now=_now("2026-08-20T00:01:00Z"))
        append_signal(res, log_path=log, now=_now("2026-08-20T18:30:00Z"))
        assert read_signal_log(log)["candle_date"].nunique() == 1


# ── Log format stability ───────────────────────────────────────────────────


class TestLogFormatStability:
    def test_columns_are_exactly_the_schema_in_order(self, tmp_path):
        log = tmp_path / "live_signals.csv"
        append_signal(_make_result("2026-08-19"), log_path=log, now=_now())
        assert list(read_signal_log(log).columns) == LOG_COLUMNS

    def test_header_written_once(self, tmp_path):
        log = tmp_path / "live_signals.csv"
        append_signal(_make_result("2026-08-19"), log_path=log, now=_now())
        append_signal(_make_result("2026-08-20"), log_path=log, now=_now())
        text = log.read_text(encoding="utf-8")
        assert text.count("candle_date,model,prob_up") == 1

    def test_append_does_not_rewrite_or_reorder_existing_rows(self, tmp_path):
        """The file after a later append must start with the earlier file, byte-for-byte."""
        log = tmp_path / "live_signals.csv"
        append_signal(_make_result("2026-08-19"), log_path=log, now=_now())
        first_bytes = log.read_bytes()

        append_signal(_make_result("2026-08-20"), log_path=log, now=_now())
        second_bytes = log.read_bytes()

        # Append-only: the original content is an exact prefix of the new file.
        assert second_bytes.startswith(first_bytes)
        assert len(second_bytes) > len(first_bytes)

    def test_candle_date_read_back_as_string(self, tmp_path):
        log = tmp_path / "live_signals.csv"
        append_signal(_make_result("2026-08-19"), log_path=log, now=_now())
        df = read_signal_log(log)
        assert df["candle_date"].iloc[0] == "2026-08-19"

    def test_read_missing_file_returns_empty_typed_frame(self, tmp_path):
        df = read_signal_log(tmp_path / "does_not_exist.csv")
        assert df.empty
        assert list(df.columns) == LOG_COLUMNS

    def test_recorded_values_roundtrip(self, tmp_path):
        log = tmp_path / "live_signals.csv"
        append_signal(
            _make_result("2026-08-19", close=69000.5, prob_lr=0.63, sig_lr="BUY"),
            log_path=log, now=_now(),
        )
        df = read_signal_log(log)
        lr = df[df["model"] == "lr"].iloc[0]
        assert lr["signal"] == "BUY"
        assert float(lr["prob_up"]) == pytest.approx(0.63)
        assert float(lr["close"]) == pytest.approx(69000.5)


# ── Forward test: accumulating gate ────────────────────────────────────────


class TestForwardTestAccumulating:
    def test_accumulating_message_format(self):
        assert accumulating_message(7, 20) == "Accumulating — 7/20 days recorded"

    def test_empty_log_reports_zero_days(self):
        result = evaluate_forward_test(
            _make_log([]), {}, horizon=1, dead_zone_pct=0.15
        )
        assert result["days_recorded"] == 0
        assert result["enough_data"] is False

    def test_under_min_days_not_enough(self):
        rows = []
        for i in range(10):  # 10 distinct dates < 20
            d = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)).date().isoformat()
            rows.append(
                {"candle_date": d, "model": "lr", "prob_up": 0.7, "signal": "BUY",
                 "threshold": 0.6, "close": 100.0, "recorded_at_utc": "x"}
            )
        result = evaluate_forward_test(_make_log(rows), {}, horizon=1, dead_zone_pct=0.15)
        assert result["days_recorded"] == 10
        assert result["enough_data"] is False

    def test_exactly_min_days_is_enough(self):
        rows = []
        for i in range(MIN_DAYS_FOR_ACCURACY):
            d = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)).date().isoformat()
            rows.append(
                {"candle_date": d, "model": "lr", "prob_up": 0.7, "signal": "SILENT",
                 "threshold": 0.6, "close": 100.0, "recorded_at_utc": "x"}
            )
        result = evaluate_forward_test(_make_log(rows), {}, horizon=1, dead_zone_pct=0.15)
        assert result["days_recorded"] == MIN_DAYS_FOR_ACCURACY
        assert result["enough_data"] is True


# ── Forward test: accuracy computation ─────────────────────────────────────


class TestForwardTestAccuracy:
    @pytest.fixture
    def alternating_fixture(self):
        """25 days of LR BUY signals; price moves +1% / -1% on alternating days.

        BUY predicts 'up'; the move from day i to i+1 is up when i is even.
        So BUY is correct on even i (0,2,...,24) = 13 of 25 = 52.0%.
        LGB is SILENT throughout (never scored).
        """
        dates = pd.date_range("2026-01-01", periods=26, freq="D")
        closes: dict[str, float] = {}
        price = 100.0
        for i, d in enumerate(dates):
            closes[d.date().isoformat()] = price
            price *= 1.01 if i % 2 == 0 else 0.99
        rows = []
        for i in range(25):
            d = dates[i].date().isoformat()
            rows.append({"candle_date": d, "model": "lr", "prob_up": 0.7,
                         "signal": "BUY", "threshold": 0.6, "close": closes[d],
                         "recorded_at_utc": "x"})
            rows.append({"candle_date": d, "model": "lgb", "prob_up": 0.5,
                         "signal": "SILENT", "threshold": 0.6, "close": closes[d],
                         "recorded_at_utc": "x"})
        return _make_log(rows), closes

    def test_accuracy_matches_hand_count(self, alternating_fixture):
        log, closes = alternating_fixture
        result = evaluate_forward_test(log, closes, horizon=1, dead_zone_pct=0.15)
        lr = result["models"]["lr"]
        assert lr["n_fired"] == 25
        assert lr["n_evaluable"] == 25
        assert lr["n_correct"] == 13
        assert lr["accuracy_pct"] == pytest.approx(52.0)

    def test_silent_model_has_no_accuracy(self, alternating_fixture):
        log, closes = alternating_fixture
        result = evaluate_forward_test(log, closes, horizon=1, dead_zone_pct=0.15)
        lgb = result["models"]["lgb"]
        assert lgb["n_fired"] == 0
        assert lgb["accuracy_pct"] is None

    def test_dead_zone_moves_are_not_scored(self):
        """A realized move inside ±dead_zone must not count as right or wrong."""
        # Two BUY signals: one with a clear +1% move, one with a +0.05% move
        # (inside the 0.15% dead zone → excluded).
        closes = {
            "2026-01-01": 100.0, "2026-01-02": 101.0,   # +1.0% clear up
            "2026-01-03": 100.0, "2026-01-04": 100.05,  # +0.05% dead zone
        }
        rows = [
            {"candle_date": "2026-01-01", "model": "lr", "prob_up": 0.7,
             "signal": "BUY", "threshold": 0.6, "close": 100.0, "recorded_at_utc": "x"},
            {"candle_date": "2026-01-03", "model": "lr", "prob_up": 0.7,
             "signal": "BUY", "threshold": 0.6, "close": 100.0, "recorded_at_utc": "x"},
        ]
        result = evaluate_forward_test(_make_log(rows), closes, horizon=1, dead_zone_pct=0.15)
        lr = result["models"]["lr"]
        assert lr["n_fired"] == 2
        assert lr["n_evaluable"] == 1  # the dead-zone one is excluded
        assert lr["n_correct"] == 1
        assert lr["accuracy_pct"] == pytest.approx(100.0)

    def test_signal_without_known_outcome_is_not_scored(self):
        """A signal whose outcome candle is absent from the lookup is skipped."""
        closes = {"2026-01-01": 100.0}  # no 2026-01-02 → outcome unknown
        rows = [
            {"candle_date": "2026-01-01", "model": "lr", "prob_up": 0.7,
             "signal": "BUY", "threshold": 0.6, "close": 100.0, "recorded_at_utc": "x"},
        ]
        result = evaluate_forward_test(_make_log(rows), closes, horizon=1, dead_zone_pct=0.15)
        lr = result["models"]["lr"]
        assert lr["n_fired"] == 1
        assert lr["n_evaluable"] == 0
        assert lr["accuracy_pct"] is None

    def test_sell_correct_on_down_move(self):
        closes = {"2026-01-01": 100.0, "2026-01-02": 98.0}  # -2% down
        rows = [
            {"candle_date": "2026-01-01", "model": "lr", "prob_up": 0.3,
             "signal": "SELL", "threshold": 0.6, "close": 100.0, "recorded_at_utc": "x"},
        ]
        result = evaluate_forward_test(_make_log(rows), closes, horizon=1, dead_zone_pct=0.15)
        lr = result["models"]["lr"]
        assert lr["n_evaluable"] == 1
        assert lr["n_correct"] == 1

    def test_wrong_direction_counts_as_incorrect(self):
        closes = {"2026-01-01": 100.0, "2026-01-02": 102.0}  # up, but we said SELL
        rows = [
            {"candle_date": "2026-01-01", "model": "lr", "prob_up": 0.3,
             "signal": "SELL", "threshold": 0.6, "close": 100.0, "recorded_at_utc": "x"},
        ]
        result = evaluate_forward_test(_make_log(rows), closes, horizon=1, dead_zone_pct=0.15)
        lr = result["models"]["lr"]
        assert lr["n_evaluable"] == 1
        assert lr["n_correct"] == 0
        assert lr["accuracy_pct"] == pytest.approx(0.0)


# ── build_close_lookup ─────────────────────────────────────────────────────


class TestBuildCloseLookup:
    def test_raw_series_and_log_merge(self):
        raw = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        )
        log = _make_log([
            {"candle_date": "2026-01-03", "model": "lr", "prob_up": 0.5,
             "signal": "SILENT", "threshold": 0.6, "close": 102.0, "recorded_at_utc": "x"},
        ])
        lookup = build_close_lookup(raw_close=raw, log_df=log)
        assert lookup["2026-01-01"] == 100.0
        assert lookup["2026-01-03"] == 102.0  # came from the log

    def test_raw_takes_precedence_over_log_for_same_date(self):
        raw = pd.Series([100.0], index=pd.to_datetime(["2026-01-01"], utc=True))
        log = _make_log([
            {"candle_date": "2026-01-01", "model": "lr", "prob_up": 0.5,
             "signal": "SILENT", "threshold": 0.6, "close": 999.0, "recorded_at_utc": "x"},
        ])
        lookup = build_close_lookup(raw_close=raw, log_df=log)
        assert lookup["2026-01-01"] == 100.0  # raw wins; log only fills gaps
