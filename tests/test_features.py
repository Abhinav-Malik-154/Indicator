"""Unit tests for the Phase 2 feature modules.

The most important tests in the repo are ``TestLeakage`` and
``TestDeliberateLeak``: features built on the full series must be identical to
features rebuilt using only data up to candle T, for every T.  If that ever
fails, something is looking ahead.  The deliberate-leak test proves the guard
really does catch future-data usage, not just silently pass clean code.
"""

from __future__ import annotations

import json
import logging
import statistics

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, candlestick, technical
from src.features.validate_features import (
    make_leaked_feature,
    validate_feature_label_alignment,
    validate_no_lookahead,
)
from src.labels.build_labels import compute_forward_return_labels

HOUR_MS = 3_600_000
BASE_MS = 1_609_459_200_000  # 2021-01-01T00:00:00Z

#: Small windows so hand-checked fixtures stay small. Longest lookback = 4 rows
#: (without RSI/MACD) or 7 rows (with MACD signal warm-up: 4+3).
SMALL_WINDOWS = {
    "return_periods": [1, 2],
    "volatility_windows": [3],
    "volume_window": 3,
    "ma_windows": [3],
    "sr_window": 3,
    "rsi_period": 3,
    "macd": {"fast_period": 2, "slow_period": 4, "signal_period": 3},
}


def make_ohlcv(n: int = 80, seed: int = 7, start_ms: int = BASE_MS) -> pd.DataFrame:
    """Deterministic random-walk OHLCV fixture with valid candle geometry."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.008, n)))
    volume = rng.lognormal(3.0, 0.5, n)
    open_time = pd.to_datetime(start_ms + HOUR_MS * np.arange(n), unit="ms", utc=True)
    return pd.DataFrame(
        {"open_time": open_time, "open": open_, "high": high, "low": low,
         "close": close, "volume": volume}
    )


def tiny_df(
    close: list[float],
    open_: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
    volume: list[float] | None = None,
) -> pd.DataFrame:
    """Hand-specified candle frame with contiguous hourly timestamps."""
    n = len(close)
    close_arr = np.asarray(close, dtype="float64")
    open_arr = np.asarray(open_ if open_ is not None else close, dtype="float64")
    high_arr = (
        np.asarray(high, dtype="float64")
        if high is not None
        else np.maximum(open_arr, close_arr) + 1.0
    )
    low_arr = (
        np.asarray(low, dtype="float64")
        if low is not None
        else np.minimum(open_arr, close_arr) - 1.0
    )
    volume_arr = np.asarray(volume if volume is not None else [10.0] * n, dtype="float64")
    open_time = pd.to_datetime(BASE_MS + HOUR_MS * np.arange(n), unit="ms", utc=True)
    return pd.DataFrame(
        {"open_time": open_time, "open": open_arr, "high": high_arr, "low": low_arr,
         "close": close_arr, "volume": volume_arr}
    )


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Technical + candlestick features with a series-independent column set."""
    tech = technical.compute_technical_features(df, **SMALL_WINDOWS)
    cdl, _ = candlestick.compute_candlestick_features(df, drop_never_fired=False)
    return pd.concat([tech, cdl], axis=1)


class TestLeakage:
    """THE leakage tests: no feature may use information from after its row."""

    def test_features_at_t_identical_when_future_is_removed(self):
        # Includes a synthetic 3-candle gap so gap handling is covered too.
        df = make_ohlcv(n=80, seed=11).drop(index=range(40, 43)).reset_index(drop=True)
        full = build_all_features(df)
        for t in range(5, len(df)):
            prefix = build_all_features(df.iloc[: t + 1])
            pd.testing.assert_frame_equal(full.iloc[[t]], prefix.iloc[[t]])

    def test_appending_future_data_does_not_change_history(self):
        df = make_ohlcv(n=100, seed=3)
        history_only = build_all_features(df.iloc[:60])
        with_future = build_all_features(df)
        pd.testing.assert_frame_equal(with_future.iloc[:60], history_only)


class TestDeliberateLeak:
    """Prove the guard actually catches lookahead, not just passes clean code.

    We deliberately shift a feature by -1 (each row gets the *next* row's
    value) and assert that ``validate_no_lookahead`` catches it.  If this
    test ever passes without raising, the guard is broken.
    """

    def test_leaked_feature_detected(self):
        df = make_ohlcv(n=30, seed=42)

        def build_with_leak(input_df: pd.DataFrame) -> pd.DataFrame:
            clean = technical.compute_technical_features(input_df, **SMALL_WINDOWS)
            leaked = make_leaked_feature(input_df["close"], "leaked_close")
            return pd.concat([clean, leaked], axis=1)

        with pytest.raises(AssertionError, match="LOOKAHEAD DETECTED"):
            validate_no_lookahead(
                df,
                build_with_leak,
                sample_points=list(range(5, len(df) - 1)),
                context="deliberate-leak test",
            )

    def test_clean_features_pass_no_lookahead(self):
        """Complementary: clean features must NOT trigger the guard."""
        df = make_ohlcv(n=30, seed=42)
        # Should not raise.
        validate_no_lookahead(
            df,
            build_all_features,
            sample_points=list(range(5, len(df))),
            context="clean features",
        )


class TestRSI:
    """Hand-checked RSI with Wilder's smoothing on a small fixture."""

    def test_rsi_hand_computed(self):
        """Verify RSI on a 6-row fixture with rsi_period=3.

        close = [100, 102, 101, 104, 103, 106]
        deltas = [NaN, +2, -1, +3, -1, +3]
        gains  = [NaN, 2, 0, 3, 0, 3]
        losses = [NaN, 0, 1, 0, 1, 0]

        alpha = 1/3; EWM with adjust=False:
          y[0] = x[0] for first non-NaN, then
          y[t] = alpha*x[t] + (1-alpha)*y[t-1].

          gain series:  [NaN, 2, 0, 3, 0, 3]
            avg_gain[1] = 2  (first value)
            avg_gain[2] = (1/3)*0 + (2/3)*2 = 4/3
            avg_gain[3] = (1/3)*3 + (2/3)*(4/3) = 17/9
            avg_gain[4] = (1/3)*0 + (2/3)*(17/9) = 34/27
            avg_gain[5] = (1/3)*3 + (2/3)*(34/27) = 149/81

          loss series:  [NaN, 0, 1, 0, 1, 0]
            avg_loss[1] = 0   -> RSI = 100 (pure gains)
            avg_loss[2] = 1/3
            avg_loss[3] = 2/9
            avg_loss[4] = 13/27
            avg_loss[5] = 26/81

        RS[2] = (4/3) / (1/3) = 4 -> RSI = 100 - 100/5 = 80
        RS[5] = (149/81) / (26/81) = 149/26
        RSI[5] = 100 - 2600/175 = 85.143...
        """
        df = tiny_df([100.0, 102.0, 101.0, 104.0, 103.0, 106.0])
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)

        # Row 0: NaN (no delta).
        assert np.isnan(out["rsi_3"].iloc[0])

        # Row 1: avg_gain=2, avg_loss=0 -> RSI=100 (pure gains edge case).
        assert out["rsi_3"].iloc[1] == pytest.approx(100.0)

        # Row 2: RS = (4/3) / (1/3) = 4, RSI = 100 - 100/5 = 80.
        assert out["rsi_3"].iloc[2] == pytest.approx(80.0)

        # Row 5: hand-computed above.
        expected_rsi_5 = 100.0 - 100.0 / (1.0 + 149.0 / 26.0)
        assert out["rsi_3"].iloc[5] == pytest.approx(expected_rsi_5)

    def test_rsi_all_gains_is_100(self):
        """Monotonically increasing prices -> RSI = 100 (after warm-up)."""
        df = tiny_df([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        # All losses are zero -> avg_loss = 0 -> RS = inf -> RSI = 100.
        for i in range(1, 6):
            assert out["rsi_3"].iloc[i] == pytest.approx(100.0)

    def test_rsi_all_losses_is_0(self):
        """Monotonically decreasing prices -> RSI ≈ 0 (after warm-up)."""
        df = tiny_df([105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        # All gains are zero -> avg_gain = 0 -> RS = 0 -> RSI = 0... but
        # avg_gain starts at 0 and stays 0, so 0/avg_loss = 0.
        # RSI = 100 - 100/(1+0) = 0.
        for i in range(1, 6):
            assert out["rsi_3"].iloc[i] == pytest.approx(0.0)


class TestMACD:
    """Hand-checked MACD with small fast=2, slow=4, signal=3."""

    def test_macd_hand_computed(self):
        """Verify MACD on a 6-row fixture.

        close = [100, 102, 101, 104, 103, 106]

        EMA fast (span=2, alpha=2/3, adjust=False):
          ema_f[0] = 100
          ema_f[1] = (2/3)*102 + (1/3)*100 = 204/3 + 100/3 = 304/3
          ema_f[2] = (2/3)*101 + (1/3)*(304/3) = 202/3 + 304/9 = 910/9
          ema_f[3] = (2/3)*104 + (1/3)*(910/9) = 208/3 + 910/27 = 2782/27
          ema_f[4] = (2/3)*103 + (1/3)*(2782/27) = 206/3 + 2782/81 = 6364/81
          (not computing ema_f[5] here, will verify row 3 instead)

        EMA slow (span=4, alpha=2/5, adjust=False):
          ema_s[0] = 100
          ema_s[1] = (2/5)*102 + (3/5)*100 = 204/5 + 300/5 = 504/5
          ema_s[2] = (2/5)*101 + (3/5)*(504/5) = 202/5 + 1512/25 = 2522/25
          ema_s[3] = (2/5)*104 + (3/5)*(2522/25) = 208/5 + 7566/125
                   = 5200/125 + 7566/125 = 12766/125

        macd_line[3] = ema_f[3] - ema_s[3]
                     = 2782/27 - 12766/125
                     = (2782*125 - 12766*27) / (27*125)
                     = (347750 - 344682) / 3375
                     = 3068/3375
        """
        df = tiny_df([100.0, 102.0, 101.0, 104.0, 103.0, 106.0])
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)

        # Row 0: both EMAs = close[0], so MACD line = 0.
        assert out["macd_line"].iloc[0] == pytest.approx(0.0)

        # Row 3: hand-computed.
        expected_macd_3 = 2782.0 / 27.0 - 12766.0 / 125.0
        assert out["macd_line"].iloc[3] == pytest.approx(expected_macd_3)

        # MACD histogram = line - signal.
        for i in range(len(df)):
            assert out["macd_hist"].iloc[i] == pytest.approx(
                out["macd_line"].iloc[i] - out["macd_signal"].iloc[i]
            )

    def test_constant_price_macd_is_zero(self):
        """With constant close, all EMAs equal close, so MACD = 0."""
        df = tiny_df([100.0] * 8)
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        assert (out["macd_line"] == 0.0).all()
        assert (out["macd_signal"] == 0.0).all()
        assert (out["macd_hist"] == 0.0).all()


class TestTechnicalHandChecked:
    def test_log_returns(self):
        df = tiny_df([100.0, 110.0, 121.0, 133.1, 146.41])
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        assert out["log_ret_1"].iloc[0] != out["log_ret_1"].iloc[0]  # NaN
        assert out["log_ret_1"].iloc[1:].tolist() == pytest.approx([np.log(1.1)] * 4)
        assert out["log_ret_2"].iloc[:2].isna().all()
        assert out["log_ret_2"].iloc[2:].tolist() == pytest.approx([2 * np.log(1.1)] * 3)

    def test_rolling_return_std(self):
        df = tiny_df([100.0, 110.0, 99.0, 108.9, 130.68])
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        rets = [np.log(1.1), np.log(0.9), np.log(1.1), np.log(1.2)]
        assert out["ret_std_3"].iloc[:3].isna().all()
        assert out["ret_std_3"].iloc[3] == pytest.approx(statistics.stdev(rets[:3]))
        assert out["ret_std_3"].iloc[4] == pytest.approx(statistics.stdev(rets[1:]))

    def test_close_vs_ma_and_sr_distances(self):
        df = tiny_df(
            [100.0, 110.0, 108.0],
            high=[110.0, 115.0, 112.0],
            low=[95.0, 98.0, 97.0],
        )
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        assert out["close_vs_ma_3"].iloc[2] == pytest.approx(108.0 / (318.0 / 3.0) - 1.0)
        assert out["dist_from_high_3"].iloc[2] == pytest.approx(108.0 / 115.0 - 1.0)
        assert out["dist_from_low_3"].iloc[2] == pytest.approx(108.0 / 95.0 - 1.0)

    def test_volume_features(self):
        df = tiny_df([100.0] * 3, volume=[10.0, 10.0, 40.0])
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        assert out["volume_vs_ma_3"].iloc[2] == pytest.approx(40.0 / 20.0)
        assert out["volume_z_3"].iloc[2] == pytest.approx(
            (40.0 - 20.0) / statistics.stdev([10.0, 10.0, 40.0])
        )

    def test_constant_volume_zscore_is_nan_not_substituted(self):
        df = tiny_df([100.0] * 4, volume=[10.0] * 4)
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        assert out["volume_z_3"].iloc[2:].isna().all()  # zero std -> undefined
        assert out["volume_vs_ma_3"].iloc[2:].tolist() == pytest.approx([1.0, 1.0])

    def test_intra_candle_shape(self):
        df = tiny_df([110.0], open_=[100.0], high=[112.0], low=[98.0])
        out = technical.compute_technical_features(df.pipe(_pad_to_len, 5), **SMALL_WINDOWS)
        assert out["body_pct"].iloc[0] == pytest.approx(10.0 / 14.0)
        assert out["upper_wick_pct"].iloc[0] == pytest.approx(2.0 / 14.0)
        assert out["lower_wick_pct"].iloc[0] == pytest.approx(2.0 / 14.0)
        assert out["close_pos_in_range"].iloc[0] == pytest.approx(12.0 / 14.0)
        assert out["range_pct"].iloc[0] == pytest.approx(14.0 / 110.0)

    def test_zero_range_candle_yields_nan_shape(self):
        df = make_ohlcv(10, seed=4)
        for col in ("open", "high", "low", "close"):
            df.loc[5, col] = 100.0
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        for col in ("body_pct", "upper_wick_pct", "lower_wick_pct", "close_pos_in_range"):
            assert np.isnan(out[col].iloc[5]), col
            assert out[col].drop(index=5).notna().all(), col
        assert out["range_pct"].iloc[5] == 0.0


def _pad_to_len(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Repeat the last candle so a single hand-made candle passes validation."""
    rows = [df.iloc[[0]]] * n
    out = pd.concat(rows, ignore_index=True)
    out["open_time"] = pd.to_datetime(BASE_MS + HOUR_MS * np.arange(n), unit="ms", utc=True)
    return out


class TestNaNPolicy:
    def test_warmup_rows_keep_nan_and_nothing_is_backfilled(self):
        df = make_ohlcv(20, seed=6)
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        # Exact first valid row per family.
        assert out["log_ret_1"].isna().tolist()[:2] == [True, False]
        assert out["log_ret_2"].isna().tolist()[:3] == [True, True, False]
        assert out["ret_std_3"].isna().tolist()[:4] == [True, True, True, False]
        assert out["range_pct_ma_3"].isna().tolist()[:3] == [True, True, False]
        assert out["close_vs_ma_3"].isna().tolist()[:3] == [True, True, False]

    def test_rsi_warmup(self):
        """RSI has NaN on first row (no delta), then produces values."""
        df = make_ohlcv(20, seed=6)
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        # Row 0: NaN (close.diff() is NaN for first row).
        assert np.isnan(out["rsi_3"].iloc[0])
        # Row 1 onward: RSI is defined (ewm seeds from first non-NaN gain/loss).
        assert out["rsi_3"].iloc[1:].notna().all()
        # RSI bounded in [0, 100].
        rsi_valid = out["rsi_3"].dropna()
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_macd_warmup(self):
        """MACD features are defined from row 0 (ewm seeds with first value)."""
        df = make_ohlcv(20, seed=6)
        out = technical.compute_technical_features(df, **SMALL_WINDOWS)
        # EWM with adjust=False produces a value from the very first row.
        assert out["macd_line"].notna().all()
        assert out["macd_signal"].notna().all()
        assert out["macd_hist"].notna().all()

    def test_defaults_longest_lookback(self):
        assert (
            technical.longest_lookback_rows(
                return_periods=[1, 3, 7, 14],
                volatility_windows=[7, 14, 30],
                volume_window=20,
                ma_windows=[7, 30],
                sr_window=30,
                rsi_period=14,
                macd={"fast_period": 12, "slow_period": 26, "signal_period": 9},
            )
            == 35  # max(15, 31, 20, 30, 30, 15, 35) = 35
        )


class TestInputValidation:
    def test_missing_column_rejected(self):
        df = make_ohlcv(10).drop(columns=["volume"])
        with pytest.raises(ValueError, match="volume"):
            technical.compute_technical_features(df, **SMALL_WINDOWS)

    def test_unsorted_open_time_rejected(self):
        df = make_ohlcv(10).iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="strictly increasing"):
            technical.compute_technical_features(df, **SMALL_WINDOWS)

    def test_bad_window_rejected(self):
        with pytest.raises(ValueError, match="volume_window"):
            technical.compute_technical_features(
                make_ohlcv(10), **{**SMALL_WINDOWS, "volume_window": 1}
            )

    def test_bad_rsi_period_rejected(self):
        with pytest.raises(ValueError, match="rsi_period"):
            technical.compute_technical_features(
                make_ohlcv(10), **{**SMALL_WINDOWS, "rsi_period": 1}
            )

    def test_bad_macd_fast_gte_slow_rejected(self):
        with pytest.raises(ValueError, match="fast_period"):
            technical.compute_technical_features(
                make_ohlcv(10),
                **{**SMALL_WINDOWS, "macd": {"fast_period": 5, "slow_period": 3,
                                              "signal_period": 3}},
            )

    def test_bad_macd_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown"):
            technical.compute_technical_features(
                make_ohlcv(10),
                **{**SMALL_WINDOWS, "macd": {"fast_period": 2, "slow_period": 4,
                                              "signal_period": 3, "bogus": 1}},
            )


class TestCandlestick:
    def test_all_patterns_present_and_values_in_allowed_set(self):
        df = make_ohlcv(300, seed=5)
        out, report = candlestick.compute_candlestick_features(df, drop_never_fired=False)
        assert out.shape[1] == len(candlestick.pattern_names())
        assert report["n_patterns_total"] == len(candlestick.pattern_names())
        for col in out.columns:
            assert col.startswith("cdl_")
            assert out[col].dtype == "int32"
            assert set(np.unique(out[col])) <= candlestick.ALLOWED_PATTERN_VALUES

    def test_doji_fires_on_crafted_candle(self):
        # 14 fat-bodied candles, then a perfect doji (open == close).
        df = tiny_df(
            close=[104.0] * 14 + [102.0],
            open_=[100.0] * 14 + [102.0],
            high=[105.0] * 15,
            low=[99.0] * 15,
        )
        out, report = candlestick.compute_candlestick_features(df)
        assert out["cdl_doji"].iloc[-1] == 100
        assert report["fire_rates"]["cdl_doji"] == pytest.approx(1 / 15)

    def test_hikkake_confirmation_emits_plus_minus_200(self):
        # Seed chosen so CDLHIKKAKE produces confirmation bars in this fixture.
        df = make_ohlcv(400, seed=0)
        out, report = candlestick.compute_candlestick_features(df)
        assert (out["cdl_hikkake"].abs() == 200).any()
        assert set(np.unique(out["cdl_hikkake"])) <= candlestick.ALLOWED_PATTERN_VALUES
        assert report["fire_rates"]["cdl_hikkake"] > 0

    def test_engulfing_near_miss_emits_plus_minus_80(self):
        # TA-Lib 0.7.1 grades near-miss engulfing/harami patterns at +-80;
        # the same seed as above is known to produce one.
        df = make_ohlcv(400, seed=0)
        out, _ = candlestick.compute_candlestick_features(df)
        assert (out["cdl_engulfing"].abs() == 80).any()
        assert set(np.unique(out["cdl_engulfing"])) <= candlestick.ALLOWED_PATTERN_VALUES

    def test_never_firing_patterns_dropped_and_logged(self, caplog):
        df = tiny_df(
            close=[104.0] * 14 + [102.0],
            open_=[100.0] * 14 + [102.0],
            high=[105.0] * 15,
            low=[99.0] * 15,
        )
        with caplog.at_level(logging.INFO):
            out, report = candlestick.compute_candlestick_features(df, context="fixture")
        assert report["dropped_never_fired"]  # plenty never fire on 15 bland candles
        assert "never fire" in caplog.text
        for col in report["dropped_never_fired"]:
            assert col not in out.columns
        assert set(report["fire_rates"]) == set(out.columns)
        assert all((out[col] != 0).any() for col in out.columns)

    def test_drop_disabled_keeps_all_columns(self):
        df = make_ohlcv(30, seed=9)
        out, report = candlestick.compute_candlestick_features(df, drop_never_fired=False)
        assert report["dropped_never_fired"] == []
        assert out.shape[1] == report["n_patterns_total"]


class TestGapAccounting:
    def _with_gap(self, n: int, drop_start: int, drop_len: int, seed: int = 2):
        df = make_ohlcv(n, seed=seed)
        removed = df.iloc[drop_start : drop_start + drop_len]
        kept = df.drop(index=range(drop_start, drop_start + drop_len)).reset_index(drop=True)
        gap = {
            "start": removed["open_time"].iloc[0].isoformat(),
            "end": removed["open_time"].iloc[-1].isoformat(),
            "n_missing": drop_len,
        }
        return kept, gap

    def test_affected_rows_marked_exactly(self):
        kept, gap = self._with_gap(n=50, drop_start=20, drop_len=3)
        mask = build_features.mark_gap_affected_rows(kept, [gap], lookback_rows=5, interval="1h")
        # Row after the jump is position 20; windows of span 5 contain the jump
        # for rows 20..23 -> exactly lookback_rows - 1 = 4 rows.
        assert mask.sum() == 4
        assert mask.iloc[20:24].all()
        assert not mask.iloc[:20].any()
        assert not mask.iloc[24:].any()

    def test_affected_rows_clipped_at_series_end(self):
        kept, gap = self._with_gap(n=50, drop_start=44, drop_len=3)
        mask = build_features.mark_gap_affected_rows(kept, [gap], lookback_rows=10, interval="1h")
        assert mask.sum() == 3  # only 3 rows remain after the jump
        assert mask.iloc[44:].all()

    def test_no_gap_records_means_no_affected_rows(self):
        df = make_ohlcv(30)
        mask = build_features.mark_gap_affected_rows(df, [], lookback_rows=10, interval="1h")
        assert not mask.any()

    def test_window_of_one_row_cannot_contain_a_gap(self):
        kept, gap = self._with_gap(n=50, drop_start=20, drop_len=3)
        mask = build_features.mark_gap_affected_rows(kept, [gap], lookback_rows=1, interval="1h")
        assert not mask.any()

    def test_recorded_gap_without_matching_jump_warns(self, caplog):
        df = make_ohlcv(30)  # contiguous — the recorded gap is bogus
        gap = {
            "start": df["open_time"].iloc[10].isoformat(),
            "end": df["open_time"].iloc[10].isoformat(),
            "n_missing": 1,
        }
        with caplog.at_level(logging.WARNING):
            mask = build_features.mark_gap_affected_rows(df, [gap], 5, "1h")
        assert not mask.any()
        assert "no matching jump" in caplog.text

    def test_load_gap_records_missing_manifest_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            gaps = build_features.load_gap_records(tmp_path / "missing.json")
        assert gaps == []
        assert "not found" in caplog.text

    def test_load_gap_records_filters_malformed(self, tmp_path, caplog):
        path = tmp_path / "m.json"
        good = {"start": "2021-01-01T00:00:00+00:00", "end": "2021-01-01T01:00:00+00:00",
                "n_missing": 2}
        path.write_text(json.dumps({"gaps": [good, {"oops": 1}]}), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            gaps = build_features.load_gap_records(path)
        assert gaps == [good]
        assert "malformed" in caplog.text


class TestBuildEndToEnd:
    def test_build_interval_writes_features_and_manifest(self, tmp_path):
        df = make_ohlcv(60, seed=13)
        removed = df.iloc[30:32]
        kept = df.drop(index=range(30, 32)).reset_index(drop=True)  # 58 rows, one gap
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        kept.to_parquet(raw_dir / "btc_usdt_1h.parquet", index=False)
        gap = {
            "start": removed["open_time"].iloc[0].isoformat(),
            "end": removed["open_time"].iloc[-1].isoformat(),
            "n_missing": 2,
        }
        (raw_dir / "btc_usdt_1h.manifest.json").write_text(
            json.dumps({"gaps": [gap]}), encoding="utf-8"
        )
        cfg = {
            "symbol": "BTCUSDT",
            "intervals": ["1h"],
            "raw_dir": str(raw_dir),
            "file_prefix": "btc_usdt",
            "processed_dir": str(tmp_path / "processed"),
            "features": SMALL_WINDOWS,
        }

        manifest = build_features.build_features_for_interval("1h", cfg)

        out_path = tmp_path / "processed" / "features_1h.parquet"
        assert out_path.is_file()
        features = pd.read_parquet(out_path)
        assert manifest["row_count"] == 58
        assert list(features.columns) == (
            ["open_time"] + manifest["features"]["technical"] + manifest["features"]["candlestick"]
        )
        assert manifest["n_features"] == features.shape[1] - 1

        # Effective lookback now accounts for RSI and MACD windows too.
        tech_lookback = technical.longest_lookback_rows(**SMALL_WINDOWS)
        expected_lookback = max(tech_lookback, candlestick.max_pattern_lookback_rows())
        assert manifest["lookback_rows"]["effective"] == expected_lookback
        assert manifest["gap_accounting"]["gap_records_from_raw_manifest"] == 1
        assert manifest["gap_accounting"]["gap_affected_rows"] == expected_lookback - 1

        on_disk = json.loads(
            (tmp_path / "processed" / "features_1h.manifest.json").read_text(encoding="utf-8")
        )
        assert on_disk == manifest

    def test_missing_raw_parquet_raises(self, tmp_path):
        cfg = {
            "symbol": "BTCUSDT",
            "intervals": ["1h"],
            "raw_dir": str(tmp_path),
            "file_prefix": "btc_usdt",
            "processed_dir": str(tmp_path / "processed"),
            "features": SMALL_WINDOWS,
        }
        with pytest.raises(FileNotFoundError, match="fetch_binance"):
            build_features.build_features_for_interval("1h", cfg)


class TestFeatureLabelAlignment:
    """Test the feature–label alignment guard from validate_features.py."""

    def _build_both(self, df: pd.DataFrame):
        """Build features and labels in-memory for a tiny fixture."""
        tech = technical.compute_technical_features(df, **SMALL_WINDOWS)
        cdl, _ = candlestick.compute_candlestick_features(
            df, drop_never_fired=False,
        )
        features = pd.concat([df[["open_time"]], tech, cdl], axis=1)
        labels = compute_forward_return_labels(
            df, horizon=1, dead_zone_pct=0.3,
        )
        # compute_forward_return_labels uses df["open_time"].values which
        # strips the tz — restore it for a clean merge.
        labels["open_time"] = pd.to_datetime(
            labels["open_time"], utc=True,
        )
        return features, labels

    def test_alignment_passes_on_correct_data(self):
        close = [100.0 + i * 0.5 for i in range(20)]
        df = tiny_df(close)
        features, labels = self._build_both(df)
        # Should not raise.
        validate_feature_label_alignment(
            features, labels, raw_candles_df=df,
            horizon=1, dead_zone_pct=0.3,
        )

    def test_duplicate_open_time_in_labels_fails(self):
        close = [100.0 + i * 0.5 for i in range(20)]
        df = tiny_df(close)
        features, labels = self._build_both(df)
        # Duplicate a row in labels.
        bad_labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
        with pytest.raises(AssertionError, match="duplicate"):
            validate_feature_label_alignment(
                features, bad_labels, raw_candles_df=df,
                horizon=1, dead_zone_pct=0.3,
            )

    def test_overlapping_column_names_fail(self):
        close = [100.0 + i * 0.5 for i in range(20)]
        df = tiny_df(close)
        features, labels = self._build_both(df)
        # Add a column to labels that also exists in features.
        labels["log_ret_1"] = 0.0
        with pytest.raises(AssertionError, match="overlap"):
            validate_feature_label_alignment(
                features, labels, raw_candles_df=df,
                horizon=1, dead_zone_pct=0.3,
            )


CONFIG_TEXT = """
data:
  symbol: BTCUSDT
  intervals: ["1h"]
  start_date: "2020-01-01"
  end_date: null

paths:
  raw_dir: "data/raw"
  file_prefix: "btc_usdt"
  processed_dir: "data/processed"

features:
  volume_window: 5
  rsi_period: 10
  macd:
    fast_period: 6
    slow_period: 13
    signal_period: 5
"""


class TestFeaturesConfig:
    def _write(self, tmp_path, text: str):
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_defaults_merged_and_overrides_respected(self, tmp_path):
        cfg = build_features.load_features_config(self._write(tmp_path, CONFIG_TEXT))
        assert cfg["processed_dir"] == "data/processed"
        assert cfg["features"]["volume_window"] == 5  # override
        assert cfg["features"]["rsi_period"] == 10  # override
        assert cfg["features"]["macd"]["fast_period"] == 6  # override
        assert cfg["features"]["return_periods"] == [1, 3, 7, 14]  # default
        assert cfg["symbol"] == "BTCUSDT"  # Phase 1 loader still applies

    def test_unknown_feature_option_rejected(self, tmp_path):
        broken = CONFIG_TEXT.replace("features:\n", "features:\n  fooo: 1\n")
        with pytest.raises(ValueError, match="fooo"):
            build_features.load_features_config(self._write(tmp_path, broken))

    def test_invalid_window_rejected(self, tmp_path):
        broken = CONFIG_TEXT.replace("volume_window: 5", "volume_window: 1")
        with pytest.raises(ValueError, match="volume_window"):
            build_features.load_features_config(self._write(tmp_path, broken))

    def test_invalid_rsi_period_rejected(self, tmp_path):
        broken = CONFIG_TEXT.replace("rsi_period: 10", "rsi_period: 1")
        with pytest.raises(ValueError, match="rsi_period"):
            build_features.load_features_config(self._write(tmp_path, broken))

    def test_missing_processed_dir_rejected(self, tmp_path):
        broken = CONFIG_TEXT.replace('  processed_dir: "data/processed"\n', "")
        with pytest.raises(ValueError, match="processed_dir"):
            build_features.load_features_config(self._write(tmp_path, broken))
