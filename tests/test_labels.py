"""Unit tests for the Phase 3 label modules.

The most important tests here are the derivation and alignment checks: every
label must be mathematically derived from exactly close[T] and close[T+N],
and a feature–label join on open_time must never shift, duplicate, or
misalign rows.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.labels import build_labels
from src.labels.build_labels import (
    class_balance_report,
    compute_forward_return_labels,
    load_labels_config,
)
from src.labels.validate_labels import validate_alignment, validate_label_derivation

HOUR_MS = 3_600_000
BASE_MS = 1_609_459_200_000  # 2021-01-01T00:00:00Z

SMALL_WINDOWS = {
    "return_periods": [1, 2],
    "volatility_windows": [3],
    "volume_window": 3,
    "ma_windows": [3],
    "sr_window": 3,
}


def _candle_df(
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


# Hand-built 12-row fixture with deliberate price movements.
# close prices designed so we can manually compute all expected labels.
HAND_CLOSE = [
    100.0,   # T=0
    101.0,   # T=1:  +1.00% from T=0  (up)
    100.5,   # T=2:  -0.495% from T=1 (down for dz=0.3%)
    100.6,   # T=3:  +0.0995% from T=2 (dead zone for dz=0.3%)
    102.0,   # T=4:  +1.392% from T=3  (up)
    101.8,   # T=5:  -0.196% from T=4  (dead zone for dz=0.3%, but clearly down for dz=0.1%)
    103.0,   # T=6:  +1.178% from T=5  (up)
    101.0,   # T=7:  -1.942% from T=6  (down)
    100.0,   # T=8:  -0.990% from T=7  (down)
    100.1,   # T=9:  +0.100% from T=8  (dead zone for dz=0.3%)
    100.0,   # T=10: -0.0999% from T=9 (dead zone for dz=0.3%)
    99.0,    # T=11: -1.000% from T=10 (last row for horizon=1, so T=10 is last valid)
]


class TestLabelComputation:
    """Hand-checked label computation with known close prices."""

    def test_horizon_1_with_dead_zone(self):
        """Manually verify every label for horizon=1, dead_zone=0.3%."""
        df = _candle_df(HAND_CLOSE)
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.3)
        threshold = 0.003
        n = len(HAND_CLOSE)

        # Hand-compute expected forward returns and labels.
        for t in range(n - 1):
            expected_ret = HAND_CLOSE[t + 1] / HAND_CLOSE[t] - 1.0
            actual_ret = out["fwd_return_1"].iloc[t]
            assert np.isclose(actual_ret, expected_ret, atol=1e-12), (
                f"Row {t}: expected ret {expected_ret}, got {actual_ret}"
            )
            if expected_ret > threshold:
                assert out["label_1"].iloc[t] == 1.0, f"Row {t} should be UP"
            elif expected_ret < -threshold:
                assert out["label_1"].iloc[t] == 0.0, f"Row {t} should be DOWN"
            else:
                assert np.isnan(out["label_1"].iloc[t]), f"Row {t} should be dead zone"

        # Last row: NaN (no close[T+1]).
        assert np.isnan(out["fwd_return_1"].iloc[-1])
        assert np.isnan(out["label_1"].iloc[-1])

    def test_horizon_3_last_three_rows_nan(self):
        """With horizon=3, the last 3 rows must be NaN."""
        df = _candle_df(HAND_CLOSE)
        out = compute_forward_return_labels(df, horizon=3, dead_zone_pct=0.3)

        # Last 3 rows: NaN.
        for t in range(len(HAND_CLOSE) - 3, len(HAND_CLOSE)):
            assert np.isnan(out["fwd_return_3"].iloc[t]), f"Row {t} fwd_return should be NaN"
            assert np.isnan(out["label_3"].iloc[t]), f"Row {t} label should be NaN"

        # Row before the tail (T=8): close[8]=100.0, close[11]=99.0 → -1%.
        expected_ret = 99.0 / 100.0 - 1.0
        assert np.isclose(out["fwd_return_3"].iloc[8], expected_ret)
        assert out["label_3"].iloc[8] == 0.0  # -1% < -0.3%

    def test_dead_zone_zero_classifies_everything(self):
        """With dead_zone=0, every row except the tail should be labelled."""
        df = _candle_df(HAND_CLOSE)
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.0)

        for t in range(len(HAND_CLOSE) - 1):
            assert not np.isnan(out["label_1"].iloc[t]), (
                f"Row {t} should be labelled (dz=0)"
            )
        # Last row still NaN (no future data).
        assert np.isnan(out["label_1"].iloc[-1])

    def test_multiple_horizons_independent(self):
        """Labels for different horizons are computed independently."""
        df = _candle_df(HAND_CLOSE)
        out1 = compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.3)
        out3 = compute_forward_return_labels(df, horizon=3, dead_zone_pct=0.3)

        # Horizon 1 has 1 NaN tail row; horizon 3 has 3.
        assert out1["fwd_return_1"].isna().sum() == 1
        assert out3["fwd_return_3"].isna().sum() == 3


class TestTailRows:
    """Assert that the last N rows are NaN because close[T+N] doesn't exist."""

    @pytest.mark.parametrize("horizon", [1, 2, 3, 5])
    def test_tail_rows_nan_for_each_horizon(self, horizon):
        close = [100.0 + i for i in range(15)]
        df = _candle_df(close)
        out = compute_forward_return_labels(df, horizon=horizon, dead_zone_pct=0.0)

        ret_col = f"fwd_return_{horizon}"
        lbl_col = f"label_{horizon}"

        # Exactly the last `horizon` rows should have NaN fwd_return.
        tail_nans = out[ret_col].iloc[-horizon:]
        assert tail_nans.isna().all(), "Tail rows must be NaN"
        assert out[lbl_col].iloc[-horizon:].isna().all(), "Tail labels must be NaN"

        # Everything before the tail should have a valid fwd_return.
        head = out[ret_col].iloc[:-horizon]
        assert head.notna().all(), "Non-tail rows must have valid fwd_return"


class TestDeadZoneBoundary:
    """Test exact boundary behaviour: strict inequalities."""

    def test_exactly_at_positive_threshold_is_excluded(self):
        """A move of exactly +dead_zone_pct is inside the dead zone (NaN)."""
        # dead_zone_pct=1.0 → threshold = 0.01.
        # close[1] / close[0] - 1 = 0.01 exactly.
        df = _candle_df([100.0, 101.0, 100.0])
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=1.0)
        assert np.isclose(out["fwd_return_1"].iloc[0], 0.01)
        assert np.isnan(out["label_1"].iloc[0]), "Exactly at +threshold should be excluded"

    def test_just_above_positive_threshold_is_up(self):
        """A move just above +dead_zone_pct is classified as up (1)."""
        # 101.01 / 100.0 - 1 = 0.01010 > 0.01.
        df = _candle_df([100.0, 101.01, 100.0])
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=1.0)
        assert out["fwd_return_1"].iloc[0] > 0.01
        assert out["label_1"].iloc[0] == 1.0

    def test_exactly_at_negative_threshold_is_excluded(self):
        """A move of exactly -dead_zone_pct is inside the dead zone (NaN)."""
        # 99.0 / 100.0 - 1 = -0.01 exactly.
        df = _candle_df([100.0, 99.0, 100.0])
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=1.0)
        assert np.isclose(out["fwd_return_1"].iloc[0], -0.01)
        assert np.isnan(out["label_1"].iloc[0]), "Exactly at -threshold should be excluded"

    def test_just_below_negative_threshold_is_down(self):
        """A move just below -dead_zone_pct is classified as down (0)."""
        # 98.99 / 100.0 - 1 = -0.0101 < -0.01.
        df = _candle_df([100.0, 98.99, 100.0])
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=1.0)
        assert out["fwd_return_1"].iloc[0] < -0.01
        assert out["label_1"].iloc[0] == 0.0

    def test_zero_return_is_excluded(self):
        """No move at all (fwd_return=0) is dead zone, unless dead_zone=0."""
        df = _candle_df([100.0, 100.0, 100.0])
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.5)
        assert out["fwd_return_1"].iloc[0] == 0.0
        assert np.isnan(out["label_1"].iloc[0])

        # With dead_zone=0: exactly 0 is still excluded (strict >).
        out0 = compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.0)
        assert np.isnan(out0["label_1"].iloc[0])


class TestDerivationValidation:
    """Test validate_label_derivation with hand-checked data."""

    def test_valid_labels_pass(self):
        df = _candle_df(HAND_CLOSE)
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.3)
        # Should not raise.
        validate_label_derivation(df, out, horizon=1, dead_zone_pct=0.3)

    def test_valid_labels_horizon_3_pass(self):
        df = _candle_df(HAND_CLOSE)
        out = compute_forward_return_labels(df, horizon=3, dead_zone_pct=0.3)
        validate_label_derivation(df, out, horizon=3, dead_zone_pct=0.3)

    def test_corrupted_label_fails(self):
        df = _candle_df(HAND_CLOSE)
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.3)
        # Flip a label that should be 1 to 0.
        for i in range(len(out)):
            if out["label_1"].iloc[i] == 1.0:
                out.iloc[i, out.columns.get_loc("label_1")] = 0.0
                break
        with pytest.raises(AssertionError, match="should be 1"):
            validate_label_derivation(df, out, horizon=1, dead_zone_pct=0.3)

    def test_filled_tail_fails(self):
        df = _candle_df(HAND_CLOSE)
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.3)
        # Fill the last row's fwd_return (should be NaN).
        out.iloc[-1, out.columns.get_loc("fwd_return_1")] = 0.0
        with pytest.raises(AssertionError, match="NaN fwd_return"):
            validate_label_derivation(df, out, horizon=1, dead_zone_pct=0.3)


class TestAlignment:
    """Test feature–label join alignment."""

    def _build_both(self, tmp_path, close: list[float], horizon: int = 1):
        """Build features and labels for a tiny fixture, return paths."""
        df = _candle_df(close)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "btc_usdt_1h.parquet"
        df.to_parquet(raw_path, index=False)
        (raw_dir / "btc_usdt_1h.manifest.json").write_text(
            json.dumps({"gaps": []}), encoding="utf-8"
        )

        proc_dir = tmp_path / "processed"
        cfg_feat = {
            "symbol": "BTCUSDT",
            "intervals": ["1h"],
            "raw_dir": str(raw_dir),
            "file_prefix": "btc_usdt",
            "processed_dir": str(proc_dir),
            "features": SMALL_WINDOWS,
        }
        build_features.build_features_for_interval("1h", cfg_feat)

        cfg_lbl = {
            "symbol": "BTCUSDT",
            "intervals": ["1h"],
            "raw_dir": str(raw_dir),
            "file_prefix": "btc_usdt",
            "processed_dir": str(proc_dir),
            "labels": {"horizons": [horizon], "dead_zone_pct": 0.3},
        }
        build_labels.build_labels_for_interval("1h", cfg_lbl)

        return (
            str(proc_dir / "features_1h.parquet"),
            str(proc_dir / "labels_1h.parquet"),
            str(raw_path),
        )

    def test_alignment_passes_on_correct_data(self, tmp_path):
        close = [100.0 + i * 0.5 for i in range(20)]
        feat_p, lbl_p, raw_p = self._build_both(tmp_path, close)
        # Should not raise.
        validate_alignment(
            feat_p, lbl_p, raw_candles_path=raw_p,
            horizon=1, dead_zone_pct=0.3,
        )

    def test_row_counts_match(self, tmp_path):
        close = [100.0 + i * 0.5 for i in range(20)]
        feat_p, lbl_p, raw_p = self._build_both(tmp_path, close)
        features = pd.read_parquet(feat_p)
        labels = pd.read_parquet(lbl_p)
        assert len(features) == len(labels), "Features and labels must have the same row count"

    def test_first_row_label_matches_forward_return(self, tmp_path):
        close = [100.0, 105.0] + [100.0 + i for i in range(18)]
        feat_p, lbl_p, raw_p = self._build_both(tmp_path, close)
        labels = pd.read_parquet(lbl_p)
        # Row 0: close[0]=100, close[1]=105 → +5%.
        assert np.isclose(labels["fwd_return_1"].iloc[0], 0.05)
        assert labels["label_1"].iloc[0] == 1.0

    def test_last_valid_row_correct(self, tmp_path):
        close = [100.0 + i for i in range(20)]
        feat_p, lbl_p, raw_p = self._build_both(tmp_path, close)
        labels = pd.read_parquet(lbl_p)
        # Last valid row for horizon=1 is index 18.
        assert labels["fwd_return_1"].iloc[18] == pytest.approx(
            close[19] / close[18] - 1.0
        )
        # Last row (index 19) is NaN.
        assert np.isnan(labels["fwd_return_1"].iloc[19])

    def test_no_column_overlap(self, tmp_path):
        close = [100.0 + i * 0.5 for i in range(20)]
        feat_p, lbl_p, raw_p = self._build_both(tmp_path, close)
        features = pd.read_parquet(feat_p)
        labels = pd.read_parquet(lbl_p)
        feat_cols = set(features.columns) - {"open_time"}
        lbl_cols = set(labels.columns) - {"open_time"}
        assert not feat_cols & lbl_cols, "Feature and label columns must not overlap"


class TestClassBalance:
    """Test the class balance report."""

    def test_counts_sum_to_total(self):
        df = _candle_df(HAND_CLOSE)
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.3)
        report = class_balance_report(
            out, horizon=1, dead_zone_pct=0.3, n_total=len(df),
        )
        total = (
            report["class_balance"]["up"]
            + report["class_balance"]["down"]
            + report["excluded_dead_zone_rows"]
            + report["excluded_tail_rows"]
        )
        assert total == len(df), "Counts must sum to total rows"

    def test_imbalance_flag_triggers_warning(self, caplog):
        # Craft data where almost everything is 'up' → imbalance.
        close = [100.0 + i * 2 for i in range(20)]  # monotonically increasing
        df = _candle_df(close)
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.0)
        with caplog.at_level(logging.WARNING):
            report = class_balance_report(
                out, horizon=1, dead_zone_pct=0.0, n_total=len(df),
            )
        assert report["imbalance_flag"] is True
        assert "IMBALANCE" in caplog.text

    def test_balanced_data_no_flag(self):
        # Alternating up/down moves.
        close = [100.0, 102.0, 100.0, 102.0, 100.0, 102.0, 100.0, 102.0,
                 100.0, 102.0, 100.0, 102.0]
        df = _candle_df(close)
        out = compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.5)
        report = class_balance_report(
            out, horizon=1, dead_zone_pct=0.5, n_total=len(df),
        )
        assert report["imbalance_flag"] is False


class TestLabelConfig:
    """Test config loading and validation."""

    CONFIG_TEXT = """\
data:
  symbol: BTCUSDT
  intervals: ["1h"]
  start_date: "2020-01-01"
  end_date: null

paths:
  raw_dir: "data/raw"
  file_prefix: "btc_usdt"
  processed_dir: "data/processed"

labels:
  horizons: [1, 3]
  dead_zone_pct: 0.2
"""

    def _write(self, tmp_path, text: str):
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_valid_config_loads(self, tmp_path):
        cfg = load_labels_config(self._write(tmp_path, self.CONFIG_TEXT))
        assert cfg["processed_dir"] == "data/processed"
        assert cfg["labels"]["horizons"] == [1, 3]
        assert cfg["labels"]["dead_zone_pct"] == 0.2
        assert cfg["symbol"] == "BTCUSDT"

    def test_defaults_applied_when_section_missing(self, tmp_path):
        text = self.CONFIG_TEXT.replace(
            "labels:\n  horizons: [1, 3]\n  dead_zone_pct: 0.2\n", ""
        )
        cfg = load_labels_config(self._write(tmp_path, text))
        assert cfg["labels"]["horizons"] == [1]
        assert cfg["labels"]["dead_zone_pct"] == 0.15

    def test_unknown_key_rejected(self, tmp_path):
        broken = self.CONFIG_TEXT + "  bogus_key: 42\n"
        with pytest.raises(ValueError, match="bogus_key"):
            load_labels_config(self._write(tmp_path, broken))

    def test_invalid_horizon_zero_rejected(self, tmp_path):
        broken = self.CONFIG_TEXT.replace("horizons: [1, 3]", "horizons: [0]")
        with pytest.raises(ValueError, match="horizons"):
            load_labels_config(self._write(tmp_path, broken))

    def test_invalid_horizon_float_rejected(self, tmp_path):
        broken = self.CONFIG_TEXT.replace("horizons: [1, 3]", "horizons: [1.5]")
        with pytest.raises(ValueError, match="horizons"):
            load_labels_config(self._write(tmp_path, broken))

    def test_negative_dead_zone_rejected(self, tmp_path):
        broken = self.CONFIG_TEXT.replace("dead_zone_pct: 0.2", "dead_zone_pct: -0.1")
        with pytest.raises(ValueError, match="dead_zone_pct"):
            load_labels_config(self._write(tmp_path, broken))

    def test_missing_processed_dir_rejected(self, tmp_path):
        broken = self.CONFIG_TEXT.replace('  processed_dir: "data/processed"\n', "")
        with pytest.raises(ValueError, match="processed_dir"):
            load_labels_config(self._write(tmp_path, broken))


class TestInputValidation:
    """Test error handling for bad inputs."""

    def test_empty_df_rejected(self):
        df = pd.DataFrame(columns=["open_time", "close"])
        with pytest.raises(ValueError, match="empty"):
            compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.1)

    def test_missing_column_rejected(self):
        df = _candle_df([100.0, 101.0]).drop(columns=["close"])
        with pytest.raises(ValueError, match="close"):
            compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.1)

    def test_unsorted_rejected(self):
        df = _candle_df([100.0, 101.0, 102.0]).iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="strictly increasing"):
            compute_forward_return_labels(df, horizon=1, dead_zone_pct=0.1)

    def test_invalid_horizon_rejected(self):
        df = _candle_df([100.0, 101.0])
        with pytest.raises(ValueError, match="horizon"):
            compute_forward_return_labels(df, horizon=0, dead_zone_pct=0.1)

    def test_negative_dead_zone_rejected(self):
        df = _candle_df([100.0, 101.0])
        with pytest.raises(ValueError, match="dead_zone_pct"):
            compute_forward_return_labels(df, horizon=1, dead_zone_pct=-0.1)


class TestBuildEndToEnd:
    """End-to-end build test: raw parquet → labels parquet + manifest."""

    def test_build_writes_labels_and_manifest(self, tmp_path):
        close = [100.0 + i * 0.3 for i in range(30)]
        df = _candle_df(close)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        df.to_parquet(raw_dir / "btc_usdt_1h.parquet", index=False)
        (raw_dir / "btc_usdt_1h.manifest.json").write_text(
            json.dumps({"gaps": []}), encoding="utf-8"
        )

        cfg = {
            "symbol": "BTCUSDT",
            "intervals": ["1h"],
            "raw_dir": str(raw_dir),
            "file_prefix": "btc_usdt",
            "processed_dir": str(tmp_path / "processed"),
            "labels": {"horizons": [1, 3], "dead_zone_pct": 0.2},
        }
        manifest = build_labels.build_labels_for_interval("1h", cfg)

        out_path = tmp_path / "processed" / "labels_1h.parquet"
        assert out_path.is_file()
        labels = pd.read_parquet(out_path)

        # Columns: open_time + fwd_return_1 + label_1 + fwd_return_3 + label_3.
        assert list(labels.columns) == [
            "open_time", "fwd_return_1", "label_1", "fwd_return_3", "label_3",
        ]
        assert len(labels) == 30

        # Manifest structure.
        assert manifest["symbol"] == "BTCUSDT"
        assert manifest["interval"] == "1h"
        assert "1" in manifest["horizons"]
        assert "3" in manifest["horizons"]
        assert manifest["horizons"]["1"]["row_count"] == 30
        assert manifest["horizons"]["3"]["excluded_tail_rows"] == 3
        assert manifest["horizons"]["1"]["excluded_tail_rows"] == 1
        assert "iron_rule" in manifest

        # Manifest on disk matches.
        on_disk = json.loads(
            (tmp_path / "processed" / "labels_1h.manifest.json").read_text(encoding="utf-8")
        )
        assert on_disk == manifest

    def test_missing_raw_parquet_raises(self, tmp_path):
        cfg = {
            "symbol": "BTCUSDT",
            "intervals": ["1h"],
            "raw_dir": str(tmp_path),
            "file_prefix": "btc_usdt",
            "processed_dir": str(tmp_path / "processed"),
            "labels": {"horizons": [1], "dead_zone_pct": 0.15},
        }
        with pytest.raises(FileNotFoundError, match="fetch_binance"):
            build_labels.build_labels_for_interval("1h", cfg)
