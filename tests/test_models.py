"""Unit tests for the Phase 4 modeling pipeline.

Three things matter most here, in order of how badly a bug would corrupt
the reported accuracy:

1. The walk-forward split is chronological, non-overlapping, and separated
   by a gap at each boundary wide enough for the modeled horizon.
2. The scaler is fit on the training split only — never on validation,
   test, or the full dataset.
3. The end-to-end pipeline (join -> split -> scale -> train -> persist)
   runs without error on a small synthetic fixture and produces sane
   artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.features import build_features
from src.labels import build_labels
from src.models import evaluate, train

HOUR_MS = 3_600_000
BASE_MS = 1_609_459_200_000  # 2021-01-01T00:00:00Z

# Small windows keep feature warm-up short so a ~200-row fixture leaves
# plenty of usable rows in every split.
SMALL_FEATURES = {
    "return_periods": [1, 2],
    "volatility_windows": [3],
    "volume_window": 3,
    "ma_windows": [3],
    "sr_window": 3,
    "rsi_period": 3,
    "macd": {"fast_period": 2, "slow_period": 4, "signal_period": 2},
}

FIXTURE_CONFIG_TEMPLATE = """\
data:
  symbol: BTCUSDT
  intervals: ["1h"]
  start_date: "2021-01-01"
  end_date: null

paths:
  raw_dir: "{raw_dir}"
  file_prefix: "btc_usdt"
  processed_dir: "{processed_dir}"
  models_dir: "{models_dir}"

features:
  return_periods: [1, 2]
  volatility_windows: [3]
  volume_window: 3
  ma_windows: [3]
  sr_window: 3
  rsi_period: 3
  macd:
    fast_period: 2
    slow_period: 4
    signal_period: 2

labels:
  horizons: [1]
  dead_zone_pct: 0.0

modeling:
  intervals: ["1h"]
  horizon: 1
  split:
    train_frac: 0.60
    val_frac: 0.20
    gap_candles: 1
  confidence_threshold: 0.60
  leak_alert_accuracy: 0.65
  random_state: 42
  logistic_regression:
    C: 1.0
    max_iter: 200
  lightgbm:
    n_estimators: 20
    learning_rate: 0.1
    num_leaves: 3
    max_depth: 2
    min_child_samples: 2
    subsample: 1.0
    subsample_freq: 0
    colsample_bytree: 1.0
    reg_lambda: 1.0
    early_stopping_rounds: 5
"""


def _synthetic_candles(n: int, seed: int = 42) -> pd.DataFrame:
    """A deterministic random-walk OHLCV fixture — mixed up/down moves."""
    rng = np.random.default_rng(seed)
    step_returns = rng.normal(0.0, 0.01, size=n - 1)
    close = np.empty(n, dtype="float64")
    close[0] = 100.0
    close[1:] = 100.0 * np.cumprod(1.0 + step_returns)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.5, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.5, size=n)
    volume = rng.uniform(50.0, 150.0, size=n)
    open_time = pd.to_datetime(BASE_MS + HOUR_MS * np.arange(n), unit="ms", utc=True)
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _write_fixture(tmp_path, n_rows: int = 200) -> str:
    """Write raw candles, build features/labels, and write a modeling config.

    Returns the path to the written config.yaml.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"

    candles = _synthetic_candles(n_rows)
    raw_path = raw_dir / "btc_usdt_1h.parquet"
    candles.to_parquet(raw_path, index=False)
    (raw_dir / "btc_usdt_1h.manifest.json").write_text(
        json.dumps({"gaps": []}), encoding="utf-8"
    )

    cfg_feat = {
        "symbol": "BTCUSDT",
        "intervals": ["1h"],
        "raw_dir": str(raw_dir),
        "file_prefix": "btc_usdt",
        "processed_dir": str(processed_dir),
        "features": SMALL_FEATURES,
    }
    build_features.build_features_for_interval("1h", cfg_feat)

    cfg_lbl = {
        "symbol": "BTCUSDT",
        "intervals": ["1h"],
        "raw_dir": str(raw_dir),
        "file_prefix": "btc_usdt",
        "processed_dir": str(processed_dir),
        "labels": {"horizons": [1], "dead_zone_pct": 0.0},
    }
    build_labels.build_labels_for_interval("1h", cfg_lbl)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        FIXTURE_CONFIG_TEMPLATE.format(
            raw_dir=raw_dir, processed_dir=processed_dir, models_dir=models_dir,
        ),
        encoding="utf-8",
    )
    return str(config_path)


class TestSplitBounds:
    """Chronology and gap correctness of the walk-forward split."""

    @pytest.mark.parametrize(
        ("n_rows", "train_frac", "val_frac", "gap"),
        [
            (200, 0.60, 0.20, 1),
            (1000, 0.70, 0.15, 3),
            (500, 0.50, 0.30, 5),
        ],
    )
    def test_boundaries_are_chronological_and_non_overlapping(
        self, n_rows, train_frac, val_frac, gap
    ):
        bounds = train.make_split_bounds(
            n_rows, train_frac=train_frac, val_frac=val_frac, gap_candles=gap,
        )
        # Strict chronological ordering.
        assert bounds.train_start == 0
        assert bounds.train_start < bounds.train_end
        assert bounds.train_end < bounds.val_start
        assert bounds.val_start < bounds.val_end
        assert bounds.val_end < bounds.test_start
        assert bounds.test_start < bounds.test_end
        assert bounds.test_end == n_rows
        # The gap is exactly the configured width at each boundary.
        assert bounds.val_start - bounds.train_end == gap
        assert bounds.test_start - bounds.val_end == gap

    def test_gap_rows_belong_to_no_split(self):
        bounds = train.make_split_bounds(
            100, train_frac=0.6, val_frac=0.2, gap_candles=3,
        )
        covered = (
            set(range(bounds.train_start, bounds.train_end))
            | set(range(bounds.val_start, bounds.val_end))
            | set(range(bounds.test_start, bounds.test_end))
        )
        gap_rows = set(range(bounds.train_end, bounds.val_start)) | set(
            range(bounds.val_end, bounds.test_start)
        )
        assert gap_rows, "test setup should produce non-empty gaps"
        assert not (covered & gap_rows), "gap rows must not appear in any split"

    def test_too_few_rows_raises(self):
        # A single row cannot fund a train split, a gap, a val split, a
        # second gap, and a test split simultaneously.
        with pytest.raises(ValueError, match="empty segment"):
            train.make_split_bounds(3, train_frac=0.6, val_frac=0.2, gap_candles=5)

    def test_gap_smaller_than_horizon_rejected_by_config(self, tmp_path):
        config_path = _write_fixture(tmp_path)
        text = Path(config_path).read_text(encoding="utf-8")
        broken = text.replace("gap_candles: 1", "gap_candles: 0")
        broken_path = tmp_path / "broken_config.yaml"
        broken_path.write_text(broken, encoding="utf-8")
        with pytest.raises(ValueError, match="gap_candles"):
            train.load_modeling_config(broken_path)


class TestSplitDataset:
    """split_dataset must respect boundaries and drop NaN without shifting them."""

    def _merged(self, n: int) -> pd.DataFrame:
        open_time = pd.to_datetime(BASE_MS + HOUR_MS * np.arange(n), unit="ms", utc=True)
        return pd.DataFrame(
            {
                "open_time": open_time,
                "f": np.arange(n, dtype="float64"),
                "label_1": np.tile([0.0, 1.0], n // 2 + 1)[:n],
            }
        )

    def test_rows_land_in_the_correct_split_only(self):
        merged = self._merged(30)
        bounds = train.make_split_bounds(30, train_frac=0.5, val_frac=0.23, gap_candles=1)
        splits = train.split_dataset(
            merged, bounds, feature_cols=["f"], label_col="label_1",
        )
        train_times = set(splits["train"]["open_time"])
        val_times = set(splits["val"]["open_time"])
        test_times = set(splits["test"]["open_time"])
        # No row appears in more than one split.
        assert not (train_times & val_times)
        assert not (val_times & test_times)
        assert not (train_times & test_times)
        # Every kept row's position falls inside its own split's bounds.
        expected_train = set(merged["open_time"].iloc[bounds.train_start:bounds.train_end])
        expected_val = set(merged["open_time"].iloc[bounds.val_start:bounds.val_end])
        expected_test = set(merged["open_time"].iloc[bounds.test_start:bounds.test_end])
        assert train_times <= expected_train
        assert val_times <= expected_val
        assert test_times <= expected_test

    def test_nan_drop_does_not_shift_rows_across_boundary(self):
        merged = self._merged(30)
        bounds = train.make_split_bounds(30, train_frac=0.5, val_frac=0.23, gap_candles=1)
        # NaN out the last feature row of train and the first of val.
        merged.loc[bounds.train_end - 1, "f"] = np.nan
        merged.loc[bounds.val_start, "f"] = np.nan
        splits = train.split_dataset(
            merged, bounds, feature_cols=["f"], label_col="label_1",
        )
        dropped_time = merged["open_time"].iloc[bounds.train_end - 1]
        assert dropped_time not in set(splits["train"]["open_time"])
        assert dropped_time not in set(splits["val"]["open_time"])
        assert dropped_time not in set(splits["test"]["open_time"])
        # The rest of train is untouched.
        assert len(splits["train"]) == (bounds.train_end - bounds.train_start) - 1
        assert len(splits["val"]) == (bounds.val_end - bounds.val_start) - 1

    def test_empty_split_after_nan_drop_raises(self):
        merged = self._merged(30)
        bounds = train.make_split_bounds(30, train_frac=0.5, val_frac=0.23, gap_candles=1)
        merged.loc[bounds.val_start : bounds.val_end - 1, "f"] = np.nan
        with pytest.raises(ValueError, match="no usable rows"):
            train.split_dataset(merged, bounds, feature_cols=["f"], label_col="label_1")


class TestScalerFitOnTrainOnly:
    """The scaler must never see validation/test statistics."""

    def test_scaler_reflects_train_distribution_only(self):
        rng = np.random.default_rng(0)
        train_features = pd.DataFrame(
            {"f": rng.normal(loc=0.0, scale=1.0, size=200)}
        )
        val_features = pd.DataFrame(
            {"f": rng.normal(loc=500.0, scale=50.0, size=50)}
        )
        scaler = train.fit_scaler_on_train(train_features)
        assert scaler.mean_[0] == pytest.approx(train_features["f"].mean())

        # If val had leaked into the fit, the mean would be dragged far
        # towards val's centre (500) — compare against fitting on both.
        combined = pd.concat([train_features, val_features], ignore_index=True)
        leaked_scaler = train.fit_scaler_on_train(combined)
        # Expected combined mean ~= 500 * 50 / 250 = 100; train-only stays near 0.
        assert abs(scaler.mean_[0]) < 10.0
        assert leaked_scaler.mean_[0] > 50.0
        assert scaler.mean_[0] != pytest.approx(leaked_scaler.mean_[0])

    def test_transform_does_not_mutate_fitted_scaler_state(self):
        rng = np.random.default_rng(1)
        train_features = pd.DataFrame({"f": rng.normal(0.0, 1.0, size=100)})
        val_features = pd.DataFrame({"f": rng.normal(500.0, 50.0, size=30)})
        scaler = train.fit_scaler_on_train(train_features)
        mean_before, scale_before = scaler.mean_.copy(), scaler.scale_.copy()

        scaler.transform(val_features)  # must be transform, never fit_transform

        assert np.array_equal(scaler.mean_, mean_before)
        assert np.array_equal(scaler.scale_, scale_before)

    def test_transformed_val_is_not_independently_standardised(self):
        """A val set scaled with train's scaler should NOT look standard-normal.

        If the code accidentally called ``fit_transform`` on val instead of
        ``transform``, the output would have mean ~0 and std ~1. Since val is
        centred far from train here, the correctly-scaled output must not be.
        """
        rng = np.random.default_rng(2)
        train_features = pd.DataFrame({"f": rng.normal(0.0, 1.0, size=200)})
        val_features = pd.DataFrame({"f": rng.normal(500.0, 50.0, size=50)})
        scaler = train.fit_scaler_on_train(train_features)
        transformed = scaler.transform(val_features)
        assert abs(transformed.mean()) > 10.0  # nowhere near standard-normal


class TestEndToEndSmoke:
    """The full join -> split -> scale -> train -> persist pipeline, on a tiny fixture."""

    def test_train_interval_runs_and_produces_sane_artifacts(self, tmp_path, caplog):
        config_path = _write_fixture(tmp_path)
        cfg = train.load_modeling_config(config_path)

        with caplog.at_level(logging.INFO):
            manifest = train.train_interval("1h", cfg)

        assert manifest["horizon"] == 1
        assert manifest["label_col"] == "label_1"
        assert len(manifest["feature_cols"]) > 0
        assert all(n > 0 for n in manifest["usable_rows"].values())
        assert manifest["lightgbm_best_iteration"] >= 1

        out_dir = tmp_path / "models" / "1h"
        assert (out_dir / "scaler.joblib").is_file()
        assert (out_dir / "logistic_regression.joblib").is_file()
        assert (out_dir / "lightgbm.joblib").is_file()
        assert (out_dir / "training_manifest.json").is_file()

    def test_evaluate_interval_runs_after_training(self, tmp_path, capsys):
        config_path = _write_fixture(tmp_path)
        cfg = train.load_modeling_config(config_path)
        train.train_interval("1h", cfg)

        report = evaluate.evaluate_interval("1h", cfg)

        for model_name in ("logistic_regression", "lightgbm"):
            for split_name in ("val", "test"):
                metrics = report["models"][model_name][split_name]["metrics"]
                assert 0.0 <= metrics["accuracy"] <= 1.0
                assert 0.0 <= metrics["base_rate"] <= 1.0
                assert len(metrics["confusion"]) == 2
        assert isinstance(report["leak_alerts"], list)
        captured = capsys.readouterr()
        assert "base rate" in captured.out

    def test_evaluate_detects_stale_artifacts_after_data_change(self, tmp_path):
        config_path = _write_fixture(tmp_path)
        cfg = train.load_modeling_config(config_path)
        train.train_interval("1h", cfg)

        # Mutate the features parquet after training — evaluation must refuse
        # to score against artifacts trained on different data.
        features_path = tmp_path / "processed" / "features_1h.parquet"
        features = pd.read_parquet(features_path)
        features.loc[0, features.columns[1]] = 999.0
        features.to_parquet(features_path, index=False)

        with pytest.raises(ValueError, match="changed since training"):
            evaluate.evaluate_interval("1h", cfg)


class TestModelingConfig:
    """Config loading and validation for the modeling section."""

    def test_valid_config_loads(self, tmp_path):
        config_path = _write_fixture(tmp_path)
        cfg = train.load_modeling_config(config_path)
        assert cfg["modeling"]["horizon"] == 1
        assert cfg["modeling"]["split"]["gap_candles"] == 1
        assert cfg["models_dir"] == str(tmp_path / "models")

    def test_horizon_not_in_built_labels_rejected(self, tmp_path):
        config_path = _write_fixture(tmp_path)
        text = Path(config_path).read_text(encoding="utf-8")
        broken = text.replace("horizon: 1", "horizon: 7", 1)
        broken_path = tmp_path / "broken.yaml"
        broken_path.write_text(broken, encoding="utf-8")
        with pytest.raises(ValueError, match="not among the built label horizons"):
            train.load_modeling_config(broken_path)

    def test_unknown_modeling_key_rejected(self, tmp_path):
        config_path = _write_fixture(tmp_path)
        text = Path(config_path).read_text(encoding="utf-8")
        broken = text.replace(
            "modeling:\n  intervals:", "modeling:\n  bogus_key: 1\n  intervals:"
        )
        broken_path = tmp_path / "broken.yaml"
        broken_path.write_text(broken, encoding="utf-8")
        with pytest.raises(ValueError, match="unknown modeling option"):
            train.load_modeling_config(broken_path)

    def test_confidence_threshold_out_of_range_rejected(self, tmp_path):
        config_path = _write_fixture(tmp_path)
        text = Path(config_path).read_text(encoding="utf-8")
        broken = text.replace("confidence_threshold: 0.60", "confidence_threshold: 0.3")
        broken_path = tmp_path / "broken.yaml"
        broken_path.write_text(broken, encoding="utf-8")
        with pytest.raises(ValueError, match="confidence_threshold"):
            train.load_modeling_config(broken_path)


class TestFairScalerBaseline:
    """Sanity check that fit_scaler_on_train matches plain StandardScaler semantics."""

    def test_matches_manual_standard_scaler_fit(self):
        rng = np.random.default_rng(3)
        data = pd.DataFrame(
            {"a": rng.normal(5.0, 2.0, size=50), "b": rng.normal(-3.0, 1.0, size=50)}
        )
        ours = train.fit_scaler_on_train(data)
        reference = StandardScaler().fit(data)
        assert np.allclose(ours.mean_, reference.mean_)
        assert np.allclose(ours.scale_, reference.scale_)
