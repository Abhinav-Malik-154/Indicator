"""Train directional models on the joined feature/label tables (Phase 4).

The training contract, in the order the code enforces it:

1. Features and labels are joined with the rehearsed one-liner
   (``features.merge(labels, on="open_time", how="left")``) and
   :func:`~src.features.validate_features.validate_feature_label_alignment`
   runs as a runtime guard before anything else happens.
2. Split boundaries are defined **positionally on the full joined table**,
   before any NaN row is dropped.  Chronological train -> gap -> validation
   -> gap -> test; the gap (>= horizon candles) stops the last training
   label from depending on prices inside the validation window.
3. NaN rows (feature warm-up, dead-zone labels, tail labels) are dropped
   **within each split** after the boundaries exist, so no global statistic
   ever touches data outside its split.
4. The scaler is fit on the training split only, then applied to
   validation/test.  Never fit on the full dataset.

Models: logistic regression (the fair, interpretable baseline) and LightGBM
(gradient boosting).  Both are persisted with joblib next to a training
manifest so evaluation can verify it is scoring exactly what was trained.

Run from the repo root::

    python -m src.models.train
    python -m src.models.train --intervals 1d --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.fetch_binance import sha256_of
from src.features.validate_features import validate_feature_label_alignment
from src.labels.build_labels import load_labels_config

logger = logging.getLogger(__name__)

_SPLIT_DEFAULTS: dict[str, Any] = {
    "train_frac": 0.70,
    "val_frac": 0.15,
    "gap_candles": 1,
}

_LOGREG_DEFAULTS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 1000,
}

_LIGHTGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
}

_PRUNING_DEFAULTS: dict[str, Any] = {
    "min_fire_rows": 30,
}

_MODELING_DEFAULTS: dict[str, Any] = {
    "intervals": ["1d"],
    "horizon": 1,
    "split": None,
    "confidence_threshold": 0.60,
    "leak_alert_accuracy": 0.65,
    "random_state": 42,
    "logistic_regression": None,
    "lightgbm": None,
    "pruning": None,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _merge_section(
    raw: dict[str, Any] | None,
    defaults: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Merge a config sub-mapping over its defaults, rejecting unknown keys."""
    section = raw or {}
    if not isinstance(section, dict):
        raise ValueError(f"config: 'modeling.{name}' must be a mapping")
    unknown = sorted(set(section) - set(defaults))
    if unknown:
        raise ValueError(f"config: unknown modeling.{name} option(s): {unknown}")
    return {**defaults, **section}


def load_modeling_config(path: str | Path) -> dict[str, Any]:
    """Load the pipeline config extended with the modeling settings.

    Builds on the Phase 3 loader (``labels`` / ``processed_dir`` and the
    Phase 1 sections), then validates ``paths.models_dir`` and the
    ``modeling`` section: defaults merged in, unknown keys rejected, split
    fractions sane, and the gap wide enough for the modeled horizon.

    Args:
        path: Path to the YAML config.

    Returns:
        The Phase 3 config dict plus ``models_dir`` and ``modeling`` keys.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If validation fails.
    """
    cfg = load_labels_config(path)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    paths = raw.get("paths") or {}
    models_dir = paths.get("models_dir")
    if not isinstance(models_dir, str) or not models_dir.strip():
        raise ValueError("config: paths.models_dir must be a non-empty string")

    section = raw.get("modeling") or {}
    if not isinstance(section, dict):
        raise ValueError("config: 'modeling' section must be a mapping")
    unknown = sorted(set(section) - set(_MODELING_DEFAULTS))
    if unknown:
        raise ValueError(f"config: unknown modeling option(s): {unknown}")
    modeling = {**_MODELING_DEFAULTS, **section}

    intervals = modeling["intervals"]
    if (
        not isinstance(intervals, (list, tuple))
        or len(intervals) == 0
        or not all(isinstance(i, str) and i.strip() for i in intervals)
    ):
        raise ValueError(
            f"config: modeling.intervals must be a non-empty list of strings, got {intervals!r}"
        )

    horizon = modeling["horizon"]
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(
            f"config: modeling.horizon must be an integer >= 1, got {horizon!r}"
        )
    if horizon not in cfg["labels"]["horizons"]:
        raise ValueError(
            f"config: modeling.horizon {horizon} is not among the built label "
            f"horizons {cfg['labels']['horizons']} — rebuild labels first"
        )

    split = _merge_section(modeling["split"], _SPLIT_DEFAULTS, "split")
    train_frac, val_frac = split["train_frac"], split["val_frac"]
    for key, value in (("train_frac", train_frac), ("val_frac", val_frac)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < 1:
            raise ValueError(f"config: modeling.split.{key} must be in (0, 1), got {value!r}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            "config: modeling.split.train_frac + val_frac must leave room for "
            f"a test split, got {train_frac} + {val_frac}"
        )
    gap = split["gap_candles"]
    if not isinstance(gap, int) or isinstance(gap, bool) or gap < 1:
        raise ValueError(
            f"config: modeling.split.gap_candles must be an integer >= 1, got {gap!r}"
        )
    if gap < horizon:
        raise ValueError(
            f"config: modeling.split.gap_candles ({gap}) is smaller than the "
            f"modeled horizon ({horizon}) — the last training label would "
            "depend on prices inside the validation window (leakage)"
        )
    modeling["split"] = split

    threshold = modeling["confidence_threshold"]
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0.5 <= threshold < 1.0
    ):
        raise ValueError(
            f"config: modeling.confidence_threshold must be in [0.5, 1), got {threshold!r}"
        )

    leak_alert = modeling["leak_alert_accuracy"]
    if (
        not isinstance(leak_alert, (int, float))
        or isinstance(leak_alert, bool)
        or not 0.5 < leak_alert <= 1.0
    ):
        raise ValueError(
            f"config: modeling.leak_alert_accuracy must be in (0.5, 1], got {leak_alert!r}"
        )

    seed = modeling["random_state"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"config: modeling.random_state must be an integer, got {seed!r}")

    modeling["logistic_regression"] = _merge_section(
        modeling["logistic_regression"], _LOGREG_DEFAULTS, "logistic_regression"
    )
    modeling["lightgbm"] = _merge_section(
        modeling["lightgbm"], _LIGHTGBM_DEFAULTS, "lightgbm"
    )
    modeling["pruning"] = _merge_section(
        modeling["pruning"], _PRUNING_DEFAULTS, "pruning"
    )
    min_fire = modeling["pruning"]["min_fire_rows"]
    if not isinstance(min_fire, int) or isinstance(min_fire, bool) or min_fire < 1:
        raise ValueError(
            f"config: modeling.pruning.min_fire_rows must be an integer >= 1, got {min_fire!r}"
        )

    return {**cfg, "models_dir": models_dir, "modeling": modeling}


# ---------------------------------------------------------------------------
# Dataset assembly (join + runtime alignment guard)
# ---------------------------------------------------------------------------


def assemble_dataset(
    interval: str, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Join features and labels for one interval, guarded against misalignment.

    Performs the rehearsed one-liner join
    (``features.merge(labels, on="open_time", how="left")``) and runs
    :func:`validate_feature_label_alignment` against the raw candles as a
    runtime guard **before** returning anything.

    Args:
        interval: Binance interval string, e.g. ``"1d"``.
        cfg: Config dict from :func:`load_modeling_config`.

    Returns:
        Tuple of (merged DataFrame, feature column names).  Feature columns
        are everything except ``open_time`` and the ``fwd_return_*`` /
        ``label_*`` columns from the label file — forward-looking columns
        must never enter the feature matrix.

    Raises:
        FileNotFoundError: If any input parquet is missing.
        AssertionError: If the alignment guard fails.
    """
    context = f"{cfg['symbol']} {interval}"
    processed = Path(cfg["processed_dir"])
    features_path = processed / f"features_{interval}.parquet"
    labels_path = processed / f"labels_{interval}.parquet"
    raw_path = Path(cfg["raw_dir"]) / f"{cfg['file_prefix']}_{interval}.parquet"
    for path, builder in (
        (features_path, "src.features.build_features"),
        (labels_path, "src.labels.build_labels"),
        (raw_path, "src.data.fetch_binance"),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{context}: {path} not found — run `python -m {builder}` first"
            )

    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    candles = pd.read_parquet(raw_path)

    horizon: int = cfg["modeling"]["horizon"]
    validate_feature_label_alignment(
        features,
        labels,
        raw_candles_df=candles,
        horizon=horizon,
        dead_zone_pct=cfg["labels"]["dead_zone_pct"],
        context=context,
    )

    merged = features.merge(labels, on="open_time", how="left")

    feature_cols = [
        col
        for col in merged.columns
        if col != "open_time"
        and not col.startswith("fwd_return_")
        and not col.startswith("label_")
    ]
    # Defence in depth: the alignment guard already enforces the label-file
    # namespace, but a forward-looking column in the feature matrix would be
    # a silent catastrophe, so re-assert it here.
    leaked = [c for c in feature_cols if c.startswith(("fwd_return_", "label_"))]
    assert not leaked, f"{context}: forward-looking columns in features: {leaked}"

    logger.info(
        "%s: assembled dataset — %d rows, %d feature columns, horizon=%d",
        context, len(merged), len(feature_cols), horizon,
    )
    return merged, feature_cols


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitBounds:
    """Positional (iloc) boundaries of the chronological split.

    Each split is the half-open row range ``[start, end)`` on the full
    joined table.  ``gap_candles`` rows are skipped between train and
    validation and between validation and test.
    """

    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    gap_candles: int


def make_split_bounds(
    n_rows: int,
    *,
    train_frac: float,
    val_frac: float,
    gap_candles: int,
    context: str = "",
) -> SplitBounds:
    """Define chronological train/val/test boundaries on the full table.

    Boundaries are purely positional and computed **before** any NaN row is
    dropped — dropping first would shift the boundaries and let the NaN
    pattern (which depends on labels, i.e. on future prices) influence which
    rows land in which split.

    Args:
        n_rows: Total row count of the full joined table.
        train_frac: Fraction of rows for training (chronologically first).
        val_frac: Fraction of rows for validation (after the first gap).
        gap_candles: Rows skipped at each boundary (>= modeled horizon).
        context: Label for error messages.

    Returns:
        A :class:`SplitBounds` with non-overlapping, ordered ranges.

    Raises:
        ValueError: If any resulting split would be empty.
    """
    label = context or "split"
    train_end = int(n_rows * train_frac)
    val_start = train_end + gap_candles
    val_end = val_start + int(n_rows * val_frac)
    test_start = val_end + gap_candles
    bounds = SplitBounds(
        train_start=0,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=n_rows,
        gap_candles=gap_candles,
    )
    if not (0 < train_end and val_start < val_end and test_start < n_rows):
        raise ValueError(
            f"{label}: split of {n_rows} rows produced an empty segment: {bounds}"
        )
    logger.info(
        "%s: train=[0, %d), gap=%d, val=[%d, %d), gap=%d, test=[%d, %d)",
        label, train_end, gap_candles, val_start, val_end,
        gap_candles, test_start, n_rows,
    )
    return bounds


def split_dataset(
    merged: pd.DataFrame,
    bounds: SplitBounds,
    *,
    feature_cols: list[str],
    label_col: str,
    context: str = "",
) -> dict[str, pd.DataFrame]:
    """Slice the joined table into splits, then drop unusable rows per split.

    NaN dropping happens **after** slicing: a row is unusable if any feature
    is NaN (warm-up) or the label is NaN (dead zone, tail).  Because the
    slice happens first, the drop can never move a row across a boundary.

    Args:
        merged: Full joined table from :func:`assemble_dataset`.
        bounds: Output of :func:`make_split_bounds` for ``len(merged)``.
        feature_cols: Feature column names.
        label_col: Label column name, e.g. ``"label_1"``.
        context: Label for log messages.

    Returns:
        Dict mapping ``"train"`` / ``"val"`` / ``"test"`` to the usable rows
        of that split (features + label + ``open_time``).

    Raises:
        ValueError: If any split has no usable rows left.
    """
    label = context or "split"
    ranges = {
        "train": (bounds.train_start, bounds.train_end),
        "val": (bounds.val_start, bounds.val_end),
        "test": (bounds.test_start, bounds.test_end),
    }
    keep_cols = ["open_time", *feature_cols, label_col]
    splits: dict[str, pd.DataFrame] = {}
    for name, (start, end) in ranges.items():
        segment = merged.iloc[start:end]
        usable = segment.dropna(subset=[*feature_cols, label_col])[keep_cols]
        if usable.empty:
            raise ValueError(f"{label}: {name} split has no usable rows")
        logger.info(
            "%s: %s rows [%d, %d): %d total, %d usable (%d dropped as NaN), "
            "%s -> %s",
            label, name, start, end, len(segment), len(usable),
            len(segment) - len(usable),
            usable["open_time"].iloc[0], usable["open_time"].iloc[-1],
        )
        splits[name] = usable
    return splits


# ---------------------------------------------------------------------------
# Scaling and model fitting
# ---------------------------------------------------------------------------


def fit_scaler_on_train(train_features: pd.DataFrame) -> StandardScaler:
    """Fit a :class:`StandardScaler` on the training feature matrix only.

    The scaler must never see validation or test rows — fitting on the full
    dataset would leak their means/variances into training.  Callers apply
    ``scaler.transform`` (never ``fit_transform``) to val/test.

    Args:
        train_features: Training-split feature matrix.

    Returns:
        The fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


def train_logistic_regression(
    train_features_scaled: pd.DataFrame,
    train_labels: pd.Series,
    *,
    params: dict[str, Any],
    random_state: int,
) -> LogisticRegression:
    """Fit the logistic-regression baseline on scaled training features.

    Args:
        train_features_scaled: Training features after the train-fit scaler.
        train_labels: Binary labels (1 = up, 0 = down).
        params: ``modeling.logistic_regression`` config section.
        random_state: Seed for reproducibility.

    Returns:
        The fitted model.
    """
    model = LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        random_state=random_state,
    )
    model.fit(train_features_scaled, train_labels)
    return model


def train_lightgbm(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: pd.DataFrame,
    val_labels: pd.Series,
    *,
    params: dict[str, Any],
    random_state: int,
) -> lgb.LGBMClassifier:
    """Fit the LightGBM classifier with early stopping on the validation split.

    Trees are scale-invariant, so LightGBM receives unscaled features.  The
    validation split is used only to pick the stopping round — standard model
    selection, never the test split.

    Args:
        train_features: Training-split feature matrix (unscaled).
        train_labels: Binary labels (1 = up, 0 = down).
        val_features: Validation-split feature matrix (unscaled).
        val_labels: Validation labels for early stopping.
        params: ``modeling.lightgbm`` config section.
        random_state: Seed for reproducibility.

    Returns:
        The fitted model.
    """
    fit_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    model = lgb.LGBMClassifier(
        **fit_params,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(
        train_features,
        train_labels,
        eval_X=val_features,
        eval_y=val_labels,
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(params["early_stopping_rounds"], verbose=False),
        ],
    )
    logger.info(
        "lightgbm: early stopping picked %s trees (cap %d)",
        model.best_iteration_, params["n_estimators"],
    )
    return model


# ---------------------------------------------------------------------------
# Candlestick feature pruning
# ---------------------------------------------------------------------------


def prune_candlestick_features(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_fire_rows: int,
    context: str = "",
) -> tuple[list[str], list[str]]:
    """Drop candlestick pattern columns that fire too rarely in the training split.

    A candlestick column fires when its value is != 0 (TA-Lib uses 0 for
    "pattern not detected").  Fire-rates are computed on ``train_df`` only —
    calling this function on the full dataset instead of the training split
    would leak future statistics into the feature-selection decision.

    Threshold justification (pre-committed, not tuned to any outcome):
    30 training occurrences — the "10 events per variable" clinical minimum
    scaled to 3×, well below which logistic-regression coefficient variance
    grows rapidly and rare binary dummies become noise amplifiers.

    Args:
        train_df: Training split DataFrame (features + optionally label).
            Only the ``cdl_*`` columns are read.
        feature_cols: Full feature column list from :func:`assemble_dataset`.
        min_fire_rows: Keep a pattern only if it fires (value != 0) in at
            least this many training rows.
        context: Label for log messages.

    Returns:
        Tuple of ``(kept_feature_cols, dropped_cdl_names)``:

        - ``kept_feature_cols``: ``feature_cols`` with rare candlestick
          patterns removed; non-candlestick features are never touched.
        - ``dropped_cdl_names``: candlestick column names that were removed.
    """
    label = context or "prune"
    cdl_cols = [c for c in feature_cols if c.startswith("cdl_")]
    if not cdl_cols:
        logger.info("%s: no candlestick columns found — nothing to prune", label)
        return list(feature_cols), []

    fire_counts = (train_df[cdl_cols] != 0).sum()
    dropped = sorted(fire_counts.index[fire_counts < min_fire_rows].tolist())
    dropped_set = set(dropped)

    kept_feature_cols = [c for c in feature_cols if c not in dropped_set]

    logger.info(
        "%s: candlestick pruning (min_fire_rows=%d): "
        "%d patterns total, dropping %d (< %d training fires), keeping %d",
        label, min_fire_rows, len(cdl_cols), len(dropped), min_fire_rows,
        len(cdl_cols) - len(dropped),
    )
    if dropped:
        drop_detail = ", ".join(
            f"{c}({int(fire_counts[c])})" for c in dropped
        )
        logger.info("%s: dropped: %s", label, drop_detail)

    return kept_feature_cols, dropped


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train_interval(interval: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the full training pipeline for one interval and persist artifacts.

    Order of operations (the whole point of this module):
    join + alignment guard -> split boundaries on the full table -> NaN drop
    per split -> scaler fit on train only -> fit both models -> save.

    Args:
        interval: Binance interval string, e.g. ``"1d"``.
        cfg: Config dict from :func:`load_modeling_config`.

    Returns:
        The training manifest that was written next to the artifacts.
    """
    context = f"{cfg['symbol']} {interval}"
    modeling = cfg["modeling"]
    horizon: int = modeling["horizon"]
    label_col = f"label_{horizon}"

    merged, feature_cols = assemble_dataset(interval, cfg)
    bounds = make_split_bounds(
        len(merged),
        train_frac=modeling["split"]["train_frac"],
        val_frac=modeling["split"]["val_frac"],
        gap_candles=modeling["split"]["gap_candles"],
        context=context,
    )
    splits = split_dataset(
        merged, bounds, feature_cols=feature_cols, label_col=label_col,
        context=context,
    )

    x_train = splits["train"][feature_cols]
    y_train = splits["train"][label_col].astype("int64")
    x_val = splits["val"][feature_cols]
    y_val = splits["val"][label_col].astype("int64")

    scaler = fit_scaler_on_train(x_train)
    x_train_scaled = pd.DataFrame(
        scaler.transform(x_train), columns=feature_cols, index=x_train.index
    )

    seed: int = modeling["random_state"]
    logreg = train_logistic_regression(
        x_train_scaled, y_train,
        params=modeling["logistic_regression"], random_state=seed,
    )
    booster = train_lightgbm(
        x_train, y_train, x_val, y_val,
        params=modeling["lightgbm"], random_state=seed,
    )

    out_dir = Path(cfg["models_dir"]) / interval
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "scaler": out_dir / "scaler.joblib",
        "logistic_regression": out_dir / "logistic_regression.joblib",
        "lightgbm": out_dir / "lightgbm.joblib",
    }
    joblib.dump(scaler, artifact_paths["scaler"])
    joblib.dump(logreg, artifact_paths["logistic_regression"])
    joblib.dump(booster, artifact_paths["lightgbm"])

    manifest: dict[str, Any] = {
        "symbol": cfg["symbol"],
        "interval": interval,
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "horizon": horizon,
        "label_col": label_col,
        "feature_cols": feature_cols,
        "n_rows_full": len(merged),
        "split_bounds": asdict(bounds),
        "usable_rows": {name: len(df) for name, df in splits.items()},
        "modeling_config": modeling,
        "lightgbm_best_iteration": booster.best_iteration_,
        "inputs": {
            "features": {
                "path": str(Path(cfg["processed_dir"]) / f"features_{interval}.parquet"),
                "sha256": sha256_of(
                    Path(cfg["processed_dir"]) / f"features_{interval}.parquet"
                ),
            },
            "labels": {
                "path": str(Path(cfg["processed_dir"]) / f"labels_{interval}.parquet"),
                "sha256": sha256_of(
                    Path(cfg["processed_dir"]) / f"labels_{interval}.parquet"
                ),
            },
        },
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "iron_rule": (
            "split boundaries on the full table before NaN drop; "
            "scaler fit on the training split only"
        ),
    }
    manifest_path = out_dir / "training_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    logger.info("%s: wrote %s and %d artifact(s)", context, manifest_path, len(artifact_paths))
    return manifest


def train_pruned_interval(interval: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the pruned training pipeline and persist artifacts to ``{interval}_pruned/``.

    Identical to :func:`train_interval` except that rare candlestick pattern
    columns are dropped **before** either model is fit.  The pruning decision
    uses the training split only (``modeling.pruning.min_fire_rows`` rows
    threshold, chosen before any val/test peek — see
    :func:`prune_candlestick_features`).

    Args:
        interval: Binance interval string, e.g. ``"1d"``.
        cfg: Config dict from :func:`load_modeling_config`.

    Returns:
        The training manifest that was written next to the artifacts.
    """
    context = f"{cfg['symbol']} {interval} (pruned)"
    modeling = cfg["modeling"]
    horizon: int = modeling["horizon"]
    label_col = f"label_{horizon}"
    min_fire_rows: int = modeling["pruning"]["min_fire_rows"]

    merged, feature_cols = assemble_dataset(interval, cfg)
    bounds = make_split_bounds(
        len(merged),
        train_frac=modeling["split"]["train_frac"],
        val_frac=modeling["split"]["val_frac"],
        gap_candles=modeling["split"]["gap_candles"],
        context=context,
    )
    # Split boundaries and NaN drop are identical to the base pipeline —
    # candlestick columns don't carry NaN so no rows shift across boundaries.
    splits = split_dataset(
        merged, bounds, feature_cols=feature_cols, label_col=label_col,
        context=context,
    )

    # Pruning uses training split only — call before ever touching val/test.
    pruned_feature_cols, dropped_patterns = prune_candlestick_features(
        splits["train"],
        feature_cols,
        min_fire_rows=min_fire_rows,
        context=context,
    )

    x_train = splits["train"][pruned_feature_cols]
    y_train = splits["train"][label_col].astype("int64")
    x_val = splits["val"][pruned_feature_cols]
    y_val = splits["val"][label_col].astype("int64")

    scaler = fit_scaler_on_train(x_train)
    x_train_scaled = pd.DataFrame(
        scaler.transform(x_train), columns=pruned_feature_cols, index=x_train.index
    )

    seed: int = modeling["random_state"]
    logreg = train_logistic_regression(
        x_train_scaled, y_train,
        params=modeling["logistic_regression"], random_state=seed,
    )
    booster = train_lightgbm(
        x_train, y_train, x_val, y_val,
        params=modeling["lightgbm"], random_state=seed,
    )

    out_dir = Path(cfg["models_dir"]) / f"{interval}_pruned"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "scaler": out_dir / "scaler.joblib",
        "logistic_regression": out_dir / "logistic_regression.joblib",
        "lightgbm": out_dir / "lightgbm.joblib",
    }
    joblib.dump(scaler, artifact_paths["scaler"])
    joblib.dump(logreg, artifact_paths["logistic_regression"])
    joblib.dump(booster, artifact_paths["lightgbm"])

    manifest: dict[str, Any] = {
        "symbol": cfg["symbol"],
        "interval": interval,
        "variant": "pruned",
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "horizon": horizon,
        "label_col": label_col,
        "feature_cols": pruned_feature_cols,
        "all_feature_cols": feature_cols,
        "dropped_cdl_patterns": dropped_patterns,
        "pruning_min_fire_rows": min_fire_rows,
        "n_rows_full": len(merged),
        "split_bounds": asdict(bounds),
        "usable_rows": {name: len(df) for name, df in splits.items()},
        "modeling_config": modeling,
        "lightgbm_best_iteration": booster.best_iteration_,
        "inputs": {
            "features": {
                "path": str(Path(cfg["processed_dir"]) / f"features_{interval}.parquet"),
                "sha256": sha256_of(
                    Path(cfg["processed_dir"]) / f"features_{interval}.parquet"
                ),
            },
            "labels": {
                "path": str(Path(cfg["processed_dir"]) / f"labels_{interval}.parquet"),
                "sha256": sha256_of(
                    Path(cfg["processed_dir"]) / f"labels_{interval}.parquet"
                ),
            },
        },
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "iron_rule": (
            "split boundaries on the full table before NaN drop; "
            "scaler fit on the training split only; "
            "candlestick fire-rates computed on the training split only"
        ),
    }
    manifest_path = out_dir / "training_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    logger.info("%s: wrote %s and %d artifact(s)", context, manifest_path, len(artifact_paths))
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Train models for every configured modeling interval.

    Args:
        argv: CLI arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 if any interval failed.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.models.train",
        description="Train directional models on the joined feature/label tables.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="path to the pipeline config (default: %(default)s)",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        metavar="INTERVAL",
        help="override modeling.intervals from the config",
    )
    parser.add_argument(
        "--pruned",
        action="store_true",
        help=(
            "run the candlestick-pruning experiment: drop rare CDL patterns "
            "(< modeling.pruning.min_fire_rows training fires) before fitting; "
            "saves to models/{interval}_pruned/"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    cfg = load_modeling_config(args.config)
    intervals: list[str] = args.intervals or cfg["modeling"]["intervals"]
    runner = train_pruned_interval if args.pruned else train_interval

    failures: list[str] = []
    for interval in intervals:
        try:
            runner(interval, cfg)
        except Exception:
            logger.exception("%s %s: training failed", cfg["symbol"], interval)
            failures.append(interval)

    if failures:
        logger.error("training failed for interval(s): %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
