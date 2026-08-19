"""Critical validation for forward-return labels.

Two categories of assertions that must hold before any label is trusted:

1. **Label derivation**: every label value is mathematically derived from
   exactly ``close[T]`` and ``close[T+N]``, with no other data involved.
   The last N rows are NaN because ``close[T+N]`` does not exist.

2. **Feature–label alignment**: when features and labels are joined on
   ``open_time``, the join is a simple left-join that neither duplicates
   rows nor shifts indices.  An off-by-one here is a classic silent leakage
   source — a label shifted one row back means the model sees the future
   price movement as if it were the current one.

These functions are called both from the test suite and (optionally) from
downstream training code as a runtime guard.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_label_derivation(
    candles: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    horizon: int,
    dead_zone_pct: float,
    context: str = "",
) -> None:
    """Assert that every label is correctly derived from forward returns.

    For each row T with a non-NaN label, recomputes the expected label from
    ``close[T]`` and ``close[T+horizon]`` and asserts an exact match.  Also
    asserts that the last ``horizon`` rows have NaN labels and NaN forward
    returns (no future data to derive them from).

    Args:
        candles: Raw candle DataFrame with ``open_time`` and ``close``.
        labels_df: Output of :func:`~src.labels.build_labels.compute_forward_return_labels`.
        horizon: Forward-return horizon.
        dead_zone_pct: Dead-zone width in percent.
        context: Label for error messages.

    Raises:
        AssertionError: On any mismatch.
    """
    label = context or "derivation check"
    ret_col = f"fwd_return_{horizon}"
    lbl_col = f"label_{horizon}"
    threshold = dead_zone_pct / 100.0

    assert len(candles) == len(labels_df), (
        f"{label}: candle and label row counts differ "
        f"({len(candles)} vs {len(labels_df)})"
    )

    close = candles["close"].values
    n = len(close)

    # --- Forward return check ---
    for t in range(n):
        actual_ret = labels_df[ret_col].iloc[t]
        if t + horizon >= n:
            # Last N rows: no future close, must be NaN.
            assert np.isnan(actual_ret), (
                f"{label}: row {t} should have NaN fwd_return (last {horizon} rows) "
                f"but got {actual_ret}"
            )
        else:
            expected_ret = close[t + horizon] / close[t] - 1.0
            assert np.isclose(actual_ret, expected_ret, atol=1e-12), (
                f"{label}: row {t} fwd_return mismatch: "
                f"expected {expected_ret}, got {actual_ret}"
            )

    # --- Label check ---
    # Use the same round-to-12dp that build_labels uses for classification.
    for t in range(n):
        actual_lbl = labels_df[lbl_col].iloc[t]
        fwd_ret = labels_df[ret_col].iloc[t]
        fwd_rounded = round(fwd_ret, 12) if not np.isnan(fwd_ret) else fwd_ret

        if t + horizon >= n:
            # Tail: must be NaN.
            assert np.isnan(actual_lbl), (
                f"{label}: row {t} label should be NaN (tail) but got {actual_lbl}"
            )
        elif fwd_rounded > threshold:
            assert actual_lbl == 1.0, (
                f"{label}: row {t} should be 1 (up, ret={fwd_ret:.6f}) "
                f"but got {actual_lbl}"
            )
        elif fwd_rounded < -threshold:
            assert actual_lbl == 0.0, (
                f"{label}: row {t} should be 0 (down, ret={fwd_ret:.6f}) "
                f"but got {actual_lbl}"
            )
        else:
            # Dead zone: must be NaN.
            assert np.isnan(actual_lbl), (
                f"{label}: row {t} should be NaN (dead zone, ret={fwd_ret:.6f}) "
                f"but got {actual_lbl}"
            )

    logger.info("%s: all %d labels verified (horizon=%d)", label, n, horizon)


def validate_alignment(
    features_path: str,
    labels_path: str,
    *,
    raw_candles_path: str,
    horizon: int,
    dead_zone_pct: float,
    context: str = "",
) -> None:
    """Assert that a feature–label join on ``open_time`` is clean.

    Loads both parquets, performs a left join (features is left), and checks:

    * The join produces exactly ``len(features)`` rows (no duplication).
    * Row 0, row 1, and the last valid labelled row have correct alignment:
      the ``open_time`` matches and the label corresponds to the correct
      forward return hand-computed from the raw candles.
    * No feature column name appears in the label columns (namespace
      separation).
    * The label file contains only ``open_time``, ``fwd_return_*``, and
      ``label_*`` columns.

    Args:
        features_path: Path to ``features_{interval}.parquet``.
        labels_path: Path to ``labels_{interval}.parquet``.
        raw_candles_path: Path to the raw candle parquet for hand-checking.
        horizon: A horizon present in the labels to spot-check.
        dead_zone_pct: Dead-zone width in percent.
        context: Label for error messages.

    Raises:
        AssertionError: On any alignment or namespace violation.
    """
    label = context or "alignment check"
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    candles = pd.read_parquet(raw_candles_path)

    # --- Namespace separation ---
    feat_cols = set(features.columns) - {"open_time"}
    lbl_cols = set(labels.columns) - {"open_time"}
    overlap = feat_cols & lbl_cols
    assert not overlap, (
        f"{label}: feature and label columns overlap: {sorted(overlap)}"
    )
    for col in lbl_cols:
        assert col.startswith("fwd_return_") or col.startswith("label_"), (
            f"{label}: unexpected label column '{col}' — "
            "label file should only contain open_time, fwd_return_*, label_*"
        )

    # --- Join check ---
    merged = features.merge(labels, on="open_time", how="left")
    assert len(merged) == len(features), (
        f"{label}: left join produced {len(merged)} rows from {len(features)} "
        "feature rows — duplication or expansion detected"
    )

    # --- Spot-check specific rows ---
    threshold = dead_zone_pct / 100.0
    ret_col = f"fwd_return_{horizon}"
    lbl_col_name = f"label_{horizon}"

    # Check that all label columns made it into the merge.
    assert ret_col in merged.columns, (
        f"{label}: {ret_col} missing from merged DataFrame"
    )
    assert lbl_col_name in merged.columns, (
        f"{label}: {lbl_col_name} missing from merged DataFrame"
    )

    close = candles["close"].values
    candle_times = candles["open_time"].values
    merged_times = merged["open_time"].values

    for row_idx in (0, 1):
        # First two rows: verify alignment by timestamp and value.
        assert merged_times[row_idx] == candle_times[row_idx], (
            f"{label}: row {row_idx} open_time mismatch after join"
        )
        if row_idx + horizon < len(close):
            expected_ret = close[row_idx + horizon] / close[row_idx] - 1.0
            actual_ret = merged[ret_col].iloc[row_idx]
            assert np.isclose(actual_ret, expected_ret, atol=1e-12), (
                f"{label}: row {row_idx} fwd_return mismatch after join: "
                f"expected {expected_ret}, got {actual_ret}"
            )

    # Last valid labelled row: index n - horizon - 1 (0-indexed).
    last_valid_idx = len(candles) - horizon - 1
    if last_valid_idx >= 0 and last_valid_idx < len(merged):
        assert merged_times[last_valid_idx] == candle_times[last_valid_idx], (
            f"{label}: last valid row ({last_valid_idx}) open_time mismatch"
        )
        expected_ret = close[last_valid_idx + horizon] / close[last_valid_idx] - 1.0
        actual_ret = merged[ret_col].iloc[last_valid_idx]
        assert np.isclose(actual_ret, expected_ret, atol=1e-12), (
            f"{label}: last valid row ({last_valid_idx}) fwd_return mismatch"
        )
        actual_lbl = merged[lbl_col_name].iloc[last_valid_idx]
        if expected_ret > threshold:
            assert actual_lbl == 1.0, f"{label}: last valid row should be up"
        elif expected_ret < -threshold:
            assert actual_lbl == 0.0, f"{label}: last valid row should be down"
        else:
            assert np.isnan(actual_lbl), f"{label}: last valid row should be dead zone"

    # Tail rows after last valid must be NaN.
    for t in range(len(candles) - horizon, len(candles)):
        if t < len(merged):
            assert np.isnan(merged[ret_col].iloc[t]), (
                f"{label}: tail row {t} should have NaN fwd_return"
            )
            assert np.isnan(merged[lbl_col_name].iloc[t]), (
                f"{label}: tail row {t} should have NaN label"
            )

    logger.info(
        "%s: alignment verified — %d rows, no duplication, "
        "spot-checked rows 0, 1, and %d",
        label, len(merged), last_valid_idx,
    )
