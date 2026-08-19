"""Critical validation for leakage-safe features.

Two categories of assertions:

1. **No-lookahead**: every feature at row T must produce the same value
   whether the full series is available or only data up to row T.  This
   proves the feature is causal — it does not depend on future rows.

2. **Feature–label alignment**: when features and labels are joined on
   ``open_time``, the join must not duplicate rows, shift indices, or
   introduce overlapping column names.

These functions are callable from both the test suite and downstream
training code as a runtime guard.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_no_lookahead(
    df: pd.DataFrame,
    build_fn: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    sample_points: list[int] | None = None,
    context: str = "",
) -> None:
    """Assert that features at row T are unchanged when future rows are removed.

    For each sample point T, truncates ``df`` to ``df.iloc[:T+1]``, rebuilds
    features via ``build_fn``, and asserts the feature row at T is identical
    to the full-build row at T.

    This is the definitive leakage test: if a feature uses ``shift(-1)`` or a
    centered window, the truncated build will differ because the future data
    that the full build relied on is absent.

    Args:
        df: Raw candle DataFrame (sorted by ``open_time``).
        build_fn: A callable that takes a candle DataFrame and returns a
            feature DataFrame (same index, one column per feature).
        sample_points: Row indices to check (0-based).  Defaults to every
            row from 5 to ``len(df)-1``.
        context: Label for error messages.

    Raises:
        AssertionError: If any feature value at a sample point differs
            between the full build and the truncated build.
    """
    label = context or "no-lookahead check"
    full = build_fn(df)

    if sample_points is None:
        sample_points = list(range(5, len(df)))

    for t in sample_points:
        prefix = build_fn(df.iloc[: t + 1])
        try:
            pd.testing.assert_frame_equal(
                full.iloc[[t]],
                prefix.iloc[[t]],
                check_names=False,
            )
        except AssertionError as exc:
            # Identify which column(s) differ for a clear error message.
            full_row = full.iloc[t]
            prefix_row = prefix.iloc[t]
            diffs = []
            for col in full.columns:
                fv = full_row[col]
                pv = prefix_row[col]
                both_nan = (
                    isinstance(fv, float)
                    and isinstance(pv, float)
                    and np.isnan(fv)
                    and np.isnan(pv)
                )
                if not both_nan and fv != pv:
                    diffs.append(f"{col}: full={fv}, truncated={pv}")
            diff_str = "; ".join(diffs) if diffs else str(exc)
            raise AssertionError(
                f"{label}: LOOKAHEAD DETECTED at row {t} — "
                f"feature value changed when future rows were removed. "
                f"Differences: {diff_str}"
            ) from exc

    logger.info(
        "%s: no-lookahead validated for %d sample point(s)",
        label,
        len(sample_points),
    )


def validate_feature_label_alignment(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    raw_candles_df: pd.DataFrame,
    horizon: int,
    dead_zone_pct: float,
    context: str = "",
) -> None:
    """Assert that a feature–label join on ``open_time`` is clean.

    Checks:

    * No duplicate ``open_time`` in either table.
    * Left join produces exactly ``len(features)`` rows (no expansion).
    * No overlapping column names (namespace separation).
    * Label columns are only ``open_time``, ``fwd_return_*``, ``label_*``.
    * Spot-check rows 0, 1, and last-valid against raw candles.

    Args:
        features_df: Feature DataFrame with ``open_time``.
        labels_df: Label DataFrame with ``open_time``.
        raw_candles_df: Raw candle DataFrame for hand-checking.
        horizon: A horizon present in the labels to spot-check.
        dead_zone_pct: Dead-zone width in percent.
        context: Label for error messages.

    Raises:
        AssertionError: On any alignment or namespace violation.
    """
    label = context or "feature-label alignment"

    # --- No duplicates ---
    assert not features_df["open_time"].duplicated().any(), (
        f"{label}: features contain duplicate open_time values"
    )
    assert not labels_df["open_time"].duplicated().any(), (
        f"{label}: labels contain duplicate open_time values"
    )

    # --- Namespace separation ---
    feat_cols = set(features_df.columns) - {"open_time"}
    lbl_cols = set(labels_df.columns) - {"open_time"}
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
    merged = features_df.merge(labels_df, on="open_time", how="left")
    assert len(merged) == len(features_df), (
        f"{label}: left join produced {len(merged)} rows from "
        f"{len(features_df)} feature rows — duplication detected"
    )

    # --- Spot-check specific rows ---
    ret_col = f"fwd_return_{horizon}"
    lbl_col_name = f"label_{horizon}"
    threshold = dead_zone_pct / 100.0

    assert ret_col in merged.columns, (
        f"{label}: {ret_col} missing from merged DataFrame"
    )
    assert lbl_col_name in merged.columns, (
        f"{label}: {lbl_col_name} missing from merged DataFrame"
    )

    close = raw_candles_df["close"].values
    candle_times = raw_candles_df["open_time"].values
    merged_times = merged["open_time"].values

    for row_idx in (0, 1):
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

    # Last valid labelled row.
    last_valid_idx = len(raw_candles_df) - horizon - 1
    if 0 <= last_valid_idx < len(merged):
        assert merged_times[last_valid_idx] == candle_times[last_valid_idx], (
            f"{label}: last valid row ({last_valid_idx}) open_time mismatch"
        )
        expected_ret = (
            close[last_valid_idx + horizon] / close[last_valid_idx] - 1.0
        )
        actual_ret = merged[ret_col].iloc[last_valid_idx]
        assert np.isclose(actual_ret, expected_ret, atol=1e-12), (
            f"{label}: last valid row ({last_valid_idx}) fwd_return mismatch"
        )
        actual_lbl = merged[lbl_col_name].iloc[last_valid_idx]
        if expected_ret > threshold:
            assert actual_lbl == 1.0, (
                f"{label}: last valid row should be up"
            )
        elif expected_ret < -threshold:
            assert actual_lbl == 0.0, (
                f"{label}: last valid row should be down"
            )
        else:
            assert np.isnan(actual_lbl), (
                f"{label}: last valid row should be dead zone"
            )

    # Tail rows after last valid must be NaN.
    for t in range(len(raw_candles_df) - horizon, len(raw_candles_df)):
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
        label,
        len(merged),
        last_valid_idx,
    )


def make_leaked_feature(
    series: pd.Series,
    feature_name: str = "leaked",
) -> pd.DataFrame:
    """Create a deliberately leaked feature for testing.

    Shifts the input series by -1 so that each row contains the *next*
    row's value — a textbook lookahead leak.  The last row becomes NaN.

    This function exists **solely** to prove that
    :func:`validate_no_lookahead` catches leakage.  It must never be used
    in production feature code.

    Args:
        series: A numeric Series (e.g. ``close``).
        feature_name: Column name for the leaked feature.

    Returns:
        Single-column DataFrame aligned to ``series``'s index.
    """
    return pd.DataFrame(
        {feature_name: series.shift(-1).values},
        index=series.index,
    )
