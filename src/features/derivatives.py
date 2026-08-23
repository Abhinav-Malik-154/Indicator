"""Derivatives & cross-asset feature engineering (Task 2).

Iron rule (identical to the on-chain module, :mod:`src.features.onchain`):
the feature at candle T must only use data from T-1 and earlier.

The raw parquet (built by :mod:`src.data.fetch_derivatives`) holds, per UTC day:
  funding_sum / funding_mean / funding_count  — perp funding settlements
  x_dxy / x_spx / x_gold                       — cross-asset closes (weekday)

We apply a configurable 1-day lag by shifting the raw index forward, so every
rolling window is computed on already-lagged data and is automatically
backward-looking.  Cross-asset closes are forward-filled a few days first so
weekends/holidays don't punch holes in the daily calendar; **returns and
z-scores are computed after the lag**, never before.

All feature names carry the ``deriv_`` prefix so they are trivially separable
from ``onchain_`` and the base technical/candlestick features in any importance
table or ablation.

Features produced (leakage-safe, 1-day lagged):
  deriv_funding            — that day's total funding rate
  deriv_funding_z7 / _z30  — 7- and 30-day rolling z-scores of daily funding
  deriv_funding_sign       — sign of daily funding (+1 longs pay / -1 shorts pay)
  deriv_funding_cum7       — trailing 7-day cumulative funding
  deriv_{asset}_ret1       — 1-day % return of each cross-asset
  deriv_{asset}_z30        — 30-day rolling z-score of that return
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FUNDING_COL = "funding_sum"
_CROSS_ASSET_COLS = ["x_dxy", "x_spx", "x_gold"]


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    """Backward-looking rolling z-score (std==0 → NaN, never inf)."""
    roll_mean = series.rolling(window, min_periods=window).mean()
    roll_std = series.rolling(window, min_periods=window).std()
    return (series - roll_mean) / roll_std.replace(0.0, np.nan)


def build_derivatives_features(
    candle_df: pd.DataFrame,
    derivatives_path: Path | str,
    *,
    lag_days: int = 1,
    z_score_windows: list[int] | tuple[int, ...] = (7, 30),
    return_z_window: int = 30,
    max_forward_fill_days: int = 3,
    context: str = "",
) -> pd.DataFrame:
    """Build 1-day-lagged derivatives + cross-asset features on the candle index.

    Args:
        candle_df: Candle DataFrame with an ``open_time`` column (UTC midnight,
            sorted ascending).
        derivatives_path: Path to ``derivatives_{interval}.parquet``.
        lag_days: Days of lag to enforce (default 1 → candle T uses T-1 data).
        z_score_windows: Windows for the funding rolling z-scores.
        return_z_window: Window for the cross-asset return z-scores.
        max_forward_fill_days: Forward-fill cross-asset closes up to this many
            days to bridge weekends/holidays.  0 disables filling.
        context: Label for log messages.

    Returns:
        DataFrame with ``open_time`` plus one ``deriv_*`` column per feature.
        Warm-up / pre-coverage rows are NaN — never backfilled.

    Raises:
        FileNotFoundError: If ``derivatives_path`` is missing.
    """
    label = context or "derivatives features"
    derivatives_path = Path(derivatives_path)
    if not derivatives_path.is_file():
        raise FileNotFoundError(
            f"{label}: derivatives data not found at {derivatives_path} — "
            "run `python -m src.data.fetch_derivatives` first"
        )

    raw = pd.read_parquet(derivatives_path)
    if "date" not in raw.columns:
        raise ValueError(f"{label}: derivatives parquet missing 'date' column")
    raw = raw.sort_values("date").reset_index(drop=True)

    idx = pd.DatetimeIndex(raw["date"])
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    raw = raw.set_index(idx).drop(columns=["date"])
    raw = raw.sort_index()

    # Fill weekend/holiday gaps in the cross-asset closes (funding is daily).
    cross_present = [c for c in _CROSS_ASSET_COLS if c in raw.columns]
    if max_forward_fill_days > 0 and cross_present:
        raw[cross_present] = raw[cross_present].ffill(limit=max_forward_fill_days)

    # Apply the lag BEFORE any rolling/return transform so every window is
    # backward-looking: data tagged for day D becomes available at D + lag.
    raw.index = raw.index + pd.Timedelta(days=lag_days)

    parts: list[pd.Series] = []

    if _FUNDING_COL in raw.columns:
        funding = raw[_FUNDING_COL].astype("float64")
        parts.append(funding.rename("deriv_funding"))
        for window in z_score_windows:
            parts.append(_rolling_z(funding, window).rename(f"deriv_funding_z{window}"))
        parts.append(np.sign(funding).rename("deriv_funding_sign"))
        parts.append(
            funding.rolling(7, min_periods=7).sum().rename("deriv_funding_cum7")
        )
    else:
        logger.warning("%s: no funding column in %s", label, derivatives_path)

    for col in cross_present:
        ret1 = raw[col].astype("float64").pct_change() * 100.0
        parts.append(ret1.rename(f"deriv_{col}_ret1"))
        parts.append(
            _rolling_z(ret1, return_z_window).rename(f"deriv_{col}_z{return_z_window}")
        )

    if not parts:
        raise ValueError(f"{label}: no usable columns in {derivatives_path}")

    feat_df = pd.concat(parts, axis=1)
    feat_df.index.name = "open_time"
    feat_df = feat_df.reset_index()

    result = candle_df[["open_time"]].merge(feat_df, on="open_time", how="left")

    deriv_cols = [c for c in result.columns if c.startswith("deriv_")]
    n_nan_rows = int(result[deriv_cols].isna().any(axis=1).sum())
    logger.info(
        "%s: built %d derivatives features for %d candle rows; "
        "%d rows have at least one NaN (warm-up or pre-series coverage)",
        label, len(deriv_cols), len(result), n_nan_rows,
    )
    return result.reset_index(drop=True)


def validate_derivatives_no_lookahead(
    candle_df: pd.DataFrame,
    derivatives_path: Path | str,
    *,
    lag_days: int = 1,
    sample_points: list[int] | None = None,
    context: str = "",
    **feature_kwargs,
) -> None:
    """Assert derivatives features at row T don't change when day-T data is hidden.

    Correct lagging means the feature at candle T uses only derivatives data
    from ≤ T-lag_days.  We rebuild features with the raw parquet truncated to
    dates strictly before candle T and require the value at T to be identical.

    Args:
        candle_df: Candle DataFrame with ``open_time``.
        derivatives_path: Path to the derivatives parquet.
        lag_days: Lag used by :func:`build_derivatives_features`.
        sample_points: Row indices to test; default 10 evenly-spaced rows from
            the latter half of the series.
        context: Label for error messages.
        **feature_kwargs: Passed through to :func:`build_derivatives_features`
            (must match the production call).

    Raises:
        AssertionError: If any sampled feature changes under truncation.
    """
    import os
    import tempfile

    label = context or "derivatives no-lookahead"
    raw = pd.read_parquet(derivatives_path).sort_values("date").reset_index(drop=True)

    full = build_derivatives_features(
        candle_df, derivatives_path, lag_days=lag_days, **feature_kwargs
    )
    deriv_cols = [c for c in full.columns if c.startswith("deriv_")]

    if sample_points is None:
        mid = len(candle_df) // 2
        step = max(1, (len(candle_df) - mid) // 10)
        sample_points = list(range(mid, len(candle_df), step))[:10]

    raw_dates = pd.DatetimeIndex(pd.to_datetime(raw["date"], utc=True))
    for t in sample_points:
        candle_time_t = candle_df["open_time"].iloc[t]
        trunc = raw[raw_dates < candle_time_t].copy()
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            trunc.to_parquet(tmp_path, engine="pyarrow", index=False)
            trunc_feat = build_derivatives_features(
                candle_df, tmp_path, lag_days=lag_days, **feature_kwargs
            )
        finally:
            os.unlink(tmp_path)

        for col in deriv_cols:
            fv = full[col].iloc[t]
            tv = trunc_feat[col].iloc[t]
            if not np.isclose(fv, tv, equal_nan=True):
                raise AssertionError(
                    f"{label}: LOOKAHEAD at row {t} "
                    f"(open_time={candle_time_t.date()}) column '{col}': "
                    f"full={fv}, truncated={tv}"
                )

    logger.info(
        "%s: no-lookahead validated for %d sample points", label, len(sample_points)
    )
