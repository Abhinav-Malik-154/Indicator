"""On-chain feature engineering for the Phase 6 experiment.

Iron rule: feature at candle T must only use data from T-1 and earlier.

On-chain data for day D is finalized at midnight UTC end of day D.  We apply
a conservative 1-day lag (configurable): the feature at candle T draws from
on-chain data through D = T-1.  Rolling z-scores and WoW changes are computed
AFTER the lag shift so every lookback window respects it automatically.

Feature names carry the ``onchain_`` prefix to make namespace separation
trivially auditable in any importance table.

For each metric (active_addresses, n_transactions, hash_rate_th, fees_usd,
volume_usd) the following features are produced:
  onchain_{metric}_z7   — 7-day rolling z-score
  onchain_{metric}_z30  — 30-day rolling z-score
  onchain_{metric}_wow  — week-over-week % change (day vs day-7)

Total: 5 metrics × 3 transformations = 15 on-chain features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_METRIC_COLS = [
    "active_addresses",
    "n_transactions",
    "hash_rate_th",
    "fees_usd",
    "volume_usd",
]


def build_onchain_features(
    candle_df: pd.DataFrame,
    onchain_path: Path | str,
    *,
    lag_days: int = 1,
    z_score_windows: list[int] | tuple[int, ...] = (7, 30),
    wow_window: int = 7,
    max_forward_fill_days: int = 3,
    context: str = "",
) -> pd.DataFrame:
    """Build on-chain features aligned to the candle open_time index.

    Applies a ``lag_days`` shift so candle T always uses on-chain data from
    at most T-1.  Rolling statistics are computed on the lagged series so
    every window is automatically backward-looking.

    Args:
        candle_df: Raw candle DataFrame with an ``open_time`` column
            (UTC midnight timestamps, sorted ascending).
        onchain_path: Path to ``btc_onchain_1d.parquet``.
        lag_days: Days of lag to enforce (default: 1 = use T-1 data for candle T).
        z_score_windows: Windows for rolling z-score normalisation.
        wow_window: Window for week-over-week % change (default: 7 = same day
            last week).
        max_forward_fill_days: Forward-fill up to this many days for small API
            publication delays.  0 = no fill.
        context: Label for log messages.

    Returns:
        DataFrame with ``open_time`` column plus one ``onchain_*`` column per
        feature.  Rows that predate the on-chain series or fall into the
        lag warm-up period have NaN values — never backfilled.

    Raises:
        FileNotFoundError: If ``onchain_path`` does not exist.
    """
    label = context or "onchain features"
    onchain_path = Path(onchain_path)
    if not onchain_path.is_file():
        raise FileNotFoundError(
            f"{label}: on-chain data not found at {onchain_path} — "
            "run `python -m src.data.fetch_onchain` first"
        )

    raw = pd.read_parquet(onchain_path)
    if "date" not in raw.columns:
        raise ValueError(f"{label}: on-chain parquet missing 'date' column")
    raw = raw.sort_values("date").reset_index(drop=True)

    present_cols = [c for c in _METRIC_COLS if c in raw.columns]
    absent = [c for c in _METRIC_COLS if c not in raw.columns]
    if absent:
        logger.warning("%s: missing metric columns: %s", label, absent)

    oc = raw.set_index("date")[present_cols].copy()
    oc.index = pd.DatetimeIndex(oc.index).tz_localize("UTC") if oc.index.tz is None else oc.index
    oc = oc.sort_index()

    # Fill small gaps (e.g. API publishing delay or weekend stalls)
    if max_forward_fill_days > 0:
        oc = oc.ffill(limit=max_forward_fill_days)

    # Apply lag: shift index forward so T-data is tagged as available at T+lag_days.
    # After this, candle with open_time D merges with on-chain data from D-lag_days.
    oc.index = oc.index + pd.Timedelta(days=lag_days)

    parts: list[pd.Series] = []
    for col in present_cols:
        series = oc[col]
        for window in z_score_windows:
            roll_mean = series.rolling(window, min_periods=window).mean()
            roll_std = series.rolling(window, min_periods=window).std()
            z = (series - roll_mean) / roll_std.replace(0.0, np.nan)
            parts.append(z.rename(f"onchain_{col}_z{window}"))

        wow_pct = (series / series.shift(wow_window) - 1.0) * 100.0
        parts.append(wow_pct.rename(f"onchain_{col}_wow"))

    feat_df = pd.concat(parts, axis=1)
    feat_df.index.name = "open_time"
    feat_df = feat_df.reset_index()

    result = candle_df[["open_time"]].merge(feat_df, on="open_time", how="left")

    onchain_cols = [c for c in result.columns if c.startswith("onchain_")]
    n_nan_rows = int(result[onchain_cols].isna().any(axis=1).sum())
    logger.info(
        "%s: built %d on-chain features for %d candle rows; "
        "%d rows have at least one NaN (lag warm-up or pre-series coverage)",
        label,
        len(onchain_cols),
        len(result),
        n_nan_rows,
    )
    return result.reset_index(drop=True)


def validate_onchain_no_lookahead(
    candle_df: pd.DataFrame,
    onchain_path: Path | str,
    *,
    lag_days: int = 1,
    z_score_windows: list[int] | tuple[int, ...] = (7, 30),
    wow_window: int = 7,
    sample_points: list[int] | None = None,
    context: str = "",
) -> None:
    """Assert on-chain features at row T are unchanged when row T is absent.

    A correctly lagged series uses only on-chain data from ≤ T-lag_days for
    candle T.  This check verifies by:

    1. Building full features.
    2. For each sample point T: building features with on-chain data truncated
       to dates strictly before candle_df.open_time[T] (i.e. ≤ T-1 on 1-day
       lag).  The feature value at T must be identical in both builds.

    Args:
        candle_df: Raw candle DataFrame.
        onchain_path: Path to ``btc_onchain_1d.parquet``.
        lag_days: Lag to test (must match what :func:`build_onchain_features`
            uses).
        z_score_windows: Must match :func:`build_onchain_features`.
        wow_window: Must match :func:`build_onchain_features`.
        sample_points: Row indices to test; defaults to 10 evenly-spaced rows
            from the latter half of the series.
        context: Label for error messages.

    Raises:
        AssertionError: If any feature at a sample point changes when only
            data up to that point is available.
    """
    label = context or "onchain no-lookahead"
    raw_oc = pd.read_parquet(onchain_path)
    raw_oc = raw_oc.sort_values("date").reset_index(drop=True)

    full_features = build_onchain_features(
        candle_df,
        onchain_path,
        lag_days=lag_days,
        z_score_windows=z_score_windows,
        wow_window=wow_window,
    )
    onchain_cols = [c for c in full_features.columns if c.startswith("onchain_")]

    if sample_points is None:
        mid = len(candle_df) // 2
        step = max(1, (len(candle_df) - mid) // 10)
        sample_points = list(range(mid, len(candle_df), step))[:10]

    import tempfile, os

    for t in sample_points:
        candle_time_t = candle_df["open_time"].iloc[t]
        # Key question: does feature at T depend on on-chain data for day T or later?
        # Correct lag means: feature at T uses only data from day T-1 and earlier.
        # So: removing day-T (and later) on-chain data must NOT change feature at T.
        # Cutoff = candle_time_t removes data for day T and later; correct lag passes.
        cutoff = candle_time_t
        oc_trunc = raw_oc[raw_oc["date"] < cutoff].copy()

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            oc_trunc.to_parquet(tmp_path, engine="pyarrow", index=False)
            trunc_features = build_onchain_features(
                candle_df,
                tmp_path,
                lag_days=lag_days,
                z_score_windows=z_score_windows,
                wow_window=wow_window,
            )
        finally:
            os.unlink(tmp_path)

        full_row = full_features[onchain_cols].iloc[t]
        trunc_row = trunc_features[onchain_cols].iloc[t]

        for col in onchain_cols:
            fv = full_row[col]
            tv = trunc_row[col]
            both_nan = isinstance(fv, float) and isinstance(tv, float) and np.isnan(fv) and np.isnan(tv)
            if not both_nan and not np.isclose(fv, tv, equal_nan=True):
                raise AssertionError(
                    f"{label}: LOOKAHEAD at row {t} (open_time={candle_time_t.date()}) "
                    f"column '{col}': full={fv}, truncated={tv}"
                )

    logger.info(
        "%s: no-lookahead validated for %d sample points", label, len(sample_points)
    )
