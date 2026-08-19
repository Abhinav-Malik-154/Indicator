"""Leakage-safe technical features computed from OHLCV candles.

Every feature at row T is a function of rows T and earlier only: returns look
back a fixed number of rows, rolling statistics use trailing windows ending at
T, and intra-candle shape features use row T alone. Nothing here shifts data
backward in time, uses centered windows, or normalises with full-series
statistics — scalers belong to the modelling phase, fit on training data only.

Rolling windows are positional (counted in candles), not wall-clock: at a
recorded exchange outage a window simply spans the gap instead of shrinking.
Rows whose lookback covers a gap are counted and reported by the build step
(see ``src.features.build_features``); candles are never synthesised to close
gaps.

NaN policy: rows without enough rolling history keep NaN — no backfill, which
would leak information backward in time. Zero-range candles (high == low) make
the intra-candle shape ratios undefined; those emit NaN as well.

Phase 2 additions: RSI (Wilder's smoothed RS) and MACD (fast/slow/signal
EMAs). Both use ``adjust=False`` exponential smoothing which is strictly
causal — each output depends only on previous outputs and the current input.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.features.common import validate_ohlcv_input

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


def validate_window_list(name: str, values: Sequence[int]) -> None:
    """Require a non-empty sequence of integers >= 1."""
    ok = (
        isinstance(values, (list, tuple))
        and len(values) > 0
        and all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in values)
    )
    if not ok:
        raise ValueError(
            f"features.{name} must be a non-empty list of integers >= 1, got {values!r}"
        )


def validate_window(name: str, value: int, minimum: int) -> None:
    """Require a single integer window >= ``minimum``."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"features.{name} must be an integer >= {minimum}, got {value!r}")


def _macd_defaults() -> dict[str, int]:
    """Default MACD parameters (fast/slow/signal EMA spans)."""
    return {"fast_period": 12, "slow_period": 26, "signal_period": 9}


def validate_macd_config(macd: Any) -> dict[str, int]:
    """Validate and normalise the ``macd`` config section.

    Args:
        macd: Raw value from config (expected to be a mapping).

    Returns:
        Dict with ``fast_period``, ``slow_period``, ``signal_period``.

    Raises:
        ValueError: On invalid or missing keys.
    """
    defaults = _macd_defaults()
    if macd is None:
        return dict(defaults)
    if not isinstance(macd, Mapping):
        raise ValueError(
            f"features.macd must be a mapping, got {type(macd).__name__}"
        )
    unknown = sorted(set(macd) - set(defaults))
    if unknown:
        raise ValueError(f"features.macd: unknown key(s) {unknown}")
    merged = {**defaults, **macd}
    for key in ("fast_period", "slow_period", "signal_period"):
        validate_window(f"macd.{key}", merged[key], minimum=2)
    if merged["fast_period"] >= merged["slow_period"]:
        raise ValueError(
            f"features.macd: fast_period ({merged['fast_period']}) must be "
            f"< slow_period ({merged['slow_period']})"
        )
    return merged


def longest_lookback_rows(
    *,
    return_periods: Sequence[int],
    volatility_windows: Sequence[int],
    volume_window: int,
    ma_windows: Sequence[int],
    sr_window: int,
    rsi_period: int = 14,
    macd: dict[str, int] | None = None,
) -> int:
    """Longest trailing span, in rows including the current one, any feature uses.

    The ``+1`` terms: a p-period return at row T needs the close at row T-p,
    and a std over w one-period returns reaches w+1 closes back.

    RSI warm-up: the Wilder EMA needs ``rsi_period`` one-period returns before
    producing its first value, so the span is ``rsi_period + 1``.

    MACD warm-up: the slow EMA needs ``slow_period`` closes, then the signal
    EMA needs ``signal_period`` MACD values, so the total span is
    ``slow_period + signal_period``.

    Args:
        return_periods: Log-return lookback periods.
        volatility_windows: Rolling windows for return std / range means.
        volume_window: Rolling window for volume mean and z-score.
        ma_windows: Rolling windows for close-vs-mean context features.
        sr_window: Rolling window for the support/resistance distances.
        rsi_period: RSI lookback period.
        macd: MACD parameters dict (fast/slow/signal).

    Returns:
        Maximum lookback span in rows.
    """
    m = macd or _macd_defaults()
    return max(
        max(return_periods) + 1,
        max(volatility_windows) + 1,
        volume_window,
        max(ma_windows),
        sr_window,
        rsi_period + 1,
        m["slow_period"] + m["signal_period"],
    )


def compute_technical_features(
    df: pd.DataFrame,
    *,
    return_periods: Sequence[int],
    volatility_windows: Sequence[int],
    volume_window: int,
    ma_windows: Sequence[int],
    sr_window: int,
    rsi_period: int = 14,
    macd: dict[str, int] | None = None,
    context: str = "",
) -> pd.DataFrame:
    """Compute the technical feature families for a candle DataFrame.

    Families (all trailing, all causal):

    * returns: ``log_ret_{p}`` for each configured period
    * volatility: ``ret_std_{w}`` (std of 1-period log returns), ``range_pct``
      and its rolling means ``range_pct_ma_{w}``
    * volume: ``volume_vs_ma_{v}`` and ``volume_z_{v}``
    * context: ``close_vs_ma_{w}``, ``dist_from_high_{s}`` / ``dist_from_low_{s}``
      against the rolling extreme of highs/lows (support/resistance proxy)
    * intra-candle: ``body_pct``, ``upper_wick_pct``, ``lower_wick_pct``,
      ``close_pos_in_range``
    * momentum: ``rsi_{period}`` (Wilder's smoothed Relative Strength Index)
    * trend: ``macd_line``, ``macd_signal``, ``macd_hist`` (MACD from EMAs)

    Args:
        df: Candle DataFrame with ``REQUIRED_COLUMNS``, sorted by ``open_time``.
        return_periods: Log-return lookback periods, e.g. ``[1, 3, 7, 14]``.
        volatility_windows: Rolling windows for return std / range means.
        volume_window: Rolling window for volume mean and z-score (>= 2).
        ma_windows: Rolling windows for close-vs-mean context features.
        sr_window: Rolling window for support/resistance distances (>= 2).
        rsi_period: RSI lookback period (>= 2).
        macd: MACD parameters dict with ``fast_period``, ``slow_period``,
            ``signal_period``. Defaults to 12/26/9.
        context: Label such as ``"BTCUSDT 1h"`` used in log messages.

    Returns:
        Float DataFrame aligned to ``df``'s index, one column per feature.
        Rows without full rolling history hold NaN by design.

    Raises:
        ValueError: On invalid windows or invalid input data.
    """
    label = context or "technical"
    validate_window_list("return_periods", return_periods)
    validate_window_list("volatility_windows", volatility_windows)
    validate_window_list("ma_windows", ma_windows)
    validate_window("volume_window", volume_window, minimum=2)
    validate_window("sr_window", sr_window, minimum=2)
    validate_window("rsi_period", rsi_period, minimum=2)
    macd_cfg = validate_macd_config(macd)
    validate_ohlcv_input(df, REQUIRED_COLUMNS, label)

    open_, high, low, close, volume = (
        df[col] for col in ("open", "high", "low", "close", "volume")
    )
    feats: dict[str, pd.Series] = {}

    # Returns (momentum family).
    for p in return_periods:
        feats[f"log_ret_{p}"] = np.log(close / close.shift(p))

    # Volatility: rolling std of 1-period log returns (computed independently
    # of the configured periods) plus high-low range features.
    log_ret_1 = np.log(close / close.shift(1))
    for w in volatility_windows:
        feats[f"ret_std_{w}"] = log_ret_1.rolling(w).std()
    candle_range = high - low
    feats["range_pct"] = candle_range / close
    for w in volatility_windows:
        feats[f"range_pct_ma_{w}"] = feats["range_pct"].rolling(w).mean()

    # Volume vs its own recent history. A zero rolling std (constant volume)
    # leaves the z-score undefined -> NaN, never a substituted value.
    volume_ma = volume.rolling(volume_window).mean()
    volume_std = volume.rolling(volume_window).std()
    feats[f"volume_vs_ma_{volume_window}"] = volume / volume_ma.where(volume_ma > 0)
    feats[f"volume_z_{volume_window}"] = (volume - volume_ma) / volume_std.where(volume_std > 0)

    # Context: close relative to its rolling mean and to recent extremes.
    for w in ma_windows:
        feats[f"close_vs_ma_{w}"] = close / close.rolling(w).mean() - 1.0
    feats[f"dist_from_high_{sr_window}"] = close / high.rolling(sr_window).max() - 1.0
    feats[f"dist_from_low_{sr_window}"] = close / low.rolling(sr_window).min() - 1.0

    # Intra-candle shape. Undefined (NaN) for zero-range candles.
    range_safe = candle_range.where(candle_range > 0)
    feats["body_pct"] = (close - open_).abs() / range_safe
    feats["upper_wick_pct"] = (high - np.maximum(open_, close)) / range_safe
    feats["lower_wick_pct"] = (np.minimum(open_, close) - low) / range_safe
    feats["close_pos_in_range"] = (close - low) / range_safe

    # RSI: Wilder's smoothed Relative Strength Index.
    # Uses adjust=False EMA with alpha = 1/period, which is equivalent to
    # Wilder's running smoothing.  Strictly causal: each value depends only
    # on the current one-period return and the previous smoothed value.
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    # Standard RSI formula edge cases: avg_loss=0 → RSI=100 (pure gains),
    # both zero → RSI=50 (neutral).  Row 0 stays NaN (no delta).
    rsi = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = avg_gain.notna() & avg_loss.notna()
    both_zero = valid & (avg_gain == 0) & (avg_loss == 0)
    loss_zero = valid & (avg_loss == 0) & (avg_gain > 0)
    normal = valid & (avg_loss > 0)
    rsi[both_zero] = 50.0
    rsi[loss_zero] = 100.0
    rsi[normal] = 100.0 - 100.0 / (1.0 + avg_gain[normal] / avg_loss[normal])
    feats[f"rsi_{rsi_period}"] = rsi

    # MACD: Moving Average Convergence Divergence.
    # Two EMAs of close (fast/slow) produce the MACD line; a signal EMA of
    # the MACD line gives the signal; histogram = line - signal.  All EMAs
    # use span-based smoothing with adjust=False (strictly causal).
    fast_p = macd_cfg["fast_period"]
    slow_p = macd_cfg["slow_period"]
    sig_p = macd_cfg["signal_period"]
    ema_fast = close.ewm(span=fast_p, adjust=False).mean()
    ema_slow = close.ewm(span=slow_p, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=sig_p, adjust=False).mean()
    feats["macd_line"] = macd_line
    feats["macd_signal"] = macd_signal
    feats["macd_hist"] = macd_line - macd_signal

    out = pd.DataFrame(feats, index=df.index)
    logger.info(
        "%s: computed %d technical feature(s) over %d row(s)", label, out.shape[1], len(out)
    )
    return out
