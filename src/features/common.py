"""Shared input validation for feature modules."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def validate_ohlcv_input(df: pd.DataFrame, required: Sequence[str], context: str = "") -> None:
    """Check that a candle DataFrame is usable as feature input.

    Feature computations assume candles are complete, ordered, and unique;
    a violation here means upstream data is corrupt, so we fail loudly rather
    than compute silently wrong features.

    Args:
        df: Candle DataFrame to validate.
        required: Column names that must be present (must include ``open_time``).
        context: Label used in error messages, e.g. ``"BTCUSDT 1h"``.

    Raises:
        ValueError: If the frame is empty, misses columns, or ``open_time`` is
            not strictly increasing.
    """
    label = context or "feature input"
    if df.empty:
        raise ValueError(f"{label}: input DataFrame is empty")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required column(s) {missing}")
    deltas = df["open_time"].diff().iloc[1:]
    if (deltas <= pd.Timedelta(0)).any():
        raise ValueError(f"{label}: open_time must be strictly increasing and unique")
