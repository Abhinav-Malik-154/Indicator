"""Candlestick pattern features from TA-Lib's CDL* family.

Applies every TA-Lib pattern-recognition function to OHLC data, one column per
pattern. All CDL functions are causal: the value at row T depends on candles at
T and earlier only (pattern lookbacks are trailing), so they respect the
no-look-ahead rule enforced by the leakage test.

Most patterns emit values in {-100, 0, 100}. Two documented exceptions, both
verified empirically against our data (TA-Lib 0.7.1):

* the Hikkake family (CDLHIKKAKE, CDLHIKKAKEMOD) emits +-200 on delayed
  confirmation bars;
* CDLENGULFING, CDLHARAMI and CDLHARAMICROSS emit +-80 for near-miss variants
  (a grading added to these patterns by the modern TA-Lib C core).

Any value outside that domain fails loudly rather than being clipped or
ignored — that check is how the +-80 behaviour was discovered.

Patterns that never fire in the given data carry no information and are dropped
(with a log of which ones); survivors' fire rates are logged and reported so
they can be recorded in the build manifest.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import talib
from talib import abstract

from src.features.common import validate_ohlcv_input

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("open_time", "open", "high", "low", "close")
#: Values a CDL function may legitimately emit: +-200 = Hikkake confirmation,
#: +-80 = engulfing/harami near-miss grading (see module docstring).
ALLOWED_PATTERN_VALUES = frozenset({-200, -100, -80, 0, 80, 100, 200})


def pattern_names() -> list[str]:
    """Return all TA-Lib pattern-recognition function names, sorted."""
    return sorted(talib.get_function_groups()["Pattern Recognition"])


def max_pattern_lookback_rows() -> int:
    """Longest trailing span, in rows including the current one, any pattern needs.

    Uses TA-Lib's own lookback metadata; ``+1`` converts a lookback (rows
    *before* the first valid output) into a window span that includes row T.
    """
    return max(abstract.Function(name).lookback for name in pattern_names()) + 1


def _column_name(pattern: str) -> str:
    """Map a TA-Lib name to a feature column, e.g. ``CDLHIKKAKE`` -> ``cdl_hikkake``."""
    return "cdl_" + pattern[3:].lower()


def compute_candlestick_features(
    df: pd.DataFrame,
    *,
    drop_never_fired: bool = True,
    context: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute all CDL* pattern columns for a candle DataFrame.

    Args:
        df: Candle DataFrame with at least ``REQUIRED_COLUMNS``, sorted by
            ``open_time``.
        drop_never_fired: Drop all-zero pattern columns (they carry no
            information for this dataset). Disabled by the leakage test so the
            column set does not depend on how much of the series is visible.
        context: Label such as ``"BTCUSDT 1h"`` used in log messages.

    Returns:
        Tuple of (features, report). ``features`` has one int32 column per kept
        pattern, aligned to ``df``'s index. ``report`` contains
        ``n_patterns_total``, ``dropped_never_fired`` and ``fire_rates`` (share
        of rows with a non-zero value, per kept pattern).

    Raises:
        ValueError: If the input is invalid or a pattern emits a value outside
            ``ALLOWED_PATTERN_VALUES``.
    """
    label = context or "candlestick"
    validate_ohlcv_input(df, REQUIRED_COLUMNS, label)
    open_, high, low, close = (
        df[col].to_numpy(dtype="float64") for col in ("open", "high", "low", "close")
    )

    names = pattern_names()
    columns: dict[str, pd.Series] = {}
    for name in names:
        values = getattr(talib, name)(open_, high, low, close)
        unexpected = set(np.unique(values)) - ALLOWED_PATTERN_VALUES
        if unexpected:
            raise ValueError(
                f"{label}: {name} emitted unexpected value(s) {sorted(unexpected)}; "
                f"allowed: {sorted(ALLOWED_PATTERN_VALUES)}"
            )
        columns[_column_name(name)] = pd.Series(values, index=df.index, dtype="int32")

    features = pd.DataFrame(columns)
    fire_rates = {col: float((features[col] != 0).mean()) for col in features.columns}
    dropped: list[str] = []
    if drop_never_fired:
        dropped = sorted(col for col, rate in fire_rates.items() if rate == 0.0)
        if dropped:
            features = features.drop(columns=dropped)
            for col in dropped:
                del fire_rates[col]
            logger.info(
                "%s: dropped %d pattern(s) that never fire in this data: %s",
                label,
                len(dropped),
                dropped,
            )
    logger.info(
        "%s: kept %d of %d pattern(s); fire rates: %s",
        label,
        features.shape[1],
        len(names),
        ", ".join(
            f"{col}={rate:.4%}"
            for col, rate in sorted(fire_rates.items(), key=lambda kv: -kv[1])
        ),
    )
    report: dict[str, Any] = {
        "n_patterns_total": len(names),
        "dropped_never_fired": dropped,
        "fire_rates": fire_rates,
    }
    return features, report
