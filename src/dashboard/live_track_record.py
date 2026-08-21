"""Leakage-immune live forward-test accuracy from the signal log (Phase 9).

This reads the append-only log written by :mod:`src.monitor.record_signal` and
scores each past signal against the price move that *actually* happened after
it was recorded.  Because every signal was written strictly before its outcome
existed (see the recorder's contract), this accuracy is out-of-sample **by
construction** — it is the one number on the dashboard that no amount of
overfitting, look-ahead, or window-picking could have inflated.

It is deliberately kept separate from the Phase 4/5 backtest figures: those are
retrospective evaluations of a frozen model on historical data; this one grows,
one honest day at a time, from signals logged in real time.

Scoring rules (mirroring :func:`src.labels.build_labels.compute_forward_return_labels`):

* Outcome for a signal at candle date ``t`` is the move to ``t + horizon`` days.
  BTC daily candles are contiguous, so a calendar shift equals the positional
  shift used at training time.
* The training dead zone applies: a realized move within ``±dead_zone_pct`` is
  "no clear move" and the signal is **not scored** (neither right nor wrong),
  exactly as those rows were excluded from the training target.
* ``SILENT`` makes no directional claim and is never scored.
* A signal is only scored once its outcome candle is available; more recent
  signals are simply "not old enough yet".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.monitor.record_signal import LOG_PATH, read_signal_log

logger = logging.getLogger(__name__)

# Minimum distinct candle dates before a forward-test number is shown at all.
# Below this the sample is too small to mean anything, so we say so plainly.
MIN_DAYS_FOR_ACCURACY = 20


# ---------------------------------------------------------------------------
# Outcome classification (mirrors the training label logic)
# ---------------------------------------------------------------------------


def _classify_move(fwd_return: float, dead_zone_ratio: float) -> str | None:
    """Classify a realized forward return as 'up' / 'down' / None (dead zone).

    Uses strict inequalities and 12-dp rounding, identical to
    :func:`src.labels.build_labels.compute_forward_return_labels`.

    Returns:
        ``"up"``, ``"down"``, or ``None`` when the move is inside the dead
        zone (too small to count either way).
    """
    r = round(float(fwd_return), 12)
    if r > dead_zone_ratio:
        return "up"
    if r < -dead_zone_ratio:
        return "down"
    return None


_SIGNAL_TO_DIRECTION = {"BUY": "up", "SELL": "down"}


# ---------------------------------------------------------------------------
# Close-price lookup
# ---------------------------------------------------------------------------


def build_close_lookup(
    raw_close: pd.Series | None = None,
    log_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Build a ``{iso_date: close}`` map from the raw candles and the log itself.

    The raw parquet is the authoritative price history; the log's own recorded
    closes extend it for the most recent days that a stale parquet may not yet
    contain.  Both sources hold closes that were final when recorded, so using
    either to score an *earlier* signal leaks nothing.

    Args:
        raw_close: Series of close prices indexed by ``open_time`` (UTC), or
            ``None`` to use only the log's closes.
        log_df: The signal log, or ``None``.

    Returns:
        Mapping from ``YYYY-MM-DD`` to close price.
    """
    lookup: dict[str, float] = {}
    if raw_close is not None and len(raw_close):
        for ts, close in raw_close.items():
            lookup[pd.Timestamp(ts).date().isoformat()] = float(close)
    if log_df is not None and not log_df.empty:
        # One close per candle_date; log rows for the same date share a close.
        for date_str, grp in log_df.groupby(log_df["candle_date"].astype("string")):
            close = grp["close"].dropna()
            if len(close):
                lookup.setdefault(str(date_str), float(close.iloc[0]))
    return lookup


# ---------------------------------------------------------------------------
# Forward-test evaluation
# ---------------------------------------------------------------------------


def evaluate_forward_test(
    log_df: pd.DataFrame,
    close_lookup: dict[str, float],
    *,
    horizon: int,
    dead_zone_pct: float,
    min_days: int = MIN_DAYS_FOR_ACCURACY,
) -> dict[str, Any]:
    """Compute leakage-immune forward-test accuracy per model from the log.

    Args:
        log_df: Rows read from ``live_signals.csv`` (see
            :func:`src.monitor.record_signal.read_signal_log`).
        close_lookup: ``{iso_date: close}`` map covering the anchor and outcome
            dates (see :func:`build_close_lookup`).
        horizon: Forward horizon in days (the modeled horizon).
        dead_zone_pct: Dead-zone width in percent (same as the label build).
        min_days: Distinct candle dates required before accuracy is reported.

    Returns:
        Dict with ``days_recorded``, ``enough_data``, ``min_days``, ``horizon``,
        ``dead_zone_pct``, ``n_evaluable_total`` and a ``models`` sub-dict.  Each
        model entry has ``n_fired``, ``n_evaluable``, ``n_correct`` and
        ``accuracy_pct`` (``None`` when nothing is evaluable yet).
    """
    dead_zone_ratio = dead_zone_pct / 100.0
    models = ("lr", "lgb")
    result: dict[str, Any] = {
        "days_recorded": 0,
        "min_days": min_days,
        "enough_data": False,
        "horizon": horizon,
        "dead_zone_pct": dead_zone_pct,
        "n_evaluable_total": 0,
        "models": {
            m: {"n_fired": 0, "n_evaluable": 0, "n_correct": 0, "accuracy_pct": None}
            for m in models
        },
    }
    if log_df is None or log_df.empty:
        return result

    dates = log_df["candle_date"].astype("string")
    result["days_recorded"] = int(dates.nunique())
    result["enough_data"] = result["days_recorded"] >= min_days

    for _, row in log_df.iterrows():
        model = str(row["model"]).lower()
        if model not in result["models"]:
            continue
        signal = str(row["signal"]).upper()
        predicted = _SIGNAL_TO_DIRECTION.get(signal)
        if predicted is None:
            continue  # SILENT — no directional claim
        result["models"][model]["n_fired"] += 1

        anchor_date = str(row["candle_date"])
        outcome_date = (pd.Timestamp(anchor_date) + pd.Timedelta(days=horizon)).date().isoformat()
        close_t = close_lookup.get(anchor_date)
        close_th = close_lookup.get(outcome_date)
        if close_t is None or close_th is None or close_t == 0:
            continue  # outcome not available yet — not old enough

        realized = _classify_move(close_th / close_t - 1.0, dead_zone_ratio)
        if realized is None:
            continue  # move fell in the dead zone — not scored, mirroring labels

        result["models"][model]["n_evaluable"] += 1
        if realized == predicted:
            result["models"][model]["n_correct"] += 1

    total_evaluable = 0
    for m in models:
        stats = result["models"][m]
        total_evaluable += stats["n_evaluable"]
        if stats["n_evaluable"] > 0:
            stats["accuracy_pct"] = 100.0 * stats["n_correct"] / stats["n_evaluable"]
    result["n_evaluable_total"] = total_evaluable
    return result


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def accumulating_message(days_recorded: int, min_days: int = MIN_DAYS_FOR_ACCURACY) -> str:
    """Return the "still accumulating" status string shown before the gate."""
    return f"Accumulating — {days_recorded}/{min_days} days recorded"


def summarize(result: dict[str, Any]) -> str:
    """Render a short human summary of the forward-test state (for CLI/logs)."""
    if not result["enough_data"]:
        return accumulating_message(result["days_recorded"], result["min_days"])
    lines = [
        f"Live forward-test accuracy ({result['days_recorded']} days recorded, "
        f"horizon {result['horizon']}d, dead zone ±{result['dead_zone_pct']}%):"
    ]
    for model in ("lr", "lgb"):
        stats = result["models"][model]
        if stats["accuracy_pct"] is None:
            lines.append(
                f"  {model.upper()}: {stats['n_fired']} fired, "
                "no signals with known outcomes yet"
            )
        else:
            lines.append(
                f"  {model.upper()}: {stats['accuracy_pct']:.1f}% "
                f"({stats['n_correct']}/{stats['n_evaluable']} evaluable, "
                f"{stats['n_fired']} fired)"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dashboard entry point
# ---------------------------------------------------------------------------


def load_forward_test(
    cfg: dict[str, Any],
    *,
    interval: str = "1d",
    log_path: str | Path = LOG_PATH,
) -> dict[str, Any]:
    """Load the log + raw closes and evaluate the forward test for the dashboard.

    Args:
        cfg: Config dict from :func:`src.models.train.load_modeling_config`.
        interval: Binance interval.
        log_path: Path to ``live_signals.csv``.

    Returns:
        The :func:`evaluate_forward_test` result dict.
    """
    log_df = read_signal_log(log_path)

    raw_close: pd.Series | None = None
    raw_path = Path(cfg["raw_dir"]) / f"{cfg['file_prefix']}_{interval}.parquet"
    if raw_path.is_file():
        raw = pd.read_parquet(raw_path, columns=["open_time", "close"])
        raw_close = raw.set_index("open_time")["close"]

    close_lookup = build_close_lookup(raw_close=raw_close, log_df=log_df)
    horizon = int(cfg["modeling"]["horizon"])
    dead_zone_pct = float(cfg["labels"]["dead_zone_pct"])
    result = evaluate_forward_test(
        log_df, close_lookup, horizon=horizon, dead_zone_pct=dead_zone_pct
    )
    logger.info(
        "forward-test: %d days recorded, %d evaluable signals",
        result["days_recorded"], result["n_evaluable_total"],
    )
    return result
