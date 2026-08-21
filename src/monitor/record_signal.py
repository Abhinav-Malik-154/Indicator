"""Record today's live signal to a permanent, append-only audit log (Phase 9).

This script is the engine of the project's **leakage-immune forward test**.
It runs independently of the Streamlit dashboard: schedule it once a day (see
``MONITORING.md``) and it appends the current model signal to a CSV *before*
the outcome is known.  Because every row is written strictly before the price
move it predicts, the accuracy computed from this log later
(:mod:`src.dashboard.live_track_record`) cannot be contaminated by hindsight —
it is out-of-sample by construction, not by discipline.

Contract:

* One record is appended **per model** (``lr`` and ``lgb``) for each closed
  daily candle.  The signal logic is imported from
  :func:`src.dashboard.signals.compute_live_signal` — never re-implemented here.
* **Idempotent**: the anchor candle date is the dedup key.  If any row for the
  latest closed candle already exists, the script writes nothing and exits 0.
  Running it twice in one day can never create a duplicate.
* **Append-only**: rows are appended with :class:`csv.DictWriter`; existing
  rows are never rewritten, reordered, or deleted.  This is a permanent trail.

Run from the repo root::

    python -m src.monitor.record_signal
    python -m src.monitor.record_signal --log-level DEBUG
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.dashboard.signals import compute_live_signal

logger = logging.getLogger(__name__)

# ── Log location + schema (single source of truth; imported by the reader) ──
LOG_DIR = Path("data/signal_log")
LOG_PATH = LOG_DIR / "live_signals.csv"

# Column order is stable and must never change once rows exist.  ``candle_date``
# is the anchor (the closed candle we predict FROM); ``close`` is that candle's
# close price — both are known at record time, so recording them leaks nothing.
# The realized outcome is deliberately NOT recorded: it lies in the future at
# record time and is only looked up later, which is what makes the forward
# test honest.
LOG_COLUMNS: list[str] = [
    "candle_date",       # ISO date of the last fully closed candle (anchor)
    "model",             # "lr" or "lgb"
    "prob_up",           # P(up) from that model
    "signal",            # BUY / SELL / SILENT at the confidence threshold
    "threshold",         # confidence threshold used
    "close",             # close price at candle_date (known now, no leakage)
    "recorded_at_utc",   # wall-clock time this row was written
]


# ---------------------------------------------------------------------------
# Log I/O
# ---------------------------------------------------------------------------


def read_signal_log(log_path: str | Path = LOG_PATH) -> pd.DataFrame:
    """Read the append-only signal log, or an empty typed frame if absent.

    Args:
        log_path: Path to ``live_signals.csv``.

    Returns:
        DataFrame with :data:`LOG_COLUMNS`.  ``candle_date`` is kept as a
        plain string (``YYYY-MM-DD``) so equality checks are exact.
    """
    path = Path(log_path)
    if not path.is_file():
        return pd.DataFrame({col: pd.Series(dtype="object") for col in LOG_COLUMNS})
    df = pd.read_csv(path, dtype={"candle_date": "string", "model": "string", "signal": "string"})
    # Preserve the canonical column order even if the file was hand-edited.
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[LOG_COLUMNS]


def _build_rows(result: dict[str, Any], *, recorded_at: str) -> list[dict[str, Any]]:
    """Turn a :func:`compute_live_signal` result into one row per model."""
    candle_date = pd.Timestamp(result["candle_date"]).date().isoformat()
    close = float(result["current_close"])
    threshold = float(result["threshold"])
    rows: list[dict[str, Any]] = []
    for model in ("lr", "lgb"):
        rows.append(
            {
                "candle_date": candle_date,
                "model": model,
                "prob_up": round(float(result[f"prob_{model}"]), 6),
                "signal": result[f"signal_{model}"],
                "threshold": threshold,
                "close": close,
                "recorded_at_utc": recorded_at,
            }
        )
    return rows


def append_signal(
    result: dict[str, Any],
    *,
    log_path: str | Path = LOG_PATH,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Idempotently append the current signal rows to the log.

    Dedup key is the anchor candle date.  If any row for that date already
    exists, nothing is written.  New rows are appended with
    :class:`csv.DictWriter` so no existing row is touched.

    Args:
        result: Output of :func:`src.dashboard.signals.compute_live_signal`.
        log_path: Path to ``live_signals.csv`` (created with header if absent).
        now: Injectable "recorded at" timestamp (defaults to UTC now).

    Returns:
        Dict with ``appended`` (bool), ``candle_date`` (str),
        ``rows_written`` (int), and ``reason`` (str).
    """
    path = Path(log_path)
    candle_date = pd.Timestamp(result["candle_date"]).date().isoformat()

    existing = read_signal_log(path)
    already = (
        not existing.empty
        and (existing["candle_date"].astype("string") == candle_date).any()
    )
    if already:
        logger.info(
            "signal for candle %s already logged — skipping (idempotent)", candle_date
        )
        return {
            "appended": False,
            "candle_date": candle_date,
            "rows_written": 0,
            "reason": "already_logged",
        }

    recorded_at = (now or pd.Timestamp.now(tz="UTC")).isoformat(timespec="seconds")
    rows = _build_rows(result, recorded_at=recorded_at)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info(
        "appended %d row(s) for candle %s to %s", len(rows), candle_date, path
    )
    return {
        "appended": True,
        "candle_date": candle_date,
        "rows_written": len(rows),
        "reason": "appended",
    }


def record(
    *,
    interval: str = "1d",
    config_path: str = "configs/config.yaml",
    log_path: str | Path = LOG_PATH,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Compute the live signal and idempotently append it to the log.

    Args:
        interval: Binance interval (matches the dashboard's default).
        config_path: Path to ``configs/config.yaml``.
        log_path: Path to the append-only log.
        now: Injectable "recorded at" timestamp.

    Returns:
        The :func:`append_signal` result dict, extended with ``data_source``
        so the caller can tell whether the underlying candle was live.
    """
    result = compute_live_signal(interval=interval, config_path=config_path)
    outcome = append_signal(result, log_path=log_path, now=now)
    outcome["data_source"] = result["data_source"]
    return outcome


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Record today's signal. Returns 0 whether or not a row was appended.

    A skip (already logged) is a success, not an error — that is the whole
    point of idempotency.  A non-zero exit means the signal could not be
    computed at all.

    Args:
        argv: CLI arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success (appended or skipped), 1 if the signal computation failed.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.monitor.record_signal",
        description="Append today's live signal to the permanent forward-test log.",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--log-path", default=str(LOG_PATH))
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    try:
        outcome = record(
            interval=args.interval,
            config_path=args.config,
            log_path=args.log_path,
        )
    except Exception:
        logger.exception("failed to compute/record the live signal")
        return 1

    if outcome["appended"]:
        logger.info(
            "recorded signal for candle %s (%d rows, source=%s)",
            outcome["candle_date"], outcome["rows_written"], outcome["data_source"],
        )
    else:
        logger.info(
            "no-op: candle %s already recorded (%s)",
            outcome["candle_date"], outcome["reason"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
