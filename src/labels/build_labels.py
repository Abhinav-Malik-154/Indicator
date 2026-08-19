"""Build forward-return labels from the Phase 1 raw candle data.

Labels are the answer key: the label at row T tells us what happened to the
price over the next N candles.  This is the *only* module that deliberately
looks forward in time.  The critical contract:

* Labels live in a **separate file** (``labels_{interval}.parquet``), never
  merged into ``features_{interval}.parquet``.
* The two are joined only at training time on ``open_time`` — a simple
  equi-join, no shift, no offset.
* The last N rows of any horizon have no ``close[T+N]`` and are NaN — never
  filled, estimated, or backfilled.

A configurable dead zone excludes tiny moves (noise) from the classification
target.  The dead zone uses strict inequalities: a forward return of exactly
±threshold is excluded (NaN), not classified.  This is the conservative choice.

Run from the repo root::

    python -m src.labels.build_labels
    python -m src.labels.build_labels --intervals 1d --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.fetch_binance import DataValidationError, load_config, sha256_of

logger = logging.getLogger(__name__)

_LABEL_DEFAULTS: dict[str, Any] = {
    "horizons": [1],
    "dead_zone_pct": 0.15,
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_labels_config(path: str | Path) -> dict[str, Any]:
    """Load the pipeline config extended with the label-build settings.

    Reuses the Phase 1 loader for the ``data`` / ``paths`` / ``binance``
    sections, then validates ``paths.processed_dir`` and the ``labels``
    section (defaults merged in, unknown keys rejected).

    Args:
        path: Path to the YAML config.

    Returns:
        The Phase 1 config dict plus ``processed_dir`` and ``labels`` keys.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If validation fails.
    """
    cfg = load_config(path)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    paths = raw.get("paths") or {}
    processed_dir = paths.get("processed_dir")
    if not isinstance(processed_dir, str) or not processed_dir.strip():
        raise ValueError("config: paths.processed_dir must be a non-empty string")

    section = raw.get("labels") or {}
    if not isinstance(section, dict):
        raise ValueError("config: 'labels' section must be a mapping")
    unknown = sorted(set(section) - set(_LABEL_DEFAULTS))
    if unknown:
        raise ValueError(f"config: unknown labels option(s): {unknown}")
    labels = {**_LABEL_DEFAULTS, **section}

    # Validate horizons.
    horizons = labels["horizons"]
    if (
        not isinstance(horizons, (list, tuple))
        or len(horizons) == 0
        or not all(isinstance(h, int) and not isinstance(h, bool) and h >= 1 for h in horizons)
    ):
        raise ValueError(
            f"config: labels.horizons must be a non-empty list of integers >= 1, got {horizons!r}"
        )

    # Validate dead_zone_pct.
    dz = labels["dead_zone_pct"]
    if not isinstance(dz, (int, float)) or isinstance(dz, bool) or dz < 0:
        raise ValueError(
            f"config: labels.dead_zone_pct must be a number >= 0, got {dz!r}"
        )

    return {**cfg, "processed_dir": processed_dir, "labels": labels}


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------


def compute_forward_return_labels(
    df: pd.DataFrame,
    *,
    horizon: int,
    dead_zone_pct: float,
    context: str = "",
) -> pd.DataFrame:
    """Compute forward-return labels for a candle DataFrame.

    The label at row T is derived from ``close[T]`` and ``close[T+horizon]``:

    * ``label = 1`` if ``fwd_return > dead_zone_pct / 100`` (up)
    * ``label = 0`` if ``fwd_return < -dead_zone_pct / 100`` (down)
    * ``label = NaN`` if inside the dead zone or if ``close[T+horizon]``
      does not exist (the last ``horizon`` rows)

    The ``fwd_return`` column is kept for downstream analysis but is clearly
    forward-looking — it must **never** be used as a feature.

    Args:
        df: Candle DataFrame with at least ``open_time`` and ``close``,
            sorted by ``open_time`` with strictly increasing timestamps.
        horizon: Number of candles to look forward.
        dead_zone_pct: Dead-zone width in percent (e.g. 0.15 means ±0.15%).
            Moves within this band are excluded (NaN). Uses strict
            inequalities: a return of exactly ±threshold is excluded.
        context: Label such as ``"BTCUSDT 1d"`` used in log messages.

    Returns:
        DataFrame with columns ``open_time``, ``fwd_return_{horizon}``,
        ``label_{horizon}``, aligned to ``df``'s index.

    Raises:
        ValueError: On invalid inputs.
    """
    label = context or "labels"
    if df.empty:
        raise ValueError(f"{label}: input DataFrame is empty")
    for col in ("open_time", "close"):
        if col not in df.columns:
            raise ValueError(f"{label}: missing required column '{col}'")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"{label}: horizon must be an integer >= 1, got {horizon!r}")
    if not isinstance(dead_zone_pct, (int, float)) or isinstance(dead_zone_pct, bool):
        raise ValueError(f"{label}: dead_zone_pct must be a number, got {dead_zone_pct!r}")
    if dead_zone_pct < 0:
        raise ValueError(f"{label}: dead_zone_pct must be >= 0, got {dead_zone_pct!r}")
    deltas = df["open_time"].diff().iloc[1:]
    if (deltas <= pd.Timedelta(0)).any():
        raise ValueError(f"{label}: open_time must be strictly increasing and unique")

    close = df["close"]
    future_close = close.shift(-horizon)
    fwd_return = future_close / close - 1.0

    threshold = dead_zone_pct / 100.0

    # Round to 12 dp before comparing: IEEE 754 can produce values like
    # 0.010000000000000009 for what is mathematically exactly 0.01.  Rounding
    # eliminates this representation noise without affecting any real
    # classification (12 dp is far beyond the precision of % returns).
    fwd_rounded = fwd_return.round(12)
    label_col = pd.Series(np.nan, index=df.index, dtype="float64")

    # Classify: strict inequalities — exactly at threshold stays NaN.
    up_mask = fwd_rounded > threshold
    down_mask = fwd_rounded < -threshold
    label_col[up_mask] = 1.0
    label_col[down_mask] = 0.0
    # Everything else (dead zone + last N rows where fwd_return is NaN) stays NaN.

    ret_col = f"fwd_return_{horizon}"
    lbl_col = f"label_{horizon}"
    out = pd.DataFrame(
        {
            "open_time": df["open_time"].values,
            ret_col: fwd_return.values,
            lbl_col: label_col.values,
        },
        index=df.index,
    )
    n_up = int(up_mask.sum())
    n_down = int(down_mask.sum())
    n_excluded = len(df) - n_up - n_down
    logger.info(
        "%s: horizon=%d, dead_zone=±%.4f%%, up=%d, down=%d, excluded=%d (of %d)",
        label, horizon, dead_zone_pct, n_up, n_down, n_excluded, len(df),
    )
    return out


# ---------------------------------------------------------------------------
# Class balance report
# ---------------------------------------------------------------------------


def class_balance_report(
    labels_df: pd.DataFrame,
    *,
    horizon: int,
    dead_zone_pct: float,
    n_total: int,
    context: str = "",
) -> dict[str, Any]:
    """Compute class balance statistics for a single horizon.

    Separates dead-zone exclusions from tail exclusions (last N rows where
    ``close[T+N]`` does not exist) for transparency.

    Args:
        labels_df: Output of :func:`compute_forward_return_labels`.
        horizon: The forward-return horizon.
        dead_zone_pct: Dead-zone width in percent.
        n_total: Total number of rows in the raw candle data.
        context: Label used in log messages.

    Returns:
        Dict with counts, percentages, and an imbalance flag.
    """
    label = context or "balance"
    lbl_col = f"label_{horizon}"
    ret_col = f"fwd_return_{horizon}"
    lbl = labels_df[lbl_col]
    fwd = labels_df[ret_col]

    n_up = int((lbl == 1).sum())
    n_down = int((lbl == 0).sum())
    n_labelled = n_up + n_down
    # Tail rows: fwd_return is NaN because close[T+N] doesn't exist.
    n_tail = int(fwd.isna().sum())
    # Dead-zone rows: fwd_return exists but the move is too small.
    n_dead_zone = n_total - n_labelled - n_tail

    up_pct = 100.0 * n_up / n_labelled if n_labelled > 0 else 0.0
    down_pct = 100.0 * n_down / n_labelled if n_labelled > 0 else 0.0
    imbalanced = n_labelled > 0 and (up_pct < 35.0 or down_pct < 35.0)

    if imbalanced:
        logger.warning(
            "%s: CLASS IMBALANCE — horizon=%d, up=%.1f%%, down=%.1f%% "
            "(one class is under 35%%)",
            label, horizon, up_pct, down_pct,
        )
    else:
        logger.info(
            "%s: horizon=%d, up=%.1f%% (%d), down=%.1f%% (%d), "
            "excluded=%d (dead_zone=%d, tail=%d)",
            label, horizon, up_pct, n_up, down_pct, n_down,
            n_dead_zone + n_tail, n_dead_zone, n_tail,
        )
    return {
        "horizon": horizon,
        "dead_zone_pct": dead_zone_pct,
        "dead_zone_ratio": dead_zone_pct / 100.0,
        "row_count": n_total,
        "labelled_rows": n_labelled,
        "excluded_tail_rows": n_tail,
        "excluded_dead_zone_rows": n_dead_zone,
        "class_balance": {
            "up": n_up,
            "down": n_down,
            "up_pct": round(up_pct, 2),
            "down_pct": round(down_pct, 2),
        },
        "imbalance_flag": imbalanced,
    }


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def build_labels_for_interval(interval: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build and persist the label table for one interval.

    Reads ``data/raw/{prefix}_{interval}.parquet`` (the same raw candle source
    as the feature pipeline — labels and features are computed through
    completely independent pipelines and only join on ``open_time`` at training
    time).

    Args:
        interval: Binance interval string, e.g. ``"1d"``.
        cfg: Config dict from :func:`load_labels_config`.

    Returns:
        The manifest dictionary that was written next to the parquet.

    Raises:
        FileNotFoundError: If the Phase 1 parquet is missing.
        DataValidationError: If the raw data fails the ordering re-check.
    """
    context = f"{cfg['symbol']} {interval}"
    raw_path = Path(cfg["raw_dir"]) / f"{cfg['file_prefix']}_{interval}.parquet"
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"{context}: raw data {raw_path} not found — "
            "run `python -m src.data.fetch_binance` first"
        )
    df = pd.read_parquet(raw_path)
    if df["open_time"].duplicated().any() or not df["open_time"].is_monotonic_increasing:
        raise DataValidationError(f"{context}: raw data is not strictly ordered by open_time")
    logger.info("%s: building labels from %s (%d rows)", context, raw_path, len(df))

    labels_cfg = cfg["labels"]
    horizons: list[int] = labels_cfg["horizons"]
    dead_zone_pct: float = labels_cfg["dead_zone_pct"]

    # Compute labels for each horizon, collecting into a single DataFrame.
    all_parts: list[pd.DataFrame] = []
    horizon_reports: dict[str, Any] = {}
    for h in horizons:
        part = compute_forward_return_labels(
            df, horizon=h, dead_zone_pct=dead_zone_pct, context=context,
        )
        report = class_balance_report(
            part, horizon=h, dead_zone_pct=dead_zone_pct,
            n_total=len(df), context=context,
        )
        # Keep only the fwd_return and label columns (open_time comes once).
        all_parts.append(part.drop(columns=["open_time"]))
        horizon_reports[str(h)] = report

    labels_out = pd.concat([df[["open_time"]]] + all_parts, axis=1)

    # Write output.
    out_dir = Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"labels_{interval}.parquet"
    labels_out.to_parquet(out_path, engine="pyarrow", index=False)

    manifest_path = Path(cfg["raw_dir"]) / f"{cfg['file_prefix']}_{interval}.manifest.json"
    manifest: dict[str, Any] = {
        "symbol": cfg["symbol"],
        "interval": interval,
        "built_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "source": {
            "parquet": str(raw_path),
            "sha256": sha256_of(raw_path),
            "manifest": str(manifest_path) if manifest_path.is_file() else None,
        },
        "horizons": horizon_reports,
        "files": {
            "parquet": {
                "path": str(out_path),
                "bytes": out_path.stat().st_size,
                "sha256": sha256_of(out_path),
            },
        },
        "iron_rule": (
            "labels use close[T+N]; features use data <= T; "
            "joined only at training time on open_time"
        ),
    }
    build_manifest_path = out_dir / f"labels_{interval}.manifest.json"
    with build_manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    logger.info(
        "%s: wrote %s (%d rows, %d horizon(s)) and %s",
        context, out_path, len(labels_out), len(horizons), build_manifest_path,
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Build labels for every configured interval.

    Args:
        argv: CLI arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 if any interval failed.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.labels.build_labels",
        description="Build forward-return labels from the raw candle data.",
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
        help="override the intervals listed in the config",
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

    cfg = load_labels_config(args.config)
    intervals: list[str] = args.intervals or cfg["intervals"]

    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for interval in intervals:
        try:
            summaries.append(build_labels_for_interval(interval, cfg))
        except Exception:
            logger.exception("%s %s: label build failed", cfg["symbol"], interval)
            failures.append(interval)

    for manifest in summaries:
        for h_str, report in manifest["horizons"].items():
            bal = report["class_balance"]
            logger.info(
                "%s %s: horizon=%s, labelled=%d/%d, up=%.1f%%, down=%.1f%%, "
                "dead_zone=%d, tail=%d%s",
                manifest["symbol"],
                manifest["interval"],
                h_str,
                report["labelled_rows"],
                report["row_count"],
                bal["up_pct"],
                bal["down_pct"],
                report["excluded_dead_zone_rows"],
                report["excluded_tail_rows"],
                " *** IMBALANCED ***" if report["imbalance_flag"] else "",
            )
    if failures:
        logger.error("label build failed for interval(s): %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
