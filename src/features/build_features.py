"""Build leakage-safe feature tables from the Phase 1 raw candle data.

Reads ``data/raw/{prefix}_{interval}.parquet`` per interval, computes the
candlestick and technical feature families, and writes
``data/processed/features_{interval}.parquet`` plus a manifest JSON describing
the build: feature list, row counts, NaN report, candlestick fire rates, and
gap accounting.

Gap policy: rolling windows are positional (over available candles), never
wall-clock. Gap records produced by the Phase 1 fetch manifests are read back
here, and every feature row whose longest lookback window spans a recorded gap
is counted and reported. Candles are never filled, interpolated, or
synthesised. Early rows without full rolling history keep NaN — backfilling
them would leak information backward in time.

Run from the repo root::

    python -m src.features.build_features
    python -m src.features.build_features --intervals 1d --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.fetch_binance import (
    DataValidationError,
    interval_to_ms,
    load_config,
    sha256_of,
)
from src.features.candlestick import compute_candlestick_features, max_pattern_lookback_rows
from src.features.technical import (
    compute_technical_features,
    longest_lookback_rows,
    validate_macd_config,
    validate_window,
    validate_window_list,
)

logger = logging.getLogger(__name__)

_FEATURE_DEFAULTS: dict[str, Any] = {
    "return_periods": [1, 3, 7, 14],
    "volatility_windows": [7, 14, 30],
    "volume_window": 20,
    "ma_windows": [7, 30],
    "sr_window": 30,
    "rsi_period": 14,
    "macd": None,  # validated as sub-mapping; None -> use technical.py defaults
}


def load_features_config(path: str | Path) -> dict[str, Any]:
    """Load the pipeline config extended with the feature-build settings.

    Reuses the Phase 1 loader for the ``data`` / ``paths`` / ``binance``
    sections, then validates ``paths.processed_dir`` and the optional
    ``features`` section (defaults merged in, unknown keys rejected).

    Args:
        path: Path to the YAML config.

    Returns:
        The Phase 1 config dict plus ``processed_dir`` and ``features`` keys.

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

    section = raw.get("features") or {}
    if not isinstance(section, dict):
        raise ValueError("config: 'features' section must be a mapping")
    unknown = sorted(set(section) - set(_FEATURE_DEFAULTS))
    if unknown:
        raise ValueError(f"config: unknown features option(s): {unknown}")
    features = {**_FEATURE_DEFAULTS, **section}
    # Window values are validated again by compute_technical_features; failing
    # here as well keeps config errors close to the config file.
    validate_window_list("return_periods", features["return_periods"])
    validate_window_list("volatility_windows", features["volatility_windows"])
    validate_window_list("ma_windows", features["ma_windows"])
    validate_window("volume_window", features["volume_window"], minimum=2)
    validate_window("sr_window", features["sr_window"], minimum=2)
    validate_window("rsi_period", features["rsi_period"], minimum=2)
    validate_macd_config(features["macd"])

    return {**cfg, "processed_dir": processed_dir, "features": features}


def load_gap_records(manifest_path: str | Path, context: str = "") -> list[dict[str, Any]]:
    """Read the gap records from a Phase 1 fetch manifest.

    Args:
        manifest_path: Path to ``{prefix}_{interval}.manifest.json``.
        context: Label used in log messages.

    Returns:
        The manifest's gap records (possibly empty). Missing or malformed
        manifests yield an empty list with a warning — gap accounting is then
        incomplete but the build itself can proceed.
    """
    label = context or "gap records"
    path = Path(manifest_path)
    if not path.is_file():
        logger.warning(
            "%s: raw manifest %s not found — gap accounting will treat the data as gap-free",
            label,
            path,
        )
        return []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("%s: raw manifest %s is not valid JSON — ignoring it", label, path)
        return []
    gaps = manifest.get("gaps", [])
    valid = [g for g in gaps if isinstance(g, dict) and {"start", "end", "n_missing"} <= set(g)]
    if len(valid) != len(gaps):
        logger.warning(
            "%s: ignoring %d malformed gap record(s) in %s", label, len(gaps) - len(valid), path
        )
    return valid


def mark_gap_affected_rows(
    df: pd.DataFrame,
    gaps: list[dict[str, Any]],
    lookback_rows: int,
    interval: str,
    context: str = "",
) -> pd.Series:
    """Flag rows whose longest lookback window spans a recorded gap.

    With positional windows, a window of span L covers rows [T-L+1, T]. The
    jump over a gap sits between the last candle before it and the first
    candle after it (position b); the window contains that jump exactly for
    rows b .. b+L-2.

    Args:
        df: Candle DataFrame the features are computed from.
        gaps: Gap records from the Phase 1 manifest.
        lookback_rows: Longest lookback span in rows (current row included).
        interval: Binance interval string, used to verify each recorded gap
            actually corresponds to a jump in the data.
        context: Label used in log messages.

    Returns:
        Boolean Series aligned to ``df``: True where the row's lookback window
        contains a recorded gap.
    """
    label = context or "gap accounting"
    interval_td = pd.Timedelta(milliseconds=interval_to_ms(interval))
    open_times = df["open_time"]
    mask = pd.Series(False, index=df.index)
    if lookback_rows < 2:
        return mask
    for gap in gaps:
        gap_end = pd.Timestamp(gap["end"])
        # First row strictly after the last missing candle = row after the jump.
        b = int(open_times.searchsorted(gap_end, side="right"))
        if b <= 0 or b >= len(df):
            logger.warning(
                "%s: recorded gap %s..%s lies outside the data range — skipped",
                label,
                gap["start"],
                gap["end"],
            )
            continue
        if open_times.iloc[b] - open_times.iloc[b - 1] <= interval_td:
            logger.warning(
                "%s: recorded gap %s..%s has no matching jump in the data — skipped",
                label,
                gap["start"],
                gap["end"],
            )
            continue
        mask.iloc[b : b + lookback_rows - 1] = True
    return mask


def build_features_for_interval(interval: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build and persist the feature table for one interval.

    Args:
        interval: Binance interval string, e.g. ``"1h"``.
        cfg: Config dict from :func:`load_features_config`.

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
    # Cheap re-check: every rolling computation assumes this ordering.
    if df["open_time"].duplicated().any() or not df["open_time"].is_monotonic_increasing:
        raise DataValidationError(f"{context}: raw data is not strictly ordered by open_time")
    logger.info("%s: building features from %s (%d rows)", context, raw_path, len(df))

    tech = compute_technical_features(df, **cfg["features"], context=context)
    cdl, cdl_report = compute_candlestick_features(df, context=context)
    features = pd.concat([df[["open_time"]], tech, cdl], axis=1)

    technical_lookback = longest_lookback_rows(**cfg["features"])
    candlestick_lookback = max_pattern_lookback_rows()
    lookback_rows = max(technical_lookback, candlestick_lookback)

    manifest_path = raw_path.with_name(f"{cfg['file_prefix']}_{interval}.manifest.json")
    gaps = load_gap_records(manifest_path, context=context)
    gap_mask = mark_gap_affected_rows(df, gaps, lookback_rows, interval, context=context)
    gap_affected = int(gap_mask.sum())
    logger.info(
        "%s: %d of %d feature rows (%.3f%%) have a recorded gap inside their "
        "%d-candle lookback window — kept as-is, never filled",
        context,
        gap_affected,
        len(features),
        100.0 * gap_affected / len(features),
        lookback_rows,
    )
    interval_td = pd.Timedelta(milliseconds=interval_to_ms(interval))
    jumps_in_data = int((df["open_time"].diff() > interval_td).sum())
    if jumps_in_data != len(gaps):
        logger.warning(
            "%s: data contains %d gap jump(s) but the raw manifest records %d — "
            "gap accounting may be incomplete",
            context,
            jumps_in_data,
            len(gaps),
        )

    feature_columns = [c for c in features.columns if c != "open_time"]
    nan_counts = features[feature_columns].isna().sum()
    columns_with_nans = {col: int(n) for col, n in nan_counts.items() if n}
    rows_with_any_nan = int(features[feature_columns].isna().any(axis=1).sum())
    logger.info(
        "%s: NaN report — %d of %d rows have at least one NaN "
        "(leading rows lack full rolling history; kept as NaN, never backfilled)",
        context,
        rows_with_any_nan,
        len(features),
    )

    out_dir = Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"features_{interval}.parquet"
    features.to_parquet(out_path, engine="pyarrow", index=False)

    manifest: dict[str, Any] = {
        "symbol": cfg["symbol"],
        "interval": interval,
        "built_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "source": {
            "parquet": str(raw_path),
            "sha256": sha256_of(raw_path),
            "manifest": str(manifest_path) if manifest_path.is_file() else None,
        },
        "windows": cfg["features"],
        "lookback_rows": {
            "technical": technical_lookback,
            "candlestick": candlestick_lookback,
            "effective": lookback_rows,
        },
        "row_count": int(len(features)),
        "n_features": len(feature_columns),
        "features": {
            "technical": list(tech.columns),
            "candlestick": list(cdl.columns),
        },
        "nan_report": {
            "rows_with_any_nan": rows_with_any_nan,
            "columns_with_nans": columns_with_nans,
            "policy": (
                "rows without full rolling history keep NaN; no backfill (backfill = leakage)"
            ),
        },
        "gap_accounting": {
            "gap_records_from_raw_manifest": len(gaps),
            "gap_affected_rows": gap_affected,
            "policy": "positional windows span gaps; affected rows counted, never filled",
        },
        "candlestick_report": cdl_report,
        "files": {
            "parquet": {
                "path": str(out_path),
                "bytes": out_path.stat().st_size,
                "sha256": sha256_of(out_path),
            }
        },
    }
    build_manifest_path = out_dir / f"features_{interval}.manifest.json"
    with build_manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    logger.info(
        "%s: wrote %s (%.2f MiB, %d features) and %s",
        context,
        out_path,
        out_path.stat().st_size / 2**20,
        len(feature_columns),
        build_manifest_path,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Build features for every configured interval.

    Args:
        argv: CLI arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 if any interval failed.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.features.build_features",
        description="Build leakage-safe feature tables from the raw candle data.",
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

    cfg = load_features_config(args.config)
    intervals: list[str] = args.intervals or cfg["intervals"]

    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for interval in intervals:
        try:
            summaries.append(build_features_for_interval(interval, cfg))
        except Exception:
            logger.exception("%s %s: feature build failed", cfg["symbol"], interval)
            failures.append(interval)

    for manifest in summaries:
        logger.info(
            "%s %s: %d rows x %d features, %d gap-affected row(s), %d row(s) with NaN",
            manifest["symbol"],
            manifest["interval"],
            manifest["row_count"],
            manifest["n_features"],
            manifest["gap_accounting"]["gap_affected_rows"],
            manifest["nan_report"]["rows_with_any_nan"],
        )
    if failures:
        logger.error("feature build failed for interval(s): %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
