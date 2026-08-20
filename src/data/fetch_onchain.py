"""Fetch Bitcoin on-chain metrics from the Blockchain.com Charts API (Phase 6).

Saves to data/raw/btc_onchain_1d.parquet.  Same discipline as fetch_binance.py:
retries with exponential backoff, data validation, gap detection, manifest.

Free metrics (no auth required):
  active_addresses  — unique sending/receiving addresses per day
  n_transactions    — confirmed transactions per day
  hash_rate_th      — estimated network hash rate (TH/s)
  fees_usd          — total miner fees in USD per day
  volume_usd        — estimated BTC transferred in USD per day

NOT available for free (documented honestly):
  exchange_netflow  — Glassnode / CryptoQuant only (paid)
  whale_movement    — paid tracking services
  hodl_waves        — Glassnode paid tier
  realized_price    — Glassnode paid tier
  nupl / sthlth     — Glassnode paid tier

Run from the repo root:
    python -m src.data.fetch_onchain
    python -m src.data.fetch_onchain --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

_ONCHAIN_DEFAULTS: dict[str, Any] = {
    "metrics": {
        "active_addresses": "n-unique-addresses",
        "n_transactions": "n-transactions",
        "hash_rate_th": "hash-rate",
        "fees_usd": "transaction-fees-usd",
        "volume_usd": "estimated-transaction-volume-usd",
    },
    "reporting_lag_days": 1,
    "z_score_windows": [7, 30],
    "wow_window": 7,
    "max_forward_fill_days": 3,
    "base_url": "https://api.blockchain.info/charts",
    "timespan": "7years",
    "request_timeout_s": 30,
    "max_retries": 3,
    "backoff_base_s": 1.0,
}


def load_onchain_config(path: str | Path) -> dict[str, Any]:
    """Load the ``onchain`` section from the pipeline config.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dict with merged onchain settings.

    Raises:
        ValueError: If required keys are missing or invalid.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    section = raw.get("onchain") or {}
    cfg: dict[str, Any] = {**_ONCHAIN_DEFAULTS, **section}

    paths = raw.get("paths") or {}
    cfg["raw_dir"] = paths.get("raw_dir", "data/raw")

    if not isinstance(cfg["reporting_lag_days"], int) or cfg["reporting_lag_days"] < 0:
        raise ValueError(
            f"onchain.reporting_lag_days must be a non-negative integer, "
            f"got {cfg['reporting_lag_days']!r}"
        )
    if not isinstance(cfg["max_retries"], int) or cfg["max_retries"] < 1:
        raise ValueError(
            f"onchain.max_retries must be a positive integer, got {cfg['max_retries']!r}"
        )
    return cfg


def _fetch_metric_raw(
    chart_name: str,
    *,
    base_url: str,
    timespan: str,
    timeout_s: float,
    max_retries: int,
    backoff_base: float,
) -> list[dict[str, Any]]:
    """Fetch one metric from the Blockchain.com Charts API with retries.

    Args:
        chart_name: API chart name, e.g. ``"n-unique-addresses"``.
        base_url: Base URL for the Charts API.
        timespan: Timespan parameter, e.g. ``"7years"``.
        timeout_s: HTTP request timeout in seconds.
        max_retries: Number of attempts before giving up.
        backoff_base: Base for exponential backoff between retries.

    Returns:
        List of ``{"x": unix_epoch_s, "y": float}`` dicts.

    Raises:
        requests.RequestException: If all retries are exhausted.
        ValueError: If the response is empty or malformed.
    """
    url = f"{base_url}/{chart_name}"
    params = {"timespan": timespan, "sampled": "false", "format": "json"}

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            values = data.get("values")
            if not values:
                raise ValueError(f"{chart_name}: API returned empty values list")
            logger.debug(
                "%s: fetched %d raw points from %s",
                chart_name,
                len(values),
                resp.url,
            )
            return values
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = backoff_base * (2**attempt)
                logger.warning(
                    "%s: attempt %d/%d failed (%s); retrying in %.1fs",
                    chart_name,
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "%s: all %d attempts failed; last error: %s",
                    chart_name,
                    max_retries,
                    exc,
                )

    raise last_exc or RuntimeError(f"{chart_name}: fetch failed with no error")


def fetch_metric(chart_name: str, col_name: str, cfg: dict[str, Any]) -> pd.Series:
    """Fetch one on-chain metric and return a UTC-indexed daily Series.

    The Blockchain.com Charts API returns Unix epoch seconds for ``x``
    (start of day UTC, verified empirically: x=1577836800 = 2020-01-01 UTC).

    Args:
        chart_name: API chart name, e.g. ``"n-unique-addresses"``.
        col_name: Column name for the returned Series.
        cfg: On-chain config dict from :func:`load_onchain_config`.

    Returns:
        Float Series with a UTC DatetimeIndex at midnight resolution,
        deduplicated and sorted ascending.  Any ``null`` API values are NaN.
    """
    raw_values = _fetch_metric_raw(
        chart_name,
        base_url=cfg["base_url"],
        timespan=cfg["timespan"],
        timeout_s=cfg["request_timeout_s"],
        max_retries=cfg["max_retries"],
        backoff_base=cfg["backoff_base_s"],
    )
    dates = pd.to_datetime(
        [v["x"] for v in raw_values], unit="s", utc=True
    ).normalize()
    vals = [
        float(v["y"]) if v.get("y") is not None else float("nan")
        for v in raw_values
    ]
    series = pd.Series(vals, index=dates, name=col_name, dtype="float64")
    series = series[~series.index.duplicated(keep="last")].sort_index()

    n_null = int(series.isna().sum())
    logger.info(
        "%s → %s: %d points, %s → %s, %d null values",
        chart_name,
        col_name,
        len(series),
        series.index[0].date(),
        series.index[-1].date(),
        n_null,
    )
    return series


def fetch_all_metrics(cfg: dict[str, Any]) -> pd.DataFrame:
    """Fetch all configured on-chain metrics and join them into one DataFrame.

    Args:
        cfg: On-chain config dict from :func:`load_onchain_config`.

    Returns:
        DataFrame with a ``date`` column (UTC midnight) and one column per
        metric.  The index is a RangeIndex; ``date`` is the join key.
        Rows with all-NaN metrics are kept so gap detection works correctly.

    Raises:
        RuntimeError: If any metric fetch fails and no fallback is available.
    """
    metrics: dict[str, str] = cfg["metrics"]
    series_list: list[pd.Series] = []

    for col_name, chart_name in metrics.items():
        logger.info("Fetching on-chain metric: %s (%s)", col_name, chart_name)
        series = fetch_metric(chart_name, col_name, cfg)
        series_list.append(series)
        if len(metrics) > 1:
            time.sleep(0.5)

    # Outer join so no data is silently dropped if one metric has wider coverage
    df = pd.concat(series_list, axis=1, sort=True)
    df.index.name = "date"
    df = df.sort_index().reset_index()

    logger.info(
        "All metrics joined: %d rows, %s → %s",
        len(df),
        df["date"].iloc[0].date(),
        df["date"].iloc[-1].date(),
    )
    return df


def validate_onchain_data(
    df: pd.DataFrame, context: str = ""
) -> tuple[list[str], list[str]]:
    """Run sanity checks on the fetched on-chain DataFrame.

    Checks:
    - No duplicate dates.
    - Dates are strictly increasing.
    - No metric is all-NaN.
    - Warn about large gaps (> 3 missing days between consecutive dates).

    Args:
        df: DataFrame from :func:`fetch_all_metrics`.
        context: Label for log messages.

    Returns:
        Tuple of (errors, warnings) — errors are fatal; warnings are logged.
    """
    label = context or "onchain validation"
    errors: list[str] = []
    warnings: list[str] = []

    if df["date"].duplicated().any():
        errors.append(f"{label}: duplicate dates found")

    if not df["date"].is_monotonic_increasing:
        errors.append(f"{label}: dates not strictly increasing")

    metric_cols = [c for c in df.columns if c != "date"]
    for col in metric_cols:
        if df[col].isna().all():
            errors.append(f"{label}: column '{col}' is entirely NaN")
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            warnings.append(f"{label}: '{col}' has {n_nan} NaN values")

    gaps = df["date"].diff().dropna()
    large_gaps = gaps[gaps > pd.Timedelta(days=3)]
    for date, gap in large_gaps.items():
        warnings.append(
            f"{label}: large gap of {gap.days} days ending at "
            f"{df.loc[date, 'date'].date()}"  # type: ignore[call-overload]
        )

    return errors, warnings


def fetch_and_save(
    cfg: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    """Fetch all metrics, validate, and save to parquet.

    Args:
        cfg: On-chain config dict from :func:`load_onchain_config`.
        out_path: Output path for the parquet file.

    Returns:
        Manifest dictionary describing what was saved.

    Raises:
        ValueError: If validation errors are found.
    """
    df = fetch_all_metrics(cfg)

    errors, warnings = validate_onchain_data(df, context="btc_onchain_1d")
    for w in warnings:
        logger.warning(w)
    if errors:
        raise ValueError(
            f"On-chain data failed validation:\n" + "\n".join(errors)
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)

    metric_cols = [c for c in df.columns if c != "date"]
    coverage: dict[str, Any] = {}
    for col in metric_cols:
        valid = df[col].dropna()
        coverage[col] = {
            "n_valid": len(valid),
            "n_nan": int(df[col].isna().sum()),
            "first_date": str(valid.index[0] if len(valid) else ""),
            "last_date": str(valid.iloc[-1] if len(valid) else ""),
        }
    # Recalculate using date column for readability
    for col in metric_cols:
        valid_mask = df[col].notna()
        if valid_mask.any():
            coverage[col]["first_date"] = str(df.loc[valid_mask, "date"].iloc[0].date())
            coverage[col]["last_date"] = str(df.loc[valid_mask, "date"].iloc[-1].date())

    manifest: dict[str, Any] = {
        "built_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "source": "Blockchain.com Charts API (free, no auth)",
        "source_url_template": f"{cfg['base_url']}/{{chart_name}}?timespan={cfg['timespan']}&sampled=false",
        "not_available_for_free": [
            "exchange_netflow (Glassnode/CryptoQuant paid)",
            "whale_movement (paid tracking services)",
            "hodl_waves (Glassnode paid)",
            "realized_price (Glassnode paid)",
            "nupl / sthlth_supply (Glassnode paid)",
        ],
        "reporting_lag_days": cfg["reporting_lag_days"],
        "metrics": cfg["metrics"],
        "row_count": len(df),
        "date_range": {
            "first": str(df["date"].iloc[0].date()),
            "last": str(df["date"].iloc[-1].date()),
        },
        "coverage_by_metric": coverage,
        "output": {
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
        },
        "iron_rule": (
            "on-chain data for day T is available at midnight UTC end of day T; "
            f"conservative lag of {cfg['reporting_lag_days']} day(s) applied in feature "
            "engineering so candle T uses only on-chain data from T-1"
        ),
    }
    manifest_path = out_path.with_suffix("").with_suffix(".manifest.json")
    manifest_path = out_path.parent / f"{out_path.stem}.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    logger.info(
        "Saved %d rows, %d metrics to %s (%.2f KiB)",
        len(df),
        len(metric_cols),
        out_path,
        out_path.stat().st_size / 1024,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Fetch on-chain metrics and save to data/raw/btc_onchain_1d.parquet.

    Args:
        argv: CLI arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.data.fetch_onchain",
        description="Fetch Bitcoin on-chain metrics from the Blockchain.com Charts API.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="path to the pipeline config (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="override output parquet path",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    cfg = load_onchain_config(args.config)
    out_path = Path(args.out) if args.out else Path(cfg["raw_dir"]) / "btc_onchain_1d.parquet"

    try:
        manifest = fetch_and_save(cfg, out_path)
        print(f"\nSaved {manifest['row_count']} daily rows to {out_path}")
        print(f"Date range: {manifest['date_range']['first']} → {manifest['date_range']['last']}")
        print("\nCoverage by metric:")
        for col, info in manifest["coverage_by_metric"].items():
            print(
                f"  {col:<22} {info['n_valid']:>4d} valid points"
                f"  ({info['first_date']} → {info['last_date']})"
                + (f"  [{info['n_nan']} NaN]" if info["n_nan"] > 0 else "")
            )
        return 0
    except Exception:
        logger.exception("fetch_onchain failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
