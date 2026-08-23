"""Fetch derivatives & cross-asset context for the accuracy experiment (Task 2).

Two free, no-auth sources, joined into one daily parquet keyed by ``date``
(UTC midnight):

* **Perp funding rate** — Binance USD-M futures ``/fapi/v1/fundingRate``.
  Settled every 8 hours; history goes back to contract launch (BTCUSDT: 2019).
  A persistently positive funding rate means longs are paying shorts (crowded
  long positioning) — a genuinely predictive sentiment signal that is *not*
  derivable from spot candles.

* **Cross-asset returns** — DXY (dollar index), S&P 500, gold, via yfinance.
  BTC's correlation to the dollar and risk assets is regime-dependent; a
  1-day-lagged cross-asset return is a cheap macro context feature.

Honest limitation (documented in the manifest): Binance's **open interest** and
**perp-spot basis** history endpoints only serve the *last 30 days*, so they
cannot be turned into a multi-year backtestable feature for free and are
deliberately excluded here.  Funding rate is the one derivatives signal with a
full free history.

Both network calls are injectable (``page_getter`` / ``downloader``) so the unit
tests never touch the network.

Run from the repo root::

    python -m src.data.fetch_derivatives
    python -m src.data.fetch_derivatives --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

BINANCE_FUTURES_URL = "https://fapi.binance.com"
FUNDING_PATH = "/fapi/v1/fundingRate"
FUNDING_PAGE_LIMIT = 1000  # Binance hard cap per request

# yfinance tickers → friendly column names used in features (``x_`` = cross-asset)
DEFAULT_CROSS_ASSETS: dict[str, str] = {
    "x_dxy": "DX-Y.NYB",   # US dollar index
    "x_spx": "^GSPC",      # S&P 500
    "x_gold": "GC=F",      # gold futures
}

# A page getter takes request params and returns the raw list of funding dicts.
PageGetter = Callable[[dict[str, Any]], list[dict[str, Any]]]
# A downloader takes (tickers, start, end) and returns a wide close-price frame.
Downloader = Callable[[list[str], str, str | None], pd.DataFrame]


# ---------------------------------------------------------------------------
# Funding rate (Binance USD-M futures)
# ---------------------------------------------------------------------------


def _requests_page_getter(
    base_url: str,
    *,
    timeout_s: float,
    max_retries: int,
    backoff_base: float,
) -> PageGetter:
    """Build a ``PageGetter`` that hits Binance with retries + backoff."""
    url = f"{base_url}{FUNDING_PATH}"

    def _get(params: dict[str, Any]) -> list[dict[str, Any]]:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=timeout_s)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    raise ValueError(f"funding API returned non-list: {type(data)}")
                return data
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait = backoff_base * (2**attempt)
                    logger.warning(
                        "funding: attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1, max_retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "funding: all %d attempts failed; last error: %s",
                        max_retries, exc,
                    )
        raise last_exc or RuntimeError("funding: fetch failed with no error")

    return _get


def fetch_funding_rate(
    symbol: str,
    *,
    start_ms: int,
    end_ms: int | None = None,
    base_url: str = BINANCE_FUTURES_URL,
    limit: int = FUNDING_PAGE_LIMIT,
    timeout_s: float = 30.0,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    page_getter: PageGetter | None = None,
) -> pd.DataFrame:
    """Fetch the full perp funding-rate history, paginating forward by time.

    The endpoint returns at most ``limit`` (1000) rows per call, so we walk
    forward using ``startTime = last fundingTime + 1`` until a short page comes
    back.  Results are de-duplicated on ``fundingTime`` and sorted ascending.

    Args:
        symbol: Futures symbol, e.g. ``"BTCUSDT"``.
        start_ms: Inclusive start time in epoch milliseconds.
        end_ms: Optional exclusive-ish upper bound in epoch milliseconds.
        base_url: Futures API base URL.
        limit: Rows per page (Binance caps at 1000).
        timeout_s: HTTP timeout.
        max_retries: Attempts per page before giving up.
        backoff_base: Exponential-backoff base seconds.
        page_getter: Injectable page fetcher (for tests); defaults to a
            retrying ``requests`` getter.

    Returns:
        DataFrame with columns ``funding_time`` (UTC datetime) and
        ``funding_rate`` (float), sorted ascending.  Empty if no data.
    """
    getter = page_getter or _requests_page_getter(
        base_url, timeout_s=timeout_s, max_retries=max_retries,
        backoff_base=backoff_base,
    )

    rows: list[dict[str, Any]] = []
    cursor = int(start_ms)
    while True:
        params: dict[str, Any] = {"symbol": symbol, "startTime": cursor, "limit": limit}
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        page = getter(params)
        if not page:
            break
        rows.extend(page)
        last_time = int(page[-1]["fundingTime"])
        if len(page) < limit:
            break
        next_cursor = last_time + 1
        if next_cursor <= cursor:  # defensive: no forward progress
            break
        cursor = next_cursor
        if end_ms is not None and cursor > int(end_ms):
            break

    if not rows:
        logger.warning("funding: no rows returned for %s", symbol)
        return pd.DataFrame(columns=["funding_time", "funding_rate"])

    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype("float64")
    df = (
        df[["funding_time", "funding_rate"]]
        .drop_duplicates(subset="funding_time", keep="last")
        .sort_values("funding_time")
        .reset_index(drop=True)
    )
    logger.info(
        "funding: %d settlements for %s, %s → %s",
        len(df), symbol,
        df["funding_time"].iloc[0].date(), df["funding_time"].iloc[-1].date(),
    )
    return df


def to_daily_funding(funding_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 8-hourly funding settlements into one row per UTC day.

    Args:
        funding_df: Output of :func:`fetch_funding_rate`.

    Returns:
        DataFrame with columns ``date`` (UTC midnight), ``funding_sum``
        (day's total funding), ``funding_mean`` (average settlement) and
        ``funding_count`` (settlements that day, normally 3).
    """
    if funding_df.empty:
        return pd.DataFrame(
            columns=["date", "funding_sum", "funding_mean", "funding_count"]
        )
    df = funding_df.copy()
    df["date"] = df["funding_time"].dt.normalize()
    daily = (
        df.groupby("date")["funding_rate"]
        .agg(funding_sum="sum", funding_mean="mean", funding_count="count")
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["funding_count"] = daily["funding_count"].astype("int64")
    return daily


# ---------------------------------------------------------------------------
# Cross-asset closes (yfinance)
# ---------------------------------------------------------------------------


def _yfinance_downloader(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Default downloader: yfinance daily closes (imported lazily)."""
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, interval="1d",
        auto_adjust=True, progress=False, threads=True,
    )
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    # With multiple tickers, columns are a (field, ticker) MultiIndex.
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return close


def fetch_cross_asset(
    tickers: dict[str, str],
    start: str,
    end: str | None = None,
    *,
    downloader: Downloader | None = None,
) -> pd.DataFrame:
    """Fetch daily closes for macro cross-assets and rename to friendly columns.

    Args:
        tickers: Mapping of ``friendly_name → yfinance_ticker``
            (e.g. ``{"x_dxy": "DX-Y.NYB"}``).
        start: Inclusive start date (``YYYY-MM-DD``).
        end: Optional exclusive end date.
        downloader: Injectable ``(tickers, start, end) → wide close frame``
            (for tests); defaults to yfinance.

    Returns:
        DataFrame with a ``date`` column (UTC midnight) and one float column
        per friendly name.  Weekday-only rows; missing tickers become all-NaN
        columns so the schema is stable.
    """
    dl = downloader or _yfinance_downloader
    close = dl(list(tickers.values()), start, end)

    frame = pd.DataFrame()
    if close is not None and len(close) > 0:
        idx = pd.to_datetime(close.index, utc=True).normalize()
        frame = pd.DataFrame(index=idx)
        for friendly, ticker in tickers.items():
            if ticker in close.columns:
                frame[friendly] = pd.to_numeric(
                    close[ticker].to_numpy(), errors="coerce"
                )
            else:
                logger.warning("cross-asset: ticker %s (%s) missing", ticker, friendly)
                frame[friendly] = float("nan")
    else:
        logger.warning("cross-asset: downloader returned no data")
        for friendly in tickers:
            frame[friendly] = pd.Series(dtype="float64")

    frame.index.name = "date"
    out = frame.sort_index().reset_index()
    if not out.empty:
        logger.info(
            "cross-asset: %d rows, %s → %s, columns=%s",
            len(out), out["date"].iloc[0].date(), out["date"].iloc[-1].date(),
            list(tickers.keys()),
        )
    return out


# ---------------------------------------------------------------------------
# Build & save
# ---------------------------------------------------------------------------


def load_derivatives_config(path: str | Path) -> dict[str, Any]:
    """Read symbol, dates and paths from the pipeline config."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = raw.get("data") or {}
    paths = raw.get("paths") or {}
    return {
        "symbol": data.get("symbol", "BTCUSDT"),
        "start_date": str(data.get("start_date", "2020-01-01")),
        "end_date": data.get("end_date"),
        "raw_dir": paths.get("raw_dir", "data/raw"),
    }


def build_derivatives_raw(
    cfg: dict[str, Any],
    out_path: Path,
    *,
    tickers: dict[str, str] | None = None,
    page_getter: PageGetter | None = None,
    downloader: Downloader | None = None,
) -> dict[str, Any]:
    """Fetch funding + cross-asset, join on ``date``, save parquet + manifest.

    Args:
        cfg: Config dict from :func:`load_derivatives_config`.
        out_path: Output parquet path.
        tickers: Cross-asset ticker map (defaults to :data:`DEFAULT_CROSS_ASSETS`).
        page_getter: Injectable funding page fetcher (tests).
        downloader: Injectable cross-asset downloader (tests).

    Returns:
        Manifest dict describing the saved file.
    """
    tickers = tickers or DEFAULT_CROSS_ASSETS
    symbol = cfg["symbol"]
    start_date = cfg["start_date"]
    end_date = cfg.get("end_date")
    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = (
        int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
        if end_date not in (None, "null", "today", "now")
        else None
    )

    funding_raw = fetch_funding_rate(
        symbol, start_ms=start_ms, end_ms=end_ms, page_getter=page_getter
    )
    funding_daily = to_daily_funding(funding_raw)
    cross = fetch_cross_asset(
        tickers, start_date, end_date if isinstance(end_date, str) else None,
        downloader=downloader,
    )

    if funding_daily.empty and cross.empty:
        raise ValueError("derivatives: both funding and cross-asset fetches were empty")

    if funding_daily.empty:
        df = cross
    elif cross.empty:
        df = funding_daily
    else:
        df = funding_daily.merge(cross, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)

    value_cols = [c for c in df.columns if c != "date"]
    coverage = {
        col: {
            "n_valid": int(df[col].notna().sum()),
            "first": (
                str(df.loc[df[col].notna(), "date"].iloc[0].date())
                if df[col].notna().any() else ""
            ),
            "last": (
                str(df.loc[df[col].notna(), "date"].iloc[-1].date())
                if df[col].notna().any() else ""
            ),
        }
        for col in value_cols
    }
    manifest: dict[str, Any] = {
        "built_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "sources": {
            "funding_rate": f"Binance USD-M futures {FUNDING_PATH} (free, no auth)",
            "cross_asset": "yfinance daily closes (free)",
        },
        "cross_asset_tickers": tickers,
        "not_available_for_free": [
            "open_interest history — Binance futures endpoint serves only the "
            "last 30 days, so it cannot back a multi-year feature",
            "perp-spot basis history — same 30-day limitation",
            "aggregated liquidations — no free historical endpoint",
        ],
        "reporting_lag_days": 1,
        "row_count": len(df),
        "date_range": {
            "first": str(df["date"].iloc[0].date()) if len(df) else "",
            "last": str(df["date"].iloc[-1].date()) if len(df) else "",
        },
        "coverage_by_column": coverage,
        "iron_rule": (
            "features apply a 1-day lag so candle T uses only derivatives data "
            "from day T-1 and earlier"
        ),
        "output": {"path": str(out_path), "bytes": out_path.stat().st_size},
    }
    manifest_path = out_path.parent / f"{out_path.stem}.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    logger.info(
        "Saved %d daily rows, %d value columns to %s (%.2f KiB)",
        len(df), len(value_cols), out_path, out_path.stat().st_size / 1024,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Fetch derivatives + cross-asset data and save to data/raw/."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.fetch_derivatives",
        description="Fetch perp funding rate + cross-asset closes (free sources).",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--interval", default="1d",
                        help="label for the output filename (default: %(default)s)")
    parser.add_argument("--out", default=None, help="override output parquet path")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    cfg = load_derivatives_config(args.config)
    out_path = (
        Path(args.out) if args.out
        else Path(cfg["raw_dir"]) / f"derivatives_{args.interval}.parquet"
    )
    try:
        manifest = build_derivatives_raw(cfg, out_path)
        print(f"\nSaved {manifest['row_count']} daily rows to {out_path}")
        print(
            f"Date range: {manifest['date_range']['first']} → "
            f"{manifest['date_range']['last']}"
        )
        print("\nCoverage by column:")
        for col, info in manifest["coverage_by_column"].items():
            print(
                f"  {col:<16} {info['n_valid']:>5d} valid"
                f"  ({info['first']} → {info['last']})"
            )
        print("\nNot available for free (excluded honestly):")
        for note in manifest["not_available_for_free"]:
            print(f"  - {note}")
        return 0
    except Exception:
        logger.exception("fetch_derivatives failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
