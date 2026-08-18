"""Fetch BTC/USDT klines from the Binance public REST API into local parquet files.

Downloads spot-market candlesticks from the unauthenticated ``/api/v3/klines``
endpoint with correct pagination (1000-candle pages, ``startTime`` advanced from
the last ``open_time``), retries transient failures with exponential backoff,
and validates the result before anything is written: UTC timestamps end to end,
strictly increasing open times, no duplicates, no nulls in OHLCV, consistent
OHLC values. Missing candles are detected and logged with their timestamps —
they are never silently dropped or filled.

Only fully *closed* candles are stored: the still-forming candle at the tail of
a pull is dropped so stored values can never change after the fact, which would
otherwise be a subtle source of look-ahead bias downstream.

Per interval, three artifacts are written to the configured raw-data directory:

* ``{prefix}_{interval}.parquet`` — canonical dataset
* ``{prefix}_{interval}.csv`` — plain-text copy for easy inspection
* ``{prefix}_{interval}.manifest.json`` — pull metadata (time, row count, date
  range, gaps, file checksums)

Run the full pull described by ``configs/config.yaml`` from the repo root::

    python -m src.data.fetch_binance
    python -m src.data.fetch_binance --intervals 1d --log-level DEBUG
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

TimestampLike = str | dt.date | dt.datetime | pd.Timestamp

DEFAULT_BASE_URL = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"
MAX_PAGE_LIMIT = 1000
USER_AGENT = "btc-behavior-indicator/0.1 (research data pipeline)"
RETRYABLE_STATUS = frozenset({418, 429})

#: Raw column layout of one kline as returned by Binance (a 12-element array).
RAW_KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "n_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)
#: Columns of the DataFrame produced by this module (``ignore`` is dropped).
EXPECTED_COLUMNS = tuple(c for c in RAW_KLINE_COLUMNS if c != "ignore")
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
FLOAT_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)

#: Fixed-length Binance intervals in milliseconds. Calendar intervals ("1M")
#: have no fixed length, so gap detection is impossible; they are unsupported.
INTERVAL_TO_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

_BINANCE_DEFAULTS: dict[str, Any] = {
    "base_url": DEFAULT_BASE_URL,
    "page_limit": MAX_PAGE_LIMIT,
    "request_timeout_s": 30.0,
    "max_retries": 5,
    "backoff_base_s": 1.0,
    "sleep_between_requests_s": 0.25,
}


class BinanceAPIError(RuntimeError):
    """The Binance API could not be queried successfully."""


class DataValidationError(ValueError):
    """Fetched data failed an integrity check and must not be saved."""


def interval_to_ms(interval: str) -> int:
    """Return the length of a Binance interval in milliseconds.

    Args:
        interval: Binance interval string, e.g. ``"30m"``, ``"1h"``, ``"1d"``.

    Returns:
        Interval length in milliseconds.

    Raises:
        ValueError: If the interval is unknown or has no fixed length (``"1M"``).
    """
    try:
        return INTERVAL_TO_MS[interval]
    except KeyError:
        raise ValueError(
            f"unsupported interval {interval!r}; supported: {sorted(INTERVAL_TO_MS)}"
        ) from None


def _ensure_utc(value: TimestampLike, name: str) -> pd.Timestamp:
    """Parse ``value`` into a timezone-aware UTC ``pd.Timestamp``.

    Naive inputs are interpreted as UTC; aware inputs are converted to UTC.
    """
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}={value!r} is not a recognisable timestamp") from exc
    if ts is pd.NaT:
        raise ValueError(f"{name}={value!r} is not a recognisable timestamp")
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _to_epoch_ms(ts: pd.Timestamp) -> int:
    """Convert a timezone-aware timestamp to integer epoch milliseconds."""
    return int(ts.value // 1_000_000)


def _ms_to_ts(ms: int) -> pd.Timestamp:
    """Convert integer epoch milliseconds to a UTC ``pd.Timestamp``."""
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def _request_page(
    session: requests.Session,
    base_url: str,
    params: dict[str, Any],
    *,
    max_retries: int,
    backoff_base_s: float,
    timeout_s: float,
) -> list[list[Any]]:
    """Request one page of klines, retrying transient failures.

    Retries connection errors, timeouts, HTTP 5xx, and rate-limit responses
    (429/418, honouring ``Retry-After``) with exponential backoff. Any other
    HTTP status is treated as non-retryable and raised immediately.

    Args:
        session: HTTP session to issue the request through.
        base_url: Binance API base URL.
        params: Query parameters for ``/api/v3/klines``.
        max_retries: Number of retries after the initial attempt.
        backoff_base_s: Backoff is ``backoff_base_s * 2**attempt`` seconds.
        timeout_s: Per-request timeout in seconds.

    Returns:
        The decoded JSON payload: a list of raw kline arrays.

    Raises:
        BinanceAPIError: On non-retryable errors or once retries are exhausted.
    """
    url = base_url.rstrip("/") + KLINES_PATH
    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout_s)
        except (requests.ConnectionError, requests.Timeout) as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if attempt == max_retries:
                raise BinanceAPIError(
                    f"request failed after {max_retries} retries ({detail})"
                ) from exc
            delay = backoff_base_s * 2**attempt
            logger.warning(
                "transient network error (%s); retry %d/%d in %.1fs",
                detail,
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
            continue

        if response.status_code == 200:
            used_weight = response.headers.get("x-mbx-used-weight-1m")
            if used_weight is not None:
                logger.debug("request weight used in 1m window: %s", used_weight)
            try:
                payload = response.json()
            except ValueError as exc:
                raise BinanceAPIError(
                    f"HTTP 200 with undecodable JSON body: {response.text[:200]!r}"
                ) from exc
            if not isinstance(payload, list):
                raise BinanceAPIError(f"unexpected response payload: {str(payload)[:200]!r}")
            return payload

        if response.status_code in RETRYABLE_STATUS or response.status_code >= 500:
            if attempt == max_retries:
                raise BinanceAPIError(
                    f"request failed after {max_retries} retries "
                    f"(HTTP {response.status_code}: {response.text[:200]!r})"
                )
            delay = backoff_base_s * 2**attempt
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    logger.debug("ignoring unparseable Retry-After header: %r", retry_after)
            logger.warning(
                "HTTP %d from Binance; retry %d/%d in %.1fs",
                response.status_code,
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
            continue

        hint = ""
        if response.status_code == 451:
            hint = (
                " (Binance blocks this endpoint for some regions; try base_url"
                " https://data-api.binance.vision, the official market-data mirror)"
            )
        raise BinanceAPIError(
            f"non-retryable HTTP {response.status_code}: {response.text[:200]!r}{hint}"
        )

    raise BinanceAPIError("request failed")  # pragma: no cover - loop always returns or raises


def _klines_to_dataframe(raw: list[list[Any]]) -> pd.DataFrame:
    """Convert raw Binance kline arrays into a typed DataFrame.

    Args:
        raw: List of 12-element kline arrays as returned by the API.

    Returns:
        DataFrame with ``EXPECTED_COLUMNS``: UTC datetimes for ``open_time`` /
        ``close_time``, float64 prices and volumes, int64 trade counts.
    """
    if raw:
        df = pd.DataFrame(raw, columns=list(RAW_KLINE_COLUMNS))
    else:
        df = pd.DataFrame(columns=list(RAW_KLINE_COLUMNS))
    df = df.drop(columns=["ignore"])
    for col in ("open_time", "close_time"):
        df[col] = pd.to_datetime(df[col].astype("int64"), unit="ms", utc=True)
    for col in FLOAT_COLUMNS:
        df[col] = df[col].astype("float64")
    df["n_trades"] = df["n_trades"].astype("int64")
    return df


def _drop_duplicates(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Remove fully identical duplicate candles (e.g. pagination overlap).

    Identical duplicates are dropped with a warning listing their timestamps.
    Duplicated ``open_time`` values with *conflicting* data are an integrity
    failure and raise instead — they must never be silently resolved.

    Args:
        df: Candle DataFrame, possibly containing duplicates.
        context: Label such as ``"BTCUSDT 1h"`` used in log messages.

    Returns:
        DataFrame with unique ``open_time`` values.

    Raises:
        DataValidationError: If duplicated timestamps carry conflicting values.
    """
    if not df["open_time"].duplicated().any():
        return df
    deduped = df.drop_duplicates(keep="first")
    conflicting = deduped["open_time"].duplicated(keep=False)
    if conflicting.any():
        conflict_ts = [str(t) for t in deduped.loc[conflicting, "open_time"].unique()[:10]]
        logger.error("%s: conflicting duplicate candles at %s", context, conflict_ts)
        raise DataValidationError(
            f"{context}: conflicting duplicate candles for open_time(s) {conflict_ts}"
        )
    dropped_ts = [str(t) for t in df.loc[df.duplicated(keep="first"), "open_time"].unique()[:10]]
    logger.warning(
        "%s: dropped %d identical duplicate candle(s) at %s (pagination overlap)",
        context,
        len(df) - len(deduped),
        dropped_ts,
    )
    return deduped.reset_index(drop=True)


def fetch_klines(
    symbol: str,
    interval: str,
    start_date: TimestampLike,
    end_date: TimestampLike | None = None,
    *,
    base_url: str = DEFAULT_BASE_URL,
    session: requests.Session | None = None,
    page_limit: int = MAX_PAGE_LIMIT,
    sleep_between_requests_s: float = 0.25,
    max_retries: int = 5,
    backoff_base_s: float = 1.0,
    request_timeout_s: float = 30.0,
) -> pd.DataFrame:
    """Fetch spot klines for ``[start_date, end_date)`` from Binance.

    Pages through ``/api/v3/klines`` (max 1000 candles per request), advancing
    ``startTime`` from the last received ``open_time``. The still-forming
    candle at the tail is dropped so only fully closed candles are returned.

    Args:
        symbol: Trading pair, e.g. ``"BTCUSDT"``.
        interval: Binance interval string, e.g. ``"1d"``, ``"1h"``, ``"30m"``.
        start_date: Inclusive range start (naive values are treated as UTC).
        end_date: Exclusive range end; ``None`` means "now", i.e. up to the
            most recent fully closed candle at call time.
        base_url: Binance API base URL.
        session: Optional HTTP session (injected in tests); a dedicated session
            is created and closed internally when omitted.
        page_limit: Candles per request, 1..1000.
        sleep_between_requests_s: Polite pause between successive pages.
        max_retries: Retries per page on transient errors.
        backoff_base_s: Exponential-backoff base in seconds.
        request_timeout_s: Per-request timeout in seconds.

    Returns:
        Validated-shape DataFrame of candles ordered as received from the API
        (one row per closed candle, columns per ``EXPECTED_COLUMNS``).

    Raises:
        ValueError: On invalid symbol, interval, range, or page limit.
        BinanceAPIError: If the API cannot be queried or pagination misbehaves.
        DataValidationError: If duplicated timestamps conflict.
    """
    interval_ms = interval_to_ms(interval)
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    symbol = symbol.strip().upper()
    if not 1 <= page_limit <= MAX_PAGE_LIMIT:
        raise ValueError(f"page_limit must be in [1, {MAX_PAGE_LIMIT}], got {page_limit}")
    start_ts = _ensure_utc(start_date, "start_date")
    end_ts = pd.Timestamp.now(tz="UTC") if end_date is None else _ensure_utc(end_date, "end_date")
    if start_ts >= end_ts:
        raise ValueError(f"start_date ({start_ts}) must be before end_date ({end_ts})")

    start_ms = _to_epoch_ms(start_ts)
    end_ms = _to_epoch_ms(end_ts)
    context = f"{symbol} {interval}"
    # Upper bound on pages for the requested range; exceeding it means the
    # pagination cursor is not advancing sanely.
    max_pages = math.ceil((end_ms - start_ms) / (interval_ms * page_limit)) + 5

    own_session = session is None
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

    rows: list[list[Any]] = []
    pages = 0
    cursor = start_ms
    logger.info(
        "%s: fetching klines from %s to %s (page size %d)",
        context,
        start_ts.isoformat(),
        end_ts.isoformat(),
        page_limit,
    )
    try:
        while cursor < end_ms:
            if pages >= max_pages:
                raise BinanceAPIError(
                    f"{context}: exceeded the expected page budget ({max_pages}); "
                    "aborting to avoid an unbounded pagination loop"
                )
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms - 1,  # Binance treats endTime as inclusive
                "limit": page_limit,
            }
            batch = _request_page(
                session,
                base_url,
                params,
                max_retries=max_retries,
                backoff_base_s=backoff_base_s,
                timeout_s=request_timeout_s,
            )
            pages += 1
            if not batch:
                break
            rows.extend(batch)
            last_open_ms = int(batch[-1][0])
            next_cursor = last_open_ms + interval_ms
            if next_cursor <= cursor:
                raise BinanceAPIError(
                    f"{context}: pagination did not advance past "
                    f"{_ms_to_ts(cursor).isoformat()} (last open_time "
                    f"{_ms_to_ts(last_open_ms).isoformat()})"
                )
            cursor = next_cursor
            if pages % 10 == 0:
                logger.info(
                    "%s: page %d, %d rows so far, through %s",
                    context,
                    pages,
                    len(rows),
                    _ms_to_ts(last_open_ms).isoformat(),
                )
            if len(batch) < page_limit:
                break
            time.sleep(sleep_between_requests_s)
    finally:
        if own_session:
            session.close()

    df = _klines_to_dataframe(rows)
    logger.info("%s: fetched %d row(s) over %d page(s)", context, len(df), pages)
    if df.empty:
        return df

    in_range = (df["open_time"] >= start_ts) & (df["open_time"] < end_ts)
    if not in_range.all():
        logger.debug(
            "%s: dropping %d row(s) outside the requested [start, end) range",
            context,
            int((~in_range).sum()),
        )
        df = df.loc[in_range].reset_index(drop=True)

    now = pd.Timestamp.now(tz="UTC")
    still_open = df["close_time"] >= now
    if still_open.any():
        open_times = [t.isoformat() for t in df.loc[still_open, "open_time"]]
        logger.info(
            "%s: dropping %d in-progress candle(s) opened at %s — only fully closed "
            "candles are kept so stored values can never change after the fact "
            "(look-ahead safety)",
            context,
            int(still_open.sum()),
            open_times,
        )
        df = df.loc[~still_open].reset_index(drop=True)

    df = _drop_duplicates(df, context)

    if not df.empty and df["open_time"].iloc[0] > start_ts + pd.Timedelta(milliseconds=interval_ms):
        logger.warning(
            "%s: first candle is %s, later than the requested start %s — the exchange "
            "has no earlier data for this range",
            context,
            df["open_time"].iloc[0].isoformat(),
            start_ts.isoformat(),
        )
    return df


def find_gaps(df: pd.DataFrame, interval: str, context: str = "") -> list[dict[str, Any]]:
    """Detect missing candles between the first and last observed ``open_time``.

    Contiguous runs of missing candles are reported as gap records and logged
    with their exact timestamps. The data itself is left untouched — gaps are
    never filled.

    Args:
        df: Candle DataFrame with a UTC ``open_time`` column, sorted ascending.
        interval: Binance interval string used to build the expected time grid.
        context: Label such as ``"BTCUSDT 1h"`` used in log messages.

    Returns:
        One record per gap: ``{"start", "end", "n_missing"}`` where ``start`` /
        ``end`` are the ISO timestamps of the first and last missing candle
        (inclusive). Empty list when no candles are missing.
    """
    label = context or "klines"
    step = pd.Timedelta(milliseconds=interval_to_ms(interval))
    if len(df) < 2:
        return []
    observed = pd.DatetimeIndex(df["open_time"])
    expected = pd.date_range(observed[0], observed[-1], freq=step)
    missing = expected.difference(observed)
    if missing.empty:
        logger.info("%s: no gaps — all %d expected candles present", label, len(expected))
        return []

    def _record(run_start: pd.Timestamp, run_end: pd.Timestamp) -> dict[str, Any]:
        return {
            "start": run_start.isoformat(),
            "end": run_end.isoformat(),
            "n_missing": int((run_end - run_start) / step) + 1,
        }

    gaps: list[dict[str, Any]] = []
    run_start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        if ts - prev != step:
            gaps.append(_record(run_start, prev))
            run_start = ts
        prev = ts
    gaps.append(_record(run_start, prev))

    for gap in gaps:
        logger.warning(
            "%s: gap of %d missing candle(s) from %s to %s (inclusive) — recorded, not filled",
            label,
            gap["n_missing"],
            gap["start"],
            gap["end"],
        )
    logger.warning(
        "%s: %d candle(s) missing in total across %d gap(s)", label, len(missing), len(gaps)
    )
    return gaps


def validate_klines(df: pd.DataFrame, interval: str, context: str = "") -> list[dict[str, Any]]:
    """Validate candle data before it may be saved.

    Hard failures (raise ``DataValidationError``): empty data, missing columns,
    non-UTC timestamps, nulls in OHLCV, duplicated or non-strictly-increasing
    ``open_time``, inconsistent OHLC values, negative volume. Gaps are *not* a
    failure: real exchange outages exist; they are logged and returned so the
    caller can record them.

    Args:
        df: Candle DataFrame as produced by :func:`fetch_klines`.
        interval: Binance interval string.
        context: Label such as ``"BTCUSDT 1h"`` used in log messages.

    Returns:
        Gap records from :func:`find_gaps`.

    Raises:
        DataValidationError: If any integrity check fails.
    """
    label = context or "klines"

    def _fail(msg: str) -> None:
        logger.error("%s: validation failed — %s", label, msg)
        raise DataValidationError(f"{label}: {msg}")

    if df.empty:
        _fail("no rows to validate")
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        _fail(f"missing columns {missing_cols}")
    for col in ("open_time", "close_time"):
        if not isinstance(df[col].dtype, pd.DatetimeTZDtype) or str(df[col].dt.tz) != "UTC":
            _fail(f"{col} must be timezone-aware UTC datetimes, got dtype {df[col].dtype}")

    null_counts = df[list(OHLCV_COLUMNS)].isna().sum()
    if null_counts.any():
        with_nulls = {col: int(n) for col, n in null_counts.items() if n}
        first_bad = df.loc[df[list(OHLCV_COLUMNS)].isna().any(axis=1), "open_time"].iloc[0]
        _fail(f"null values in OHLCV columns {with_nulls}, first at {first_bad.isoformat()}")

    duplicated = df["open_time"].duplicated()
    if duplicated.any():
        dup_ts = [str(t) for t in df.loc[duplicated, "open_time"].unique()[:10]]
        _fail(f"{int(duplicated.sum())} duplicated open_time value(s), e.g. {dup_ts}")

    deltas = df["open_time"].diff().iloc[1:]
    non_increasing = deltas <= pd.Timedelta(0)
    if non_increasing.any():
        first_bad = df["open_time"].iloc[1:][non_increasing].iloc[0]
        _fail(f"open_time is not strictly increasing at {first_bad.isoformat()}")

    inconsistent = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["volume"] < 0)
    )
    if inconsistent.any():
        first_bad = df.loc[inconsistent, "open_time"].iloc[0]
        _fail(
            f"{int(inconsistent.sum())} candle(s) with inconsistent OHLCV values, "
            f"first at {first_bad.isoformat()}"
        )

    return find_gaps(df, interval, context=label)


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_klines(
    df: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    raw_dir: str | Path,
    file_prefix: str,
    gaps: list[dict[str, Any]],
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    source_base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Write the canonical parquet, a CSV copy, and a manifest JSON.

    Args:
        df: Validated, non-empty candle DataFrame.
        symbol: Trading pair the data belongs to.
        interval: Binance interval string.
        raw_dir: Output directory (created if missing).
        file_prefix: File-name prefix, e.g. ``"btc_usdt"``.
        gaps: Gap records returned by :func:`validate_klines`.
        requested_start: Inclusive range start of the pull.
        requested_end: Exclusive range end of the pull.
        source_base_url: API base URL recorded in the manifest.

    Returns:
        The manifest dictionary that was written to disk.
    """
    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{file_prefix}_{interval}"
    parquet_path = out_dir / f"{stem}.parquet"
    csv_path = out_dir / f"{stem}.csv"
    manifest_path = out_dir / f"{stem}.manifest.json"

    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    df.to_csv(csv_path, index=False)

    n_missing = sum(gap["n_missing"] for gap in gaps)
    manifest: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "pulled_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "source": {"base_url": source_base_url, "endpoint": KLINES_PATH},
        "requested_start_utc": requested_start.isoformat(),
        "requested_end_utc": requested_end.isoformat(),
        "first_open_time_utc": df["open_time"].iloc[0].isoformat(),
        "last_open_time_utc": df["open_time"].iloc[-1].isoformat(),
        "row_count": int(len(df)),
        "expected_row_count": int(len(df) + n_missing),
        "missing_candle_count": int(n_missing),
        "gaps": gaps,
        "files": {
            "parquet": {
                "path": str(parquet_path),
                "bytes": parquet_path.stat().st_size,
                "sha256": sha256_of(parquet_path),
            },
            "csv": {"path": str(csv_path), "bytes": csv_path.stat().st_size},
        },
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    logger.info(
        "%s %s: wrote %s (%.2f MiB), %s (%.2f MiB) and %s",
        symbol,
        interval,
        parquet_path,
        parquet_path.stat().st_size / 2**20,
        csv_path,
        csv_path.stat().st_size / 2**20,
        manifest_path,
    )
    return manifest


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the pipeline configuration file.

    Args:
        path: Path to the YAML config (see ``configs/config.yaml``).

    Returns:
        Normalised config: ``symbol``, ``intervals``, ``start_date``,
        ``end_date`` (``None`` means "now"), ``raw_dir``, ``file_prefix`` and a
        ``binance`` dict with defaults merged in.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid YAML or fails schema validation.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"config file {config_path} is not valid YAML: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"config file {config_path} must contain a YAML mapping")

    data = cfg.get("data")
    if not isinstance(data, dict):
        raise ValueError("config: missing 'data' section")
    symbol = data.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("config: data.symbol must be a non-empty string")
    intervals = data.get("intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("config: data.intervals must be a non-empty list")
    unknown_intervals = [i for i in intervals if i not in INTERVAL_TO_MS]
    if unknown_intervals:
        raise ValueError(
            f"config: unsupported interval(s) {unknown_intervals}; "
            f"supported: {sorted(INTERVAL_TO_MS)}"
        )
    start_date = data.get("start_date")
    if start_date is None:
        raise ValueError("config: data.start_date is required")
    _ensure_utc(start_date, "data.start_date")
    end_date = data.get("end_date")
    if isinstance(end_date, str) and end_date.strip().lower() in {"today", "now"}:
        end_date = None
    if end_date is not None:
        _ensure_utc(end_date, "data.end_date")

    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("config: missing 'paths' section")
    raw_dir = paths.get("raw_dir")
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        raise ValueError("config: paths.raw_dir must be a non-empty string")
    file_prefix = paths.get("file_prefix")
    if not isinstance(file_prefix, str) or not file_prefix.strip():
        raise ValueError("config: paths.file_prefix must be a non-empty string")

    binance = cfg.get("binance") or {}
    if not isinstance(binance, dict):
        raise ValueError("config: 'binance' section must be a mapping")
    unknown_keys = sorted(set(binance) - set(_BINANCE_DEFAULTS))
    if unknown_keys:
        raise ValueError(f"config: unknown binance option(s): {unknown_keys}")
    binance_cfg = {**_BINANCE_DEFAULTS, **binance}
    if not 1 <= int(binance_cfg["page_limit"]) <= MAX_PAGE_LIMIT:
        raise ValueError(f"config: binance.page_limit must be in [1, {MAX_PAGE_LIMIT}]")

    return {
        "symbol": symbol.strip().upper(),
        "intervals": list(intervals),
        "start_date": start_date,
        "end_date": end_date,
        "raw_dir": raw_dir,
        "file_prefix": file_prefix,
        "binance": binance_cfg,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the full pull described by the config file.

    Args:
        argv: CLI arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 if any interval failed.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.data.fetch_binance",
        description="Fetch, validate and store Binance klines as configured.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="path to the pipeline config (default: %(default)s)",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        choices=sorted(INTERVAL_TO_MS),
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

    cfg = load_config(args.config)
    intervals: list[str] = args.intervals or cfg["intervals"]
    requested_start = _ensure_utc(cfg["start_date"], "start_date")
    requested_end = (
        pd.Timestamp.now(tz="UTC")
        if cfg["end_date"] is None
        else _ensure_utc(cfg["end_date"], "end_date")
    )

    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for interval in intervals:
        context = f"{cfg['symbol']} {interval}"
        try:
            df = fetch_klines(
                cfg["symbol"],
                interval,
                requested_start,
                requested_end,
                **cfg["binance"],
            )
            gaps = validate_klines(df, interval, context=context)
            manifest = save_klines(
                df,
                symbol=cfg["symbol"],
                interval=interval,
                raw_dir=cfg["raw_dir"],
                file_prefix=cfg["file_prefix"],
                gaps=gaps,
                requested_start=requested_start,
                requested_end=requested_end,
                source_base_url=cfg["binance"]["base_url"],
            )
            summaries.append(manifest)
        except Exception:
            logger.exception("%s: pull failed", context)
            failures.append(interval)

    for manifest in summaries:
        logger.info(
            "%s %s: %d rows, %s -> %s, %d missing candle(s) in %d gap(s)",
            manifest["symbol"],
            manifest["interval"],
            manifest["row_count"],
            manifest["first_open_time_utc"],
            manifest["last_open_time_utc"],
            manifest["missing_candle_count"],
            len(manifest["gaps"]),
        )
    if failures:
        logger.error("pull failed for interval(s): %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
