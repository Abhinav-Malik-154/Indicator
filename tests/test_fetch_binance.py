"""Unit tests for ``src.data.fetch_binance``.

All network access is mocked; an autouse fixture makes any real HTTP request
fail the test immediately.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest
import requests

from src.data import fetch_binance as fb

HOUR_MS = 3_600_000
BASE_MS = 1_609_459_200_000  # 2021-01-01T00:00:00Z


def ts(ms: int) -> pd.Timestamp:
    """Epoch milliseconds -> UTC Timestamp."""
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def make_kline(open_ms: int, interval_ms: int = HOUR_MS, price: float = 100.0) -> list[Any]:
    """Build one raw 12-element kline array as Binance returns it (strings!)."""
    return [
        open_ms,
        f"{price:.2f}",
        f"{price * 1.01:.2f}",
        f"{price * 0.99:.2f}",
        f"{price * 1.005:.2f}",
        "12.34",
        open_ms + interval_ms - 1,
        "1234.56",
        42,
        "6.17",
        "617.28",
        "0",
    ]


def make_page(start_ms: int, n: int, interval_ms: int = HOUR_MS) -> list[list[Any]]:
    """Build ``n`` consecutive klines starting at ``start_ms``."""
    return [make_kline(start_ms + i * interval_ms, interval_ms) for i in range(n)]


def clean_df(n: int = 48, interval_ms: int = HOUR_MS) -> pd.DataFrame:
    """A well-formed candle DataFrame with ``n`` consecutive hourly rows."""
    return fb._klines_to_dataframe(make_page(BASE_MS, n, interval_ms))


@dataclass
class FakeResponse:
    status_code: int = 200
    payload: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""

    def json(self) -> Any:
        if self.payload is None:
            raise ValueError("no JSON body")
        return self.payload


class FakeSession:
    """Stands in for ``requests.Session``; replays a scripted response list.

    Script items may be raw kline pages (wrapped in a 200 response), prepared
    ``FakeResponse`` objects, or exceptions to raise.
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict | None = None, timeout: float | None = None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        if not self._script:
            raise AssertionError("FakeSession ran out of scripted responses")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, FakeResponse):
            return item
        return FakeResponse(payload=item)

    def close(self) -> None:  # pragma: no cover - fetcher only closes owned sessions
        pass


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    """Fail loudly if any test ever reaches for the real network."""

    def _blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("test attempted real network access")

    monkeypatch.setattr("requests.sessions.Session.request", _blocked)


@pytest.fixture()
def sleeps(monkeypatch):
    """Record ``time.sleep`` calls instead of actually sleeping."""
    recorded: list[float] = []
    monkeypatch.setattr(fb.time, "sleep", recorded.append)
    return recorded


class TestIntervalToMs:
    def test_known_intervals(self):
        assert fb.interval_to_ms("30m") == 30 * 60 * 1000
        assert fb.interval_to_ms("1h") == HOUR_MS
        assert fb.interval_to_ms("1d") == 24 * HOUR_MS

    def test_unknown_interval_raises(self):
        with pytest.raises(ValueError, match="unsupported interval"):
            fb.interval_to_ms("7x")

    def test_calendar_month_rejected(self):
        with pytest.raises(ValueError, match="unsupported interval"):
            fb.interval_to_ms("1M")


class TestKlinesToDataframe:
    def test_columns_and_dtypes(self):
        df = clean_df(3)
        assert list(df.columns) == list(fb.EXPECTED_COLUMNS)
        # Millisecond resolution — Binance's native timestamp precision.
        assert str(df["open_time"].dtype) == "datetime64[ms, UTC]"
        assert str(df["close_time"].dtype) == "datetime64[ms, UTC]"
        for col in fb.FLOAT_COLUMNS:
            assert df[col].dtype == "float64", col
        assert df["n_trades"].dtype == "int64"

    def test_values_parsed_from_strings(self):
        df = fb._klines_to_dataframe([make_kline(BASE_MS, price=250.0)])
        assert df.loc[0, "open"] == pytest.approx(250.0)
        assert df.loc[0, "high"] == pytest.approx(252.5)
        assert df.loc[0, "n_trades"] == 42
        assert df.loc[0, "open_time"] == ts(BASE_MS)
        assert df.loc[0, "close_time"] == ts(BASE_MS + HOUR_MS - 1)

    def test_timestamps_are_utc(self):
        df = clean_df(2)
        assert str(df["open_time"].dt.tz) == "UTC"
        assert str(df["close_time"].dt.tz) == "UTC"


class TestPagination:
    def test_advances_start_time_and_concatenates(self, sleeps):
        end_ms = BASE_MS + 1500 * HOUR_MS
        session = FakeSession(
            [make_page(BASE_MS, 1000), make_page(BASE_MS + 1000 * HOUR_MS, 500)]
        )
        df = fb.fetch_klines(
            "BTCUSDT",
            "1h",
            ts(BASE_MS),
            ts(end_ms),
            session=session,
            sleep_between_requests_s=0.05,
        )
        assert len(df) == 1500
        assert len(session.calls) == 2
        first, second = (call["params"] for call in session.calls)
        assert first["startTime"] == BASE_MS
        assert first["endTime"] == end_ms - 1
        assert first["limit"] == 1000
        # startTime advances from the last open_time by exactly one interval.
        assert second["startTime"] == BASE_MS + 1000 * HOUR_MS
        assert df["open_time"].is_monotonic_increasing
        assert df["open_time"].is_unique
        # One polite sleep between the two pages, none after the final page.
        assert sleeps == [0.05]

    def test_stops_after_partial_page(self, sleeps):
        session = FakeSession([make_page(BASE_MS, 300)])
        df = fb.fetch_klines(
            "BTCUSDT", "1h", ts(BASE_MS), ts(BASE_MS + 5000 * HOUR_MS), session=session
        )
        assert len(df) == 300
        assert len(session.calls) == 1

    def test_stops_on_empty_page(self, sleeps):
        session = FakeSession([[]])
        df = fb.fetch_klines(
            "BTCUSDT", "1h", ts(BASE_MS), ts(BASE_MS + 10 * HOUR_MS), session=session
        )
        assert df.empty
        assert list(df.columns) == list(fb.EXPECTED_COLUMNS)
        assert len(session.calls) == 1

    def test_runaway_pagination_aborts(self, sleeps):
        # A server bug that keeps returning the same page must abort, not loop.
        page = make_page(BASE_MS, 1000)
        session = FakeSession([page, page])
        with pytest.raises(fb.BinanceAPIError, match="did not advance"):
            fb.fetch_klines(
                "BTCUSDT", "1h", ts(BASE_MS), ts(BASE_MS + 3000 * HOUR_MS), session=session
            )

    def test_incomplete_final_candle_dropped(self, sleeps, caplog):
        now_ms = fb._to_epoch_ms(pd.Timestamp.now(tz="UTC"))
        current_open = (now_ms // HOUR_MS) * HOUR_MS
        closed = [make_kline(current_open - i * HOUR_MS) for i in (3, 2, 1)]
        session = FakeSession([closed + [make_kline(current_open)]])
        with caplog.at_level(logging.INFO):
            df = fb.fetch_klines(
                "BTCUSDT", "1h", ts(current_open - 3 * HOUR_MS), None, session=session
            )
        assert len(df) == 3
        assert (df["close_time"] < pd.Timestamp.now(tz="UTC")).all()
        assert "in-progress" in caplog.text

    def test_rejects_invalid_range(self):
        with pytest.raises(ValueError, match="before end_date"):
            fb.fetch_klines("BTCUSDT", "1h", ts(BASE_MS), ts(BASE_MS), session=FakeSession([]))

    def test_rejects_bad_page_limit(self):
        with pytest.raises(ValueError, match="page_limit"):
            fb.fetch_klines(
                "BTCUSDT",
                "1h",
                ts(BASE_MS),
                ts(BASE_MS + HOUR_MS),
                session=FakeSession([]),
                page_limit=1001,
            )


class TestDuplicates:
    def test_identical_duplicates_dropped_and_logged(self, sleeps, caplog):
        page1 = make_page(BASE_MS, 1000)
        # Second page overlaps by one candle, byte-for-byte identical.
        page2 = [page1[-1]] + make_page(BASE_MS + 1000 * HOUR_MS, 10)
        session = FakeSession([page1, page2])
        with caplog.at_level(logging.WARNING):
            df = fb.fetch_klines(
                "BTCUSDT",
                "1h",
                ts(BASE_MS),
                ts(BASE_MS + 2000 * HOUR_MS),
                session=session,
                sleep_between_requests_s=0,
            )
        assert len(df) == 1010
        assert df["open_time"].is_unique
        assert "duplicate" in caplog.text

    def test_conflicting_duplicates_raise(self, sleeps):
        page1 = make_page(BASE_MS, 1000)
        conflicting = list(page1[-1])
        conflicting[4] = "999.99"  # same open_time, different close price
        page2 = [conflicting] + make_page(BASE_MS + 1000 * HOUR_MS, 10)
        session = FakeSession([page1, page2])
        with pytest.raises(fb.DataValidationError, match="conflicting"):
            fb.fetch_klines(
                "BTCUSDT",
                "1h",
                ts(BASE_MS),
                ts(BASE_MS + 2000 * HOUR_MS),
                session=session,
                sleep_between_requests_s=0,
            )


class TestRetries:
    def test_retries_5xx_then_succeeds(self, sleeps):
        session = FakeSession(
            [
                FakeResponse(status_code=500, text="server error"),
                FakeResponse(status_code=502, text="bad gateway"),
                make_page(BASE_MS, 5),
            ]
        )
        df = fb.fetch_klines(
            "BTCUSDT",
            "1h",
            ts(BASE_MS),
            ts(BASE_MS + 10 * HOUR_MS),
            session=session,
            max_retries=3,
            backoff_base_s=1.0,
        )
        assert len(df) == 5
        assert len(session.calls) == 3
        assert sleeps == [1.0, 2.0]  # exponential backoff

    def test_respects_retry_after_on_429(self, sleeps):
        session = FakeSession(
            [
                FakeResponse(status_code=429, headers={"Retry-After": "7"}, text="rate limited"),
                make_page(BASE_MS, 5),
            ]
        )
        df = fb.fetch_klines(
            "BTCUSDT",
            "1h",
            ts(BASE_MS),
            ts(BASE_MS + 10 * HOUR_MS),
            session=session,
            backoff_base_s=1.0,
        )
        assert len(df) == 5
        assert sleeps == [7.0]

    def test_retries_connection_errors(self, sleeps):
        session = FakeSession([requests.ConnectionError("boom"), make_page(BASE_MS, 5)])
        df = fb.fetch_klines(
            "BTCUSDT", "1h", ts(BASE_MS), ts(BASE_MS + 10 * HOUR_MS), session=session
        )
        assert len(df) == 5
        assert len(session.calls) == 2

    def test_gives_up_after_max_retries(self, sleeps):
        session = FakeSession([FakeResponse(status_code=503, text="down")] * 3)
        with pytest.raises(fb.BinanceAPIError, match="after 2 retries"):
            fb.fetch_klines(
                "BTCUSDT",
                "1h",
                ts(BASE_MS),
                ts(BASE_MS + 10 * HOUR_MS),
                session=session,
                max_retries=2,
            )
        assert len(session.calls) == 3  # initial attempt + 2 retries

    def test_non_retryable_status_fails_immediately(self, sleeps):
        session = FakeSession(
            [FakeResponse(status_code=400, text='{"code":-1121,"msg":"Invalid symbol."}')]
        )
        with pytest.raises(fb.BinanceAPIError, match="non-retryable"):
            fb.fetch_klines(
                "BTCUSDT", "1h", ts(BASE_MS), ts(BASE_MS + 10 * HOUR_MS), session=session
            )
        assert len(session.calls) == 1
        assert sleeps == []


class TestValidation:
    def test_clean_data_passes(self):
        assert fb.validate_klines(clean_df(), "1h", context="test") == []

    def test_empty_raises(self):
        with pytest.raises(fb.DataValidationError, match="no rows"):
            fb.validate_klines(fb._klines_to_dataframe([]), "1h")

    def test_null_in_ohlcv_raises_and_logs(self, caplog):
        df = clean_df()
        df.loc[5, "close"] = None
        with caplog.at_level(logging.ERROR):
            with pytest.raises(fb.DataValidationError, match="null values"):
                fb.validate_klines(df, "1h", context="BTCUSDT 1h")
        assert "validation failed" in caplog.text
        assert "close" in caplog.text

    def test_duplicate_open_time_raises(self, caplog):
        df = clean_df()
        df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
        with caplog.at_level(logging.ERROR):
            with pytest.raises(fb.DataValidationError, match="duplicated open_time"):
                fb.validate_klines(df, "1h")
        assert "validation failed" in caplog.text

    def test_non_increasing_open_time_raises(self):
        df = clean_df().iloc[::-1].reset_index(drop=True)
        with pytest.raises(fb.DataValidationError, match="strictly increasing"):
            fb.validate_klines(df, "1h")

    def test_ohlc_inconsistency_raises(self):
        df = clean_df()
        df.loc[3, "high"] = df.loc[3, "low"] - 1.0
        with pytest.raises(fb.DataValidationError, match="inconsistent OHLCV"):
            fb.validate_klines(df, "1h")

    def test_negative_volume_raises(self):
        df = clean_df()
        df.loc[2, "volume"] = -5.0
        with pytest.raises(fb.DataValidationError, match="inconsistent OHLCV"):
            fb.validate_klines(df, "1h")

    def test_naive_timestamps_raise(self):
        df = clean_df()
        df["open_time"] = df["open_time"].dt.tz_localize(None)
        with pytest.raises(fb.DataValidationError, match="UTC"):
            fb.validate_klines(df, "1h")


class TestGapDetection:
    def test_no_gaps_on_contiguous_data(self, caplog):
        with caplog.at_level(logging.INFO):
            assert fb.find_gaps(clean_df(), "1h") == []
        assert "no gaps" in caplog.text

    def test_single_gap_detected_and_logged_with_timestamps(self, caplog):
        df = clean_df(48).drop(index=[10, 11, 12]).reset_index(drop=True)
        with caplog.at_level(logging.WARNING):
            gaps = fb.find_gaps(df, "1h", context="BTCUSDT 1h")
        assert gaps == [
            {
                "start": "2021-01-01T10:00:00+00:00",
                "end": "2021-01-01T12:00:00+00:00",
                "n_missing": 3,
            }
        ]
        assert "2021-01-01T10:00:00+00:00" in caplog.text
        assert "2021-01-01T12:00:00+00:00" in caplog.text

    def test_multiple_gaps_reported_separately(self):
        df = clean_df(48).drop(index=[5, 20, 21]).reset_index(drop=True)
        gaps = fb.find_gaps(df, "1h")
        assert [gap["n_missing"] for gap in gaps] == [1, 2]
        assert gaps[0]["start"] == gaps[0]["end"] == "2021-01-01T05:00:00+00:00"
        assert gaps[1]["start"] == "2021-01-01T20:00:00+00:00"
        assert gaps[1]["end"] == "2021-01-01T21:00:00+00:00"

    def test_validate_reports_gaps_without_failing(self, caplog):
        df = clean_df(48).drop(index=[7]).reset_index(drop=True)
        with caplog.at_level(logging.WARNING):
            gaps = fb.validate_klines(df, "1h", context="BTCUSDT 1h")
        assert [gap["n_missing"] for gap in gaps] == [1]
        assert "missing" in caplog.text


class TestSaveKlines:
    def test_writes_parquet_csv_and_manifest(self, tmp_path):
        df = clean_df(24)
        gaps = [
            {
                "start": "2021-01-02T00:00:00+00:00",
                "end": "2021-01-02T01:00:00+00:00",
                "n_missing": 2,
            }
        ]
        manifest = fb.save_klines(
            df,
            symbol="BTCUSDT",
            interval="1h",
            raw_dir=tmp_path,
            file_prefix="btc_usdt",
            gaps=gaps,
            requested_start=pd.Timestamp("2021-01-01", tz="UTC"),
            requested_end=pd.Timestamp("2021-01-03", tz="UTC"),
        )
        parquet_path = tmp_path / "btc_usdt_1h.parquet"
        csv_path = tmp_path / "btc_usdt_1h.csv"
        manifest_path = tmp_path / "btc_usdt_1h.manifest.json"
        assert parquet_path.is_file() and csv_path.is_file() and manifest_path.is_file()

        # Parquet round-trips exactly, including tz-aware dtypes.
        pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), df)
        # CSV has a header plus one line per candle.
        assert len(csv_path.read_text().strip().splitlines()) == 25

        on_disk = json.loads(manifest_path.read_text())
        assert on_disk == manifest
        assert on_disk["row_count"] == 24
        assert on_disk["missing_candle_count"] == 2
        assert on_disk["expected_row_count"] == 26
        assert on_disk["gaps"] == gaps
        assert on_disk["first_open_time_utc"] == df["open_time"].iloc[0].isoformat()
        assert on_disk["last_open_time_utc"] == df["open_time"].iloc[-1].isoformat()
        expected_sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        assert on_disk["files"]["parquet"]["sha256"] == expected_sha


VALID_CONFIG = """
data:
  symbol: BTCUSDT
  intervals: ["1d", "1h", "30m"]
  start_date: "2020-01-01"
  end_date: null

binance:
  base_url: "https://api.binance.com"

paths:
  raw_dir: "data/raw"
  file_prefix: "btc_usdt"
"""


class TestLoadConfig:
    def _write(self, tmp_path, text: str):
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_valid_config_merges_defaults(self, tmp_path):
        cfg = fb.load_config(self._write(tmp_path, VALID_CONFIG))
        assert cfg["symbol"] == "BTCUSDT"
        assert cfg["intervals"] == ["1d", "1h", "30m"]
        assert cfg["end_date"] is None
        assert cfg["raw_dir"] == "data/raw"
        assert cfg["file_prefix"] == "btc_usdt"
        assert cfg["binance"]["base_url"] == "https://api.binance.com"
        # Defaults are merged in for options the file does not set.
        assert cfg["binance"]["page_limit"] == 1000
        assert cfg["binance"]["max_retries"] == 5

    def test_end_date_today_normalised_to_none(self, tmp_path):
        cfg = fb.load_config(self._write(tmp_path, VALID_CONFIG.replace("null", '"today"')))
        assert cfg["end_date"] is None

    def test_unsupported_interval_rejected(self, tmp_path):
        broken = VALID_CONFIG.replace('["1d", "1h", "30m"]', '["1h", "7x"]')
        with pytest.raises(ValueError, match="7x"):
            fb.load_config(self._write(tmp_path, broken))

    def test_missing_symbol_rejected(self, tmp_path):
        broken = VALID_CONFIG.replace("  symbol: BTCUSDT\n", "")
        with pytest.raises(ValueError, match="symbol"):
            fb.load_config(self._write(tmp_path, broken))

    def test_unknown_binance_option_rejected(self, tmp_path):
        broken = VALID_CONFIG.replace("binance:\n", "binance:\n  fooo: 1\n")
        with pytest.raises(ValueError, match="fooo"):
            fb.load_config(self._write(tmp_path, broken))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            fb.load_config(tmp_path / "nope.yaml")
