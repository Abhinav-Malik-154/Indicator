"""Tests for Task 2: derivatives + cross-asset features (network mocked).

Covers:
- Funding-rate pagination / de-dup / daily aggregation.
- Cross-asset fetch with an injected downloader (rename, missing ticker).
- build_derivatives_raw end-to-end with both network calls stubbed.
- Derivatives feature engineering: deriv_ namespace, 1-day lag, and a
  no-lookahead assertion on synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.fetch_derivatives import (
    build_derivatives_raw,
    fetch_cross_asset,
    fetch_funding_rate,
    to_daily_funding,
)
from src.features.derivatives import (
    build_derivatives_features,
    validate_derivatives_no_lookahead,
)

_8H_MS = 8 * 60 * 60 * 1000


def _funding_page(start_ms: int, n: int, rate: float = 0.0001) -> list[dict]:
    """Build ``n`` fake 8-hourly funding rows starting at ``start_ms``."""
    return [
        {
            "symbol": "BTCUSDT",
            "fundingTime": start_ms + i * _8H_MS,
            "fundingRate": f"{rate:.8f}",
            "markPrice": "40000.0",
        }
        for i in range(n)
    ]


# ── Funding rate fetch / aggregation ───────────────────────────────────────


class TestFundingRate:
    def test_single_short_page_stops(self):
        page = _funding_page(0, 5)
        calls: list[dict] = []

        def getter(params):
            calls.append(params)
            return page

        df = fetch_funding_rate("BTCUSDT", start_ms=0, limit=1000, page_getter=getter)
        assert len(df) == 5
        assert len(calls) == 1  # short page → no second request
        assert list(df.columns) == ["funding_time", "funding_rate"]
        assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)

    def test_paginates_until_short_page(self):
        # First call returns a full page (limit=3), second returns a short page.
        pages = [_funding_page(0, 3), _funding_page(3 * _8H_MS, 2)]

        def getter(params):
            return pages.pop(0) if pages else []

        df = fetch_funding_rate("BTCUSDT", start_ms=0, limit=3, page_getter=getter)
        assert len(df) == 5
        assert df["funding_time"].is_monotonic_increasing

    def test_dedupes_overlapping_times(self):
        # Overlapping fundingTime across pages must collapse to one row.
        pages = [_funding_page(0, 3), _funding_page(2 * _8H_MS, 2)]

        def getter(params):
            return pages.pop(0) if pages else []

        df = fetch_funding_rate("BTCUSDT", start_ms=0, limit=3, page_getter=getter)
        assert df["funding_time"].is_unique

    def test_empty_returns_empty_frame(self):
        df = fetch_funding_rate("BTCUSDT", start_ms=0, page_getter=lambda p: [])
        assert df.empty
        assert list(df.columns) == ["funding_time", "funding_rate"]

    def test_to_daily_aggregates_three_per_day(self):
        raw = fetch_funding_rate(
            "BTCUSDT", start_ms=0, limit=1000,
            page_getter=lambda p: _funding_page(0, 6, rate=0.001),
        )
        daily = to_daily_funding(raw)
        assert list(daily.columns) == [
            "date", "funding_sum", "funding_mean", "funding_count",
        ]
        assert (daily["funding_count"] == 3).all()  # 3 settlements per day
        assert daily["funding_sum"].iloc[0] == pytest.approx(0.003)
        assert daily["funding_mean"].iloc[0] == pytest.approx(0.001)

    def test_to_daily_empty(self):
        daily = to_daily_funding(pd.DataFrame(columns=["funding_time", "funding_rate"]))
        assert daily.empty


# ── Cross-asset fetch ──────────────────────────────────────────────────────


def _fake_close(tickers, start, end):
    idx = pd.bdate_range("2024-01-01", periods=5)  # weekdays only
    return pd.DataFrame(
        {t: np.linspace(100, 104, 5) + j for j, t in enumerate(tickers)},
        index=idx,
    )


class TestCrossAsset:
    def test_rename_to_friendly(self):
        tickers = {"x_dxy": "DX-Y.NYB", "x_spx": "^GSPC"}
        df = fetch_cross_asset(tickers, "2024-01-01", downloader=_fake_close)
        assert "date" in df.columns
        assert set(df.columns) == {"date", "x_dxy", "x_spx"}
        assert len(df) == 5

    def test_missing_ticker_becomes_nan_column(self):
        def partial(tickers, start, end):
            idx = pd.bdate_range("2024-01-01", periods=3)
            return pd.DataFrame({"DX-Y.NYB": [1.0, 2.0, 3.0]}, index=idx)

        df = fetch_cross_asset(
            {"x_dxy": "DX-Y.NYB", "x_gold": "GC=F"}, "2024-01-01", downloader=partial
        )
        assert df["x_dxy"].notna().all()
        assert df["x_gold"].isna().all()

    def test_empty_downloader(self):
        df = fetch_cross_asset(
            {"x_dxy": "DX-Y.NYB"}, "2024-01-01",
            downloader=lambda t, s, e: pd.DataFrame(),
        )
        assert "x_dxy" in df.columns


# ── End-to-end build ───────────────────────────────────────────────────────


class TestBuildDerivativesRaw:
    def test_build_and_manifest(self, tmp_path: Path):
        cfg = {
            "symbol": "BTCUSDT", "start_date": "2024-01-01",
            "end_date": None, "raw_dir": str(tmp_path),
        }
        out = tmp_path / "derivatives_1d.parquet"
        manifest = build_derivatives_raw(
            cfg, out,
            tickers={"x_dxy": "DX-Y.NYB"},
            page_getter=lambda p: _funding_page(
                int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000), 9
            ),
            downloader=_fake_close,
        )
        assert out.is_file()
        df = pd.read_parquet(out)
        assert "date" in df.columns
        assert "funding_sum" in df.columns
        assert "x_dxy" in df.columns
        # Manifest documents the honest 30-day free-data limitation.
        assert any("30" in note for note in manifest["not_available_for_free"])
        man_file = tmp_path / "derivatives_1d.manifest.json"
        assert json.loads(man_file.read_text())["row_count"] == len(df)


# ── Feature engineering ────────────────────────────────────────────────────


@pytest.fixture
def synthetic_setup(tmp_path: Path):
    """A 60-day candle frame + a matching derivatives parquet."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    candles = pd.DataFrame({"open_time": dates})

    rng = np.random.default_rng(0)
    deriv = pd.DataFrame({
        "date": dates,
        "funding_sum": rng.normal(0.0003, 0.0005, 60),
        "funding_mean": rng.normal(0.0001, 0.0002, 60),
        "funding_count": 3,
        "x_dxy": 100 + np.cumsum(rng.normal(0, 0.5, 60)),
        "x_spx": 4000 + np.cumsum(rng.normal(0, 5, 60)),
        "x_gold": 2000 + np.cumsum(rng.normal(0, 3, 60)),
    })
    path = tmp_path / "derivatives_1d.parquet"
    deriv.to_parquet(path, index=False)
    return candles, path


class TestDerivativesFeatures:
    def test_deriv_namespace_and_columns(self, synthetic_setup):
        candles, path = synthetic_setup
        feats = build_derivatives_features(candles, path)
        deriv_cols = [c for c in feats.columns if c.startswith("deriv_")]
        assert deriv_cols  # non-empty
        assert all(c.startswith("deriv_") for c in feats.columns if c != "open_time")
        for expected in ("deriv_funding", "deriv_funding_z7", "deriv_funding_sign",
                         "deriv_funding_cum7", "deriv_x_dxy_ret1", "deriv_x_spx_z30"):
            assert expected in feats.columns

    def test_row_count_matches_candles(self, synthetic_setup):
        candles, path = synthetic_setup
        feats = build_derivatives_features(candles, path)
        assert len(feats) == len(candles)

    def test_one_day_lag(self, synthetic_setup):
        """deriv_funding at candle T must equal raw funding_sum at day T-1."""
        candles, path = synthetic_setup
        raw = pd.read_parquet(path)
        feats = build_derivatives_features(candles, path, lag_days=1)
        # Row 5 candle → day index 4 raw funding.
        assert feats["deriv_funding"].iloc[5] == pytest.approx(raw["funding_sum"].iloc[4])
        # Row 0 has no prior day → NaN.
        assert np.isnan(feats["deriv_funding"].iloc[0])

    def test_missing_file_raises(self, tmp_path):
        candles = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=3, tz="UTC")})
        with pytest.raises(FileNotFoundError):
            build_derivatives_features(candles, tmp_path / "nope.parquet")

    def test_no_lookahead(self, synthetic_setup):
        candles, path = synthetic_setup
        # Should not raise: features at T use only ≤ T-1 data.
        validate_derivatives_no_lookahead(candles, path, lag_days=1)
