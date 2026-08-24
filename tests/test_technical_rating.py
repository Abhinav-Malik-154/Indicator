"""Tests for the live technical-rating signal call (Live tab).

Covers the pure pieces: the symbol/interval mapping, the score→label
classification boundaries, and the aggregate rating direction on synthetic
trends.  The live fetch (`rate_symbol`) is exercised as an integration probe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dashboard.technical_rating import (
    classify,
    compute_technical_rating,
    to_binance,
)


class TestToBinance:
    def test_binance_pair_passthrough(self):
        assert to_binance("BINANCE:BTCUSDT", "1m") == ("BTCUSDT", "1m")

    def test_usd_maps_to_usdt_proxy(self):
        assert to_binance("BITSTAMP:BTCUSD", "1D") == ("BTCUSDT", "1d")
        assert to_binance("COINBASE:BTCUSD", "1h") == ("BTCUSDT", "1h")

    def test_eth_and_weekly(self):
        assert to_binance("BINANCE:ETHUSDT", "1W") == ("ETHUSDT", "1w")

    def test_unknown_interval_defaults_1m(self):
        assert to_binance("BINANCE:BTCUSDT", "bogus")[1] == "1m"


class TestClassify:
    @pytest.mark.parametrize("score,label", [
        (1.0, "STRONG BUY"),
        (0.5, "STRONG BUY"),
        (0.3, "BUY"),
        (0.1, "BUY"),
        (0.0, "NEUTRAL"),
        (-0.05, "NEUTRAL"),
        (-0.3, "SELL"),
        (-0.5, "STRONG SELL"),
        (-1.0, "STRONG SELL"),
    ])
    def test_boundaries(self, score, label):
        assert classify(score) == label


class TestComputeRating:
    def _trend(self, drift):
        return pd.DataFrame({"close": list(100 * np.exp(np.cumsum(np.full(80, drift))))})

    def test_uptrend_is_buy_side(self):
        r = compute_technical_rating(self._trend(0.01))
        assert r["call"] in {"BUY", "STRONG BUY"}
        assert r["score"] > 0
        assert r["n_up"] >= r["n_down"]

    def test_downtrend_is_sell_side(self):
        r = compute_technical_rating(self._trend(-0.01))
        assert r["call"] in {"SELL", "STRONG SELL"}
        assert r["score"] < 0

    def test_contract_keys(self):
        r = compute_technical_rating(self._trend(0.005))
        assert set(r) >= {"call", "score", "votes", "n_up", "n_down", "rsi", "price"}
        assert set(r["votes"]) == {
            "EMA 10/30", "Price vs SMA50", "RSI(14)", "MACD", "Momentum(10)",
        }
        assert all(v in (-1, 0, 1) for v in r["votes"].values())

    def test_too_few_candles_raises(self):
        with pytest.raises(ValueError):
            compute_technical_rating(pd.DataFrame({"close": [100.0] * 10}))


class TestRateSymbolIntegration:
    def test_live_rating(self):
        pytest.importorskip("requests")
        from src.dashboard.technical_rating import rate_symbol

        try:
            r = rate_symbol("BTCUSDT", "1m")
        except Exception:  # pragma: no cover - offline
            pytest.skip("Binance not reachable")
        assert r["call"] in {
            "STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL",
        }
        assert -1.0 <= r["score"] <= 1.0
