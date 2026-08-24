"""Tests for the honest 'Next-candle outlook' panel.

Covers the pure pieces — expected-move math, keyword sentiment, RSS parsing
(mocked) — and a light integration check of the volatility-regime model when
the trained artefacts and data are present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.dashboard.outlook import (
    DEFAULT_NEWS_URL,
    DIRECTION_CV,
    VOLDIR_CV_ACCURACY,
    expected_move,
    fetch_news,
    predict_volatility_regime,
    summarize_sentiment,
    tag_sentiment,
)

# ── Direction guardrail constant ───────────────────────────────────────────


class TestDirectionGuardrail:
    def test_accuracy_within_ci(self):
        assert DIRECTION_CV["ci_low"] <= DIRECTION_CV["accuracy"] <= DIRECTION_CV["ci_high"]

    def test_base_rate_inside_ci_means_no_edge(self):
        # The whole point of the guardrail: base rate sits inside the CI.
        assert DIRECTION_CV["ci_low"] <= DIRECTION_CV["base_rate"] <= DIRECTION_CV["ci_high"]

    def test_edge_sign_matches_accuracy_vs_base(self):
        expected = (DIRECTION_CV["accuracy"] - DIRECTION_CV["base_rate"]) * 100
        assert DIRECTION_CV["edge_pp"] == pytest.approx(expected, abs=0.15)

    def test_no_edge_verdict(self):
        assert "coin flip" in DIRECTION_CV["verdict"]
        assert DIRECTION_CV["n"] > 1000

# ── Expected move (magnitude) ──────────────────────────────────────────────


class TestExpectedMove:
    def test_band_brackets_current_price(self):
        rng = np.random.default_rng(0)
        closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 100))))
        em = expected_move(closes, closes[-1])
        assert em["low_1sigma"] < closes[-1] < em["high_1sigma"]
        assert em["sigma_pct"] > 0
        assert em["window"] == 20

    def test_higher_vol_gives_wider_band(self):
        calm = list(100 * np.exp(np.cumsum(np.array([0.001, -0.001] * 30))))
        wild = list(100 * np.exp(np.cumsum(np.array([0.05, -0.05] * 30))))
        assert (
            expected_move(wild, wild[-1])["sigma_pct"]
            > expected_move(calm, calm[-1])["sigma_pct"]
        )

    def test_typical_move_usd_scales_with_price(self):
        rng = np.random.default_rng(1)
        closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 50))))
        em = expected_move(closes, 50_000.0)
        # ~1σ in dollars = price * sigma
        assert em["typical_move_usd"] == pytest.approx(50_000.0 * em["sigma_pct"] / 100)

    def test_too_few_closes_raises(self):
        with pytest.raises(ValueError):
            expected_move([100.0, 101.0], 101.0)

    def test_bad_price_raises(self):
        with pytest.raises(ValueError):
            expected_move([100.0, 101.0, 102.0], 0.0)

    def test_short_series_uses_available_window(self):
        em = expected_move([100.0, 101.0, 100.5, 101.5, 102.0], 102.0)
        assert em["window"] == 4  # len - 1


# ── Keyword sentiment ──────────────────────────────────────────────────────


class TestSentiment:
    def test_bullish(self):
        assert tag_sentiment("Bitcoin surges to record high on ETF inflows") == "▲"

    def test_bearish(self):
        assert tag_sentiment("BTC plunges after exchange hack and selloff") == "▼"

    def test_neutral(self):
        assert tag_sentiment("Bitcoin trades sideways ahead of data") == "–"

    def test_case_and_punctuation_insensitive(self):
        assert tag_sentiment("RALLY! Bitcoin gains, adoption climbs.") == "▲"


# ── RSS news feed (mocked) ─────────────────────────────────────────────────

_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>Bitcoin surges to new record</title>
        <link>https://x/1</link><pubDate>Mon, 23 Aug 2026</pubDate></item>
  <item><title>SEC delays spot ETF decision</title>
        <link>https://x/2</link><pubDate>Mon, 23 Aug 2026</pubDate></item>
  <item><title>Miners hold steady</title>
        <link>https://x/3</link><pubDate>Mon, 23 Aug 2026</pubDate></item>
</channel></rss>"""


class TestFetchNews:
    def test_parses_and_tags(self):
        news = fetch_news(fetcher=lambda url: _RSS, limit=6)
        assert len(news) == 3
        assert news[0]["title"] == "Bitcoin surges to new record"
        assert news[0]["tag"] == "▲"
        assert news[1]["tag"] == "▼"  # "delays" is bearish
        assert news[0]["link"] == "https://x/1"

    def test_respects_limit(self):
        assert len(fetch_news(fetcher=lambda url: _RSS, limit=2)) == 2

    def test_bad_feed_returns_empty(self):
        assert fetch_news(fetcher=lambda url: "not xml <<<") == []

    def test_network_error_returns_empty(self):
        def boom(url):
            raise RuntimeError("network down")

        assert fetch_news(fetcher=boom) == []

    def test_default_feed_is_bitcoin_focused(self):
        assert "bitcoin" in DEFAULT_NEWS_URL.lower()


class TestSummarizeSentiment:
    def test_net_bullish(self):
        news = fetch_news(fetcher=lambda url: _RSS)  # ▲, ▼, – → net 0 = mixed
        s = summarize_sentiment(news)
        assert s["n_bull"] == 1 and s["n_bear"] == 1 and s["n_neutral"] == 1
        assert s["net"] == 0 and s["label"] == "mixed"

    def test_bullish_label(self):
        s = summarize_sentiment([{"tag": "▲"}, {"tag": "▲"}, {"tag": "–"}])
        assert s["label"] == "bullish" and s["net"] == 2

    def test_bearish_label(self):
        s = summarize_sentiment([{"tag": "▼"}, {"tag": "▼"}, {"tag": "▲"}])
        assert s["label"] == "bearish" and s["net"] == -1

    def test_empty(self):
        s = summarize_sentiment([])
        assert s["label"] == "—" and s["net"] == 0


# ── Volatility-regime model (integration; skips without data/models) ───────


class TestVolRegimeIntegration:
    def test_predicts_when_data_present(self):
        if not Path("models/1d_pruned").is_dir() or not Path(
            "data/raw/btc_usdt_1d.parquet"
        ).is_file():
            pytest.skip("trained model / data not present")
        from src.models.train import load_modeling_config

        vr = predict_volatility_regime(load_modeling_config("configs/config.yaml"))
        assert 0.0 <= vr["p_expand"] <= 1.0
        assert vr["regime"] in {"EXPAND", "CONTRACT"}
        assert vr["cv_accuracy"] == VOLDIR_CV_ACCURACY
