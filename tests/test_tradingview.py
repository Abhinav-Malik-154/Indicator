"""Tests for the live TradingView embed helper (dashboard 'Live' tab).

The widget itself is TradingView's remote JS; we can only unit-test the HTML we
generate — that it carries the right symbol / interval / theme and loads the
official ``tv.js`` script.
"""

from __future__ import annotations

import json
import re

import pytest

from src.dashboard.tradingview import (
    DEFAULT_INTERVAL_LABEL,
    DEFAULT_SYMBOL_LABEL,
    INTERVALS,
    SYMBOLS,
    tradingview_html,
)


def _extract_config(html: str) -> dict:
    """Pull the JSON passed to ``TradingView.widget({...})`` out of the HTML."""
    m = re.search(r"TradingView\.widget\((\{.*?\})\);", html, re.DOTALL)
    assert m, "widget config not found in HTML"
    return json.loads(m.group(1))


class TestTradingViewHtml:
    def test_embeds_symbol_and_interval(self):
        html = tradingview_html("BINANCE:BTCUSDT", interval="1")
        cfg = _extract_config(html)
        assert cfg["symbol"] == "BINANCE:BTCUSDT"
        assert cfg["interval"] == "1"

    def test_loads_official_script(self):
        html = tradingview_html("BINANCE:BTCUSDT")
        assert "https://s3.tradingview.com/tv.js" in html

    def test_theme_light_and_dark(self):
        assert _extract_config(tradingview_html("X", theme="light"))["theme"] == "light"
        dark = _extract_config(tradingview_html("X", theme="dark"))
        assert dark["theme"] == "dark"
        assert dark["toolbar_bg"] == "#0e1117"

    def test_container_id_is_unique_per_key(self):
        a = tradingview_html("X", container_id="tv_a")
        b = tradingview_html("X", container_id="tv_b")
        assert 'id="tv_a"' in a and 'id="tv_b"' in b

    def test_config_is_valid_json(self):
        cfg = _extract_config(tradingview_html("BINANCE:BTCUSDT", interval="60"))
        assert cfg["autosize"] is True
        assert cfg["style"] == "1"  # candlesticks


class TestMappings:
    def test_defaults_are_valid_keys(self):
        assert DEFAULT_SYMBOL_LABEL in SYMBOLS
        assert DEFAULT_INTERVAL_LABEL in INTERVALS

    def test_default_symbol_is_project_pair(self):
        assert SYMBOLS[DEFAULT_SYMBOL_LABEL] == "BINANCE:BTCUSDT"

    @pytest.mark.parametrize("label,code", list(INTERVALS.items()))
    def test_interval_codes_render(self, label, code):
        cfg = _extract_config(tradingview_html("X", interval=code))
        assert cfg["interval"] == code
