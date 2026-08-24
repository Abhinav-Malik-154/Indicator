"""Tests for the live price + candle-close countdown badge.

The badge's countdown and price refresh run as browser JS, so we can only
unit-test the HTML we emit: that it carries the symbol, the correct candle
interval in ms, the last-close reference, both colours, and the client-side
countdown / ticker wiring.
"""

from __future__ import annotations

import pytest

from src.dashboard.live_ticker import (
    _DOWN_RED,
    _UP_GREEN,
    INTERVAL_MS,
    live_badge_html,
)


class TestLiveBadgeHtml:
    def test_embeds_symbol_and_prev_close(self):
        html = live_badge_html("BTCUSDT", interval="1d", prev_close=77559)
        assert "BTCUSDT" in html
        assert "77,559" in html  # formatted initial price
        assert '"prevClose": 77559' in html

    def test_interval_ms_matches_table(self):
        html = live_badge_html("BTCUSDT", interval="1h", prev_close=100)
        assert f'"intervalMs": {INTERVAL_MS["1h"]}' in html

    def test_unknown_interval_falls_back_to_daily(self):
        html = live_badge_html("BTCUSDT", interval="bogus", prev_close=100)
        assert f'"intervalMs": {INTERVAL_MS["1d"]}' in html

    def test_carries_both_colours(self):
        html = live_badge_html("BTCUSDT", prev_close=100)
        assert _UP_GREEN in html and _DOWN_RED in html

    def test_has_countdown_and_ticker_wiring(self):
        html = live_badge_html("BTCUSDT", prev_close=100)
        # countdown ticks every second; price polls periodically
        assert "setInterval(countdown, 1000)" in html
        assert "api/v3/ticker/price" in html
        assert "setInterval(pull," in html

    def test_countdown_targets_next_interval_boundary(self):
        html = live_badge_html("BTCUSDT", prev_close=100)
        # next close = next epoch-aligned multiple of the interval
        assert "Math.floor(now / C.intervalMs) + 1" in html

    @pytest.mark.parametrize("interval", list(INTERVAL_MS))
    def test_all_intervals_render(self, interval):
        html = live_badge_html("BTCUSDT", interval=interval, prev_close=100)
        assert f'"intervalMs": {INTERVAL_MS[interval]}' in html
