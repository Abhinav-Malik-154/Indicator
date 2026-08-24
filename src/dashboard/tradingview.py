"""Live TradingView Advanced Chart embed for the dashboard's 'Live' view.

This is the real TradingView widget (the same one on tradingview.com): a
genuinely live candlestick chart with volume, an OHLC readout, the full
timeframe toolbar and drawing tools.  It streams via TradingView's own feed —
no API key, no polling on our side.  It renders inside a sandboxed iframe via
``streamlit.components.v1.html``.

Because it is TradingView's own canvas, our model's hindsight signal markers and
the live "Today" dot cannot be drawn on it — those live on the companion
"Signals" chart (:mod:`src.dashboard.chart`).
"""

from __future__ import annotations

import json

import streamlit as st

# Friendly label → TradingView symbol.  Default is BINANCE:BTCUSDT to match the
# project's own data source (Binance USDT pair); BITSTAMP:BTCUSD matches the
# reference screenshots.
SYMBOLS: dict[str, str] = {
    "BTC / USDT (Binance)": "BINANCE:BTCUSDT",
    "BTC / USD (Bitstamp)": "BITSTAMP:BTCUSD",
    "BTC / USD (Coinbase)": "COINBASE:BTCUSD",
    "ETH / USDT (Binance)": "BINANCE:ETHUSDT",
}
DEFAULT_SYMBOL_LABEL = "BTC / USDT (Binance)"

# Friendly label → TradingView interval code.
INTERVALS: dict[str, str] = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "4h": "240", "1D": "D", "1W": "W",
}
DEFAULT_INTERVAL_LABEL = "1m"


def tradingview_html(
    symbol: str,
    *,
    interval: str = "1",
    theme: str = "light",
    height: int = 620,
    container_id: str = "tv_advanced_chart",
) -> str:
    """Build the TradingView Advanced Chart widget HTML.

    Args:
        symbol: TradingView symbol, e.g. ``"BINANCE:BTCUSDT"``.
        interval: TradingView interval code (``"1"``, ``"60"``, ``"D"`` …).
        theme: ``"light"`` or ``"dark"``.
        height: Widget height in pixels.
        container_id: DOM id for the widget container (unique per render).

    Returns:
        A self-contained HTML string embedding the live widget.
    """
    toolbar_bg = "#0e1117" if theme == "dark" else "#ffffff"
    config = {
        "autosize": True,
        "symbol": symbol,
        "interval": interval,
        "timezone": "Etc/UTC",
        "theme": theme,
        "style": "1",  # candles
        "locale": "en",
        "toolbar_bg": toolbar_bg,
        "enable_publishing": False,
        "allow_symbol_change": True,
        "hide_side_toolbar": False,
        "withdateranges": True,
        "details": True,
        "calendar": False,
        "container_id": container_id,
    }
    return f"""
    <div class="tradingview-widget-container" style="height:{height}px;width:100%">
      <div id="{container_id}" style="height:{height}px;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({json.dumps(config)});
      </script>
    </div>
    """


def render_tradingview(
    symbol: str,
    *,
    interval: str = "1",
    theme: str = "light",
    height: int = 620,
    key: str = "tv",
) -> None:
    """Render the live TradingView widget in the current Streamlit container.

    Args:
        symbol: TradingView symbol.
        interval: TradingView interval code.
        theme: ``"light"`` or ``"dark"``.
        height: Widget height in pixels.
        key: Suffix for the container id so multiple embeds don't collide.
    """
    html = tradingview_html(
        symbol, interval=interval, theme=theme, height=height,
        container_id=f"tv_advanced_chart_{key}",
    )
    # A little headroom so the widget's own toolbar isn't clipped.
    st.iframe(html, height=height + 20, width="stretch")
