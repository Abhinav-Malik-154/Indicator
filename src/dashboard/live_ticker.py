"""TradingView-style live price + candle-close countdown badge.

The static price tag drawn on the Plotly chart (:func:`src.dashboard.chart.
_add_live_price_tag`) cannot *tick* — Plotly annotations only update when
Streamlit reruns (every few minutes).  TradingView's green/red box shows a
**live price** and a **countdown to the current candle's close** that both move
every second.  To reproduce that faithfully we render a tiny self-contained
HTML/JS badge (via ``st.iframe``) that runs in the browser:

* the **countdown** is pure client-side clock arithmetic — it ticks every second
  with no network and no Streamlit rerun;
* the **price** is refreshed client-side straight from Binance's public ticker
  every few seconds, so the number is genuinely live, and the box turns green
  when at/above the last close and red when below.

This is display-only: it drives no model logic and makes no prediction.
"""

from __future__ import annotations

import json

import streamlit as st

# Candle length in milliseconds; UTC-epoch-aligned so the close time is just the
# next multiple of the interval.
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000,
}

_UP_GREEN = "#0ecb81"
_DOWN_RED = "#f6465d"


def live_badge_html(
    symbol: str,
    *,
    interval: str = "1d",
    prev_close: float,
    up_color: str = _UP_GREEN,
    down_color: str = _DOWN_RED,
    poll_ms: int = 3000,
) -> str:
    """Build the self-contained live price + countdown badge HTML.

    Args:
        symbol: Binance symbol for the client-side ticker fetch, e.g. ``"BTCUSDT"``.
        interval: Candle interval whose close the countdown targets.
        prev_close: Last **closed** candle price — the green/red reference and the
            initial number shown before the first live tick.
        up_color: Box colour when the live price is at/above ``prev_close``.
        down_color: Box colour when below.
        poll_ms: How often (ms) the browser re-fetches the live price.

    Returns:
        An HTML string embedding the ticking badge.
    """
    interval_ms = INTERVAL_MS.get(interval, INTERVAL_MS["1d"])
    cfg = {
        "symbol": symbol,
        "intervalMs": interval_ms,
        "prevClose": float(prev_close),
        "up": up_color,
        "down": down_color,
        "pollMs": int(poll_ms),
    }
    return f"""
    <div id="tvbox" style="
        display:inline-flex;flex-direction:column;align-items:center;
        background:{up_color};color:#fff;border-radius:6px;
        padding:6px 16px;min-width:96px;
        font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
        box-shadow:0 1px 4px rgba(0,0,0,0.25);">
      <span id="tvprice" style="font-weight:700;font-size:17px;line-height:1.1;">
        {prev_close:,.0f}</span>
      <span id="tvcd" style="font-size:12px;opacity:0.95;letter-spacing:0.5px;">
        --:--</span>
    </div>
    <script>
      const C = {json.dumps(cfg)};
      const box = document.getElementById('tvbox');
      const pe = document.getElementById('tvprice');
      const ce = document.getElementById('tvcd');
      let price = C.prevClose;
      const fmt = n => n.toLocaleString('en-US', {{maximumFractionDigits: 0}});
      function paint() {{ box.style.background = price >= C.prevClose ? C.up : C.down; }}
      async function pull() {{
        try {{
          const r = await fetch(
            'https://api.binance.com/api/v3/ticker/price?symbol=' + C.symbol);
          const j = await r.json();
          if (j && j.price) {{ price = parseFloat(j.price); pe.textContent = fmt(price); paint(); }}
        }} catch (e) {{ /* offline / CORS: keep last price, countdown still ticks */ }}
      }}
      function countdown() {{
        const now = Date.now();
        const next = (Math.floor(now / C.intervalMs) + 1) * C.intervalMs;
        let s = Math.max(0, Math.round((next - now) / 1000));
        const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
        const pad = v => String(v).padStart(2, '0');
        ce.textContent = h > 0
          ? h + ':' + pad(m) + ':' + pad(sec)
          : pad(m) + ':' + pad(sec);
      }}
      paint(); pull(); countdown();
      setInterval(countdown, 1000);
      setInterval(pull, C.pollMs);
    </script>
    """


def render_live_badge(
    symbol: str,
    *,
    interval: str = "1d",
    prev_close: float,
    height: int = 78,
) -> None:
    """Render the live price + countdown badge in the current Streamlit container.

    Args:
        symbol: Binance symbol.
        interval: Candle interval for the countdown.
        prev_close: Last closed candle price (colour reference + initial value).
        height: Iframe height in pixels.
    """
    html = live_badge_html(symbol, interval=interval, prev_close=prev_close)
    st.iframe(html, height=height, width="content")
