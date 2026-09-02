"""Honest 'Next-candle outlook' panel (direction, magnitude, vol-regime, news).

This module deliberately does **not** claim to know the next candle's price. The
project measured next-day direction and found no edge (≈50%, base rate inside the
CI), so the outlook is built only from things that are *actually* forecastable,
each labelled with its real accuracy:

* :func:`expected_move` — the **magnitude** of the next candle from recent
  realized volatility (a genuine, well-behaved estimate); the "how much".
* :func:`predict_volatility_regime` — will volatility **expand or contract** next
  window?  This is Task 3's statistically-significant edge (~69% CV).
* **direction** — surfaced from the existing model P(up), shown *with* the honest
  ~50% accuracy so it is never read as a certainty.
* :func:`fetch_news` — today's BTC headlines from a free RSS feed, with a crude
  keyword sentiment tag (context, not a price prediction).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Walk-forward CV accuracy of the volatility-direction model (LR), from
# `python -m src.models.reframe --target voldir`: 69.3% [66.6%, 71.9%].
# Stored as a documented measured constant (same pattern as HISTORICAL_ACCURACY).
VOLDIR_CV_ACCURACY: tuple[float, float, float] = (0.693, 0.666, 0.719)

# Walk-forward CV accuracy of next-day **direction** (LR), from
# `python -m src.models.cross_validate --intervals 1d --folds 8 --embargo 1`:
# 49.4% [46.4%, 52.3%] over 1108 pooled OOS days, base rate 50.8% — the base rate
# sits inside the CI, so the model has **no measured directional edge**. This is
# the precise number the Direction guardrail is stamped with.
DIRECTION_CV: dict[str, Any] = {
    "accuracy": 0.494,
    "ci_low": 0.464,
    "ci_high": 0.523,
    "base_rate": 0.508,
    "edge_pp": -1.4,
    "n": 1108,
    "verdict": "indistinguishable from a coin flip (no edge)",
}

# Free **Bitcoin-focused** news RSS (no API key). Cointelegraph's bitcoin tag →
# BTC-specific headlines, matching the "learn what moves BTC" goal. Any RSS 2.0
# feed with <item> works if this is overridden.
DEFAULT_NEWS_URL = "https://cointelegraph.com/rss/tag/bitcoin"

# Backup feeds tried in order if the primary times out or fails, so one slow host
# never leaves the news panel blank. All free, no API key, RSS 2.0.
FALLBACK_NEWS_URLS: tuple[str, ...] = (
    "https://cointelegraph.com/rss/tag/bitcoin",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://cointelegraph.com/rss",
)


# ---------------------------------------------------------------------------
# Expected move (magnitude) — the honest "how much"
# ---------------------------------------------------------------------------


def expected_move(
    closes: list[float] | pd.Series | np.ndarray,
    current_price: float,
    *,
    window: int = 20,
) -> dict[str, Any]:
    """Estimate the next candle's typical move from recent realized volatility.

    Magnitude (unlike direction) is forecastable: returns cluster, so the recent
    per-candle volatility ``σ`` is a good estimate of the next candle's typical
    size.  We report the 1σ band — about 2 candles in 3 close within it.

    Args:
        closes: Recent close prices (chronological).
        current_price: Price to centre the band on (live price or last close).
        window: Number of returns used for the volatility estimate.

    Returns:
        Dict with ``sigma_pct`` (per-candle σ in %), ``typical_move_usd`` (≈1σ in
        dollars), ``low_1sigma`` / ``high_1sigma`` (price band) and ``window``.

    Raises:
        ValueError: If there are too few closes or the price is non-positive.
    """
    c = np.asarray(closes, dtype="float64")
    c = c[np.isfinite(c) & (c > 0)]
    if len(c) < 3:
        raise ValueError("expected_move needs at least 3 valid closes")
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    eff = min(window, len(c) - 1)
    log_ret = np.diff(np.log(c[-(eff + 1):]))
    sigma = float(np.std(log_ret, ddof=1))
    return {
        "sigma_pct": sigma * 100.0,
        "typical_move_usd": current_price * sigma,
        "low_1sigma": current_price * float(np.exp(-sigma)),
        "high_1sigma": current_price * float(np.exp(sigma)),
        "window": eff,
        "current_price": float(current_price),
    }


# ---------------------------------------------------------------------------
# Volatility-regime prediction — the real edge (Task 3)
# ---------------------------------------------------------------------------


def predict_volatility_regime(
    cfg: dict[str, Any],
    *,
    interval: str = "1d",
    vol_window: int = 7,
) -> dict[str, Any]:
    """Predict whether next-window volatility expands or contracts.

    Trains the volatility-direction logistic model (Task 3) on all history whose
    label is known, then applies it to the latest feature row — a live use of the
    same model that scored ~69% in walk-forward CV.  Leakage-safe: the latest row
    uses only features ``≤`` today; its forward-vol label is unknown (future).

    Args:
        cfg: Config dict from :func:`src.models.train.load_modeling_config`.
        interval: Binance interval.
        vol_window: Window for the volatility-direction target.

    Returns:
        Dict with ``p_expand``, ``regime`` ("EXPAND"/"CONTRACT"), ``current_vol``,
        ``as_of`` (feature date) and ``cv_accuracy`` (point, low, high).

    Raises:
        ValueError: If there is not enough labelled history to train.
    """
    from src.labels.alt_targets import compute_volatility_direction_labels
    from src.models.train import (
        assemble_dataset,
        fit_scaler_on_train,
        train_logistic_regression,
    )

    merged, feature_cols = assemble_dataset(interval, cfg)
    raw_path = Path(cfg["raw_dir"]) / f"{cfg['file_prefix']}_{interval}.parquet"
    candles = pd.read_parquet(raw_path).sort_values("open_time").reset_index(drop=True)
    voldir = compute_volatility_direction_labels(candles, vol_window=vol_window)[
        ["open_time", "label_voldir", "current_vol"]
    ]

    joined = merged.merge(voldir, on="open_time", how="left")
    feat_ok = joined.dropna(subset=feature_cols).sort_values("open_time")
    train = feat_ok.dropna(subset=["label_voldir"])
    if len(train) < 100:
        raise ValueError(f"{interval}: not enough labelled history for vol-regime model")

    seed = int(cfg["modeling"]["random_state"])
    x_train = train[feature_cols]
    y_train = train["label_voldir"].astype("int64")
    scaler = fit_scaler_on_train(x_train)
    x_train_scaled = pd.DataFrame(
        scaler.transform(x_train), columns=feature_cols, index=x_train.index
    )
    lr = train_logistic_regression(
        x_train_scaled, y_train,
        params=cfg["modeling"]["logistic_regression"], random_state=seed,
    )

    latest_idx = feat_ok.index[-1]
    x_latest = pd.DataFrame(
        scaler.transform(feat_ok.loc[[latest_idx], feature_cols]),
        columns=feature_cols, index=[latest_idx],
    )
    p_expand = float(lr.predict_proba(x_latest)[:, 1][0])
    cur_vol = feat_ok.loc[latest_idx, "current_vol"]
    return {
        "p_expand": p_expand,
        "regime": "EXPAND" if p_expand >= 0.5 else "CONTRACT",
        "current_vol": None if pd.isna(cur_vol) else float(cur_vol),
        "as_of": pd.Timestamp(feat_ok.loc[latest_idx, "open_time"]),
        "cv_accuracy": VOLDIR_CV_ACCURACY,
        "vol_window": vol_window,
    }


# ---------------------------------------------------------------------------
# Daily news feed (free RSS + crude keyword sentiment)
# ---------------------------------------------------------------------------

_BULLISH = {
    "surge", "surges", "rally", "rallies", "soar", "soars", "gain", "gains",
    "jump", "jumps", "record", "high", "adopt", "adoption", "approval", "approve",
    "approved", "inflow", "inflows", "bullish", "buy", "buys", "rise", "rises",
    "top", "breakout", "boom", "milestone", "surging", "climb", "climbs", "up",
    "outperform", "accumulate", "accumulation",
}
_BEARISH = {
    "crash", "crashes", "plunge", "plunges", "drop", "drops", "fall", "falls",
    "sell", "sells", "selloff", "ban", "bans", "hack", "hacked", "outflow",
    "outflows", "bearish", "dump", "dumps", "low", "reject", "rejected", "delay",
    "delays", "lawsuit", "fear", "liquidation", "liquidations", "slump", "slide",
    "down", "warning", "risk", "fraud",
}


def tag_sentiment(title: str) -> str:
    """Crude keyword sentiment for a headline: ``▲`` / ``▼`` / ``–``.

    Deliberately simple and transparent — a bag-of-words vote, **not** a model.
    It signals tone, never a price prediction.
    """
    words = {w.strip(".,!?:;\"'()").lower() for w in title.split()}
    bull = len(words & _BULLISH)
    bear = len(words & _BEARISH)
    if bull > bear:
        return "▲"
    if bear > bull:
        return "▼"
    return "–"


def _parse_feed(text: str, limit: int) -> list[dict[str, Any]]:
    """Parse RSS 2.0 XML into tagged headline dicts (raises on bad XML)."""
    root = ET.fromstring(text)
    items: list[dict[str, Any]] = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        items.append({
            "title": title,
            "link": (item.findtext("link") or "").strip(),
            "published": (item.findtext("pubDate") or "").strip(),
            "tag": tag_sentiment(title),
        })
    return items


def fetch_news(
    *,
    url: str = DEFAULT_NEWS_URL,
    limit: int = 6,
    timeout_s: float = 12.0,
    retries: int = 2,
    fetcher: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent BTC headlines from free RSS feeds, with retries + fallbacks.

    Tries ``url`` first, then :data:`FALLBACK_NEWS_URLS`, each up to ``retries``
    times, and returns the first feed that yields headlines — so a single slow or
    down host (the timeout you saw) no longer leaves the panel blank.

    Args:
        url: Preferred RSS 2.0 feed URL (tried first).
        limit: Max headlines to return.
        timeout_s: Per-request HTTP timeout (ignored when ``fetcher`` is given).
        retries: Attempts per feed before moving to the next.
        fetcher: Injectable ``url -> xml_text`` (for tests); defaults to
            ``requests.get``.

    Returns:
        List of dicts with ``title``, ``link``, ``published`` and ``tag``.
        Empty list only if **every** feed fails.
    """
    import time

    feeds = [url, *(u for u in FALLBACK_NEWS_URLS if u != url)]
    last_exc: Exception | None = None
    for feed_url in feeds:
        for attempt in range(max(1, retries)):
            try:
                if fetcher is not None:
                    text = fetcher(feed_url)
                else:
                    resp = requests.get(
                        feed_url, timeout=timeout_s,
                        headers={"User-Agent": "Mozilla/5.0 (BTC-dashboard)"},
                    )
                    resp.raise_for_status()
                    text = resp.text
                items = _parse_feed(text, limit)
                if items:
                    logger.info("news: %d headlines from %s", len(items), feed_url)
                    return items
            except Exception as exc:  # noqa: BLE001 - try the next feed/attempt
                last_exc = exc
                # Back off only for real network calls, never in tests.
                if fetcher is None and attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))
    if last_exc is not None:
        logger.warning("news: all feeds failed (last error: %s)", last_exc)
    return []


def summarize_sentiment(news: list[dict[str, Any]]) -> dict[str, Any]:
    """Tally the headline tags into a net bull/bear read of the day's tape.

    Args:
        news: Output of :func:`fetch_news`.

    Returns:
        Dict with ``n_bull`` / ``n_bear`` / ``n_neutral`` counts, ``net``
        (bull − bear) and a ``label`` ("bullish" / "bearish" / "mixed" / "—").
    """
    n_bull = sum(1 for n in news if n.get("tag") == "▲")
    n_bear = sum(1 for n in news if n.get("tag") == "▼")
    n_neutral = sum(1 for n in news if n.get("tag") == "–")
    net = n_bull - n_bear
    if not news:
        label = "—"
    elif net > 0:
        label = "bullish"
    elif net < 0:
        label = "bearish"
    else:
        label = "mixed"
    return {
        "n_bull": n_bull, "n_bear": n_bear, "n_neutral": n_neutral,
        "net": net, "label": label,
    }
