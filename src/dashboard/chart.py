"""Candlestick chart with retrospective, hindsight-coloured signal markers (Phase 9).

The chart shows the last ~180 daily candles.  On top of it, markers are drawn
**only** on the out-of-sample test split — the same signals the pruned model
produced in Phase 5 evaluation — and coloured by whether the direction turned
out correct N days later (green = right, red = wrong).

Two honesty guarantees are baked in:

* Markers are **retrospective**.  They are not a live prediction feed; they are
  a "how did the model's calls actually land" overlay on data the model never
  trained on.  Every marked row already has a known outcome (its training-time
  label survived the NaN-drop), so there is no dead-zone ambiguity to fudge.
* Nothing is ever drawn past the last fully closed candle.  The only forward-
  looking element is the "today" badge, which shows the current *live* signal
  at the latest closed candle and makes no claim about the future.

Markers use the logistic-regression signals: LightGBM fires zero signals on the
OOS split at the 0.60 threshold (documented in Phase 5/7), so it has nothing to
plot.  This is stated in the caption returned by :func:`chart_caption`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from src.backtest.report import build_test_signals
from src.backtest.simulate import signals_from_proba
from src.dashboard.signals import compute_live_signal, fetch_live_candles

logger = logging.getLogger(__name__)

_MARKER_GREEN = "#2ca02c"
_MARKER_RED = "#d62728"
_BADGE_COLOUR = {"BUY": "#2ca02c", "SELL": "#d62728", "SILENT": "#7f7f7f"}

CHART_CANDLES = 180


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_recent_ohlc(
    cfg: dict[str, Any],
    *,
    interval: str = "1d",
    n_candles: int = CHART_CANDLES,
) -> pd.DataFrame:
    """Return the last ``n_candles`` fully closed OHLC candles.

    Tries the live Binance feed first (freshest, already drops the still-forming
    candle); falls back to the raw parquet if the fetch fails.

    Args:
        cfg: Config dict from :func:`src.models.train.load_modeling_config`.
        interval: Binance interval.
        n_candles: Number of candles to return.

    Returns:
        DataFrame with ``open_time``, ``open``, ``high``, ``low``, ``close``.
    """
    try:
        candles = fetch_live_candles(cfg["symbol"], interval, n_candles=n_candles)
        logger.info("chart: using %d live candles", len(candles))
        return candles
    except Exception as exc:  # pragma: no cover - network fallback
        logger.warning("chart: live fetch failed (%s); using raw parquet", exc)
        raw_path = Path(cfg["raw_dir"]) / f"{cfg['file_prefix']}_{interval}.parquet"
        raw = pd.read_parquet(raw_path, columns=["open_time", "open", "high", "low", "close"])
        return raw.tail(n_candles).reset_index(drop=True)


def compute_historical_markers(
    cfg: dict[str, Any],
    *,
    interval: str = "1d",
    model_variant: str = "pruned",
) -> pd.DataFrame:
    """Build retrospective, correctness-coloured markers for the OOS test split.

    Uses the pruned logistic-regression signals from Phase 5 and each row's
    training-time label (already known, NaN rows dropped) as the realized
    outcome.  Only fired (non-silent) signals produce a marker.

    Args:
        cfg: Config dict from :func:`src.models.train.load_modeling_config`.
        interval: Binance interval.
        model_variant: Model directory variant (default ``"pruned"``).

    Returns:
        DataFrame with ``date``, ``signal`` (BUY/SELL), ``price`` (close at that
        candle), ``realized`` (up/down), ``correct`` (bool).  Empty if the
        model fired nothing.
    """
    sig_data = build_test_signals(interval, cfg, model_variant=model_variant)
    test: pd.DataFrame = sig_data["test_split"]
    threshold: float = sig_data["threshold"]
    label_col = f"label_{cfg['modeling']['horizon']}"

    sig_lr = signals_from_proba(sig_data["prob_lr"], threshold=threshold)

    raw_path = Path(cfg["raw_dir"]) / f"{cfg['file_prefix']}_{interval}.parquet"
    close_map = (
        pd.read_parquet(raw_path, columns=["open_time", "close"])
        .set_index("open_time")["close"]
        .to_dict()
    )

    dates = test["open_time"].tolist()
    labels = test[label_col].tolist()
    rows: list[dict[str, Any]] = []
    for i, sig in enumerate(sig_lr):
        if sig == 0:
            continue  # silent — no directional claim, no marker
        date = dates[i]
        predicted = "up" if sig == 1 else "down"
        realized = "up" if float(labels[i]) == 1.0 else "down"
        rows.append(
            {
                "date": date,
                "signal": "BUY" if sig == 1 else "SELL",
                "price": float(close_map.get(date, float("nan"))),
                "realized": realized,
                "correct": predicted == realized,
            }
        )
    markers = pd.DataFrame(rows)
    logger.info(
        "chart: %d retrospective markers (%d correct)",
        len(markers), int(markers["correct"].sum()) if not markers.empty else 0,
    )
    return markers


# ---------------------------------------------------------------------------
# Figure assembly (pure — no I/O, easy to smoke-test)
# ---------------------------------------------------------------------------


def build_candlestick_figure(
    ohlc: pd.DataFrame,
    markers: pd.DataFrame,
    today_badge: dict[str, Any] | None = None,
) -> go.Figure:
    """Assemble the candlestick figure with markers and the today badge.

    Args:
        ohlc: DataFrame with ``open_time``, ``open``, ``high``, ``low``, ``close``.
        markers: Output of :func:`compute_historical_markers` (may be empty).
        today_badge: Optional dict with ``date``, ``price``, ``signal`` (and
            optionally ``text``) for the live-signal badge at the last candle.

    Returns:
        A Plotly :class:`~plotly.graph_objects.Figure`.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=ohlc["open_time"],
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            name="BTC/USDT",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        )
    )

    if markers is not None and not markers.empty:
        window_start = ohlc["open_time"].min()
        visible = markers[markers["date"] >= window_start]
        for correct, colour, name in (
            (True, _MARKER_GREEN, "Signal correct (hindsight)"),
            (False, _MARKER_RED, "Signal wrong (hindsight)"),
        ):
            subset = visible[visible["correct"] == correct]
            if subset.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset["price"],
                    mode="markers",
                    name=name,
                    marker=dict(
                        size=10,
                        color=colour,
                        symbol=[
                            "triangle-up" if s == "BUY" else "triangle-down"
                            for s in subset["signal"]
                        ],
                        line=dict(width=1, color="white"),
                    ),
                    customdata=subset[["signal", "realized"]].to_numpy(),
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>signal=%{customdata[0]}"
                        "<br>outcome=%{customdata[1]}<br>price=%{y:.0f}<extra></extra>"
                    ),
                )
            )

    if today_badge is not None:
        sig = today_badge["signal"]
        colour = _BADGE_COLOUR.get(sig, "#7f7f7f")
        text = today_badge.get("text", f"Today: {sig}")
        fig.add_trace(
            go.Scatter(
                x=[today_badge["date"]],
                y=[today_badge["price"]],
                mode="markers",
                name="Today (live)",
                marker=dict(
                    size=16, color=colour, symbol="star",
                    line=dict(width=1.5, color="black"),
                ),
                hovertext=[text],
                hoverinfo="text",
            )
        )
        fig.add_annotation(
            x=today_badge["date"],
            y=today_badge["price"],
            text=text,
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            bgcolor=colour,
            font=dict(color="white", size=11),
            bordercolor="black",
            borderwidth=1,
        )

    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="Price (USDT)",
    )
    return fig


def chart_caption() -> str:
    """The mandatory honesty caption shown under the chart."""
    return (
        "Historical markers are retrospective (test period only) — not a live "
        "prediction feed. Green = the model's call turned out correct N days "
        "later; red = wrong. Markers use the logistic-regression signals "
        "(LightGBM fires none on this split). The ★ shows today's live signal "
        "at the last closed candle and makes no claim about the future."
    )


# ---------------------------------------------------------------------------
# Dashboard entry point
# ---------------------------------------------------------------------------


def load_chart_figure(
    cfg: dict[str, Any],
    *,
    interval: str = "1d",
    live_result: dict[str, Any] | None = None,
) -> go.Figure:
    """Load data and build the full candlestick figure for the dashboard.

    Args:
        cfg: Config dict from :func:`src.models.train.load_modeling_config`.
        interval: Binance interval.
        live_result: Optional pre-computed :func:`compute_live_signal` result
            (reused to avoid a second network fetch); computed if omitted.

    Returns:
        The assembled Plotly figure.
    """
    ohlc = load_recent_ohlc(cfg, interval=interval)
    try:
        markers = compute_historical_markers(cfg, interval=interval)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("chart: could not build markers (%s)", exc)
        markers = pd.DataFrame()

    result = live_result or compute_live_signal(interval=interval)
    last = ohlc.iloc[-1]
    badge = {
        "date": last["open_time"],
        "price": float(last["close"]),
        "signal": result["signal_lr"],
        "text": (
            f"Today {pd.Timestamp(result['candle_date']).date()}: "
            f"LR {result['signal_lr']} (P={result['prob_lr']:.2f})"
        ),
    }
    return build_candlestick_figure(ohlc, markers, badge)
