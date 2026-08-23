"""Price chart with retrospective, hindsight-coloured signal markers (Phase 9).

The default view is a Binance-style gold line with a gradient area fill on a
clean dark canvas (:func:`build_price_figure`); a candlestick view is also
available (:func:`build_candlestick_figure`).  On top of either, markers are drawn
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

# ── Binance-style palette ─────────────────────────────────────────────────
BINANCE_GOLD = "#F0B90B"        # brand gold (fill)
BINANCE_GOLD_LINE = "#F3BA2F"   # slightly brighter gold for the line
_UP_GREEN = "#0ecb81"           # Binance green
_DOWN_RED = "#f6465d"           # Binance red
_AXIS_TEXT = "#848e9c"          # muted gray axis/label text
_GRID = "rgba(255,255,255,0.05)"  # near-invisible horizontal gridlines
_SPIKE = "#5e6673"              # hover crosshair colour

_MARKER_GREEN = _UP_GREEN
_MARKER_RED = _DOWN_RED
_BADGE_COLOUR = {"BUY": _UP_GREEN, "SELL": _DOWN_RED, "SILENT": _AXIS_TEXT}

CHART_CANDLES = 180             # default candlestick window
CHART_FETCH_CANDLES = 365       # fetched once, then sliced per selected range

# Selectable time ranges (days back from the last candle); "YTD" is special.
RANGE_DAYS = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
CHART_RANGES = ["1M", "3M", "6M", "YTD", "1Y"]


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


def _add_signal_markers(
    fig: go.Figure,
    window_start: Any,
    markers: pd.DataFrame,
    *,
    show: bool = True,
) -> None:
    """Overlay retrospective green/red correctness markers within the window."""
    if not show or markers is None or markers.empty:
        return
    visible = markers[markers["date"] >= window_start]
    for correct, colour in ((True, _MARKER_GREEN), (False, _MARKER_RED)):
        subset = visible[visible["correct"] == correct]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["date"], y=subset["price"], mode="markers",
                marker=dict(
                    size=8, color=colour,
                    symbol=[
                        "triangle-up" if s == "BUY" else "triangle-down"
                        for s in subset["signal"]
                    ],
                    line=dict(width=1, color="#0b0e11"),
                ),
                customdata=subset[["signal", "realized"]].to_numpy(),
                hovertemplate=(
                    "%{x|%b %d}<br>signal=%{customdata[0]}"
                    "<br>outcome=%{customdata[1]}<extra></extra>"
                ),
                showlegend=False,
            )
        )


def _add_today_badge(fig: go.Figure, today_badge: dict[str, Any] | None) -> None:
    """Add the dashed current-price line, the live-signal dot, and its label."""
    if today_badge is None:
        return
    price = float(today_badge["price"])
    sig = today_badge["signal"]
    colour = _BADGE_COLOUR.get(sig, _AXIS_TEXT)
    fig.add_hline(
        y=price, line_dash="dot", line_color=_SPIKE, line_width=1,
        annotation_text=f" ${price:,.0f} ", annotation_position="right",
        annotation_font=dict(color="#eaecef", size=11),
        annotation_bgcolor="#2b3139",
    )
    fig.add_trace(
        go.Scatter(
            x=[today_badge["date"]], y=[price], mode="markers",
            marker=dict(size=10, color=colour, symbol="circle",
                        line=dict(width=2, color="#0b0e11")),
            hovertext=[today_badge.get("text", f"Today: {sig}")],
            hoverinfo="text", showlegend=False,
        )
    )
    fig.add_annotation(
        x=today_badge["date"], y=price, text=f"● Today: {sig}",
        showarrow=False, xanchor="right", yanchor="bottom", yshift=10,
        font=dict(color=colour, size=11),
    )


def _apply_dark_layout(fig: go.Figure, x: pd.Series, lo: float, hi: float) -> None:
    """Apply the shared Binance-style dark theme, grid, axes, and hover crosshair."""
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=70, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_AXIS_TEXT, size=12),
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(
            showgrid=False, showline=False, zeroline=False, color=_AXIS_TEXT,
            showspikes=True, spikemode="across", spikethickness=1,
            spikecolor=_SPIKE, spikedash="dot",
            range=[x.min(), x.max()],
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=_GRID, gridwidth=1, showline=False,
            zeroline=False, color=_AXIS_TEXT, tickprefix="$", tickformat=".3s",
            range=[lo, hi],
        ),
    )


def build_candlestick_figure(
    ohlc: pd.DataFrame,
    markers: pd.DataFrame,
    today_badge: dict[str, Any] | None = None,
    *,
    show_markers: bool = True,
) -> go.Figure:
    """Assemble a Binance-style candlestick figure with markers and today badge.

    Same clean dark canvas, faint gridlines, dashed current-price line, and
    live-signal dot as :func:`build_price_figure`, but with green/red OHLC
    candles instead of the line + gradient area.

    Args:
        ohlc: DataFrame with ``open_time``, ``open``, ``high``, ``low``, ``close``.
        markers: Output of :func:`compute_historical_markers` (may be empty).
        today_badge: Optional dict with ``date``, ``price``, ``signal``, ``text``.
        show_markers: When ``False``, draw only the candles.

    Returns:
        A Plotly :class:`~plotly.graph_objects.Figure`.
    """
    fig = go.Figure()
    if ohlc.empty:
        return fig

    x = ohlc["open_time"]
    lo = float(ohlc["low"].min())
    hi = float(ohlc["high"].max())
    pad = (hi - lo) * 0.06 or hi * 0.02
    lo, hi = lo - pad, hi + pad

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=ohlc["open"], high=ohlc["high"],
            low=ohlc["low"], close=ohlc["close"],
            name="BTC/USDT",
            increasing_line_color=_UP_GREEN, decreasing_line_color=_DOWN_RED,
            increasing_fillcolor=_UP_GREEN, decreasing_fillcolor=_DOWN_RED,
            line=dict(width=1),
            showlegend=False,
        )
    )
    _add_signal_markers(fig, x.min(), markers, show=show_markers)
    _add_today_badge(fig, today_badge)
    _apply_dark_layout(fig, x, lo, hi)
    return fig


def slice_by_range(ohlc: pd.DataFrame, range_label: str) -> pd.DataFrame:
    """Slice an OHLC frame to a selectable time range (Binance-style pills).

    Args:
        ohlc: DataFrame with an ``open_time`` column, sorted ascending.
        range_label: One of :data:`CHART_RANGES` (``"1M"``…``"1Y"``, ``"YTD"``).

    Returns:
        The rows within the requested range (always ends at the last candle).
    """
    if ohlc.empty:
        return ohlc
    last = pd.Timestamp(ohlc["open_time"].iloc[-1])
    if range_label == "YTD":
        start = pd.Timestamp(year=last.year, month=1, day=1, tz=last.tz)
    else:
        start = last - pd.Timedelta(days=RANGE_DAYS.get(range_label, 90))
    return ohlc[ohlc["open_time"] >= start].reset_index(drop=True)


def build_price_figure(
    ohlc: pd.DataFrame,
    markers: pd.DataFrame,
    today_badge: dict[str, Any] | None = None,
    *,
    show_markers: bool = True,
) -> go.Figure:
    """Assemble a Binance-style gold line + gradient-area price figure.

    A thin, crisp gold line (linear — peaks stay sharp) over a vertical gradient
    fill fading to transparent, on a clean dark canvas with faint horizontal
    gridlines and a dashed current-price line. The project's retrospective
    green/red signal markers and the live "today" badge are overlaid on top and
    can be toggled off for the pure price view.

    Args:
        ohlc: DataFrame with ``open_time`` and ``close`` (open/high/low ignored).
        markers: Output of :func:`compute_historical_markers` (may be empty).
        today_badge: Optional dict with ``date``, ``price``, ``signal``, ``text``.
        show_markers: When ``False``, draw only the clean price line + area.

    Returns:
        A Plotly :class:`~plotly.graph_objects.Figure`.
    """
    fig = go.Figure()
    if ohlc.empty:
        return fig

    x = ohlc["open_time"]
    y = ohlc["close"].astype(float)
    ymin, ymax = float(y.min()), float(y.max())
    pad = (ymax - ymin) * 0.10 or ymax * 0.02
    lo, hi = ymin - pad, ymax + pad

    # Invisible baseline at the bottom of the visible band so the gradient fills
    # only the band between the line and the floor (not all the way to zero).
    fig.add_trace(
        go.Scatter(
            x=x, y=[lo] * len(x), mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", name="BTC",
            line=dict(color=BINANCE_GOLD_LINE, width=2, shape="linear"),
            fill="tonexty",
            fillgradient=dict(
                type="vertical",
                colorscale=[
                    [0.0, "rgba(240,185,11,0.0)"],
                    [1.0, "rgba(240,185,11,0.35)"],
                ],
            ),
            hovertemplate="%{x|%b %d, %Y}<br><b>$%{y:,.0f}</b><extra></extra>",
            showlegend=False,
        )
    )

    _add_signal_markers(fig, x.min(), markers, show=show_markers)
    _add_today_badge(fig, today_badge)
    _apply_dark_layout(fig, x, lo, hi)
    return fig


def chart_caption() -> str:
    """The mandatory honesty caption shown under the chart."""
    return (
        "Historical markers are retrospective (test period only) — not a live "
        "prediction feed. Green = the model's call turned out correct N days "
        "later; red = wrong. Markers use the logistic-regression signals "
        "(LightGBM fires none on this split). The ● dot shows today's live "
        "signal at the last closed candle and makes no claim about the future."
    )


# ---------------------------------------------------------------------------
# Dashboard entry point
# ---------------------------------------------------------------------------


def load_chart_data(
    cfg: dict[str, Any],
    *,
    interval: str = "1d",
    n_candles: int = CHART_FETCH_CANDLES,
    live_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch the OHLC window, markers, and today badge once for the dashboard.

    Fetching the full range up front lets the UI switch time ranges (1M…1Y)
    without re-hitting the network — slice the returned ``ohlc`` with
    :func:`slice_by_range`.

    Args:
        cfg: Config dict from :func:`src.models.train.load_modeling_config`.
        interval: Binance interval.
        n_candles: How many candles to fetch (covers the widest range).
        live_result: Optional pre-computed :func:`compute_live_signal` result.

    Returns:
        Dict with ``ohlc`` (DataFrame), ``markers`` (DataFrame), ``badge`` (dict).
    """
    ohlc = load_recent_ohlc(cfg, interval=interval, n_candles=n_candles)
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
    return {"ohlc": ohlc, "markers": markers, "badge": badge}


def load_chart_figure(
    cfg: dict[str, Any],
    *,
    interval: str = "1d",
    live_result: dict[str, Any] | None = None,
    range_label: str = "3M",
    show_markers: bool = True,
    chart_type: str = "candlestick",
) -> go.Figure:
    """Load data and build the price figure for the dashboard.

    Args:
        cfg: Config dict from :func:`src.models.train.load_modeling_config`.
        interval: Binance interval.
        live_result: Optional pre-computed :func:`compute_live_signal` result.
        range_label: One of :data:`CHART_RANGES`.
        show_markers: Whether to overlay the retrospective signal markers.
        chart_type: ``"candlestick"`` (default) or ``"area"`` (Binance line).

    Returns:
        The assembled Plotly figure.
    """
    data = load_chart_data(cfg, interval=interval, live_result=live_result)
    ohlc = slice_by_range(data["ohlc"], range_label)
    if chart_type == "area":
        return build_price_figure(
            ohlc, data["markers"], data["badge"], show_markers=show_markers
        )
    return build_candlestick_figure(
        ohlc, data["markers"], data["badge"], show_markers=show_markers
    )
