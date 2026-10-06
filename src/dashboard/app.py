"""BTC indicator dashboard — research instrument, not financial advice."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (trading/) is on sys.path so 'src' is importable
# when Streamlit launches the file directly (e.g. `streamlit run src/dashboard/app.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.dashboard.alerts import render_alerts
from src.dashboard.chart import (
    CHART_RANGES,
    build_candlestick_figure,
    build_price_figure,
    chart_caption,
    compute_expected_band,
    load_chart_data,
    slice_by_range,
)
from src.dashboard.confluence import (
    confluence,
    gather_timeframe_calls,
    setup_verdict,
)
from src.dashboard.freshness import MAX_AGE_HOURS, check_freshness, retrain_with_validation
from src.dashboard.ledger import build_scorecard, build_signal_ledger
from src.dashboard.live_predictor import (
    accuracy_over_time,
    load_predictions,
    poll_predictor,
    predictions_path,
    predictions_table,
    save_predictions,
)
from src.dashboard.live_ticker import render_live_badge
from src.dashboard.live_track_record import accumulating_message, load_forward_test
from src.dashboard.outlook import (
    DIRECTION_CV,
    expected_move,
    fetch_news,
    predict_volatility_regime,
    summarize_sentiment,
)
from src.dashboard.paper_trader import (
    STARTING_CAPITAL,
    buy_and_hold,
    load_portfolio,
    max_drawdown,
    portfolio_summary,
    reset_portfolio,
    save_portfolio,
    trade_stats,
)
from src.dashboard.risk_trader import (
    REWARD_RISK,
    RISK_FRAC,
    poll_risk_trader,
    risk_path,
)
from src.dashboard.signals import (
    HISTORICAL_ACCURACY,
    compute_live_signal,
    fetch_live_price,
)
from src.dashboard.technical_rating import RATING_STYLE, rate_symbol, to_binance
from src.dashboard.tradingview import (
    DEFAULT_INTERVAL_LABEL,
    DEFAULT_SYMBOL_LABEL,
    INTERVALS,
    SYMBOLS,
    render_tradingview,
)
from src.models.train import load_modeling_config

st.set_page_config(
    page_title="BTC Signal Dashboard",
    layout="wide",
)

_INTERVAL = "1d"
_MANIFEST_PATH = "models/1d_pruned/training_manifest.json"


@st.cache_data(ttl=3600)
def _load_cfg() -> dict:
    return load_modeling_config("configs/config.yaml")


@st.cache_data(ttl=300)
def _load_signal() -> dict:
    return compute_live_signal()


@st.cache_data(ttl=300)
def _load_chart_data(_cfg: dict) -> dict:
    return load_chart_data(_cfg, interval=_INTERVAL)


@st.cache_data(ttl=3600)
def _load_vol_regime(_cfg: dict) -> dict:
    return predict_volatility_regime(_cfg, interval=_INTERVAL)


@st.cache_data(ttl=900)
def _load_news() -> list:
    return fetch_news(limit=6)


@st.cache_data(ttl=20)
def _fresh_live_price(symbol: str) -> float:
    return fetch_live_price(symbol)


@st.fragment(run_every="20s")
def _render_signals_chart(_cfg: dict) -> None:
    """Auto-refreshing model candlestick with a live BUY/SELL marker.

    Reruns every ~20s so the live price tag and the directional signal marker
    move with the market without reloading the whole page.
    """
    try:
        chart_data = _load_chart_data(_cfg)
        badge = dict(chart_data["badge"])  # copy so we never mutate the cache
        try:
            badge["live_price"] = _fresh_live_price(_cfg["symbol"])
            badge["expected_move"] = compute_expected_band(
                chart_data["ohlc"], badge["live_price"]
            )
        except Exception:  # pragma: no cover - keep cached price on failure
            pass
        c1, c2, c3 = st.columns([2, 1.4, 1])
        range_label = (
            c1.segmented_control(
                "Range", CHART_RANGES, default="3M",
                label_visibility="collapsed", key="sig_range",
            )
            or "3M"
        )
        chart_type = (
            c2.segmented_control(
                "Type", ["Candlestick", "Line"], default="Candlestick",
                label_visibility="collapsed", key="sig_type",
            )
            or "Candlestick"
        )
        show_markers = c3.toggle("Signal markers", value=True, key="sig_markers")
        ohlc_slice = slice_by_range(chart_data["ohlc"], range_label)
        builder = (
            build_candlestick_figure if chart_type == "Candlestick"
            else build_price_figure
        )
        fig = builder(ohlc_slice, chart_data["markers"], badge, show_markers=show_markers)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Could not build the chart: {exc}")
    st.caption(chart_caption())


@st.fragment(run_every="15s")
def _live_predictor_panel(binance_symbol: str, interval: str) -> None:
    """Self-scoring next-candle predictor table; advances every ~15s.

    Predictions accumulate in ``st.session_state`` (per symbol+interval) so the
    table grows and scores itself live as candles close.
    """
    key = f"live_preds_{binance_symbol}_{interval}"
    path = predictions_path(binance_symbol, interval)
    # Restore from disk on first load of this session so the scoreboard survives
    # a browser refresh, logout, or reopen (session_state alone is ephemeral).
    if key not in st.session_state:
        st.session_state[key] = load_predictions(path)
    preds = st.session_state[key]
    try:
        preds = poll_predictor(binance_symbol, interval, preds)
    except Exception as exc:  # pragma: no cover - network/defensive UI guard
        st.caption(f"Live predictor unavailable: {exc}")
        return
    st.session_state[key] = preds
    save_predictions(preds, path)  # persist every refresh

    table, summ = predictions_table(preds)
    need = summ["min_scored"]

    def _rate_display(stats: dict) -> str:
        """Show a % only once the sample is big enough; else 'too small'.

        Works for both the overall summary (``n_scored``) and the per-direction
        stats (``n_calls``).
        """
        if stats["hit_rate"] is None:
            return "—"
        n = stats.get("n_calls", stats.get("n_scored", 0))
        if not stats["reliable"]:
            return f"— ({n}/{need})"
        return f"{stats['hit_rate']:.0f}%"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predictions", len(preds))
    m2.metric("Scored", summ["n_scored"])
    m3.metric("Correct", summ["n_correct"])
    m4.metric("Hit rate", _rate_display(summ))

    if summ["n_scored"] > 0 and not summ["reliable"]:
        st.caption(
            f"Sample too small to trust — only {summ['n_scored']} scored "
            f"call(s). A hit rate isn't shown until **{need}+** (2/3 = 67% is one "
            "lucky flip). The number will drift toward ~50% as data builds."
        )

    # Per-direction detail: how the BUY (up) and SELL (down) calls each do.
    up, dn = summ["by_call"]["UP"], summ["by_call"]["DOWN"]

    def _dir_line(stats: dict) -> str:
        if stats["hit_rate"] is None:
            return f"{stats['n_calls']} scored · {stats['n_pending']} pending"
        trust = "" if stats["reliable"] else "  ·  _too small_"
        return (
            f"**{stats['hit_rate']:.0f}%** "
            f"({stats['n_correct']}/{stats['n_calls']}) · "
            f"{stats['n_pending']} pending{trust}"
        )

    d1, d2 = st.columns(2)
    d1.metric("▲ Buy (UP) hit rate", _rate_display(up), _dir_line(up))
    d2.metric("▼ Sell (DOWN) hit rate", _rate_display(dn), _dir_line(dn))

    if table.empty:
        st.caption("Warming up… the first call appears on the next refresh.")
    else:
        st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        f"Each new {interval} candle, a **regime-adaptive** signal calls UP/DOWN "
        "**before** it closes: it **follows** a strong trend and **mean-reverts** "
        "in the chop (so it stops blindly fighting trends). **Conf** = how much "
        "the sub-signals *agree* (firm/mild/faint), **not** a probability of being "
        "right. Correct/incorrect fills in when the candle closes. Auto-refreshes ~15s. Still a "
        f"rule-based indicator, **not** a proven edge — with {need}+ scored calls "
        "expect ~50%."
    )


_GRADE_COLOUR = {"A": "#0ecb81", "B": "#F0B90B", "C": "#848e9c", "N": "#848e9c"}
_TF_ARROW = {"UP": "▲", "DOWN": "▼", "NEUTRAL": "■"}


@st.fragment(run_every="20s")
def _confluence_panel(_cfg: dict) -> None:
    """Multi-timeframe confluence + volatility-regime gate (Elder triple-screen).

    Runs the next-candle signal on 1m/5m/15m/1h, takes a higher-timeframe-weighted
    vote, and grades the moment against the volatility-regime model — so the panel
    says *when a disciplined trader would act, and when to sit out*.
    """
    symbol = _cfg["symbol"]
    try:
        calls = gather_timeframe_calls(symbol)
    except Exception as exc:  # pragma: no cover - network/defensive UI guard
        st.caption(f"Confluence unavailable: {exc}")
        return
    if not calls:
        st.caption("Confluence warming up… (waiting on live candles)")
        return

    conf = confluence(calls)
    try:
        vr = _load_vol_regime(_cfg)
    except Exception:  # pragma: no cover - vol model optional here
        vr = None
    verdict = setup_verdict(conf, vr)
    colour = _GRADE_COLOUR.get(verdict["grade"][0], "#848e9c")

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"<div style='font-size:1.6rem;font-weight:700;color:{colour}'>"
            f"{verdict['grade']}</div>"
            f"<div style='color:#848e9c'>setup grade</div>",
            unsafe_allow_html=True,
        )
        st.metric(
            "Confluence",
            f"{verdict['direction']}  ({conf['agree']}/{conf['n_tf']} agree)",
            f"vol {'EXPAND ▲' if verdict['expanding'] else 'contract'}",
        )
    with right:
        cols = st.columns(len(calls))
        for col, c in zip(cols, calls, strict=True):
            col.metric(
                c.timeframe,
                f"{_TF_ARROW.get(c.predicted, '■')} {c.predicted}",
                f"{c.regime}",
            )
    st.caption(f"**Read:** {verdict['action']}")
    st.caption(
        "Higher timeframes are weighted more (the *tide*, per Elder's Triple "
        "Screen). Grade **A** = timeframes aligned **and** volatility expanding; "
        "**No setup** = timeframes disagree → sit out. This is a **discipline "
        "filter that flags better moments** — not a price oracle; direction is "
        "still ≈50%."
    )


_VOTE_ARROW = {1: "▲", -1: "▼", 0: "·"}
_VOTE_COLOUR = {1: "#0ecb81", -1: "#f6465d", 0: "#848e9c"}
_DECISION_COLOUR = {"BUY": "#0ecb81", "SELL": "#f6465d", "HOLD": "#848e9c"}

# Trading-terminal palette for the styled metric cards.
_CARD_BG, _CARD_BORDER, _CARD_FG, _CARD_MUTED = "#181a20", "#2b3139", "#eaecef", "#848e9c"
_POS_COL, _NEG_COL = "#0ecb81", "#f6465d"


def _pnl_col(v: float) -> str:
    return _POS_COL if v >= 0 else _NEG_COL


def _metric_card(
    label: str, value: str, sub: str = "",
    *, value_colour: str = _CARD_FG, sub_colour: str = _CARD_MUTED,
) -> str:
    """A styled KPI card (HTML) that reads like a pro trading terminal tile."""
    sub_html = (
        f"<div style='color:{sub_colour};font-size:0.8rem;margin-top:3px'>{sub}</div>"
        if sub else ""
    )
    return (
        f"<div style='background:{_CARD_BG};border:1px solid {_CARD_BORDER};"
        f"border-radius:12px;padding:14px 16px'>"
        f"<div style='color:{_CARD_MUTED};font-size:0.7rem;text-transform:uppercase;"
        f"letter-spacing:0.7px;font-weight:600'>{label}</div>"
        f"<div style='color:{value_colour};font-size:1.65rem;font-weight:700;"
        f"line-height:1.3;margin-top:4px'>{value}</div>{sub_html}</div>"
    )


def _equity_figure(
    curve: list[dict], start: float, first_price: float | None, trades: list[dict],
):
    """Pro equity chart: strategy (area-filled) vs buy-and-hold, with trade markers."""
    import plotly.graph_objects as go

    df = pd.DataFrame(curve)
    df["time"] = pd.to_datetime(df["time"])
    fig = go.Figure()
    if first_price:
        hold = start * (1 - 0.001) * df["price"] / first_price
        fig.add_trace(go.Scatter(
            x=df["time"], y=hold, name="Buy & hold", mode="lines",
            line={"color": "#848e9c", "width": 1.4, "dash": "dot"},
            hovertemplate="Hold ₹%{y:,.0f}<extra></extra>",
        ))
    # Invisible baseline so the strategy area fills to ₹start (never down to 0).
    fig.add_trace(go.Scatter(
        x=df["time"], y=[start] * len(df), mode="lines", line={"width": 0},
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["equity"], name="Strategy", mode="lines",
        line={"color": "#F0B90B", "width": 2.4, "shape": "spline",
              "smoothing": 0.4},
        fill="tonexty", fillcolor="rgba(240,185,11,0.10)",
        hovertemplate="Equity ₹%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=start, line={"color": "#5e6673", "width": 1, "dash": "dash"})
    # BUY / SELL markers, snapped onto the equity line at each trade's time.
    for side, colour, symbol in (
        ("BUY", "#0ecb81", "triangle-up"), ("SELL", "#f6465d", "triangle-down"),
    ):
        pts = [t for t in (trades or []) if t.get("side") == side]
        if not pts:
            continue
        xs, ys = [], []
        for t in pts:
            idx = (df["time"] - pd.to_datetime(t["time"])).abs().idxmin()
            xs.append(df["time"].iloc[idx])
            ys.append(df["equity"].iloc[idx])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=side,
            marker={"color": colour, "size": 11, "symbol": symbol,
                    "line": {"color": "#0b0e11", "width": 1}},
            hovertemplate=f"{side} ₹%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        height=260, margin={"l": 0, "r": 0, "t": 30, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#848e9c", "size": 11}, hovermode="x unified",
        legend={"orientation": "h", "y": 1.18, "x": 0, "bgcolor": "rgba(0,0,0,0)"},
        xaxis={"showgrid": False, "showspikes": True, "spikecolor": "#5e6673",
               "spikethickness": 1, "spikemode": "across", "spikedash": "dot"},
        yaxis={"gridcolor": "rgba(255,255,255,0.05)", "tickprefix": "₹",
               "tickformat": ",.0f", "zeroline": False},
    )
    return fig


@st.fragment(run_every="20s")
def _paper_trade_panel(binance_symbol: str, interval: str) -> None:
    """Paper-trading simulator (₹10,000 fake money) driven by the strategy ensemble.

    Loads/persists the run from disk, steps it each candle, and shows equity vs a
    buy-and-hold benchmark, P&L (fees included), win rate, drawdown, the current
    ensemble decision, the equity curve and the trade log.  Fake money only.
    """
    key = f"risk_{binance_symbol}_{interval}"
    path = risk_path(binance_symbol, interval)
    if key not in st.session_state:
        st.session_state[key] = load_portfolio(path)
    state = st.session_state[key]

    try:
        state = poll_risk_trader(binance_symbol, interval, state)
    except Exception as exc:  # pragma: no cover - network/defensive UI guard
        st.caption(f"Risk trader unavailable: {exc}")
        return
    st.session_state[key] = state
    save_portfolio(state, path)

    price = float(state.get("last_price") or 0.0)
    summ = portfolio_summary(state, price)
    hold = buy_and_hold(state, price)
    stats = trade_stats(state)
    mdd = max_drawdown(state)
    sig = state.get("last_signal", {})
    vs_hold = summ["pnl"] - hold["pnl"]

    start_cap = state["starting_capital"]

    # ── Status banner: position, live stop/target, and the volatility gate ─
    vol_on = state.get("vol_expanding", True)
    up_on = state.get("uptrend", False)
    trend = (
        f"<span style='color:{_POS_COL}'>trend up ▲</span>" if up_on
        else f"<span style='color:{_NEG_COL}'>trend down ▼</span>"
    )
    gate = (
        f"<span style='color:{_POS_COL}'>vol expanding ▲</span>"
        if vol_on else
        f"<span style='color:{_CARD_MUTED}'>vol contracting</span>"
    )
    if summ["holding"]:
        status = (
            f"<span style='color:{_POS_COL};font-weight:700'>● LONG BTC</span> "
            f"from ${summ['entry_price']:,.0f} · unrealized "
            f"<span style='color:{_pnl_col(summ['unrealized'])};font-weight:600'>"
            f"₹{summ['unrealized']:+,.0f}</span> · "
            f"<span style='color:{_NEG_COL}'>stop ${state['stop_price']:,.0f}</span> / "
            f"<span style='color:{_POS_COL}'>target ${state['target_price']:,.0f}</span>"
        )
    else:
        status = (
            f"<span style='color:{_CARD_MUTED};font-weight:700'>○ FLAT (cash)</span>"
            " · waiting for a setup"
        )
    st.markdown(
        f"<div style='font-size:0.95rem;margin-bottom:10px'>{status}"
        f"<br><span style='color:{_CARD_MUTED};font-size:0.9rem'>"
        f"BTC <b>${price:,.0f}</b> &nbsp;·&nbsp; {trend} &nbsp;·&nbsp; {gate} "
        f"&nbsp;→&nbsp; long when <b>both</b> are favourable</span></div>",
        unsafe_allow_html=True,
    )

    # ── Money cards: equity, P&L, and the benchmark that matters ───────────
    c1, c2, c3 = st.columns(3)
    c1.markdown(_metric_card(
        "Equity", f"₹{summ['equity']:,.0f}", f"{summ['pnl_pct']:+.2f}%",
        sub_colour=_pnl_col(summ["pnl"])), unsafe_allow_html=True)
    c2.markdown(_metric_card(
        "Net P&L", f"₹{summ['pnl']:+,.0f}", f"on ₹{start_cap:,.0f} start",
        value_colour=_pnl_col(summ["pnl"])), unsafe_allow_html=True)
    c3.markdown(_metric_card(
        "vs Buy & Hold", f"₹{vs_hold:+,.0f}",
        f"hold {hold['pnl_pct']:+.2f}% · {'ahead' if vs_hold >= 0 else 'behind'}",
        value_colour=_pnl_col(vs_hold), sub_colour=_pnl_col(vs_hold)),
        unsafe_allow_html=True)

    st.write("")
    # ── Risk / activity cards ──────────────────────────────────────────────
    d, e, f, g = st.columns(4)
    wr = f"{stats['win_rate']:.0f}%" if stats["win_rate"] is not None else "—"
    d.markdown(_metric_card("Win rate", wr,
        f"{stats['n_wins']}W / {stats['n_losses']}L"), unsafe_allow_html=True)
    e.markdown(_metric_card("Max drawdown", f"{mdd:.2f}%", "peak → trough",
        value_colour=_NEG_COL if mdd < 0 else _CARD_FG), unsafe_allow_html=True)
    f.markdown(_metric_card("Trades", f"{summ['n_trades']}", "buys + sells"),
        unsafe_allow_html=True)
    g.markdown(_metric_card("Fees paid", f"₹{summ['fees_paid']:,.1f}", "0.1% / trade"),
        unsafe_allow_html=True)

    st.write("")
    # ── Live ensemble decision + coloured votes (in a card) ────────────────
    if sig:
        colour = _DECISION_COLOUR.get(sig["decision"], "#848e9c")
        votes = "  ·  ".join(
            f"<span style='color:{_VOTE_COLOUR.get(v, '#848e9c')}'>"
            f"{name} {_VOTE_ARROW.get(v, '·')}</span>"
            for name, v in sig["votes"].items()
        )
        st.markdown(
            f"<div style='background:{_CARD_BG};border:1px solid {_CARD_BORDER};"
            f"border-radius:12px;padding:12px 16px'>"
            f"<b>Ensemble now:</b> <span style='color:{colour};font-weight:700;"
            f"font-size:1.05rem'>{sig['decision']}</span> "
            f"<span style='color:{_CARD_MUTED}'>· net {sig['net']:+.2f}</span>"
            f"<div style='margin-top:6px'>{votes}</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    # ── Equity curve vs buy-and-hold (auto-scaled, trade markers) ──────────
    curve = state.get("equity_curve", [])
    if len(curve) >= 2:
        st.plotly_chart(
            _equity_figure(curve, start_cap, state.get("first_price"),
                           state.get("trades", [])),
            width="stretch", config={"displayModeBar": False},
        )
    else:
        st.caption("Equity curve builds as the run progresses…")

    # ── Trade log + reset ──────────────────────────────────────────────────
    trades = list(reversed(state.get("trades", [])))[:8]
    if trades:
        tdf = pd.DataFrame([
            {
                "Time": pd.to_datetime(t["time"]).strftime("%m-%d %H:%M"),
                "Side": t["side"],
                "Price": f"${t['price']:,.0f}",
                "Qty (BTC)": f"{t['qty']:.6f}",
                "Fee": f"₹{t['fee']:,.2f}",
                "Realized": f"₹{t['realized']:+,.1f}" if "realized" in t else "—",
                "Reason": t.get("reason", "entry" if t["side"] == "BUY" else "—"),
            }
            for t in trades
        ])
        st.dataframe(tdf, hide_index=True, width="stretch")
    if st.button("↺ Reset to ₹10,000", key=f"reset_{key}"):
        st.session_state[key] = reset_portfolio(path)
        st.rerun()

    st.caption(
        f"**Fake ₹{STARTING_CAPITAL:,.0f}, risk-managed — trend-following long/flat.** "
        "It goes **long** when the market is in an **uptrend** *and* **volatility is "
        "expanding** (backed by the 5-strategy ensemble), risking just "
        f"**{RISK_FRAC:.0%} of equity** per trade with an **ATR stop-loss and a "
        f"{REWARD_RISK:.0f}:1 take-profit** — so winners outrun losers even below a "
        "50% hit rate (*expectancy*, the real source of profit, not accuracy); "
        "otherwise it stays in **cash**. Gold = strategy, dotted = **buy & hold** "
        "(the bar to beat). Fake money; real exchanges add fees + 1% TDS. **Not "
        "financial advice.**"
    )


@st.fragment(run_every="20s")
def _compare_panel(_cfg: dict) -> None:
    """Investment vs Prediction on one shared time axis, + a Strategy-vs-Hold verdict.

    Reads both runs from disk (their panels persist every refresh), so it stays
    current: equity curve (strategy vs buy-and-hold) on top, the live predictor's
    cumulative hit-rate below — same time axis, directly comparable.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    symbol = _cfg["symbol"]
    paper = load_portfolio(risk_path(symbol, "15m"))
    pred_interval = st.session_state.get("pred_interval") or "1m"
    preds = load_predictions(predictions_path(symbol, pred_interval))

    price = float(paper.get("last_price") or 0.0)
    if price <= 0:
        st.caption("Comparison builds once the investment run has ticked…")
        return
    summ = portfolio_summary(paper, price)
    hold = buy_and_hold(paper, price)
    beat = summ["pnl"] - hold["pnl"]

    # ── Verdict: did the strategy beat simply holding? ─────────────────────
    v1, v2, v3 = st.columns(3)
    v1.markdown(_metric_card(
        "Strategy", f"₹{summ['equity']:,.0f}", f"{summ['pnl_pct']:+.2f}%",
        value_colour=_pnl_col(summ["pnl"])), unsafe_allow_html=True)
    v2.markdown(_metric_card(
        "Buy & hold BTC", f"₹{hold['value']:,.0f}", f"{hold['pnl_pct']:+.2f}%",
        value_colour=_pnl_col(hold["pnl"])), unsafe_allow_html=True)
    verdict = "BEAT hold" if beat >= 0 else "LAGGED hold"
    v3.markdown(_metric_card(
        "Verdict", f"₹{beat:+,.0f}", verdict,
        value_colour=_pnl_col(beat), sub_colour=_pnl_col(beat)),
        unsafe_allow_html=True)

    # ── Shared-time-axis chart: equity (top) + prediction hit-rate (bottom) ─
    curve = paper.get("equity_curve", [])
    acc = accuracy_over_time(preds)
    if len(curve) < 2:
        st.caption("Equity curve is still warming up…")
        return
    cdf = pd.DataFrame(curve)
    cdf["time"] = pd.to_datetime(cdf["time"])
    start = paper["starting_capital"]
    fp = paper.get("first_price")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
        subplot_titles=("Investment: strategy vs buy & hold (₹)",
                        f"Prediction accuracy over time ({pred_interval}, → ~50%)"),
    )
    if fp:
        fig.add_trace(go.Scatter(
            x=cdf["time"], y=start * (1 - 0.001) * cdf["price"] / fp, mode="lines",
            name="Buy & hold", line={"color": "#848e9c", "width": 1.4, "dash": "dot"}),
            row=1, col=1)
    fig.add_trace(go.Scatter(
        x=cdf["time"], y=cdf["equity"], mode="lines", name="Strategy",
        line={"color": "#F0B90B", "width": 2.2}), row=1, col=1)
    fig.add_hline(y=start, line={"color": "#5e6673", "width": 1, "dash": "dash"},
                  row=1, col=1)
    if not acc.empty:
        fig.add_trace(go.Scatter(
            x=acc["time"], y=acc["hit_rate"], mode="lines", name="Hit rate %",
            line={"color": "#0ecb81", "width": 2}), row=2, col=1)
    fig.add_hline(y=50, line={"color": "#f6465d", "width": 1, "dash": "dash"},
                  row=2, col=1)
    fig.update_layout(
        height=430, margin={"l": 0, "r": 0, "t": 40, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#848e9c", "size": 11}, hovermode="x unified",
        showlegend=True, legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", range=[0, 100], row=2, col=1)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Same time axis: **top** = your ₹10,000 investment (gold) vs just holding "
        "BTC (dotted); **bottom** = the 1-minute predictor's running accuracy, which "
        "drifts to **~50%** (a coin flip). The honest takeaway: prediction doesn't "
        "beat chance — value comes from disciplined *investing*, not calling candles."
    )


@st.fragment(run_every="15s")
def _live_signal_call(symbol_tv: str, interval_label: str) -> None:
    """Auto-refreshing BUY/SELL/NEUTRAL call for the selected market + interval.

    Runs every 15s on its own (via ``st.fragment(run_every=...)``) so the signal
    updates live without rerunning the whole page.
    """
    binance_symbol, binance_interval = to_binance(symbol_tv, interval_label)
    st.markdown(f"**Signal call · {binance_symbol} · {interval_label}**")
    try:
        r = rate_symbol(binance_symbol, binance_interval)
    except Exception as exc:  # pragma: no cover - network/defensive UI guard
        st.caption(f"Signal call unavailable: {exc}")
        return
    colour = RATING_STYLE.get(r["call"], "#848e9c")
    arrow = "▲" if r["score"] > 0 else ("▼" if r["score"] < 0 else "■")
    st.markdown(
        f"<div style='display:inline-block;background:{colour};color:#fff;"
        f"font-weight:700;font-size:18px;border-radius:6px;padding:6px 16px;'>"
        f"{arrow} {r['call']}</div>"
        f"<span style='color:#848e9c;margin-left:10px;'>"
        f"{r['n_up']}↑ / {r['n_down']}↓ · RSI {r['rsi']:.0f}</span>",
        unsafe_allow_html=True,
    )
    votes = "  ".join(
        f"{'▲' if v > 0 else ('▼' if v < 0 else '–')} {name}"
        for name, v in r["votes"].items()
    )
    st.caption(votes)
    st.caption(
        "Aggregated technical rating (EMA/RSI/MACD/momentum) from live candles — "
        "same idea as TradingView's gauge. A rule-based indicator, **not** a "
        "proven-profit signal; short-horizon direction has no measured edge. "
        "Auto-refreshes ~15s."
    )


@st.cache_data(ttl=300)
def _load_track_record(_cfg: dict) -> dict:
    return load_forward_test(_cfg, interval=_INTERVAL)


cfg = _load_cfg()

# ── Always-visible disclaimer ─────────────────────────────────────────────
st.warning(
    "**Research instrument only.** This dashboard visualises output from a "
    "backtesting project. The model has **no demonstrated directional edge** on "
    "out-of-sample data. Do not use these signals for trading decisions. "
    "**Not financial advice.**"
)

# ── Model freshness + validated retrain ───────────────────────────────────
freshness = check_freshness(_MANIFEST_PATH, max_age_hours=MAX_AGE_HOURS)
if freshness["is_stale"]:
    banner = st.container()
    banner.warning(freshness['message'])
    if banner.button("Retrain now (validated)", type="primary"):
        with st.spinner("Retraining into staging and running the validation gate…"):
            outcome = retrain_with_validation(_INTERVAL, cfg)
        if outcome["promoted"]:
            st.success(outcome["message"])
        else:
            st.error(outcome["message"])
        st.caption(
            "The retrain trains into a staging directory and only promotes the "
            "new model if it passes the same leak-tripwire gate as the original. "
            "On failure the previous model stays live. Every attempt is logged to "
            "`data/signal_log/retrain_audit.csv`."
        )
        _load_signal.clear()
        _load_chart_data.clear()
        st.rerun()
else:
    st.caption(freshness['message'])

# ── Fetch live signal ─────────────────────────────────────────────────────
with st.spinner("Fetching latest candle from Binance…"):
    result = _load_signal()

if result["error"]:
    st.error(
        f"Live fetch failed — showing cached data instead.\n\nReason: {result['error']}"
    )

# ── Command Center hero: live price + the three genuinely-useful signals ───
_hero_price = float(result["current_close"])
try:
    _hero_vr = _load_vol_regime(cfg)
    _vr_expand = _hero_vr["regime"] == "EXPAND"
    _vr_value = "EXPAND ▲" if _vr_expand else "CONTRACT ▼"
    _vr_sub = f"P={_hero_vr['p_expand']:.0%} · backtested 69% (real edge)"
except Exception:  # pragma: no cover - model warming up / data missing
    _vr_expand, _vr_value, _vr_sub = False, "—", "vol model warming up"
try:
    _hero_cd = _load_chart_data(cfg)
    _hero_bp = float(_hero_cd["badge"].get("live_price") or _hero_cd["badge"]["price"])
    _hero_em = expected_move(_hero_cd["ohlc"]["close"].tolist(), _hero_bp)
    _em_value = f"±{_hero_em['sigma_pct']:.1f}%"
    _em_sub = f"~${_hero_em['typical_move_usd']:,.0f} · typical 1-day move"
except Exception:  # pragma: no cover - defensive
    _em_value, _em_sub = "—", "n/a"

st.markdown(
    f"<div style='background:linear-gradient(135deg,#181a20,#0b0e11);"
    f"border:1px solid #2b3139;border-radius:16px;padding:22px 26px;"
    f"margin-bottom:14px;display:flex;justify-content:space-between;"
    f"align-items:center;flex-wrap:wrap;gap:14px'>"
    f"<div><div style='color:#F0B90B;font-size:1.55rem;font-weight:800;"
    f"letter-spacing:0.3px'>₿ BTC Command Center</div>"
    f"<div style='color:#848e9c;font-size:0.85rem'>Honest signals — built to "
    f"inform, not to gamble · {result['candle_date'].date()}</div></div>"
    f"<div style='text-align:right'><div style='color:#eaecef;font-size:2.1rem;"
    f"font-weight:800;line-height:1'>${_hero_price:,.0f}</div>"
    f"<div style='color:#848e9c;font-size:0.78rem'>latest close · "
    f"{result['data_source']}</div></div></div>",
    unsafe_allow_html=True,
)
hc1, hc2, hc3 = st.columns(3)
hc1.markdown(_metric_card(
    "① Volatility regime · the real edge", _vr_value, _vr_sub,
    value_colour=_POS_COL if _vr_expand else _CARD_FG), unsafe_allow_html=True)
hc2.markdown(_metric_card(
    "② Expected move · the honest 'how much'", _em_value, _em_sub),
    unsafe_allow_html=True)
hc3.markdown(_metric_card(
    "③ Direction · coin-flip guardrail", f"{result['prob_lr']:.0%} up",
    "≈ 50% — size risk, don't chase", sub_colour=_NEG_COL),
    unsafe_allow_html=True)
st.markdown(
    f"<div style='color:#848e9c;font-size:0.9rem;margin-top:10px'>"
    f"<b>Today's read:</b> volatility likely to "
    f"<b style='color:{'#0ecb81' if _vr_expand else '#eaecef'}'>"
    f"{'expand' if _vr_expand else 'contract'}</b>; a typical day moves "
    f"<b>{_em_value}</b>; direction is a coin flip — so <b>manage risk, don't "
    f"predict</b>.</div>",
    unsafe_allow_html=True,
)

# ── Live signal alert (banner + browser notification while tab open) ──────
render_alerts(result)

st.divider()

# ── Price charts: live TradingView embed + model-signal candlestick ───────
st.subheader("BTC price")
tab_live, tab_signals = st.tabs(["Live (TradingView)", "Signals (model)"])

with tab_live:
    lc1, lc2, lc3 = st.columns([2, 1.4, 1])
    tv_symbol_label = (
        lc1.selectbox("Market", list(SYMBOLS), index=list(SYMBOLS).index(
            DEFAULT_SYMBOL_LABEL), label_visibility="collapsed")
        or DEFAULT_SYMBOL_LABEL
    )
    tv_interval_label = (
        lc2.segmented_control(
            "Interval", list(INTERVALS), default=DEFAULT_INTERVAL_LABEL,
            label_visibility="collapsed",
        )
        or DEFAULT_INTERVAL_LABEL
    )
    tv_dark = lc3.toggle("Dark", value=False)
    _live_signal_call(SYMBOLS[tv_symbol_label], tv_interval_label)
    render_tradingview(
        SYMBOLS[tv_symbol_label],
        interval=INTERVALS[tv_interval_label],
        theme="dark" if tv_dark else "light",
        height=620,
        key="main",
    )
    st.caption(
        "Live feed streamed by TradingView (no API key). Model buy/sell markers "
        "live on the **Signals** tab, not on this canvas."
    )

with tab_signals:
    _sig_badge = _load_chart_data(cfg)["badge"]
    head_l, head_r = st.columns([3, 1])
    head_l.caption(
        "Hindsight-coloured signals on the model's own daily candlestick, with a "
        "**live BUY/SELL marker** at the current price (auto-refreshes ~20s). The "
        "live badge → shows the current price and the countdown to the daily "
        "candle close (UTC)."
    )
    with head_r:
        render_live_badge(
            cfg["symbol"], interval=_INTERVAL,
            prev_close=float(_sig_badge["price"]),
        )
    _render_signals_chart(cfg)

st.divider()

# ── Signal ledger + scorecard (when/where the model called BUY/SELL, and hits/misses)
st.subheader("Signal ledger & scorecard")

_now = "BUY ▲" if result["signal_lr"] == "BUY" else (
    "SELL ▼" if result["signal_lr"] == "SELL" else "SILENT (no position)"
)
st.markdown(
    f"**Right now ({result['candle_date'].date()}):** LR says **{_now}** "
    f"· P(up) `{result['prob_lr']:.3f}` · threshold `{result['threshold']}`"
)

_markers = _load_chart_data(cfg)["markers"]
score_tbl, summ = build_scorecard(_markers, limit=30)

if summ["accuracy_pct"] is not None:
    s1, s2, s3 = st.columns(3)
    s1.metric("Correct", summ["n_correct"])
    s2.metric("Wrong", summ["n_wrong"])
    s3.metric("Hit rate", f"{summ['accuracy_pct']:.0f}%")

led_col, score_col = st.columns([1.1, 1.5], vertical_alignment="top")

with led_col:
    st.markdown("**When & where — the model's BUY/SELL calls**")
    ledger = build_signal_ledger(_markers, limit=30)
    if ledger.empty:
        st.caption("No directional calls yet (the 0.60 confidence gate hasn't fired).")
    else:
        st.dataframe(ledger, hide_index=True, use_container_width=True)
    st.caption("Entry = the close on the day the model signalled. Most recent first.")

with score_col:
    st.markdown("**Right vs wrong — how those calls landed**")
    if summ["accuracy_pct"] is None:
        st.caption("No scored calls yet.")
    else:
        st.dataframe(score_tbl, hide_index=True, use_container_width=True)
    st.caption(
        "Outcome = actual move one day later (right / wrong). These are "
        "out-of-sample calls on data the model never trained on — an honest track "
        "record, **not** a promise (measured edge is ≈0)."
    )

st.divider()

# ── Live next-candle predictor (self-scoring, updates every refresh) ───────
st.subheader("Live next-candle predictor  ·  self-scoring")
st.caption(
    "Forward-looking: calls the **next** candle before it closes, then scores "
    "itself when it does — a running, honest track record that updates live."
)
pc1, _pc2 = st.columns([1, 3])
_pred_interval = (
    pc1.segmented_control(
        "Predictor interval", ["1m", "5m", "15m"], default="1m",
        label_visibility="collapsed", key="pred_interval",
    )
    or "1m"
)
_live_predictor_panel(cfg["symbol"], _pred_interval)

st.divider()

# ── Multi-timeframe confluence + volatility gate (expert discipline filter) ─
st.subheader("Multi-timeframe confluence  ·  when to act, when to sit out")
st.caption(
    "The disciplined-trader view: do 1m/5m/15m/1h **agree**, and is volatility "
    "expanding? Most moments are **No setup** — that's the point. A filter for "
    "*better moments*, not a prediction."
)
_confluence_panel(cfg)

st.divider()

# ── Paper-trading simulator (₹10,000 fake money, strategy ensemble) ─────────
st.subheader("Risk-managed paper trading  ·  ₹10,000  ·  vol-gated, stop/target")
st.caption(
    "The professional version: the strategy ensemble only enters when **volatility "
    "is expanding**, every trade has an **ATR stop-loss + a bigger take-profit**, "
    "and each risks a fixed slice of equity. This attacks **expectancy** (winners > "
    "losers), the real source of profit — not the ~50% hit rate. Fake money only."
)
# Fixed investment horizon — no interval selector. This is a long/flat
# *investment* view (hold through trends), not a scalping toy you retune.
_paper_trade_panel(cfg["symbol"], "15m")

st.divider()

# ── Investment vs Prediction: one shared time axis + a verdict ─────────────
st.subheader("Investment vs Prediction  ·  same time axis  ·  the honest verdict")
st.caption(
    "Did disciplined **investing** beat just holding — and does short-term "
    "**prediction** actually work? Both on one time axis so you can see the truth."
)
_compare_panel(cfg)

st.divider()

# ── Next-candle outlook (honest: magnitude + vol-regime are real; direction ~50%)
st.subheader("Next-candle outlook")
st.caption(
    "Ordered by how much you can trust it. **Volatility-regime and magnitude are "
    "backtested, real edges** — lead with these. **Direction is ≈ a coin flip** "
    "(measured) — it comes last, as a guardrail. Not financial advice."
)
oc1, oc2, oc3 = st.columns(3)

# Volatility regime FIRST — the real, statistically-significant edge (Task 3).
with oc1:
    st.markdown("**① Volatility regime**  ·  _the real edge_")
    try:
        vr = _load_vol_regime(cfg)
        arrow = "EXPAND ▲" if vr["regime"] == "EXPAND" else "CONTRACT ▼"
        st.metric("Next window", arrow, f"P(expand)={vr['p_expand']:.0%}")
        pt, lo, hi = vr["cv_accuracy"]
        st.caption(
            f"✓ Backtested **{pt * 100:.0f}%** [{lo * 100:.0f}%, {hi * 100:.0f}%] "
            f"(walk-forward CV) — a **real** edge. Predicts vol size, not direction."
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.caption(f"Volatility-regime model unavailable: {exc}")

# Expected move SECOND — the legitimate, forecastable "how much".
with oc2:
    st.markdown("**② Expected move**  ·  _the honest 'how much'_")
    try:
        _cd = _load_chart_data(cfg)
        base_price = float(
            _cd["badge"].get("live_price") or _cd["badge"]["price"]
        )
        em = expected_move(_cd["ohlc"]["close"].tolist(), base_price)
        st.metric("Typical ± (1σ)", f"±{em['sigma_pct']:.1f}%",
                  f"~${em['typical_move_usd']:,.0f}")
        st.caption(
            f"~2 candles in 3 close within **${em['low_1sigma']:,.0f} – "
            f"${em['high_1sigma']:,.0f}** (from {em['window']}-day realized vol)."
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.caption(f"Expected move unavailable: {exc}")

# Direction LAST — a guardrail, not a signal: P(up) stamped with its precise
# measured accuracy so an up/down "lean" is never mistaken for knowledge.
with oc3:
    st.markdown("**③ Direction**  ·  _coin flip — guardrail only_")
    lean = "UP ▲" if result["prob_lr"] >= 0.5 else "DOWN ▼"
    dist_pp = (result["prob_lr"] - 0.5) * 100.0
    st.metric("P(up) · LR", f"{result['prob_lr']:.1%}", f"{lean}  ({dist_pp:+.1f}pp)")
    d = DIRECTION_CV
    st.caption(
        f"Backtested {d['accuracy']:.1%} "
        f"[{d['ci_low']:.1%}, {d['ci_high']:.1%}]** over {d['n']:,} out-of-sample "
        f"days (walk-forward CV) · base rate {d['base_rate']:.1%} sits **inside** "
        f"the CI → **{d['verdict']}** (edge {d['edge_pp']:+.1f}pp)."
    )
    st.caption(
        "So this P(up) is a **lean, not a forecast** — even at 55% the honest read "
        "is 'basically a coin flip.' Use it to size *down* conviction, never up."
    )

with st.expander("📰 Today's BTC news  ·  learn what moves BTC", expanded=True):
    news = _load_news()
    if not news:
        st.caption("News feed unavailable right now.")
    else:
        tape = summarize_sentiment(news)
        # Juxtapose the day's headline tone against BTC's actual move so you can
        # eyeball whether the tape lined up with price — the honest way to learn
        # what moves BTC (correlation to notice, never a causal claim).
        _b = _load_chart_data(cfg)["badge"]
        move_txt = ""
        if _b.get("live_price"):
            chg = (_b["live_price"] / _b["price"] - 1.0) * 100.0
            move_txt = f"  ·  BTC **{chg:+.1f}%** since last close"
        st.markdown(
            f"**Today's tape:** {tape['n_bull']}▲ {tape['n_bear']}▼ "
            f"{tape['n_neutral']}– → net **{tape['label']}**{move_txt}"
        )
        for n in news:
            title = f"[{n['title']}]({n['link']})" if n["link"] else n["title"]
            meta = f"  ·  _{n['published']}_" if n["published"] else ""
            st.markdown(f"{n['tag']}  {title}{meta}")
    st.caption(
        "BTC-focused headlines from a public RSS feed (Cointelegraph). The ▲/▼/– "
        "tag is a crude keyword vote and the tape-vs-move line is a **juxtaposition "
        "to learn from, not a causal claim** — news is context, never a prediction."
    )

st.divider()

# ── Panel 1: Current signal ───────────────────────────────────────────────
st.subheader("1. Current Signal")

_COLOUR = {"BUY": "green", "SELL": "red", "SILENT": "orange"}

lr_sig   = result["signal_lr"]
lgb_sig  = result["signal_lgb"]
lr_prob  = result["prob_lr"]
lgb_prob = result["prob_lgb"]
threshold = result["threshold"]

lr_col, lgb_col = st.columns(2)

with lr_col:
    st.markdown(
        f"**Logistic Regression**  \n"
        f":{_COLOUR[lr_sig]}[**{lr_sig}**]  \n"
        f"P(up) = `{lr_prob:.4f}`  |  threshold = `{threshold}`"
    )
    lr_hist = HISTORICAL_ACCURACY["lr"]
    if lr_sig != "SILENT":
        wr = lr_hist["win_rate_pct"]
        st.caption(
            f"When LR fired a signal on the OOS test split, it was right "
            f"**{wr:.1f}%** of the time ({lr_hist['n_trades']} trades, "
            f"base rate {lr_hist['base_rate_pct']:.1f}%)."
        )
    else:
        st.caption(
            f"LR P(up) is within the SILENT band ({1 - threshold:.2f}–{threshold:.2f}). "
            f"No position implied. Historical OOS win rate when it did fire: "
            f"{lr_hist['win_rate_pct']:.1f}%."
        )

with lgb_col:
    st.markdown(
        f"**LightGBM**  \n"
        f":{_COLOUR[lgb_sig]}[**{lgb_sig}**]  \n"
        f"P(up) = `{lgb_prob:.4f}`  |  threshold = `{threshold}`"
    )
    lgb_hist = HISTORICAL_ACCURACY["lgb"]
    st.caption(
        f"LGB fired **{lgb_hist['n_trades']} signals** on the entire OOS test split "
        f"({HISTORICAL_ACCURACY['test_period']}) — it never exceeded the "
        f"{threshold} confidence threshold. SILENT is its only state."
    )

st.divider()

# ── Panel 2: Historical accuracy context ─────────────────────────────────
st.subheader("2. Historical Accuracy  (OOS test split, data model never saw)")
st.caption(
    f"Measured on the test split: {HISTORICAL_ACCURACY['test_period']}. "
    "These are fixed historical numbers — not live estimates, not promises."
)

lr  = HISTORICAL_ACCURACY["lr"]
lgb = HISTORICAL_ACCURACY["lgb"]
bah = HISTORICAL_ACCURACY["buyhold_test_return_pct"]

lr_acc  = f"{lr['test_accuracy_pct']:.1f}%"
lgb_acc = f"{lgb['test_accuracy_pct']:.1f}%"
lr_br   = f"{lr['base_rate_pct']:.1f}%"
lgb_br  = f"{lgb['base_rate_pct']:.1f}%"
lr_ep   = f"{lr['edge_pp']:+.1f} pp"
lgb_ep  = f"{lgb['edge_pp']:+.1f} pp"
lr_ret  = f"{lr['test_return_pct']:+.1f}%"
lgb_ret = f"{lgb['test_return_pct']:+.1f}%"
bah_ret = f"{bah:+.1f}%"

acc_df = pd.DataFrame(
    {
        "Model":         ["Logistic Regression", "LightGBM", "Buy-and-hold"],
        "Test accuracy": [lr_acc, lgb_acc, "—"],
        "Base rate":     [lr_br, lgb_br, "—"],
        "Edge vs base":  [lr_ep, lgb_ep, "—"],
        "OOS return":    [lr_ret, lgb_ret, bah_ret],
        "# trades":      [str(lr["n_trades"]), str(lgb["n_trades"]), "1"],
    }
)
st.table(acc_df)

st.caption(
    "LR accuracy was **below** the base rate on the test split (edge = −7 pp). "
    "LGB fired zero signals — effective accuracy undefined. "
    "Neither model beat buy-and-hold net of fees."
)

st.divider()

# ── Live forward-test track record (leakage-immune by construction) ───────
st.subheader("Live forward-test accuracy  (recorded before outcomes were known)")
st.caption(
    "**Distinct from the backtest above.** Each signal here was written to a "
    "permanent log by `python -m src.monitor.record_signal` *before* the price "
    "move it predicts existed — so this accuracy cannot be inflated by "
    "hindsight, window-picking, or leakage. It grows one honest day at a time. "
    "See MONITORING.md for the daily-logging setup."
)

track = _load_track_record(cfg)
if not track["enough_data"]:
    st.info(
        f"{accumulating_message(track['days_recorded'], track['min_days'])}. "
        f"A forward-test accuracy will appear once at least {track['min_days']} "
        "days of signals have been logged — showing a number before then would "
        "be noise, not evidence."
    )
else:
    st.markdown(
        f"Over **{track['days_recorded']} days recorded** "
        f"(horizon {track['horizon']}d, dead zone ±{track['dead_zone_pct']}%):"
    )
    ft_cols = st.columns(2)
    for col, model in zip(ft_cols, ("lr", "lgb"), strict=True):
        stats = track["models"][model]
        label = "Logistic Regression" if model == "lr" else "LightGBM"
        if stats["accuracy_pct"] is None:
            col.metric(f"{label} — live accuracy", "—")
            col.caption(
                f"{stats['n_fired']} signal(s) fired; none old enough to score yet."
            )
        else:
            col.metric(
                f"{label} — live accuracy",
                f"{stats['accuracy_pct']:.1f}%",
                help="Correct directional calls / evaluable calls, recorded live.",
            )
            col.caption(
                f"{stats['n_correct']}/{stats['n_evaluable']} evaluable calls "
                f"correct ({stats['n_fired']} fired in total)."
            )

st.divider()

# ── Panel 3: Volatility regime ────────────────────────────────────────────
st.subheader("3. Current Volatility Regime")

vol    = result["vol_regime"]
colour = {"calm": "green", "elevated": "orange", "high": "red"}[vol["regime"]]

st.markdown(f"Regime: :{colour}[**{vol['label']}**]")
st.write(vol["explanation"])
st.metric(
    "ret_std_30",
    f"{result['ret_std_30']:.5f}",
    help="30-day rolling std of log returns (from live features, same calc as training)",
)

st.divider()

# ── Panel 4: Reality check ────────────────────────────────────────────────
st.subheader("4. Reality Check — Fee Drag vs Measured Edge")

rt_cost = HISTORICAL_ACCURACY["round_trip_cost_pct"]
lr_edge = lr["edge_pp"]
lr_return_vs_bah = lr["test_return_pct"] - bah

c1, c2, c3 = st.columns(3)
c1.metric(
    "Round-trip cost",
    f"{rt_cost:.2f}%",
    help="0.1% taker fee + 0.1% slippage, applied to each of the two legs",
)
c2.metric(
    "LR OOS edge",
    f"{lr_edge:+.1f} pp",
    help="Test accuracy minus base rate. Negative = below base rate.",
)
c3.metric(
    "LR return vs B&H (OOS)",
    f"{lr_return_vs_bah:+.1f} pp",
    help="LR total return minus buy-and-hold return on the same period.",
)

st.info(
    f"Every round-trip costs ~{rt_cost:.1f}% in fee + slippage. "
    f"LR's measured directional edge on the OOS split was {lr_edge:+.1f} pp — "
    "already negative before costs. "
    "Acting on signals with negative edge **and** paying transaction costs is "
    "expected to underperform doing nothing. "
    "LGB avoids this by staying in cash, which outperformed in the 2025-26 bear "
    "market but would underperform in any sustained bull run."
)

st.divider()

# ── Expander: How to read this dashboard ─────────────────────────────────
with st.expander("How to read this dashboard"):
    st.markdown(
        f"""
**What this is:** a live readout from a price-direction research project
(BTCUSDT 1d, pruned logistic-regression + LightGBM models).
Data is fetched from the Binance public API and processed through the same
feature pipeline used during training.

**What this is NOT:** a trading system, a financial advisor, or a reliable edge.
The OOS test result is flat vs buy-and-hold after fees.

| Panel | What it shows |
|---|---|
| Freshness banner | Flags a model older than {MAX_AGE_HOURS:.0f}h; offers a validated retrain |
| Signal alert | Banner + browser notification when a model is not silent (with accuracy context) |
| Price chart | Binance-style line (1M–1Y); test-period markers green/red by hindsight |
| Current Signal | P(up) from each model; BUY/SELL/SILENT at the {threshold} threshold |
| Historical Accuracy | Measured OOS backtest performance on {HISTORICAL_ACCURACY["test_period"]} |
| Live forward-test | Accuracy of signals logged in real time, before outcomes were known |
| Volatility Regime | Current 30-day vol vs training-split p25/p75 thresholds |
| Reality Check | Fee cost per trade vs measured edge (negative edge + fees = expected loss) |

**Backtest vs live forward-test:** the *Historical Accuracy* panel is a
retrospective evaluation of a frozen model on past data; the *Live forward-test*
panel scores signals that were recorded before their outcomes existed. The
latter is leakage-immune by construction and is the more trustworthy of the two.

**Data freshness:** cached for 5 minutes (`ttl=300`). Each reload re-fetches
the latest complete daily candle from Binance and recomputes all features live.
The candle date shown is the **most recent fully closed** daily bar — the current
incomplete bar is never used.

See `DASHBOARD.md` for run instructions, `MONITORING.md` for the live-logging
setup, and `PHASES.md` for full methodology.
        """
    )
