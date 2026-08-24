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
from src.dashboard.freshness import MAX_AGE_HOURS, check_freshness, retrain_with_validation
from src.dashboard.ledger import build_scorecard, build_signal_ledger
from src.dashboard.live_predictor import poll_predictor, predictions_table
from src.dashboard.live_ticker import render_live_badge
from src.dashboard.live_track_record import accumulating_message, load_forward_test
from src.dashboard.outlook import (
    DIRECTION_CV,
    expected_move,
    fetch_news,
    predict_volatility_regime,
    summarize_sentiment,
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
    page_icon="📊",
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
    preds = st.session_state.get(key, {})
    try:
        preds = poll_predictor(binance_symbol, interval, preds)
    except Exception as exc:  # pragma: no cover - network/defensive UI guard
        st.caption(f"Live predictor unavailable: {exc}")
        return
    st.session_state[key] = preds

    table, summ = predictions_table(preds)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predictions", len(preds))
    m2.metric("Scored", summ["n_scored"])
    m3.metric("Correct", summ["n_correct"])
    m4.metric(
        "Hit rate",
        f"{summ['hit_rate']:.0f}%" if summ["hit_rate"] is not None else "—",
    )

    # Per-direction detail: how the BUY (up) and SELL (down) calls each do.
    up, dn = summ["by_call"]["UP"], summ["by_call"]["DOWN"]

    def _dir_line(stats: dict) -> str:
        if stats["hit_rate"] is None:
            return f"{stats['n_calls']} scored · {stats['n_pending']} pending"
        return (
            f"**{stats['hit_rate']:.0f}%** "
            f"({stats['n_correct']}/{stats['n_calls']}) · "
            f"{stats['n_pending']} pending"
        )

    d1, d2 = st.columns(2)
    d1.metric("▲ Buy (UP) hit rate",
              f"{up['hit_rate']:.0f}%" if up["hit_rate"] is not None else "—",
              _dir_line(up))
    d2.metric("▼ Sell (DOWN) hit rate",
              f"{dn['hit_rate']:.0f}%" if dn["hit_rate"] is not None else "—",
              _dir_line(dn))

    if table.empty:
        st.caption("Warming up… the first call appears on the next refresh.")
    else:
        st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        f"Each new {interval} candle, a **two-sided mean-reversion** signal calls "
        "UP/DOWN **before** it closes (leans DOWN when price is stretched up, UP "
        "when dipped — so it calls both ways, not just the trend); ✅/❌ is filled "
        "in once the candle closes. Auto-refreshes ~15s. A rule-based indicator, "
        "**not** a proven edge — expect the hit rate to settle near ~50%."
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
    banner.warning(f"🕒 {freshness['message']}")
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
    st.caption(f"🟢 {freshness['message']}")

# ── Fetch live signal ─────────────────────────────────────────────────────
with st.spinner("Fetching latest candle from Binance…"):
    result = _load_signal()

if result["error"]:
    st.error(
        f"Live fetch failed — showing cached data instead.\n\nReason: {result['error']}"
    )

# ── Page header ───────────────────────────────────────────────────────────
st.title("BTC/USDT Signal Dashboard  ·  1d pruned model")
h1, h2, h3 = st.columns(3)
h1.metric("Latest candle", str(result["candle_date"].date()))
h2.metric("BTC close", f"${result['current_close']:,.2f}")
h3.metric("Data source", result["data_source"])

# ── Live signal alert (banner + browser notification while tab open) ──────
render_alerts(result)

st.divider()

# ── Price charts: live TradingView embed + model-signal candlestick ───────
st.subheader("BTC price")
tab_live, tab_signals = st.tabs(["📈 Live (TradingView)", "🎯 Signals (model)"])

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
st.subheader("📋 Signal ledger & scorecard")

_now = "BUY ▲" if result["signal_lr"] == "BUY" else (
    "SELL ▼" if result["signal_lr"] == "SELL" else "SILENT (no position)"
)
st.markdown(
    f"**Right now ({result['candle_date'].date()}):** LR says **{_now}** "
    f"· P(up) `{result['prob_lr']:.3f}` · threshold `{result['threshold']}`"
)

_markers = _load_chart_data(cfg)["markers"]
led_col, score_col = st.columns([1, 1.4])

with led_col:
    st.markdown("**When & where — the model's BUY/SELL calls**")
    ledger = build_signal_ledger(_markers, limit=30)
    if ledger.empty:
        st.caption("No directional calls yet (the 0.60 confidence gate hasn't fired).")
    else:
        st.dataframe(ledger, hide_index=True, width="stretch")
    st.caption("Entry = the close on the day the model signalled. Most recent first.")

with score_col:
    st.markdown("**Right vs wrong — how those calls landed**")
    score_tbl, summ = build_scorecard(_markers, limit=30)
    if summ["accuracy_pct"] is None:
        st.caption("No scored calls yet.")
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric("Correct", summ["n_correct"])
        s2.metric("Wrong", summ["n_wrong"])
        s3.metric("Hit rate", f"{summ['accuracy_pct']:.0f}%")
        st.dataframe(score_tbl, hide_index=True, width="stretch")
    st.caption(
        "Outcome = actual move one day later (✅ right / ❌ wrong). These are "
        "out-of-sample calls on data the model never trained on — an honest track "
        "record, **not** a promise (measured edge is ≈0)."
    )

st.divider()

# ── Live next-candle predictor (self-scoring, updates every refresh) ───────
st.subheader("🔮 Live next-candle predictor  ·  self-scoring")
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

# ── Next-candle outlook (honest: magnitude + vol-regime are real; direction ~50%)
st.subheader("🔮 Next-candle outlook")
st.caption(
    "What's *actually* forecastable before the next daily candle. Magnitude and "
    "volatility-regime are backtested and real; direction is ≈ a coin flip "
    "(measured). Not financial advice."
)
oc1, oc2, oc3 = st.columns(3)

# Direction — a guardrail, not a signal: P(up) stamped with its precise
# measured accuracy so an up/down "lean" is never mistaken for knowledge.
with oc1:
    st.markdown("**Direction**  ·  _guardrail, not a signal_")
    lean = "UP ▲" if result["prob_lr"] >= 0.5 else "DOWN ▼"
    dist_pp = (result["prob_lr"] - 0.5) * 100.0
    st.metric("P(up) · LR", f"{result['prob_lr']:.1%}", f"{lean}  ({dist_pp:+.1f}pp)")
    d = DIRECTION_CV
    st.caption(
        f"⚠ **Backtested {d['accuracy']:.1%} "
        f"[{d['ci_low']:.1%}, {d['ci_high']:.1%}]** over {d['n']:,} out-of-sample "
        f"days (walk-forward CV) · base rate {d['base_rate']:.1%} sits **inside** "
        f"the CI → **{d['verdict']}** (edge {d['edge_pp']:+.1f}pp)."
    )
    st.caption(
        "So this P(up) is a **lean, not a forecast** — even at 55% the honest read "
        "is 'basically a coin flip.' Use it to size *down* conviction, never up."
    )

# Expected move — the legitimate "how much".
with oc2:
    st.markdown("**Expected move**")
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

# Volatility regime — the real, significant edge (Task 3).
with oc3:
    st.markdown("**Volatility regime**")
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
        f"📈 {accumulating_message(track['days_recorded'], track['min_days'])}. "
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
