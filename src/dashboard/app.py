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
from src.dashboard.chart import chart_caption, load_chart_figure
from src.dashboard.freshness import MAX_AGE_HOURS, check_freshness, retrain_with_validation
from src.dashboard.live_track_record import accumulating_message, load_forward_test
from src.dashboard.signals import HISTORICAL_ACCURACY, compute_live_signal
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
def _load_chart(_cfg: dict):
    return load_chart_figure(_cfg, interval=_INTERVAL)


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
        _load_chart.clear()
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

# ── Candlestick chart with retrospective, hindsight-coloured markers ──────
st.subheader("Price chart — last 180 days with hindsight-coloured signals")
with st.spinner("Building candlestick chart…"):
    try:
        fig = _load_chart(cfg)
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Could not build the chart: {exc}")
st.caption(chart_caption())

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
| Price chart | Last 180 candles; test-period markers coloured green/red by hindsight correctness |
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
