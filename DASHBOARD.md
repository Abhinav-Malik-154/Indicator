# BTC Signal Dashboard

A Streamlit dashboard that fetches live BTC/USDT candles from Binance,
runs them through the project's feature pipeline, and displays the current
model signal — with honest context about what the signal historically means.

## Running

With the virtualenv active, run from **anywhere** — the app adds the project
root to `sys.path` automatically:

```bash
# From the project root (most common)
streamlit run src/dashboard/app.py

# Or with an absolute path from any directory
streamlit run /path/to/trading/src/dashboard/app.py
```

This opens a browser tab at `http://localhost:8501`.

**Requirements:** `pip install -r requirements.txt` (includes `streamlit==1.62.0`
and `plotly==5.24.1`).

**No API key needed.** Live candles are fetched from the Binance public
REST API (`GET /api/v3/klines`, no authentication required).

## What each panel shows

### Disclaimer (always visible)

A permanent banner at the top of every page. The model has no demonstrated
directional edge on out-of-sample data; the disclaimer is not dismissable.

### Model freshness banner + validated retrain

Just under the disclaimer, the dashboard checks the training manifest's
`trained_at_utc`. If the model is older than **24 hours** it shows a banner with
a **"Retrain now (validated)"** button; otherwise it shows a small "model is
fresh" note.

Clicking retrain does **not** blindly overwrite the live model. It:
1. Trains a fresh model into a **staging directory**.
2. Runs the same **leak-tripwire validation gate** as the original
   (`src.models.evaluate`).
3. **Only promotes** the staged model to live (atomic directory swap) **if the
   gate passes**. On failure or error the previous model stays live and a clear
   failure message is shown — an unvalidated model is never silently served.
4. Appends the attempt (timestamp, pass/fail, accuracy) to
   `data/signal_log/retrain_audit.csv`.

### Signal alert (banner + browser notification)

When either model emits a non-silent signal, a warning banner appears and — if
you granted permission — a browser notification fires. **Every alert inlines the
model's historical hit rate**, e.g. *"LR: BUY (confidence 0.63) — historically
right 38.6% of the time."* A bare directional call is never shown. When both
models are silent, a neutral "nothing to act on" note appears instead.

Alerts fire **only while the dashboard tab is open**. For an always-on record
that does not depend on anyone watching, use the daily recorder in
`MONITORING.md`.

### Price chart (candlestick + hindsight-coloured markers)

A Plotly candlestick of the **last 180 daily candles**. On top of it:

- **Green / red markers** on the **out-of-sample test period only**, showing the
  pruned logistic-regression signals coloured by whether the call turned out
  correct N days later (green = right, red = wrong). These are **retrospective**,
  not a live prediction feed. LightGBM fires no signals on this split, so it has
  no markers.
- A **★ "today" badge** at the last closed candle showing the current live
  signal. Nothing is ever drawn past the last fully closed candle.

### 1. Current Signal

The latest probability estimate from each model, computed from the most
recent complete daily candle.

| Field | Meaning |
|---|---|
| **LR signal** | BUY / SELL / SILENT from the logistic-regression model |
| **LGB signal** | BUY / SELL / SILENT from LightGBM |
| **P(up)** | Raw probability that the next close is higher than the current close |
| **Threshold** | Confidence gate (default 0.60); signals only fire at the extremes |

Signal rules:
- BUY if P(up) > threshold
- SELL if P(up) < 1 − threshold
- SILENT otherwise

The caption underneath each signal shows the historically measured win rate
when that model did fire a signal on the OOS test split.

**Data freshness:** candle data is cached for 5 minutes (`ttl=300`). Each
page load that is more than 5 minutes old re-fetches from Binance and
recomputes all features live. The date shown is the most recent *complete*
daily bar — the still-forming current bar is never used.

If the live Binance fetch fails (network error, rate limit), the dashboard
falls back to the cached features parquet from `data/processed/` and shows
a red error banner explaining this.

### 2. Historical Accuracy Context

A table of the model's measured performance on the **OOS test split** —
data the model never saw during training or early stopping.

- **Test accuracy**: fraction of predictions that matched the actual direction
- **Base rate**: fraction of days the market moved up (random-guess benchmark)
- **Edge vs base**: accuracy minus base rate (negative = below random guessing)
- **OOS return**: total return of the strategy during the test period
- **# trades**: how many round-trips were executed at the confidence threshold

These numbers are **fixed historical measurements**, not live estimates.
They describe what actually happened during `2025-08-16 → 2026-08-11`.

### Live forward-test accuracy (leakage-immune)

**This is distinct from — and more trustworthy than — the backtest above.** It
scores signals that were written to `data/signal_log/live_signals.csv` by the
daily recorder (`python -m src.monitor.record_signal`) **before** their outcomes
existed. Because a signal cannot be tuned to an outcome that has not happened
yet, this accuracy is out-of-sample by construction.

- Until **20 distinct days** have been logged it shows
  *"Accumulating — N/20 days recorded"* — no premature number.
- After that it shows each model's live accuracy: correct directional calls ÷
  evaluable calls (a call is evaluable once its outcome candle exists and the
  realized move clears the dead zone).

Setup for the daily logging job is a **one-time manual step** — see
`MONITORING.md`. Until it is scheduled, this section stays at 0/20.

### 3. Current Volatility Regime

Classifies the current 30-day rolling volatility (`ret_std_30`) relative to
the training-split distribution:

| Regime | Condition | What it means |
|---|---|---|
| **Calm** (green) | ret_std_30 < p25 | Below-average volatility; smaller expected moves |
| **Elevated** (orange) | p25 ≤ ret_std_30 ≤ p75 | Normal range |
| **High** (red) | ret_std_30 > p75 | Above-average volatility; fee drag is larger fraction of any move |

Thresholds (p25/p75 of the training split):
- Calm threshold: 0.023273
- High threshold: 0.037508

### 4. Reality Check

Compares the fee cost of acting on a signal against the model's historically
measured edge:

| Field | Value | Meaning |
|---|---|---|
| Round-trip cost | 0.40% | 0.1% taker fee + 0.1% slippage × 2 sides |
| LR OOS edge | −7.0 pp | LR was 7 percentage points *below* the base rate |
| LR return vs B&H | difference | LR total return minus buy-and-hold on the same period |

The info box explains: negative edge plus transaction costs means acting on
any signal is expected to lose money compared to doing nothing.

### Expander: How to read this dashboard

Collapsed by default. Explains the signal threshold, data freshness,
the difference between test accuracy and win rate, and links to
`PHASES.md` and the project README for deeper context.

## Architecture

```
src/dashboard/
    __init__.py            # empty package marker
    signals.py             # fetch → features → predict → result dict
    chart.py               # plotly candlestick + retrospective markers + today badge
    alerts.py              # non-silent → banner + browser notification (with accuracy context)
    freshness.py           # staleness check + validated retrain-and-promote + audit
    live_track_record.py   # leakage-immune forward-test accuracy from the signal log
    app.py                 # Streamlit UI (render only — calls the modules above)
src/monitor/
    record_signal.py       # standalone daily recorder (no Streamlit dependency)
```

The computation modules are pure Python (no Streamlit imports at their core, so
they are independently testable); `app.py` handles all layout. `record_signal.py`
imports its signal logic from `signals.py` rather than duplicating it. Tests:
`tests/test_dashboard.py` (signals + alerts + chart), `tests/test_record_signal.py`
(recorder idempotency, log stability, forward-test accuracy), and
`tests/test_freshness.py` (retrain gate) — all network-free.

## Troubleshooting

**"ModuleNotFoundError: No module named 'src'"** — this should not happen
with the current version of `app.py`, which inserts the project root into
`sys.path` at startup. If it does occur, make sure the virtualenv is active
(`source venv/bin/activate`) and dependencies are installed
(`pip install -r requirements.txt`).

**Live fetch fails / shows cached data** — Binance rate-limits heavy
scrapers. If the error banner appears, wait a few minutes and reload.
The cached data in `data/processed/features_1d.parquet` is used as fallback.

**Dashboard loads but candle date is stale** — Streamlit caches the result
for 5 minutes (`ttl=300`). Use Ctrl+F5 or the re-run button in the top
right to force a fresh fetch.

**Freshness banner says the model is stale** — the model manifest is older than
24h. Click "Retrain now (validated)" to retrain through the validation gate, or
run `python -m src.models.train --pruned --intervals 1d` from the shell.

**Live forward-test stuck at "Accumulating — 0/20"** — the daily recorder has
not been scheduled yet (or has run fewer than 20 days). Follow `MONITORING.md`
to set it up, or run `python -m src.monitor.record_signal` manually to add a day.
