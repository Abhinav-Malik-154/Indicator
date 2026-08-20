# BTC Signal Dashboard

A Streamlit dashboard that fetches live BTC/USDT candles from Binance,
runs them through the project's feature pipeline, and displays the current
model signal — with honest context about what the signal historically means.

## Running

From the repository root, with the virtualenv active:

```bash
streamlit run src/dashboard/app.py
```

This opens a browser tab at `http://localhost:8501`.

**Requirements:** `pip install -r requirements.txt` (includes `streamlit==1.62.0`).

**No API key needed.** Live candles are fetched from the Binance public
REST API (`GET /api/v3/klines`, no authentication required).

## What each panel shows

### Disclaimer (always visible)

A permanent banner at the top of every page. The model has no demonstrated
directional edge on out-of-sample data; the disclaimer is not dismissable.

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
    __init__.py      # empty package marker
    signals.py       # all computation: fetch → features → predict → result dict
    app.py           # Streamlit UI (render only — calls signals.py)
```

`signals.py` is pure Python (no Streamlit imports). `app.py` imports from
it and handles all layout. This separation makes `signals.py` independently
testable: see `tests/test_dashboard.py` (35 tests, no network calls).

## Troubleshooting

**"ModuleNotFoundError: No module named 'src'"** — run from the repository
root, not from inside `src/`. The command is:
```bash
streamlit run src/dashboard/app.py
```

**Live fetch fails / shows cached data** — Binance rate-limits heavy
scrapers. If the error banner appears, wait a few minutes and reload.
The cached data in `data/processed/features_1d.parquet` is used as fallback.

**Dashboard loads but candle date is stale** — Streamlit caches the result
for 5 minutes (`ttl=300`). Use Ctrl+F5 or the re-run button in the top
right to force a fresh fetch.
