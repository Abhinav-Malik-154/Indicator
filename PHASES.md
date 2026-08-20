# Phase-by-Phase Technical Notes

This document contains the detailed methodology, data, and results for each
phase of the BTC Price-Behaviour Indicator project.

See [README.md](README.md) for the executive summary.

---

## Phase 1 — Data Pipeline

**Goal:** production-grade BTC/USDT candle data with no silent failures.

**Key decisions:**
- Only fully closed candles are stored. The still-forming candle at the tail of
  a pull is dropped — its OHLCV values would change before the bar closes.
- Missing candles (real exchange outages) are detected, logged with their exact
  UTC timestamps, and written to a manifest. They are never silently dropped,
  filled, or interpolated.
- Validation fails loudly: UTC timestamps end-to-end, strictly increasing open
  times, no duplicates, no nulls in OHLCV, OHLC consistency.

**Running:**
```bash
python -m src.data.fetch_binance                          # all intervals
python -m src.data.fetch_binance --intervals 1d
python -m src.data.fetch_binance --log-level DEBUG
```

**Outputs** (in `data/raw/`):

| File | Purpose |
|---|---|
| `btc_usdt_{interval}.parquet` | canonical dataset |
| `btc_usdt_{interval}.csv` | plain-text copy |
| `btc_usdt_{interval}.manifest.json` | pull time, row count, gaps, checksums |

---

## Phase 2 — Feature Engineering

**Goal:** technical + candlestick features with zero look-ahead leakage.

**Iron rule:** a feature at candle T uses only information from candles T and
earlier. No centered windows, no full-series normalisation, no backfill. This is
enforced by `validate_no_lookahead` and the deliberate-leak test in
`tests/test_features.py`, which rebuilds features using only data up to T for
every T and asserts the values are identical to the full-series build.

**Feature families:**

| Family | Features |
|---|---|
| returns | `log_ret_{1,3,7,14}` |
| volatility | `ret_std_{7,14,30}`, `range_pct`, `range_pct_ma_{7,14,30}` |
| volume | `volume_vs_ma_20`, `volume_z_20` |
| context | `close_vs_ma_{7,30}`, `dist_from_high_30`, `dist_from_low_30` |
| intra-candle | `body_pct`, `upper_wick_pct`, `lower_wick_pct`, `close_pos_in_range` |
| momentum | `rsi_14` (Wilder's smoothed RSI) |
| trend | `macd_line`, `macd_signal`, `macd_hist` (12/26/9 EMA-based) |
| candlestick | all TA-Lib `CDL*` patterns that fire in the data (one int column each) |

Candlestick pattern values live in `{-200, -100, -80, 0, 80, 100, 200}`.
Hikkake confirmations are graded ±200 and engulfing/harami near-misses ±80 by
the modern TA-Lib core (verified empirically; out-of-domain values fail the build).

**Gap policy:** rolling windows are positional (over available candles), never
wall-clock. Gap records from Phase 1 manifests are read back during the build;
every feature row whose longest lookback spans a recorded gap is counted and
written to the build manifest. Rows without full rolling history keep NaN.

**Running:**
```bash
python -m src.features.build_features                     # all intervals
python -m src.features.build_features --intervals 1h 30m
```

---

## Phase 3 — Label Design

**Goal:** forward-return labels with noise exclusion.

The label iron rule: the label at row T is the answer key — it deliberately
looks forward to `close[T+N]`. Labels live in a separate file
(`labels_{interval}.parquet`), never merged into the features file. They are
joined only at training time on `open_time`.

| Setting | Default | Purpose |
|---|---|---|
| `horizons` | `[1]` | forward-return horizon in candles |
| `dead_zone_pct` | `0.15` | moves within ±0.15% are excluded (NaN) |

Classification: `label = 1` if `close[T+N]/close[T] - 1 > +threshold`,
`label = 0` if `< -threshold`, `NaN` if inside the dead zone or tail.

**Running:**
```bash
python -m src.labels.build_labels                          # all intervals
python -m src.labels.build_labels --intervals 1d
```

---

## Phase 4 — Modeling

**Goal:** honest directional accuracy measurement on held-out data.

**Split strategy — walk-forward, never random:**
1. Chronological ordering: training always precedes validation, which always
   precedes test. Boundaries computed positionally before any NaN is dropped.
2. No overlap: a configurable gap (`modeling.split.gap_candles`, must be
   `>= modeling.horizon`) separates each split.
3. NaN handling after boundaries, not before.
4. Scaler fit on training data only: `StandardScaler` is `.fit()` on train
   and `.transform()` on validation and test.

**Models:**
- **Logistic regression** — interpretable baseline. Trained on scaled features.
- **LightGBM** — gradient boosting with early stopping on val log loss.

**Confidence gating:** P(up) > 0.60 → BUY, P(up) < 0.40 → SELL, else SILENT.

**Leak tripwire:** if OOS accuracy exceeds 65% or a single feature carries
>50% of total importance, `evaluate.py` stamps the report **PROBABLE LEAK**
and exits with code 2.

**Honest results — BTCUSDT 1d, horizon 1:**

Split: train 1,557 rows (2020-01-31 → 2024-08-16), val 329 rows
(2024-08-18 → 2025-08-14), test 328 rows (2025-08-17 → 2026-08-10).

| Model | Split | Accuracy | Base rate | Edge |
|---|---|---|---|---|
| Logistic regression | val | 52.9% | 53.2% (up) | −0.3 pp |
| Logistic regression | test | 44.2% | 52.1% (down) | −7.9 pp |
| LightGBM | val | 50.8% | 53.2% (up) | −2.4 pp |
| LightGBM | test | 51.8% | 52.1% (down) | −0.3 pp |

No leak alert fired. Neither model beats the base rate.

**Running:**
```bash
python -m src.models.train                     # trains all modeling.intervals
python -m src.models.train --intervals 1d
python -m src.models.evaluate --intervals 1d
```

### Feature Pruning Experiment (Phase 4 add-on)

**Motivation:** logistic-regression top coefficients were dominated by rare
candlestick patterns firing in only 1–28 training rows. A coefficient from
that few observations is effectively a coincidence, not a signal.

**Threshold (pre-committed):** drop every candlestick pattern firing in fewer
than 30 training rows. The threshold is the "10 events per variable" clinical
minimum scaled by 3×, chosen before looking at any val/test metric.

Fire-rates are computed on the training split only. This is enforced by
`prune_candlestick_features()` and tested in `TestPruneCandlestickFeatures`.

**Patterns dropped (20 of 38):** `cdl_3inside`, `cdl_3linestrike`,
`cdl_advanceblock`, `cdl_dojistar`, `cdl_dragonflydoji`, `cdl_eveningdojistar`,
`cdl_eveningstar`, `cdl_gapsidesidewhite`, `cdl_gravestonedoji`,
`cdl_hangingman`, `cdl_hikkakemod`, `cdl_identical3crows`,
`cdl_invertedhammer`, `cdl_morningstar`, `cdl_risefall3methods`,
`cdl_separatinglines`, `cdl_shootingstar`, `cdl_stalledpattern`,
`cdl_tristar`, `cdl_xsidegap3methods`.

**Accuracy delta:** ≤1 pp on every split — pruning confirmed noise removal.

**Importances after pruning (LR):** `cdl_longline`, `cdl_rickshawman`,
`close_pos_in_range`, `range_pct_ma_30`, `dist_from_high_30`, `cdl_belthold`.
All fire ≥223 training times. Technical features are now visible; rare
coincidence patterns are gone.

**Running:**
```bash
python -m src.models.train --pruned --intervals 1d
python -m src.models.evaluate --pruned --intervals 1d
```

---

## Phase 5 — Backtesting

**Goal:** measure whether the model produces real returns net of fees.

**Methodology:** uses the pruned model's predictions on the test split only.
Val was used for early stopping (model selection) and is not a clean proxy.

**Signal-to-trade:** on days with a confidence-gated signal, take a full
position at that day's close. On SILENT days, hold cash.

**Cost structure:**

| Cost item | Rate | Applied |
|---|---|---|
| Taker fee | 0.1% per side | every entry and exit |
| Slippage | 0.1% per side | every entry and exit |
| Total round-trip | ~0.4% | per trade |

Buy-and-hold pays the same per-side rate but only twice — once in, once out.

**Iron rule:** position at T is set from signal[T], which uses only features
from close[T]. The return is `close[T+1] / close[T] − 1`.

**Results — BTCUSDT 1d, test split (2025-08-17 → 2026-08-11):**

BTC fell −46% during this window.

| Metric | LR (pruned) | LGB (pruned) | Buy-and-hold |
|---|---|---|---|
| Total return | −46.8% | 0.0% | −46.0% |
| Sharpe ratio | −1.80 | — | −1.08 |
| # trades | 70 | 0 | 1 |
| Win rate | 38.6% | — | n/a |

**Honest verdict:** LR returned −46.8% vs B&H −46.0% — essentially the same
outcome, with 70 round-trips generating ~14 pp of fee drag that the model
had to overcome just to match buy-and-hold. It didn't.

LGB fired zero signals at the 0.60 threshold. Being mostly in cash during a
bear market produced 0% nominal return, which outperformed buy-and-hold
mechanically — but zero-signal is not a strategy, and the same model would
miss any sustained bull run entirely.

**Running:**
```bash
python -m src.backtest.report
python -m src.backtest.report --intervals 1d
```

---

## Phase 6 — On-chain Features

**Goal:** measure whether free on-chain data improves directional accuracy.

**Source:** Blockchain.com Charts API (free, no auth required). Coverage:
2019-08-22 → present.

**Metrics fetched:**

| Metric | API chart name |
|---|---|
| Active addresses | `n-unique-addresses` |
| Transaction count | `n-transactions` |
| Hash rate (TH/s) | `hash-rate` |
| Fees (USD) | `transaction-fees-usd` |
| Volume (USD) | `estimated-transaction-volume-usd` |

**NOT available free:** exchange netflows, whale movement, HODL waves, realized
price, NUPL, STH/LTH supply — all require Glassnode or CryptoQuant paid tiers.

**Feature engineering:**
- Conservative 1-day lag: on-chain data for day T is used at candle T+1 only.
- Rolling z-scores: 7-day and 30-day normalised values for each metric.
- Week-over-week % change for each metric.
- Total: 5 metrics × 3 transformations = **15 on-chain features**, all prefixed `onchain_`.

**3-way comparison (BTCUSDT 1d, horizon 1):**

| Variant | Model | Val accuracy | Val edge | Test accuracy | Test edge |
|---|---|---|---|---|---|
| base | LR | 52.9% | −0.3 pp | 45.4% | −6.7 pp |
| base | LGB | 53.2% | +0.0 pp | 47.9% | −4.3 pp |
| pruned | LR | 53.5% | +0.3 pp | 45.4% | −6.7 pp |
| pruned | LGB | 53.2% | +0.0 pp | 47.9% | −4.3 pp |
| **onchain** | **LR** | **53.8%** | **+0.6 pp** | **47.0%** | **−5.2 pp** |
| **onchain** | **LGB** | **53.8%** | **+0.6 pp** | **47.6%** | **−4.6 pp** |

**Honest verdict:** on-chain features produce a marginal, non-material
improvement (+0.3–0.6 pp on val, within noise on test). At daily granularity,
free on-chain metrics trend together with price — their z-scores carry
correlated directional information already present in price features. The
on-chain metrics most useful for regime detection (exchange netflows, realized
price, STH/LTH ratio) require paid services. Result: neutral.

**Running:**
```bash
python -m src.data.fetch_onchain
python -m src.features.build_features --intervals 1d
python -m src.models.train --intervals 1d --onchain
python -m src.models.evaluate --intervals 1d --onchain
```

---

## Phase 7 — Multi-regime Backtesting

**Goal:** test whether IS bear-market results reflect learned skill or memorisation.

**Methodology:** five historical BTC windows were defined in
`src/backtest/multi_regime.py` **before** any backtest was run, preventing
post-hoc cherry-picking. Windows were chosen from publicly documented BTC
market history. The in-sample flag is set mechanically — any window overlapping
the training period `[2020-01-01, 2024-08-16]` is labelled IS.

**Regime windows:**

| Regime | IS/OOS | Start | End | Rationale |
|---|---|---|---|---|
| COVID crash | IS | 2020-02-01 | 2020-04-30 | Sharp drawdown in training data |
| 2020-21 bull run | IS | 2020-10-01 | 2021-11-10 | Major bull run, in training data |
| 2022 bear market | IS | 2022-01-01 | 2022-12-31 | Sustained bear, in training data |
| 2023 recovery | IS | 2023-01-01 | 2023-12-31 | Recovery year, in training data |
| Phase 5 test | **OOS** | **2025-08-16** | **2026-08-11** | The held-out test split |

**Results (BTCUSDT 1d, pruned model):**

| Regime | IS/OOS | Days | LR return | LGB return | B&H return | LR Sharpe | LR trades |
|---|---|---|---|---|---|---|---|
| COVID crash | IS | 90 | +116.3% | 0.0% | −6.3% | 3.10 | 28 |
| 2020-21 bull run | IS | 212 | +6.5% | 0.0% | +442.1% | 0.39 | 37 |
| 2022 bear market | IS | 365 | +14.0% | 0.0% | −65.3% | 0.48 | 85 |
| 2023 recovery | IS | 365 | +17.8% | 0.0% | +164.8% | 0.74 | 76 |
| **Phase 5 test** | **OOS** | **360** | **−46.8%** | **0.0%** | **−46.0%** | **−1.80** | **70** |

LGB fires zero signals in every window — IS and OOS alike.

**Honest verdict:**

- IS bear markets (COVID, 2022): LR substantially outperforms B&H. This is what
  overfitting to bear conditions in the training data looks like — not skill.
- IS bull markets (2020-21, 2023): LR severely underperforms B&H. The same
  conservatism that avoids bears causes it to miss sustained uptrends entirely.
- OOS (the only result that counts): LR −46.8% vs B&H −46.0%. Even in a bear
  regime that superficially resembles the 2022 training data, LR fails to
  replicate the IS result. 70 round-trips generated ~14 pp of fee drag for
  essentially the same terminal outcome as buy-and-hold.

**Conclusion:** Phase 7 confirms Phase 5. The model has no demonstrated
directional edge. IS bear-market performance is memorisation, not predictive
ability.

**Running:**
```bash
python -m src.backtest.multi_regime
python -m src.backtest.multi_regime --log-level DEBUG
```

---

## Phase 8 — Dashboard

**Goal:** make the project presentable and demoable.

Outputs:
- `src/dashboard/signals.py` — live signal computation (fetch → features → predict)
- `src/dashboard/app.py` — Streamlit dashboard (5 panels + disclaimer)
- `tests/test_dashboard.py` — 35 smoke tests (pure functions, no network)
- `DASHBOARD.md` — run instructions and panel descriptions

**Running:**
```bash
streamlit run src/dashboard/app.py
```

See [DASHBOARD.md](DASHBOARD.md) for full documentation.
