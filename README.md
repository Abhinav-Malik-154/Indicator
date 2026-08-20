# BTC Price-Behaviour Indicator

A research project that studies short-horizon BTC/USDT price behaviour and builds an
indicator on top of it — engineered like a data product, not a notebook experiment.

## Honest methodology, stated up front

Directional accuracy for systems of this kind realistically lands around **50–58%**.
Anything far above that range is almost always a symptom of look-ahead bias or data
leakage, so this project treats data integrity as a first-class requirement:

- Only **fully closed candles** are ever stored. The still-forming candle at the tail
  of a pull is dropped, because its values would change after the fact.
- Missing candles (real exchange outages) are **detected, logged with their exact
  timestamps, and recorded in a manifest — never silently dropped or filled**.
- Validation is strict and fails loudly: UTC timestamps end to end, strictly
  increasing open times, no duplicates, no nulls in OHLCV, consistent OHLC values.

## Project status

**Phase 1 (done):** production-grade BTC/USDT data pipeline against the Binance
public REST API. **Phase 2 (done):** leakage-safe feature engineering —
candlestick patterns, technical indicators (returns, volatility, volume,
context, intra-candle shape), RSI, MACD, standalone validation guards
(`validate_no_lookahead`, `validate_feature_label_alignment`), and a
deliberate-leak test that proves the guard catches future-data usage.
**Phase 3 (done):** forward-return label design with dead-zone noise
exclusion, alignment validation, and class-balance reporting.
**Phase 4 (done):** walk-forward-split modeling — logistic regression
baseline, LightGBM gradient boosting, confidence gating, an automated
leak tripwire, and honest accuracy-vs-base-rate reporting.
**Phase 5 (done):** backtesting with realistic fees and slippage —
strategy equity curves, comparison vs buy-and-hold, and an honest verdict.
**Phase 6 (done):** Bitcoin on-chain features from the Blockchain.com Charts API
(free, no auth) — 5 metrics × 3 transformations = 15 features added to the
pipeline, with a 1-day conservative lag, no-lookahead validation, and a 3-way
comparison against price-only models.
**Phase 7 (done):** multi-regime backtesting across 5 historical BTC windows
(COVID crash, 2020-21 bull, 2022 bear, 2023 recovery, Phase 5 OOS test) with
explicit in-sample / out-of-sample labeling committed before running any backtest.

## Repository layout

```
configs/config.yaml               # symbol, intervals, date range, API, windows, paths, modeling
src/data/fetch_binance.py         # fetch -> validate -> save pipeline (CLI entry point)
src/features/candlestick.py       # TA-Lib CDL* pattern features + fire-rate reporting
src/features/technical.py         # returns / volatility / volume / context / RSI / MACD
src/features/build_features.py    # feature build CLI: gap accounting + manifests
src/features/validate_features.py # no-lookahead guard + feature-label alignment check
src/labels/build_labels.py        # forward-return labels with dead zone + class balance
src/labels/validate_labels.py     # derivation correctness + feature-label alignment checks
src/models/train.py               # walk-forward split, scaler-on-train-only, logreg + LightGBM
src/models/evaluate.py            # accuracy vs base rate, importances, gating, leak tripwire
src/backtest/simulate.py          # day-by-day equity simulation with per-side fees + slippage
src/backtest/baseline.py          # buy-and-hold baseline (1 entry + 1 exit fee)
src/backtest/report.py            # comparison table + CLI for Phase 5
src/backtest/multi_regime.py      # Phase 7: 5-regime backtest with IS/OOS labeling
src/data/fetch_onchain.py         # Blockchain.com Charts API fetcher (Phase 6, free)
src/features/onchain.py           # on-chain feature engineering: lag + z-scores + WoW
tests/                            # unit tests incl. leakage + deliberate-leak + alignment
notebooks/01_feature_sanity.ipynb # visual sanity checks only — no logic in notebooks
notebooks/02_label_sanity.ipynb   # label colour overlay, balance chart, return distribution
notebooks/03_feature_sanity.ipynb # Phase 2: RSI/MACD plots, correlation matrix, spot-checks
notebooks/04_model_evaluation.ipynb # Phase 4: probability dist, confusion matrix, importances,
                                   #   accuracy-by-confidence-bucket
notebooks/05_backtest_results.ipynb # Phase 5: equity curves, drawdown, comparison table
notebooks/06_onchain_analysis.ipynb # Phase 6: on-chain metrics vs price, 3-way comparison
notebooks/07_multi_regime.ipynb   # Phase 7: small-multiple equity curves, IS/OOS table
data/raw/                         # candles: parquet + CSV + manifest (git-ignored)
data/processed/                   # feature / label tables + manifests (git-ignored)
models/                           # trained artifacts + training manifest (git-ignored)
.github/workflows/ci.yml          # ruff + pytest on every push / PR
```

## Setup

Requires Python 3.11+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Running the data pull

From the repository root:

```bash
python -m src.data.fetch_binance                          # all intervals from configs/config.yaml
python -m src.data.fetch_binance --intervals 1d           # subset of intervals
python -m src.data.fetch_binance --log-level DEBUG        # verbose logging
```

The pull is configured by `configs/config.yaml` (symbol, intervals, UTC date range,
retry/backoff and pagination settings, output paths). `end_date: null` means "up to
the most recent fully closed candle at pull time".

Each interval produces three artifacts in `data/raw/`:

| File | Purpose |
| --- | --- |
| `btc_usdt_{interval}.parquet` | canonical dataset |
| `btc_usdt_{interval}.csv` | plain-text copy for easy inspection |
| `btc_usdt_{interval}.manifest.json` | pull time, row count, date range, gaps, checksums |

## Building features

```bash
python -m src.features.build_features                     # all intervals -> data/processed/
python -m src.features.build_features --intervals 1h 30m  # subset
```

The iron rule: a feature for candle T uses information from candles T and
earlier only. No centered windows, no full-series normalisation (scalers are
fit on training data only, in a later phase), no backfill. This is enforced by
the leakage test in `tests/test_features.py`, which rebuilds features using
only data up to candle T for every T and asserts the values at T are identical
to the full-series build.

Feature families (windows configurable in `configs/config.yaml`):

| Family | Features |
| --- | --- |
| returns | `log_ret_{1,3,7,14}` |
| volatility | `ret_std_{7,14,30}`, `range_pct`, `range_pct_ma_{7,14,30}` |
| volume | `volume_vs_ma_20`, `volume_z_20` |
| context | `close_vs_ma_{7,30}`, `dist_from_high_30`, `dist_from_low_30` |
| intra-candle | `body_pct`, `upper_wick_pct`, `lower_wick_pct`, `close_pos_in_range` |
| momentum | `rsi_14` (Wilder's smoothed RSI) |
| trend | `macd_line`, `macd_signal`, `macd_hist` (12/26/9 EMA-based) |
| candlestick | all TA-Lib `CDL*` patterns that fire in the data (one int column each) |

Candlestick values live in `{-200, -100, -80, 0, 80, 100, 200}`: Hikkake
confirmations are graded ±200 and engulfing/harami near-misses ±80 by the
modern TA-Lib core (verified empirically; out-of-domain values fail the build).

Gap policy: rolling windows are positional (over available candles), never
wall-clock. Gap records from the Phase 1 manifests are read back during the
build, and every feature row whose longest lookback window spans a recorded
gap is counted and written to the build manifest — candles are never filled,
interpolated, or synthesised. Rows without full rolling history keep NaN.

## Building labels

```bash
python -m src.labels.build_labels                          # all intervals -> data/processed/
python -m src.labels.build_labels --intervals 1d           # subset
```

The label iron rule: the label at row T is the *answer key* — it deliberately
looks forward to `close[T+N]`. But labels live in a **separate file**
(`labels_{interval}.parquet`), never merged into the features file. The two
are joined only at training time on `open_time`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `horizons` | `[1]` | forward-return horizon in candles |
| `dead_zone_pct` | `0.15` | moves within ±0.15% are excluded (NaN) — noise filter |

Classification: `label = 1` if `close[T+N]/close[T] - 1 > +threshold`,
`label = 0` if `< -threshold`, `NaN` if inside the dead zone or if
`close[T+N]` does not exist (the last N rows). The dead zone uses strict
inequalities: a return of exactly ±threshold is excluded.

Alignment between features and labels is validated by
`src/labels/validate_labels.py` and `src/features/validate_features.py`,
which check derivation correctness, join integrity (no off-by-one, no
duplication), and namespace separation. The deliberate-leak test in
`tests/test_features.py` intentionally shifts a feature by -1 and asserts
that `validate_no_lookahead` catches it — this proves the guard works,
not just that it passes clean code.

## Modeling (Phase 4)

```bash
python -m src.models.train                     # trains all modeling.intervals -> models/
python -m src.models.train --intervals 1d       # subset
python -m src.models.evaluate --intervals 1d    # honest evaluation report
```

### Split strategy — time-based walk-forward, not random shuffle

1. **Chronological ordering**: the split respects time — training data
   always precedes validation, which always precedes test. Boundaries are
   computed positionally on the full joined feature+label table **before**
   any NaN row is dropped, so no statistic derived from the split (or from
   the NaN pattern, which depends on labels) can influence where a row
   lands.
2. **No overlap**: a configurable gap (`modeling.split.gap_candles`, must be
   `>= modeling.horizon`) separates training from validation and validation
   from test, so the label at the last row of one segment never depends on
   a close price inside the next segment.
3. **NaN handling after boundaries, not before**: within each split, rows
   with a warm-up feature (insufficient rolling history) or a dead-zone/tail
   label are dropped. Because this happens after the split boundaries are
   fixed, dropped rows never shift a row across a boundary.
4. **Scaler fit on training data only**: `StandardScaler` is `.fit()` on the
   training split's features and only `.transform()`-ed onto validation and
   test — verified by `tests/test_models.py`, which asserts the fitted
   scaler's mean/scale are unaffected by data it never saw.

### Models

- **Logistic regression** — the fair, interpretable baseline. Trained on
  scaled features (mean 0, unit variance, scaler fit on train only).
- **LightGBM** — gradient boosting, trained on unscaled features (trees are
  scale-invariant) with early stopping on the validation split's log loss.

Both are interval-agnostic: `modeling.intervals` in `configs/config.yaml`
controls which intervals get trained; 1d runs first, 1h/30m can be added
later without code changes.

### Confidence gating

A configurable probability threshold (`modeling.confidence_threshold`,
default 0.60) turns raw probabilities into a three-way signal: emit "up" if
`P(up) > threshold`, "down" if `P(up) < 1 - threshold`, otherwise stay
silent. `evaluate.py` reports accuracy for "all predictions" and "signals
fired" separately, plus signal coverage — the gated number is what an
indicator built on this would actually use.

### Leak tripwire

If any model's out-of-sample accuracy exceeds `modeling.leak_alert_accuracy`
(default 65%), `evaluate.py` logs an error, stamps the report
**PROBABLE LEAK**, and exits with code 2 instead of presenting the number as
a result. A single feature carrying more than half of total importance
triggers the same kind of warning — both are checked automatically, not left
to manual review.

### Honest results — BTCUSDT 1d, horizon 1

Trained on `data/processed/features_1d.parquet` + `labels_1d.parquet`
(2,415 rows). Split: train 1,557 usable rows (2020-01-31 → 2024-08-16), val
329 rows (2024-08-18 → 2025-08-14), test 328 rows (2025-08-17 → 2026-08-10),
gap of 1 candle at each boundary.

| Model | Split | Accuracy | Base rate | Edge |
| --- | --- | --- | --- | --- |
| Logistic regression | val | 52.9% | 53.2% (up) | −0.3pp |
| Logistic regression | test | 44.2% | 52.1% (down) | −7.9pp |
| LightGBM | val | 50.8% | 53.2% (up) | −2.4pp |
| LightGBM | test | 51.8% | 52.1% (down) | −0.3pp |

No leak alert fired (nothing exceeds 65%) and no single feature dominates
importance (LightGBM's top feature, `log_ret_1`, carries 9.6% of gain).
Read plainly: on this split, neither model beats the base rate — consistent
with the 50–58% honest-accuracy expectation stated at the top of this
README, and with directional prediction on daily BTC being a genuinely hard
problem. Confidence gating does not yet help either: gated ("signals fired")
accuracy is below ungated accuracy for logistic regression on both splits,
and LightGBM's gated signals are thin (16–30 predictions, all one-directional
on both splits) — see `notebooks/04_model_evaluation.ipynb` for the
accuracy-by-confidence-bucket chart this conclusion is based on. These
numbers are reported as-is, not smoothed over; Phase 5 backtesting will show
whether either model is useful net of fees, which is a separate question
from raw directional accuracy.

### Feature pruning experiment (Phase 4 add-on)

**Motivation.** In Phase 4, logistic regression's top coefficients were dominated by
rare candlestick pattern dummies that fired in only 1–28 training rows.  With so few
observations, the logistic-regression coefficient is estimated from a vanishingly thin
slice of data: its value is effectively a coincidence, not a signal.  An overfit-to-regime
coefficient may flip sign the next time the pattern fires, making the importance table
untrustworthy even if accuracy looks fine.

**Threshold (pre-committed, never tuned).** Drop every candlestick pattern that fires
in fewer than 30 training rows.  The threshold is the clinical "10 events per variable"
minimum scaled by 3× — a general statistical stability rule chosen *before* looking at
any validation or test metric.  Using val/test outcomes to choose the cutoff would be
feature-selection leakage (the same category of mistake the rest of the project avoids).

Fire-rates are computed on the **training split only** (never val/test); this is
enforced by `prune_candlestick_features()` and tested in `TestPruneCandlestickFeatures`.

**Patterns dropped (20 of 38 candlestick features, all < 30 training fires):**
`cdl_3inside` (7), `cdl_3linestrike` (4), `cdl_advanceblock` (25), `cdl_dojistar` (13),
`cdl_dragonflydoji` (28), `cdl_eveningdojistar` (2), `cdl_eveningstar` (2),
`cdl_gapsidesidewhite` (2), `cdl_gravestonedoji` (20), `cdl_hangingman` (19),
`cdl_hikkakemod` (0), `cdl_identical3crows` (0), `cdl_invertedhammer` (6),
`cdl_morningstar` (2), `cdl_risefall3methods` (2), `cdl_separatinglines` (9),
`cdl_shootingstar` (3), `cdl_stalledpattern` (2), `cdl_tristar` (1),
`cdl_xsidegap3methods` (7).  All technical features are kept untouched.

**Side-by-side accuracy (BTCUSDT 1d, horizon 1):**

| Model | Variant | Split | Accuracy | Base rate | Edge |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | baseline | val | 52.9% | 53.2% (up) | −0.3pp |
| Logistic regression | **pruned** | val | 52.3% | 53.2% (up) | −0.9pp |
| Logistic regression | baseline | test | 44.2% | 52.1% (down) | −7.9pp |
| Logistic regression | **pruned** | test | 45.1% | 52.1% (down) | −7.0pp |
| LightGBM | baseline | val | 50.8% | 53.2% (up) | −2.4pp |
| LightGBM | **pruned** | val | 50.8% | 53.2% (up) | −2.4pp |
| LightGBM | baseline | test | 51.8% | 52.1% (down) | −0.3pp |
| LightGBM | **pruned** | test | 51.8% | 52.1% (down) | −0.3pp |

Accuracy deltas are within ±1pp for every split and model — the experiment confirms that
the rare dummies added noise, not signal: removing them does not hurt performance.

**Feature importances — the actual success criterion.**

*Logistic regression top 6 (baseline):*
`cdl_rickshawman` (+0.22), `cdl_longline` (+0.21), **`cdl_gapsidesidewhite` (+0.20, 2 fires)**,
**`cdl_risefall3methods` (−0.20, 2 fires)**, `close_pos_in_range` (−0.17),
**`cdl_dragonflydoji` (+0.16, 28 fires)**, **`cdl_tristar` (−0.15, 1 fire)**.
Four of the top seven were rare patterns that fired in ≤28 rows — classic overfit-to-regime
signatures.

*Logistic regression top 6 (pruned):*
`cdl_longline` (+0.20), `cdl_rickshawman` (+0.16), `close_pos_in_range` (−0.16),
`range_pct_ma_30` (−0.13), `dist_from_high_30` (−0.12), `cdl_belthold` (−0.12).
All six fire ≥223 training times.  Technical features (`close_pos_in_range`,
`range_pct_ma_30`, `dist_from_high_30`) are now visible; the top surviving candlestick
patterns are high-frequency ones whose coefficients are estimated from a real sample.

*LightGBM (identical for both variants):* `log_ret_1` (9.6%), `upper_wick_pct` (8.9%),
`close_pos_in_range` (6.6%) — tree ensembles are naturally resistant to low-fire-rate
dummies because gain-splitting on a column that is almost always 0 carries little reward.
No pruning change was needed or observed.

**Verdict.** Success on the stated criterion: logistic-regression importances are now
economically sensible — momentum (`log_ret_14`), volatility (`ret_std_30`, `range_pct_ma_30`),
structure (`dist_from_high_30`, `close_pos_in_range`), and well-observed candlestick patterns
(`cdl_longline`, `cdl_belthold`).  Accuracy was flat, as expected; no leak alert fired.

Run with:
```bash
python -m src.models.train --pruned --intervals 1d    # saves to models/1d_pruned/
python -m src.models.evaluate --pruned --intervals 1d  # loads from models/1d_pruned/
```

## Backtesting (Phase 5)

```bash
python -m src.backtest.report                     # runs on modeling.intervals
python -m src.backtest.report --intervals 1d       # subset
python -m src.backtest.report --log-level DEBUG    # verbose
```

### Methodology

The backtest uses the **pruned model's predictions on the test split only.**
Val was used for early stopping (model selection) in Phase 4, so it is not a
clean out-of-sample proxy — only the test split qualifies as "real future".

**Signal-to-trade logic:** on days with a confidence-gated signal
(`P(up) > 0.60` → long; `P(up) < 0.40` → short), the simulation takes a full
position at that day's close.  On silent days, it holds cash.

**Cost structure (configured in `backtest:` in `configs/config.yaml`):**

| Cost item | Rate | Applied |
| --- | --- | --- |
| Taker fee | 0.1% per side | every entry **and** every exit |
| Slippage | 0.1% per side | every entry **and** every exit |
| Total round-trip | ~0.4% | per trade |

Buy-and-hold pays the same per-side rate but only once on entry and once on
exit — one trade total, no intermediate fees (no intermediate trades to charge).

**Iron rule:** position at T is set from signal[T] which uses only features
known at close[T].  The return that position earns is
`close[T+1] / close[T] − 1` — applied strictly after signal[T] is committed.

### Results — BTCUSDT 1d, test split (2025-08-17 → 2026-08-11, 359 days)

BTC fell **−46%** during this window.

| Metric | LR (pruned) | LGB (pruned) | Buy-and-hold |
| --- | --- | --- | --- |
| Total return | −48.8% | **−33.5%** | −46.1% |
| CAGR | −49.4% | −33.9% | −46.6% |
| Sharpe ratio | −2.10 | −1.68 | −1.08 |
| Max drawdown | −50.2% | −37.5% | −53.0% |
| # trades | 53 | 23 | 1 |
| Win rate | 34.0% | 47.8% | n/a |
| Avg trade P&L | −1.03% | −1.40% | −45.9% |

### Honest verdict

**LGB appears to outperform buy-and-hold (−33.5% vs −46.1%), but this is a
regime-specific illusion, not directional edge.**

LGB fired only 30 signals in 328 days — 91% cash.  Being mostly absent from a
market that fell 46% naturally produces better-looking numbers than buy-and-hold.
The same low-coverage model would massively underperform in a bull market, missing
most of the upside while paying fees on the few signals it does fire.

**LR underperforms buy-and-hold (−48.8% vs −46.1%).**  Its 49 long signals
landed in downtrend windows, and 53 round-trips consumed roughly 10 percentage
points in fees alone.

**No model should be traded with real money based on these results.**  Phase 4
found no directional edge over the base rate; Phase 5 confirms that real costs
(0.4% per round-trip) eliminate any marginal gain once friction is applied.  The
Sharpe ratio is negative for all three configurations, which is the expected
outcome for any system operating in a sustained downtrend with no demonstrated
edge.

See `notebooks/05_backtest_results.ipynb` for equity curves and the drawdown chart.

## On-chain features (Phase 6)

### What was available for free

**Source**: Blockchain.com Charts API (`https://api.blockchain.info/charts/{metric}?timespan=7years&sampled=false&format=json`).
No API key required.  Coverage: 2019-08-22 → present.

| Metric | API chart name | Coverage (at fetch time) |
| --- | --- | --- |
| Active addresses | `n-unique-addresses` | 2019-08-22 → 2026-08-16 |
| Transaction count | `n-transactions` | 2019-08-22 → 2026-08-16 |
| Hash rate (TH/s) | `hash-rate` | 2019-08-22 → 2026-08-16 |
| Fees (USD) | `transaction-fees-usd` | 2019-08-22 → 2026-08-16 |
| Volume (USD) | `estimated-transaction-volume-usd` | 2019-08-22 → 2026-08-19 |

**NOT available for free (stated honestly):**
- **Exchange netflows** — Glassnode / CryptoQuant paid tier
- **Whale movement / large-transaction tracking** — paid services
- **HODL waves / coin age distribution** — Glassnode paid
- **Realized price, NUPL, STH/LTH supply** — Glassnode paid

### Feature engineering

- **Reporting lag**: conservative 1-day shift.  On-chain data for day T is finalized at midnight UTC end of T.  We use T-1 data for candle T — so no decision ever depends on the same day's blockchain activity.
- **Rolling z-scores**: 7-day and 30-day normalised values for each metric.
- **Week-over-week % change**: `(value / value.shift(7) - 1) × 100` for each metric.
- **Total**: 5 metrics × 3 transformations = **15 on-chain features**, all prefixed `onchain_`.

### Running Phase 6

```bash
# Step 1: fetch raw on-chain data
python -m src.data.fetch_onchain

# Step 2: rebuild features (on-chain columns added automatically)
python -m src.features.build_features --intervals 1d

# Step 3: retrain all variants
python -m src.models.train --intervals 1d          # base (price-only)
python -m src.models.train --intervals 1d --pruned # pruned (price-only)
python -m src.models.train --intervals 1d --onchain # price + on-chain

# Step 4: evaluate
python -m src.models.evaluate --intervals 1d
python -m src.models.evaluate --intervals 1d --pruned
python -m src.models.evaluate --intervals 1d --onchain
```

### 3-way comparison (BTCUSDT 1d, horizon 1)

Same split as Phase 4/5: train 1,557 rows (2020-01-31 → 2024-08-16), val 329 rows, test 328 rows (2025-08-17 → 2026-08-10).

| Variant | Model | Val accuracy | Val edge | Test accuracy | Test edge |
| --- | --- | --- | --- | --- | --- |
| base | LR | 52.9% | −0.3pp | 45.4% | −6.7pp |
| base | LGB | 53.2% | +0.0pp | 47.9% | −4.3pp |
| pruned | LR | 53.5% | +0.3pp | 45.4% | −6.7pp |
| pruned | LGB | 53.2% | +0.0pp | 47.9% | −4.3pp |
| **onchain** | **LR** | **53.8%** | **+0.6pp** | **47.0%** | **−5.2pp** |
| **onchain** | **LGB** | **53.8%** | **+0.6pp** | **47.6%** | **−4.6pp** |

No leak alert fired (all well below the 65% tripwire).

### Honest verdict

On-chain features produce a **marginal but not material** improvement.  At daily resolution:

- Validation: the on-chain variant gains +0.3–0.6pp on both models vs pruned price-only.  This is within noise.
- Test: on-chain LR is +1.6pp over pruned LR, LGB is +0.3pp.  No consistent win.
- LightGBM early-stopping selected 13 trees for the on-chain model vs 1 tree for base/pruned — it found slightly more learnable signal, but the resulting accuracy gain is not robust across val/test.

**Why**: at daily granularity, the free on-chain metrics (active addresses, hash rate, fees, volume) trend together with price — their z-scores and WoW changes carry correlated directional information to the price features already present.  The on-chain metrics historically most useful for regime detection (exchange netflows, realized-price deviation, STH/LTH ratio) require paid services that were not used.

**Conclusion**: free on-chain data is neither harmful nor transformative here.  The result is neutral, which is the honest finding.  Including the features does not hurt the pipeline and is kept in the feature table for any future experiment that explores non-daily resolution or longer horizons.

See `notebooks/06_onchain_analysis.ipynb` for metric time-series plots, correlation with the label, LightGBM feature importances by category (on-chain vs price), and the full 3-way comparison table.

## Multi-regime backtesting (Phase 7)

### Methodology

Five historical BTC windows were committed **before running any backtest** to prevent
post-hoc cherry-picking.  Windows were chosen from publicly documented BTC market
history.  The in-sample flag is set mechanically — any window overlapping the
training period `[2020-01-01, 2024-08-16]` is labelled IN-SAMPLE.

**Critical distinction:**
- **IN-SAMPLE (IS):** the pruned model was trained on this data. Results are
  diagnostic only — good IS performance is expected from a model that memorised the
  training distribution, not evidence of predictive skill.
- **OUT-OF-SAMPLE (OOS):** model never saw this data during training or early
  stopping. This is the only window where performance carries meaning.

### Regime comparison table (BTCUSDT 1d, pruned model)

| Regime | IS/OOS | Days | LR return | LGB return | B&H return | LR Sharpe | LR trades |
| --- | --- | --- | --- | --- | --- | --- | --- |
| COVID crash (Feb-Apr 2020) | **IS** | 90 | **+116.3%** | 0.0% | −6.3% | 3.10 | 28 |
| 2020-2021 bull run | **IS** | 212 | +6.5% | 0.0% | **+442.1%** | 0.39 | 37 |
| 2022 bear market | **IS** | 365 | +14.0% | 0.0% | −65.3% | 0.48 | 85 |
| 2023 recovery | **IS** | 365 | +17.8% | 0.0% | +164.8% | 0.74 | 76 |
| **Phase 5 test window** | **OOS** | 360 | **−46.8%** | **0.0%** | **−46.0%** | **−1.80** | **70** |

LGB fires **zero signals in every window** (IS and OOS alike) at the 0.60 confidence
threshold.  This is not a regime-dependent outcome — the model simply never exceeds
the threshold in either direction regardless of market conditions.

### Honest verdict

The only result that counts is the out-of-sample window.  LR returned **−46.8%** vs
buy-and-hold **−46.0%** — essentially the same outcome, with 70 round-trips generating
roughly 14 percentage points of fee drag that the model had to overcome just to match
buy-and-hold.  It didn't.

The in-sample results are instructive about what this model has memorised:

- **Bear regimes (COVID crash, 2022):** LR substantially outperforms B&H in both
  windows.  This reflects a model that learned to go short or go cash when its
  in-sample training data showed prices falling.  It is what overfitting to bear
  conditions looks like.
- **Bull regimes (2020-21 bull, 2023 recovery):** LR severely underperforms B&H
  (+6.5% vs +442%, +18% vs +165%).  The same conservatism that avoids bear drawdowns
  causes it to miss sustained uptrends almost entirely.
- **The OOS window (the falling BTC of 2025-26):** even in a bear regime that
  superficially resembles the 2022 period the model was trained on, LR fails to
  replicate the IS bear-market result.  It takes nearly as much damage as buy-and-hold
  (−46.8% vs −46.0%) while paying fees on 70 trades.

**Conclusion:** phase 7 confirms phase 5's finding by adding historical context.  The
model has no demonstrated directional edge.  IS bear-market results are a function of
memorisation, not predictive ability — the OOS result is the honest signal, and it is
flat-to-negative vs buy-and-hold after fees.  No model variant should be used for
live trading.

### Running Phase 7

```bash
python -m src.backtest.multi_regime
python -m src.backtest.multi_regime --log-level DEBUG
```

See `notebooks/07_multi_regime.ipynb` for equity curves (small multiples, IS regimes
visually distinguished from OOS) and the formatted comparison table.

## Tests and lint

```bash
pytest          # unit tests — never touch the real API
ruff check .    # lint
```

Both run in CI on every push and pull request.
