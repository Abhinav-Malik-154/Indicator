# Accuracy experiments (Tasks 1–3)

After Phases 1–9 established an honest pipeline whose headline finding was
**"no demonstrated directional edge"**, this follow-up asks the obvious next
question — *what could actually make it more accurate?* — and answers it with
three experiments, each built to be measured honestly rather than to flatter the
model. Every accuracy below is an **out-of-sample, purged walk-forward** number
reported with a **Wilson 95% confidence interval** and an explicit significance
test against the base rate.

The one-line result: **two of three directions changed nothing; reframing the
target from price-direction to volatility-direction produced a large, robust,
statistically-significant edge.**

---

## Task 1 — Measure it properly

A single train/val/test split gives one accuracy from one market regime. That is
high-variance and easy to over-read. Task 1 replaces it with regime-averaged,
leakage-safe measurement.

| Module | What it does |
| --- | --- |
| `src/models/stats.py` | Wilson score CIs + one-proportion significance vs a base rate |
| `src/models/cross_validate.py` | Purged & embargoed walk-forward CV; pools predictions across folds |
| `src/models/calibrate.py` | Isotonic probability calibration (fit on val), Brier score, gated accuracy |

* **Purge + embargo:** a gap of `horizon + embargo` rows sits between every
  train block and its test block, so a training label that peeks `horizon`
  candles ahead can never overlap the test window.
* **Full feature set only:** CV uses the base features (no candlestick pruning)
  so no feature-selection decision leaks across folds.

**Result (1d, 8 folds, ~1100 pooled OOS predictions, base rate 50.8%):**

```
logistic_regression  49.4% [46.4%, 52.3%]  edge -1.4pp  → indistinguishable from base rate
lightgbm             51.5% [48.6%, 54.5%]  edge +0.7pp  → indistinguishable from base rate
```

The base rate sits inside both CIs. The scary single-window 45.1% from Phase 5
was mostly regime noise; the regime-averaged truth is simply **"no edge, either
way."** Calibration lowers the Brier score and makes the 0.60 confidence gate
fire more honestly, but it cannot manufacture edge (it is a monotonic re-map).

Reproduce:

```bash
python -m src.models.cross_validate --intervals 1d --folds 8 --embargo 1
python -m src.models.calibrate --pruned --intervals 1d
```

---

## Task 2 — Add predictive features

Hypothesis: spot candles miss information that lives in the derivatives market
and the macro tape. Task 2 adds **perp funding rate** (Binance USD-M futures,
free history back to 2020) and **cross-asset returns** (DXY, S&P 500, gold via
yfinance), all 1-day-lagged and leakage-safe.

| Module | What it does |
| --- | --- |
| `src/data/fetch_derivatives.py` | Fetches funding (paginated) + cross-asset closes; both network calls injectable |
| `src/features/derivatives.py` | 11 `deriv_*` features: funding level / z-scores / sign / 7-day cum, cross-asset ret1 + z-score |

The base-vs-derivatives comparison runs both feature sets on the **identical**
usable rows (the `deriv_*` columns always constrain the NaN-drop) so any
difference is attributable to the features, not the sample.

**Result (1d, identical 1092 OOS predictions, base rate 51.0%):**

```
                 base                         +derivatives                 Δ
logistic_regression  49.9% [46.9,52.9]  →  50.5% [47.6,53.5]   +0.6pp
lightgbm             52.5% [49.5,55.4]  →  50.3% [47.3,53.2]   -2.2pp
```

Both CIs still straddle the base rate with and without the features. Funding and
cross-asset context **did not create a directional edge** on daily BTC — LR
barely moves, LGB gets slightly worse.

### Honest data limitation

Binance's **open-interest** and **perp-spot basis** history endpoints only serve
the **last ~30 days**, so they cannot back a multi-year feature for free and are
deliberately excluded (see `not_available_for_free` in the derivatives
manifest). Funding rate is the one derivatives signal with a full free history.

Reproduce:

```bash
python -m src.data.fetch_derivatives --interval 1d
python -m src.models.cross_validate --intervals 1d --derivatives
```

---

## Task 3 — Reframe the target

If price direction is a martingale, stop predicting it. Task 3 swaps the
**question** and re-measures with the same walk-forward machinery.

| Target (`src/labels/alt_targets.py`) | Idea |
| --- | --- |
| `voldir` | Will realized volatility **expand or contract** next window? (vol clusters → autocorrelated) |
| `triple` | Triple-barrier: which of vol-scaled TP / SL / time-limit is hit first? |
| `meta` | Meta-labelling: a secondary model gates a primary direction call (act / don't-act) |

**Results (1d, purged walk-forward, ~1100–1190 pooled OOS predictions):**

```
Volatility-direction:
  logistic_regression  69.3% [66.6%, 71.9%]  edge +18.6pp  → SIGNIFICANTLY beats base rate
  lightgbm             65.4% [62.7%, 68.1%]  edge +14.7pp  → SIGNIFICANTLY beats base rate

Triple-barrier:
  logistic_regression  49.2% [46.3%, 52.0%]  → indistinguishable from base rate
  lightgbm             50.8% [47.9%, 53.6%]  → indistinguishable from base rate

Meta-labelling (primary=LGB direction, act@0.50):
  primary on ALL 1108 rows: 50.5% [47.5%, 53.4%]
  meta ACTED on 589 rows (coverage 53.2%): 50.3% [46.2%, 54.3%]  → no lift
```

**Volatility-direction is genuinely predictable.** The edge is large (+18.6pp),
tight, and — critically — **robust to the purge width**: widening the embargo to
`gap=17` leaves it at 69.5%, so it is not a boundary-overlap artifact. This is
exactly the expected consequence of volatility clustering: the *level* of
volatility is strongly autocorrelated and mean-reverting, so its *direction* is
forecastable even when price direction is not. The same base features that score
~50% on price direction score ~69% here — the change is entirely in the target,
which rules out a new feature-side leak.

Triple-barrier and meta-labelling honestly **did not help**: path-dependent
labels stayed at the base rate, and a meta-model cannot rescue an already-random
primary side (it gates ~half the rows to the same ~50%).

Reproduce:

```bash
python -m src.models.reframe --intervals 1d --target all
python -m src.models.reframe --intervals 1d --target voldir --embargo 10   # robustness
```

---

## Bottom line

| Direction | Outcome |
| --- | --- |
| Measure it properly (Task 1) | Confirmed **no price-direction edge** — regime-averaged, with CIs |
| Add derivatives/macro features (Task 2) | **No edge gained** — funding + cross-asset don't move the needle |
| Reframe the target (Task 3) | **Volatility-direction is ~69% accurate, significant and robust** |

The accuracy was never going to come from a better price-direction model on this
data — it came from **predicting a different, more structured quantity.** That is
the honest, reproducible lesson of these three experiments.

> Scope note: the volatility-direction model is a *research result*, not wired
> into the live dashboard signal (which remains the honestly-reported
> price-direction indicator). Productionizing a vol-regime signal would be its
> own phase.
