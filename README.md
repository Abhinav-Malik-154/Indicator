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
public REST API. **Phase 2 (current):** leakage-safe feature engineering.
Later phases will design labels with equal care, compare predictability across
timeframes (1d vs 1h vs 30m), and evaluate models honestly.

## Repository layout

```
configs/config.yaml               # symbol, intervals, date range, API, windows, paths
src/data/fetch_binance.py         # fetch -> validate -> save pipeline (CLI entry point)
src/features/candlestick.py       # TA-Lib CDL* pattern features + fire-rate reporting
src/features/technical.py         # returns / volatility / volume / context / intra-candle
src/features/build_features.py    # feature build CLI: gap accounting + manifests
tests/                            # unit tests incl. the no-look-ahead leakage test
notebooks/01_feature_sanity.ipynb # visual sanity checks only — no logic in notebooks
data/raw/                         # candles: parquet + CSV + manifest (git-ignored)
data/processed/                   # feature tables + manifests (git-ignored)
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
| candlestick | all TA-Lib `CDL*` patterns that fire in the data (one int column each) |

Candlestick values live in `{-200, -100, -80, 0, 80, 100, 200}`: Hikkake
confirmations are graded ±200 and engulfing/harami near-misses ±80 by the
modern TA-Lib core (verified empirically; out-of-domain values fail the build).

Gap policy: rolling windows are positional (over available candles), never
wall-clock. Gap records from the Phase 1 manifests are read back during the
build, and every feature row whose longest lookback window spans a recorded
gap is counted and written to the build manifest — candles are never filled,
interpolated, or synthesised. Rows without full rolling history keep NaN.

## Tests and lint

```bash
pytest          # unit tests — never touch the real API
ruff check .    # lint
```

Both run in CI on every push and pull request.
