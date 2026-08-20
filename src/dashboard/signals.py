"""Live signal computation for the BTC indicator dashboard.

Fetches the latest complete daily candles from Binance, builds features
using the same pipeline the model was trained on, and returns the current
signal from the pruned model.

Design choices:
- Fetches 90 candles to provide enough warm-up history for the 30-day rolling
  windows (the longest lookback in the feature set).
- Drops any candle whose close_time is in the future (still forming).
- Fills missing candlestick pattern columns with 0 (a pattern that never
  fired in the live window is genuinely absent, not unknown).
- Never modifies disk state: this module is read-only with respect to
  data/processed/ and models/.

Historical accuracy numbers are fixed constants from the Phase 4/5 OOS
evaluation — they describe the model's measured performance on data it
never saw during training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.features.candlestick import compute_candlestick_features
from src.features.technical import compute_technical_features
from src.models.evaluate import load_artifacts
from src.models.train import load_modeling_config

logger = logging.getLogger(__name__)

# ── Binance klines column layout ───────────────────────────────────────────
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

# ── Volatility regime thresholds (ret_std_30 from training split, p25/p75) ─
# Computed from the first 1690 rows of features_1d.parquet (the training split).
# These are fixed at training time — using them at inference is correct.
_VOL_CALM_THRESHOLD     = 0.023273   # p25 of training ret_std_30
_VOL_ELEVATED_THRESHOLD = 0.037508   # p75 of training ret_std_30

# ── Historical accuracy (Phase 4/5 OOS test split: 2025-08-16 → 2026-08-11) ─
# Source: python -m src.models.evaluate --pruned / src/backtest/report.py
HISTORICAL_ACCURACY: dict[str, Any] = {
    "lr": {
        "test_accuracy_pct": 45.1,
        "base_rate_pct":     52.1,
        "edge_pp":           -7.0,
        "test_return_pct":   -46.8,
        "n_trades":          70,
        "win_rate_pct":      38.6,
    },
    "lgb": {
        "test_accuracy_pct": 51.8,
        "base_rate_pct":     52.1,
        "edge_pp":           -0.3,
        "test_return_pct":   0.0,   # 0 signals fired → flat
        "n_trades":          0,
        "win_rate_pct":      float("nan"),
    },
    "buyhold_test_return_pct": -46.0,
    "test_period": "2025-08-16 → 2026-08-11 (360 days)",
    "confidence_threshold": 0.60,
    "round_trip_cost_pct": 0.40,    # fee + slippage, both sides
}


# ── Binance fetch ─────────────────────────────────────────────────────────

def fetch_live_candles(
    symbol: str,
    interval: str,
    n_candles: int = 90,
    *,
    base_url: str = "https://api.binance.com",
    timeout_s: float = 15.0,
) -> pd.DataFrame:
    """Fetch the most recent complete candles from the Binance public API.

    Requests ``n_candles + 1`` to handle the case where the most recent
    candle is still forming.  Any candle whose close_time lies in the future
    is dropped before returning.

    Args:
        symbol: Trading pair, e.g. ``"BTCUSDT"``.
        interval: Binance interval string, e.g. ``"1d"``.
        n_candles: Number of complete candles to return.
        base_url: Override for testing.
        timeout_s: HTTP request timeout.

    Returns:
        DataFrame with columns ``open_time`` (UTC DatetimeIndex),
        ``open``, ``high``, ``low``, ``close``, ``volume`` as float64.
        Always has at least ``n_candles`` rows unless the API returns fewer.

    Raises:
        requests.RequestException: On network failure.
        ValueError: If the response is empty or malformed.
    """
    resp = requests.get(
        f"{base_url}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": n_candles + 1},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        raise ValueError(f"Binance returned empty klines for {symbol} {interval}")

    df = pd.DataFrame(raw, columns=_KLINE_COLS)
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    df = df[df["close_time"].astype(int) <= now_ms].copy()

    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df = df[["open_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    logger.info(
        "Binance: fetched %d complete %s %s candles (%s → %s)",
        len(df), symbol, interval,
        df["open_time"].iloc[0].date(), df["open_time"].iloc[-1].date(),
    )
    return df


# ── Feature building ──────────────────────────────────────────────────────

def build_live_features(
    candles: pd.DataFrame,
    feature_cfg: dict[str, Any],
    required_cols: list[str],
) -> pd.DataFrame:
    """Build technical + candlestick features for a live candle DataFrame.

    Builds the same feature families as the training pipeline, then aligns
    the output to exactly ``required_cols`` (fills any missing candlestick
    columns with 0).

    Args:
        candles: DataFrame with ``open_time``, ``open``, ``high``, ``low``,
            ``close``, ``volume``, sorted ascending.
        feature_cfg: The ``features`` sub-dict from the pipeline config.
        required_cols: Exact ordered list of columns the model expects.

    Returns:
        DataFrame with one row per input candle and columns = ``required_cols``.
    """
    feat = feature_cfg
    tech = compute_technical_features(
        candles,
        return_periods=feat["return_periods"],
        volatility_windows=feat["volatility_windows"],
        volume_window=feat["volume_window"],
        ma_windows=feat["ma_windows"],
        sr_window=feat["sr_window"],
        rsi_period=feat["rsi_period"],
        macd=feat.get("macd"),
    )
    cdl, _ = compute_candlestick_features(candles, drop_never_fired=False)
    # Both tech and cdl use a plain RangeIndex (0..n-1); concat aligns correctly.
    combined = pd.concat([tech, cdl], axis=1)

    # Align to required_cols: fill missing cdl columns (never fired → 0)
    for col in required_cols:
        if col not in combined.columns:
            combined[col] = 0
    return combined[required_cols]


# ── Signal derivation ─────────────────────────────────────────────────────

def _signal_label(prob_up: float, threshold: float) -> str:
    if prob_up > threshold:
        return "BUY"
    if prob_up < 1.0 - threshold:
        return "SELL"
    return "SILENT"


def get_volatility_regime(ret_std_30: float) -> dict[str, str]:
    """Classify the current volatility regime from the 30-day return std.

    Thresholds are the p25/p75 of ret_std_30 in the training split —
    fixed at training time, never updated from live data.

    Returns:
        Dict with ``regime`` ("calm" / "elevated" / "high"), ``label``,
        and ``explanation``.
    """
    if ret_std_30 < _VOL_CALM_THRESHOLD:
        return {
            "regime": "calm",
            "label": "Calm",
            "colour": "green",
            "explanation": (
                f"30-day volatility ({ret_std_30:.4f}) is below the historical 25th "
                f"percentile ({_VOL_CALM_THRESHOLD:.4f}). "
                "Lower volatility means smaller expected moves in either direction."
            ),
        }
    if ret_std_30 <= _VOL_ELEVATED_THRESHOLD:
        return {
            "regime": "elevated",
            "label": "Elevated",
            "colour": "orange",
            "explanation": (
                f"30-day volatility ({ret_std_30:.4f}) is between historical p25 "
                f"({_VOL_CALM_THRESHOLD:.4f}) and p75 ({_VOL_ELEVATED_THRESHOLD:.4f}). "
                "Normal range — neither unusually calm nor stressed."
            ),
        }
    return {
        "regime": "high",
        "label": "High",
        "colour": "red",
        "explanation": (
            f"30-day volatility ({ret_std_30:.4f}) exceeds the historical 75th "
            f"percentile ({_VOL_ELEVATED_THRESHOLD:.4f}). "
            "High volatility expands fee drag as a fraction of expected move."
        ),
    }


def compute_live_signal(
    interval: str = "1d",
    config_path: str = "configs/config.yaml",
    model_variant: str = "pruned",
) -> dict[str, Any]:
    """Fetch live candles, build features, return the current signal.

    This is the main entry point for the dashboard.

    Args:
        interval: Binance interval (e.g. ``"1d"``).
        config_path: Path to ``configs/config.yaml``.
        model_variant: Which trained model to use.

    Returns:
        Dict with keys:
            ``signal_lr``, ``signal_lgb``: "BUY" / "SELL" / "SILENT"
            ``prob_lr``, ``prob_lgb``: float P(up)
            ``candle_date``: pd.Timestamp of the last complete candle
            ``current_close``: most recent close price
            ``ret_std_30``: 30-day rolling volatility
            ``vol_regime``: volatility regime dict from :func:`get_volatility_regime`
            ``data_source``: "live" or "cached" with explanation
            ``n_candles_fetched``: how many candles were used
            ``error``: None or an error message if live fetch failed

    Raises:
        FileNotFoundError: If model artifacts or config are missing.
    """
    cfg = load_modeling_config(config_path)
    artifacts = load_artifacts(interval, cfg, model_variant=model_variant)
    manifest = artifacts["manifest"]
    feature_cols: list[str] = manifest["feature_cols"]
    threshold: float = cfg["modeling"]["confidence_threshold"]
    symbol: str = cfg["symbol"]
    feat_cfg: dict[str, Any] = {}

    # Load feature config for building live features
    import yaml
    raw_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    from src.features.build_features import _FEATURE_DEFAULTS
    feat_cfg = {**_FEATURE_DEFAULTS, **(raw_cfg.get("features") or {})}

    error: str | None = None
    data_source = "live"

    try:
        candles = fetch_live_candles(symbol, interval, n_candles=90)
        live_features = build_live_features(candles, feat_cfg, feature_cols)
        last_row = live_features.iloc[[-1]].dropna(subset=feature_cols)
        if len(last_row) == 0:
            raise ValueError("Last live candle has NaN features (warm-up period)")
        last_candle = candles.iloc[-1]
        n_used = len(candles)
        data_source = "live"
    except Exception as exc:
        logger.warning("Live fetch failed (%s); falling back to cached features", exc)
        error = str(exc)
        data_source = "cached (live fetch failed)"
        # Fall back to the processed features parquet
        feat_path = Path(cfg["processed_dir"]) / f"features_{interval}.parquet"
        cached = pd.read_parquet(feat_path)
        last_row = cached[feature_cols].iloc[[-1]]
        raw_path = Path(cfg["raw_dir"]) / f"{cfg['file_prefix']}_{interval}.parquet"
        raw = pd.read_parquet(raw_path, columns=["open_time", "close"])
        last_candle = raw.iloc[-1]
        n_used = len(cached)

    X = last_row[feature_cols]
    X_scaled = pd.DataFrame(
        artifacts["scaler"].transform(X),
        columns=feature_cols,
        index=X.index,
    )
    prob_lr  = float(artifacts["logistic_regression"].predict_proba(X_scaled)[:, 1][0])
    prob_lgb = float(artifacts["lightgbm"].predict_proba(X)[:, 1][0])

    ret_std_30 = float(last_row["ret_std_30"].iloc[0])
    vol_regime = get_volatility_regime(ret_std_30)

    candle_date = pd.Timestamp(last_candle["open_time"])
    current_close = float(last_candle["close"])

    return {
        "signal_lr":        _signal_label(prob_lr, threshold),
        "signal_lgb":       _signal_label(prob_lgb, threshold),
        "prob_lr":          prob_lr,
        "prob_lgb":         prob_lgb,
        "threshold":        threshold,
        "candle_date":      candle_date,
        "current_close":    current_close,
        "ret_std_30":       ret_std_30,
        "vol_regime":       vol_regime,
        "data_source":      data_source,
        "n_candles_fetched": n_used,
        "error":            error,
        "symbol":           symbol,
        "interval":         interval,
        "model_variant":    model_variant,
    }
