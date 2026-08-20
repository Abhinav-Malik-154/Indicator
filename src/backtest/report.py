"""Load pruned-model signals and produce the Phase 5 backtest comparison report.

Evaluation order (iron rule: no future data used to size or time any trade):

1. Re-assemble the dataset through the training code path (same join + alignment
   guard as Phase 4).
2. Derive the test split using the SAME boundaries as training — never re-fit
   them here.
3. Score both pruned models (logistic regression + LightGBM) on the TEST split
   only.  Val was used for early-stopping (model selection) so it is not a
   clean out-of-sample proxy.
4. Convert probabilities to signals with the same confidence gate as Phase 4.
5. Run ``simulate_strategy`` for each model and ``simulate_buyhold`` for the
   baseline over the test window.

Run from the repo root::

    python -m src.backtest.report
    python -m src.backtest.report --intervals 1d --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.backtest.baseline import simulate_buyhold
from src.backtest.simulate import BacktestResult, signals_from_proba, simulate_strategy
from src.models.evaluate import load_artifacts
from src.models.train import (
    SplitBounds,
    assemble_dataset,
    load_modeling_config,
    split_dataset,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_backtest_config(path: str | Path) -> dict[str, Any]:
    """Read ``backtest`` section from the pipeline config.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dict with ``fee_rate``, ``slippage_rate``, ``starting_notional``.

    Raises:
        ValueError: If required keys are missing or out of range.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    bt = raw.get("backtest") or {}
    cfg: dict[str, Any] = {
        "fee_rate": float(bt.get("fee_rate", 0.001)),
        "slippage_rate": float(bt.get("slippage_rate", 0.001)),
        "starting_notional": float(bt.get("starting_notional", 10_000.0)),
    }
    if not 0.0 <= cfg["fee_rate"] < 0.1:
        raise ValueError(f"backtest.fee_rate must be in [0, 0.1), got {cfg['fee_rate']}")
    if not 0.0 <= cfg["slippage_rate"] < 0.1:
        raise ValueError(
            f"backtest.slippage_rate must be in [0, 0.1), got {cfg['slippage_rate']}"
        )
    if cfg["starting_notional"] <= 0:
        raise ValueError(
            f"backtest.starting_notional must be > 0, got {cfg['starting_notional']}"
        )
    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_test_prices(interval: str, cfg: dict[str, Any], bounds: SplitBounds) -> pd.Series:
    """Return close prices for the test window plus one extra candle.

    The extra candle is needed to compute the return from the LAST signal day
    to the following day.  Its close price is read from the raw parquet — it
    is never used to generate a signal, only to evaluate the return that
    follows the final signal.

    Args:
        interval: Binance interval string, e.g. ``"1d"``.
        cfg: Config dict from :func:`~src.models.train.load_modeling_config`.
        bounds: Split boundaries from training manifest.

    Returns:
        Series of close prices with DatetimeIndex, length >= 2.
    """
    raw_path = Path(cfg["raw_dir"]) / f"{cfg['file_prefix']}_{interval}.parquet"
    raw = pd.read_parquet(raw_path, columns=["open_time", "close"])
    raw = raw.sort_values("open_time").reset_index(drop=True)

    # Test split positional range in the raw parquet.
    # The raw parquet contains every candle; the merged table and the raw table
    # share the same open_time values (features are built from the raw candles).
    # We read the raw rows at the test positions plus one beyond test_end.
    test_raw = raw.iloc[bounds.test_start : bounds.test_end + 1].copy()

    if len(test_raw) < 2:
        raise ValueError(
            f"{interval}: raw price window has fewer than 2 rows "
            f"(test_start={bounds.test_start}, test_end={bounds.test_end})"
        )

    test_raw = test_raw.set_index("open_time")["close"]
    logger.info(
        "%s: test price window %s -> %s (%d candles including +1 for final return)",
        interval,
        test_raw.index[0],
        test_raw.index[-1],
        len(test_raw),
    )
    return test_raw


def build_test_signals(
    interval: str,
    cfg: dict[str, Any],
    *,
    model_variant: str = "pruned",
) -> dict[str, Any]:
    """Compute signals for both models on the test split only.

    Returns a dict with keys:
    ``"feature_cols"``, ``"test_split"``, ``"bounds"``,
    ``"prob_lr"`` (np.ndarray), ``"prob_lgb"`` (np.ndarray),
    ``"threshold"`` (float).
    """
    artifacts = load_artifacts(interval, cfg, model_variant=model_variant)
    manifest = artifacts["manifest"]

    include_onchain = (model_variant == "onchain")
    merged, all_feature_cols = assemble_dataset(
        interval, cfg, include_onchain=include_onchain
    )
    feature_cols: list[str] = manifest["feature_cols"]
    missing = [c for c in feature_cols if c not in all_feature_cols]
    if missing:
        raise ValueError(f"{interval}: manifest feature cols missing: {missing}")

    bounds_d = manifest["split_bounds"]
    bounds = SplitBounds(**bounds_d)
    label_col: str = manifest["label_col"]

    splits = split_dataset(
        merged, bounds, feature_cols=feature_cols, label_col=label_col,
        context=f"{cfg['symbol']} {interval} ({model_variant})",
    )
    test = splits["test"]

    scaler = artifacts["scaler"]
    logreg = artifacts["logistic_regression"]
    lgbm = artifacts["lightgbm"]

    x_test = test[feature_cols]
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test), columns=feature_cols, index=x_test.index
    )
    prob_lr = logreg.predict_proba(x_test_scaled)[:, 1]
    prob_lgb = lgbm.predict_proba(x_test)[:, 1]

    threshold: float = cfg["modeling"]["confidence_threshold"]
    return {
        "bounds": bounds,
        "test_split": test,
        "feature_cols": feature_cols,
        "prob_lr": prob_lr,
        "prob_lgb": prob_lgb,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------


def _align_prices_to_signals(
    prices: pd.Series, test_split: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Match raw price series to the test split's open_time dates.

    The test split may have NaN-dropped rows that are absent from the usable
    set but present in the raw price series.  We align prices to the usable
    test rows (keeping the extra candle for returns) so that ``prices[i]``
    corresponds to ``signals[i]`` and ``prices[i+1]`` to the return from that
    signal day.

    Returns:
        Tuple of ``(aligned_prices, signal_dates_index)``.
        ``aligned_prices`` has ``len(test_split) + 1`` entries if the next
        candle is available, otherwise ``len(test_split)`` (last signal skipped).
    """
    signal_dates = test_split["open_time"]
    # Find each signal date in the price series
    price_idx = prices.index
    matched: list[Any] = []
    missing: list[Any] = []
    for dt in signal_dates:
        if dt in price_idx:
            matched.append(dt)
        else:
            missing.append(dt)
    if missing:
        logger.warning(
            "align: %d test-split dates not found in raw prices; "
            "first missing: %s",
            len(missing), missing[0],
        )

    matched_arr = pd.DatetimeIndex(matched)
    # Find the date right after the last matched date for the final return
    last_signal_loc = price_idx.get_loc(matched_arr[-1])
    if last_signal_loc + 1 < len(price_idx):
        extra_date = price_idx[last_signal_loc + 1]
        all_dates = matched_arr.append(pd.DatetimeIndex([extra_date]))
    else:
        logger.warning("align: no candle after last signal date; last return excluded")
        all_dates = matched_arr[:-1]  # drop last signal to keep contract

    aligned_prices = prices.loc[all_dates]
    n_signals = len(aligned_prices) - 1
    return aligned_prices, test_split.head(n_signals)["open_time"]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: float, fmt: str = ".2f") -> str:
    nan = isinstance(value, float) and np.isnan(value)
    return "n/a" if (value is None or nan) else f"{value:{fmt}}"


def comparison_table(results: list[BacktestResult]) -> str:
    """Render a side-by-side comparison table for multiple backtest results."""
    metrics = [
        ("Total return", "total_return_pct", "%"),
        ("CAGR", "cagr_pct", "%"),
        ("Sharpe ratio", "sharpe_ratio", ""),
        ("Max drawdown", "max_drawdown_pct", "%"),
        ("# trades", "n_trades", ""),
        ("Win rate", "win_rate_pct", "%"),
        ("Avg trade P&L", "avg_trade_pct", "%"),
        ("Days simulated", "n_days", ""),
    ]

    col_w = 20
    header = f"{'Metric':<25}" + "".join(
        f"{(r.label or f'result_{i}')[:col_w]:>{col_w}}" for i, r in enumerate(results)
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for display_name, key, unit in metrics:
        row = f"{display_name + unit:<25}"
        for r in results:
            val = getattr(r, key)
            if isinstance(val, int):
                row += f"{val:>{col_w}d}"
            else:
                row += f"{_fmt(val) + unit:>{col_w}}"
        rows.append(row)
    return "\n".join(rows)


def run_backtest(
    interval: str,
    cfg: dict[str, Any],
    bt_cfg: dict[str, Any],
    *,
    model_variant: str = "pruned",
) -> dict[str, BacktestResult]:
    """Run strategy + buy-and-hold for one interval, return all results."""
    context = f"{cfg['symbol']} {interval}"
    signals_data = build_test_signals(interval, cfg, model_variant=model_variant)

    test_split: pd.DataFrame = signals_data["test_split"]
    bounds: SplitBounds = signals_data["bounds"]
    threshold: float = signals_data["threshold"]
    prob_lr: np.ndarray = signals_data["prob_lr"]
    prob_lgb: np.ndarray = signals_data["prob_lgb"]

    sig_lr = signals_from_proba(prob_lr, threshold=threshold)
    sig_lgb = signals_from_proba(prob_lgb, threshold=threshold)

    # Get raw prices aligned to usable test rows + 1 extra candle
    prices_raw = load_test_prices(interval, cfg, bounds)
    aligned_prices, signal_index = _align_prices_to_signals(prices_raw, test_split)

    n = len(signal_index)
    lr_signals_series = pd.Series(sig_lr[:n], index=signal_index)
    lgb_signals_series = pd.Series(sig_lgb[:n], index=signal_index)

    fee = bt_cfg["fee_rate"]
    slip = bt_cfg["slippage_rate"]
    notional = bt_cfg["starting_notional"]

    logger.info(
        "%s: test window %s -> %s (%d signal days), "
        "fee=%.3f%% slip=%.3f%%, notional=%.0f",
        context,
        signal_index.iloc[0], signal_index.iloc[-1], n,
        fee * 100, slip * 100, notional,
    )
    logger.info(
        "%s: LR signals — %d fired (%d long, %d short) of %d",
        context,
        int((sig_lr[:n] != 0).sum()),
        int((sig_lr[:n] == 1).sum()),
        int((sig_lr[:n] == -1).sum()), n,
    )
    logger.info(
        "%s: LGB signals — %d fired (%d long, %d short) of %d",
        context,
        int((sig_lgb[:n] != 0).sum()),
        int((sig_lgb[:n] == 1).sum()),
        int((sig_lgb[:n] == -1).sum()), n,
    )

    result_lr = simulate_strategy(
        aligned_prices, lr_signals_series,
        fee_rate=fee, slippage_rate=slip, starting_notional=notional,
        label=f"LR ({model_variant})",
    )
    result_lgb = simulate_strategy(
        aligned_prices, lgb_signals_series,
        fee_rate=fee, slippage_rate=slip, starting_notional=notional,
        label=f"LGB ({model_variant})",
    )
    result_bah = simulate_buyhold(
        aligned_prices, fee_rate=fee, slippage_rate=slip, starting_notional=notional,
    )
    result_bah.label = "buy-and-hold BTC"

    return {"lr": result_lr, "lgb": result_lgb, "buyhold": result_bah}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run Phase 5 backtests and print the comparison table.

    Returns:
        0 on success, 1 if any interval failed.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.backtest.report",
        description="Phase 5: backtest the pruned model signals vs buy-and-hold.",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--intervals", nargs="+", metavar="INTERVAL")
    parser.add_argument(
        "--model-variant", default="pruned",
        help="which model variant to backtest (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    cfg = load_modeling_config(args.config)
    bt_cfg = load_backtest_config(args.config)
    intervals: list[str] = args.intervals or cfg["modeling"]["intervals"]

    failures: list[str] = []
    for interval in intervals:
        try:
            results = run_backtest(
                interval, cfg, bt_cfg, model_variant=args.model_variant
            )
            ordered = [results["lr"], results["lgb"], results["buyhold"]]
            table = comparison_table(ordered)
            print(
                f"\n===== {cfg['symbol']} {interval} — Phase 5 backtest "
                f"(test split only, model_variant={args.model_variant}) =====\n"
            )
            print(table)
            print()

            # Honest verdict
            strat_best = max(results["lr"].total_return_pct, results["lgb"].total_return_pct)
            bah = results["buyhold"].total_return_pct
            if strat_best > bah:
                print(
                    "VERDICT: the strategy outperforms buy-and-hold on this "
                    "test window — but the test window covers only one regime "
                    "and one horizon; do not over-read a single period."
                )
            else:
                print(
                    "VERDICT: buy-and-hold outperforms (or matches) both model "
                    "strategies on the test window, net of fees and slippage. "
                    "Given the absence of directional edge in Phase 4, this is "
                    "the expected result — real costs eliminate any marginal gain."
                )
        except Exception:
            logger.exception("%s %s: backtest failed", cfg["symbol"], interval)
            failures.append(interval)

    if failures:
        logger.error("backtest failed for: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
