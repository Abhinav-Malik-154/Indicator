"""Multi-regime backtesting for Phase 7 (BTCUSDT 1d, pruned model).

Regime windows are defined HERE, before any backtest result is computed, to
prevent post-hoc cherry-picking.  Each window represents a well-documented
historical BTC market regime chosen from public knowledge.

Critical label:
  IN-SAMPLE  — the model was trained on data overlapping this window.
               Results show how the model behaves on memorised data.
               Good performance here is NOT evidence of skill.
  OUT-OF-SAMPLE — the model never saw this data during training or early stopping.
               This is the only meaningful performance signal.

Training period (pruned model, BTCUSDT 1d):
  train: 2020-01-01 → 2024-08-16  (rows [0, 1690), used for fitting)
  val:   2024-08-18 → 2025-08-14  (rows [1691, 2053), used for LGB early stopping)
  test:  2025-08-16 → 2026-08-11  (rows [2054, 2415), genuinely held-out)

Run from the repo root:
    python -m src.backtest.multi_regime
    python -m src.backtest.multi_regime --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.baseline import simulate_buyhold
from src.backtest.report import load_backtest_config
from src.backtest.simulate import BacktestResult, signals_from_proba, simulate_strategy
from src.models.evaluate import load_artifacts
from src.models.train import assemble_dataset, load_modeling_config

logger = logging.getLogger(__name__)

# ── Training period boundaries (from models/1d_pruned/training_manifest.json) ─
_TRAIN_PERIOD_START = pd.Timestamp("2020-01-01", tz="UTC")
_TRAIN_PERIOD_END   = pd.Timestamp("2024-08-16", tz="UTC")
_VAL_PERIOD_END     = pd.Timestamp("2025-08-14", tz="UTC")


# ── Regime definitions (COMMITTED BEFORE RUNNING ANY BACKTEST) ────────────────
#
# Selection methodology:
#   Each window was chosen from publicly documented BTC market history.
#   No backtest results were examined before these dates were written.
#   The in_sample flag is set mechanically — any window that overlaps
#   [2020-01-01, 2024-08-16] is in-sample.
#
# The val period (2024-08-18 → 2025-08-14) was used for LGB early stopping
# (model selection), so it is also "seen" data — but none of the 5 defined
# regimes fall in this range.

@dataclass(frozen=True)
class RegimeWindow:
    """A pre-committed historical BTC regime window."""

    name: str
    label: str
    start: str   # ISO date string, inclusive (UTC)
    end: str     # ISO date string, inclusive (UTC)
    rationale: str
    in_sample: bool  # mechanically derived from training period overlap


REGIMES: list[RegimeWindow] = [
    RegimeWindow(
        name="covid_crash",
        label="COVID crash (Feb-Apr 2020)",
        start="2020-02-01",
        end="2020-04-30",
        rationale=(
            "BTC fell ~65% ($10 400 → $3 600) as global markets seized in March 2020. "
            "March 12 ('Black Thursday') was approximately −50% in a single day. "
            "Classic sharp bear driven by forced liquidations and macro panic."
        ),
        in_sample=True,
    ),
    RegimeWindow(
        name="bull_2020_2021",
        label="2020-2021 bull run (Oct 2020-Apr 2021)",
        start="2020-10-01",
        end="2021-04-30",
        rationale=(
            "BTC rose from ~$10 000 to its April 2021 ATH of ~$64 000. "
            "Driven by institutional adoption (MicroStrategy, Tesla treasury), "
            "PayPal integration, and 2020 halving aftermath. "
            "Strong, sustained uptrend with few major retracements."
        ),
        in_sample=True,
    ),
    RegimeWindow(
        name="bear_2022",
        label="2022 bear market (Jan-Dec 2022)",
        start="2022-01-01",
        end="2022-12-31",
        rationale=(
            "BTC fell ~65% ($47 000 → $16 000) through 2022. "
            "Key catalysts: LUNA/Terra collapse (May), Three Arrows Capital "
            "insolvency (June), FTX collapse (November). "
            "Sustained bear with serial credit implosions."
        ),
        in_sample=True,
    ),
    RegimeWindow(
        name="recovery_2023",
        label="2023 recovery (Jan-Dec 2023)",
        start="2023-01-01",
        end="2023-12-31",
        rationale=(
            "Choppy recovery: BTC ranged $16 000–$45 000 with no clear trend. "
            "Multiple 20–30% retracements. BlackRock spot ETF application (June) "
            "began building the 2024 ETF-approval narrative. "
            "Representative of a trendless, noisy regime."
        ),
        in_sample=True,
    ),
    RegimeWindow(
        name="phase5_test",
        label="Phase 5 test window (Aug 2025-Aug 2026)",
        start="2025-08-16",
        end="2026-08-11",
        rationale=(
            "The held-out test split from Phase 4/5 (model never saw this data). "
            "BTC fell ~46% during this period. "
            "This is the ONLY genuinely out-of-sample window in this analysis. "
            "Result should be consistent with the Phase 5 backtest report."
        ),
        in_sample=False,
    ),
]


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class RegimeSummary:
    """Backtest results for one regime window."""

    regime: RegimeWindow
    lr: BacktestResult
    lgb: BacktestResult
    buyhold: BacktestResult
    n_window_rows: int    # total rows in date range (including NaN)
    n_usable_rows: int    # rows after dropping NaN features/labels
    n_nan_dropped: int    # rows dropped (feature warm-up or dead-zone label)


# ── Core scoring ──────────────────────────────────────────────────────────────

def score_regime(
    regime: RegimeWindow,
    merged: pd.DataFrame,
    feature_cols: list[str],
    artifacts: dict[str, Any],
    raw_close: pd.Series,
    cfg: dict[str, Any],
    bt_cfg: dict[str, Any],
) -> RegimeSummary:
    """Run the backtest for one regime window.

    The model is scored on ALL usable rows in the window (including in-sample
    rows it was trained on).  The in_sample flag on the regime makes the
    diagnostic status explicit — do not interpret in-sample results as evidence
    of skill.

    Args:
        regime: Pre-defined :class:`RegimeWindow`.
        merged: Full feature+label DataFrame from :func:`assemble_dataset`.
        feature_cols: Feature columns the pruned model was trained on.
        artifacts: Dict with ``scaler``, ``logistic_regression``, ``lightgbm``.
        raw_close: Close price Series indexed by UTC open_time (all candles).
        cfg: Modeling config from :func:`load_modeling_config`.
        bt_cfg: Backtest config from :func:`load_backtest_config`.

    Returns:
        :class:`RegimeSummary` with LR, LGB, and buy-and-hold results.

    Raises:
        ValueError: If fewer than 5 usable rows fall in the window.
    """
    start_ts = pd.Timestamp(regime.start, tz="UTC")
    end_ts   = pd.Timestamp(regime.end,   tz="UTC")

    mask = (merged["open_time"] >= start_ts) & (merged["open_time"] <= end_ts)
    window_df = merged[mask]
    usable_df = window_df.dropna(subset=feature_cols)

    n_total  = int(len(window_df))
    n_usable = int(len(usable_df))

    if n_usable < 5:
        raise ValueError(
            f"{regime.name}: only {n_usable} usable rows in window "
            f"({regime.start} → {regime.end}); minimum 5 required"
        )

    scaler    = artifacts["scaler"]
    logreg    = artifacts["logistic_regression"]
    lgb_model = artifacts["lightgbm"]
    threshold: float = cfg["modeling"]["confidence_threshold"]

    X = usable_df[feature_cols]
    X_scaled = pd.DataFrame(
        scaler.transform(X), columns=feature_cols, index=X.index
    )
    prob_lr  = logreg.predict_proba(X_scaled)[:, 1]
    prob_lgb = lgb_model.predict_proba(X)[:, 1]

    sig_lr  = signals_from_proba(prob_lr,  threshold=threshold)
    sig_lgb = signals_from_proba(prob_lgb, threshold=threshold)

    signal_dates = pd.DatetimeIndex(usable_df["open_time"])

    # simulate_strategy needs prices with len(signals)+1 entries.
    # Find the raw candle immediately after the last signal date.
    last_date = signal_dates[-1]
    pos = raw_close.index.searchsorted(last_date)
    if pos + 1 < len(raw_close):
        extra_date = raw_close.index[pos + 1]
    else:
        # Last candle in the dataset — use the original last date as exit,
        # and drop the last signal (can't compute its forward return).
        extra_date = last_date
        signal_dates = signal_dates[:-1]
        sig_lr  = sig_lr[:-1]
        sig_lgb = sig_lgb[:-1]
        logger.warning(
            "%s: no candle after last signal date — dropped last signal row",
            regime.name,
        )

    n_sig = len(signal_dates)
    price_dates = list(signal_dates) + [extra_date]
    prices = raw_close.loc[price_dates]

    lr_signals  = pd.Series(sig_lr[:n_sig],  index=signal_dates, name="signal")
    lgb_signals = pd.Series(sig_lgb[:n_sig], index=signal_dates, name="signal")

    fee      = bt_cfg["fee_rate"]
    slip     = bt_cfg["slippage_rate"]
    notional = bt_cfg["starting_notional"]

    result_lr  = simulate_strategy(
        prices, lr_signals,
        fee_rate=fee, slippage_rate=slip, starting_notional=notional,
        label=f"LR ({regime.label})",
    )
    result_lgb = simulate_strategy(
        prices, lgb_signals,
        fee_rate=fee, slippage_rate=slip, starting_notional=notional,
        label=f"LGB ({regime.label})",
    )
    result_bah = simulate_buyhold(
        prices, fee_rate=fee, slippage_rate=slip, starting_notional=notional,
    )
    result_bah.label = f"buy-and-hold ({regime.label})"

    logger.info(
        "%s [%s]: %d usable rows, %d NaN dropped; LR=%+.1f%% LGB=%+.1f%% B&H=%+.1f%%",
        regime.name,
        "IS" if regime.in_sample else "OOS",
        n_sig,
        n_total - n_usable,
        result_lr.total_return_pct,
        result_lgb.total_return_pct,
        result_bah.total_return_pct,
    )

    return RegimeSummary(
        regime=regime,
        lr=result_lr,
        lgb=result_lgb,
        buyhold=result_bah,
        n_window_rows=n_total,
        n_usable_rows=n_sig,
        n_nan_dropped=n_total - n_usable,
    )


def run_multi_regime(
    interval: str,
    cfg: dict[str, Any],
    bt_cfg: dict[str, Any],
    *,
    model_variant: str = "pruned",
    regimes: list[RegimeWindow] | None = None,
) -> list[RegimeSummary]:
    """Load data + model, then run backtests for all defined regimes.

    Args:
        interval: Binance interval string, e.g. ``"1d"``.
        cfg: Modeling config from :func:`load_modeling_config`.
        bt_cfg: Backtest config from :func:`load_backtest_config`.
        model_variant: Which trained model to use (default: ``"pruned"``).
        regimes: List of :class:`RegimeWindow`; defaults to :data:`REGIMES`.

    Returns:
        List of :class:`RegimeSummary`, one per regime, in definition order.

    Raises:
        FileNotFoundError: If artifacts or data files are missing.
        ValueError: If a regime has fewer than 5 usable rows.
    """
    if regimes is None:
        regimes = REGIMES

    artifacts = load_artifacts(interval, cfg, model_variant=model_variant)
    manifest  = artifacts["manifest"]
    feature_cols: list[str] = manifest["feature_cols"]

    merged, _ = assemble_dataset(interval, cfg, include_onchain=False)

    raw_path  = Path(cfg["raw_dir"]) / f"{cfg['file_prefix']}_{interval}.parquet"
    raw_close = (
        pd.read_parquet(raw_path, columns=["open_time", "close"])
        .set_index("open_time")["close"]
        .sort_index()
    )

    summaries: list[RegimeSummary] = []
    for regime in regimes:
        s = score_regime(
            regime, merged, feature_cols, artifacts, raw_close, cfg, bt_cfg
        )
        summaries.append(s)

    return summaries


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt(value: float, fmt: str = ".1f") -> str:
    nan = isinstance(value, float) and (value != value)
    return "n/a" if nan else f"{value:{fmt}}"


def format_regime_block(summary: RegimeSummary) -> str:
    """Format one regime result as a detailed text block."""
    r = summary.regime
    is_label = "[IN-SAMPLE]" if r.in_sample else "[OUT-OF-SAMPLE]"
    lines = [
        f"===== {r.label}  {is_label} =====",
        f"Window: {r.start} → {r.end}   "
        f"usable rows: {summary.n_usable_rows}   "
        f"NaN dropped: {summary.n_nan_dropped}",
    ]
    if r.in_sample:
        lines.append(
            "  !! IN-SAMPLE: model was trained on this data. "
            "Results are diagnostic only — not evidence of skill."
        )
    else:
        lines.append(
            "  ** OUT-OF-SAMPLE: model never saw this data. "
            "This is the only meaningful performance window."
        )

    col_w = 18
    header = f"{'Metric':<28}" + "".join(
        f"{lbl[:col_w]:>{col_w}}"
        for lbl in ("LR (pruned)", "LGB (pruned)", "buy-and-hold")
    )
    sep = "-" * len(header)
    lines += [header, sep]

    stat_rows = [
        ("Total return %",   "total_return_pct"),
        ("Sharpe ratio",     "sharpe_ratio"),
        ("Max drawdown %",   "max_drawdown_pct"),
        ("# trades",         "n_trades"),
        ("Win rate %",       "win_rate_pct"),
        ("Avg trade P&L %",  "avg_trade_pct"),
    ]
    for display, key in stat_rows:
        row = f"{display:<28}"
        for res in (summary.lr, summary.lgb, summary.buyhold):
            val = getattr(res, key)
            if isinstance(val, int):
                row += f"{val:>{col_w}d}"
            else:
                row += f"{_fmt(val):>{col_w}}"
        lines.append(row)

    return "\n".join(lines)


def format_summary_table(summaries: list[RegimeSummary]) -> str:
    """Format the cross-regime summary table with IS/OOS labels."""
    col = 10
    header_row = (
        f"{'Regime':<34} {'IS/OOS':<6} {'Rows':>5}  "
        + f"{'LR ret%':>{col}} {'LGB ret%':>{col}} {'B&H ret%':>{col}}  "
        + f"{'LR Shp':>{col}} {'LGB Shp':>{col}}  "
        + f"{'LR tr':>6} {'LGB tr':>6}"
    )
    sep = "-" * len(header_row)
    lines = [
        "===== MULTI-REGIME SUMMARY (BTCUSDT 1d, pruned model) =====",
        "Regime windows committed before running any backtest.",
        "IS results reflect memorised training data — NOT evidence of skill.",
        "",
        header_row,
        sep,
    ]

    for s in summaries:
        r = s.regime
        is_tag = "IS" if r.in_sample else "OOS"
        lines.append(
            f"{r.label:<34} {is_tag:<6} {s.n_usable_rows:>5}  "
            + f"{_fmt(s.lr.total_return_pct):>{col}} "
            + f"{_fmt(s.lgb.total_return_pct):>{col}} "
            + f"{_fmt(s.buyhold.total_return_pct):>{col}}  "
            + f"{_fmt(s.lr.sharpe_ratio, '.2f'):>{col}} "
            + f"{_fmt(s.lgb.sharpe_ratio, '.2f'):>{col}}  "
            + f"{s.lr.n_trades:>6} {s.lgb.n_trades:>6}"
        )

    lines.append(sep)
    lines.append("IS = in-sample (trained on)   OOS = out-of-sample (never seen)")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Run multi-regime backtests and print results to stdout.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.backtest.multi_regime",
        description="Phase 7: multi-regime backtesting with in-sample/OOS labeling.",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--model-variant", default="pruned",
        help="which model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--intervals", nargs="+", metavar="INTERVAL",
        help="override modeling.intervals from config",
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

    cfg    = load_modeling_config(args.config)
    bt_cfg = load_backtest_config(args.config)
    intervals: list[str] = args.intervals or cfg["modeling"]["intervals"]

    for interval in intervals:
        print(f"\n{'='*70}")
        print(f"  {cfg['symbol']} {interval} — Phase 7 multi-regime backtest")
        print(f"  Model variant: {args.model_variant}")
        print(f"{'='*70}")
        print()
        print("Regime windows (committed before running any backtest):")
        for reg in REGIMES:
            is_tag = "[IS] " if reg.in_sample else "[OOS]"
            print(f"  {is_tag} {reg.label}: {reg.start} → {reg.end}")
        print()
        print("IS = model trained on this data  |  OOS = never seen by model")
        print("In-sample results are NOT evidence of skill.")
        print()

        try:
            summaries = run_multi_regime(
                interval, cfg, bt_cfg, model_variant=args.model_variant
            )
        except Exception:
            logger.exception(
                "Multi-regime backtest failed for %s %s", cfg["symbol"], interval
            )
            return 1

        for s in summaries:
            print(format_regime_block(s))
            print()

        print(format_summary_table(summaries))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
