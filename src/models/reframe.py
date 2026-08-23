"""Honest walk-forward evaluation of reframed targets (Task 3).

Task 1 measured next-day **price direction** and found no edge; Task 2 added
features and still found none.  This module tests whether **changing the
question** helps, using the same purged walk-forward machinery
(:mod:`src.models.cross_validate`) so the comparison to the direction baseline
is apples-to-apples:

* ``voldir``   — volatility-direction (expand vs contract), a target with real
  autocorrelation.
* ``triple``   — triple-barrier labels (volatility-scaled TP/SL + time limit).
* ``meta``     — meta-labelling: a secondary model gates a primary direction
  call, reported as *acted* precision and coverage.

Each target's forward span sets the purge gap, so no future candle used to build
a label can also sit in the training window.

Run from the repo root::

    python -m src.models.reframe --target voldir
    python -m src.models.reframe --target triple
    python -m src.models.reframe --target meta
    python -m src.models.reframe --target all
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.labels.alt_targets import (
    compute_meta_labels,
    compute_triple_barrier_labels,
    compute_volatility_direction_labels,
)
from src.models.cross_validate import (
    _score_fold,
    make_walk_forward_folds,
)
from src.models.evaluate import base_rate
from src.models.stats import accuracy_vs_base_rate, wilson_interval
from src.models.train import assemble_dataset, load_modeling_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared: assemble base features + a raw candle frame
# ---------------------------------------------------------------------------


def _load_raw_candles(interval: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the raw OHLC candles used to build the reframed targets."""
    from pathlib import Path

    raw_path = Path(cfg["raw_dir"]) / f"{cfg['file_prefix']}_{interval}.parquet"
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"{interval}: raw candles {raw_path} not found — "
            "run `python -m src.data.fetch_binance` first"
        )
    return pd.read_parquet(raw_path).sort_values("open_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Direct binary targets (voldir, triple-barrier)
# ---------------------------------------------------------------------------


@dataclass
class TargetCVReport:
    """Pooled walk-forward result for one reframed binary target."""

    interval: str
    target: str
    n_folds: int
    n_usable: int
    gap: int
    forward_span: int
    pooled: dict[str, Any]


def run_target_cv(
    interval: str,
    cfg: dict[str, Any],
    *,
    target: str,
    n_folds: int = 8,
    embargo: int = 1,
    initial_frac: float = 0.5,
    voldir_window: int = 7,
    tb_vol_window: int = 20,
    tb_mult: float = 1.5,
    tb_max_horizon: int = 10,
) -> TargetCVReport:
    """Purged walk-forward CV for a reframed binary target on the base features.

    Args:
        interval: Binance interval.
        cfg: Config dict from :func:`load_modeling_config`.
        target: ``"voldir"`` or ``"triple"``.
        n_folds, embargo, initial_frac: Fold geometry.
        voldir_window: Window for the volatility-direction target.
        tb_vol_window, tb_mult, tb_max_horizon: Triple-barrier parameters.

    Returns:
        A :class:`TargetCVReport` with pooled accuracy, Wilson CI and verdict.
    """
    modeling = cfg["modeling"]
    seed = int(modeling["random_state"])
    merged, feature_cols = assemble_dataset(interval, cfg)
    candles = _load_raw_candles(interval, cfg)

    if target == "voldir":
        tgt = compute_volatility_direction_labels(
            candles, vol_window=voldir_window, context=f"{interval} voldir"
        )[["open_time", "label_voldir"]]
        label_col, forward_span = "label_voldir", voldir_window
    elif target == "triple":
        tgt = compute_triple_barrier_labels(
            candles, vol_window=tb_vol_window, upper_mult=tb_mult,
            lower_mult=tb_mult, max_horizon=tb_max_horizon,
            context=f"{interval} triple",
        )[["open_time", "tb_label"]]
        label_col, forward_span = "tb_label", tb_max_horizon
    else:
        raise ValueError(f"unknown direct target {target!r} (use voldir/triple)")

    gap = forward_span + max(0, int(embargo))
    joined = merged.merge(tgt, on="open_time", how="left")
    usable = (
        joined.dropna(subset=[*feature_cols, label_col])
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    folds = make_walk_forward_folds(
        len(usable), n_folds=n_folds, gap=gap, initial_frac=initial_frac
    )
    if not folds:
        raise ValueError(f"{interval}/{target}: no usable folds after purging")

    y_all, lr_all, lgb_all = [], [], []
    for fold in folds:
        _, y_true, lr_pred, lgb_pred = _score_fold(
            usable, fold, feature_cols=feature_cols, label_col=label_col, seed=seed,
            lr_params=modeling["logistic_regression"],
            lgb_params=modeling["lightgbm"],
        )
        y_all.append(y_true)
        lr_all.append(lr_pred)
        lgb_all.append(lgb_pred)

    y_pool = np.concatenate(y_all)
    pooled_base, pooled_majority = base_rate(y_pool)
    pooled: dict[str, Any] = {
        "n": int(len(y_pool)), "base_rate": pooled_base,
        "majority_class": pooled_majority, "models": {},
    }
    for name, pred in (("logistic_regression", np.concatenate(lr_all)),
                       ("lightgbm", np.concatenate(lgb_all))):
        k = int((pred == y_pool).sum())
        ci = wilson_interval(k, len(y_pool))
        sig = accuracy_vs_base_rate(k, len(y_pool), pooled_base)
        pooled["models"][name] = {
            "accuracy": k / len(y_pool), "ci_low": ci.low, "ci_high": ci.high,
            "edge_pp": sig.edge_pp, "verdict": sig.verdict,
            "beats_base_rate": sig.beats_base_rate,
        }
    return TargetCVReport(
        interval=interval, target=target, n_folds=len(folds), n_usable=len(usable),
        gap=gap, forward_span=forward_span, pooled=pooled,
    )


def format_target_report(report: TargetCVReport) -> str:
    """Render a reframed-target CV report."""
    p = report.pooled
    titles = {"voldir": "Volatility-direction (expand vs contract)",
              "triple": "Triple-barrier (vol-scaled TP/SL + time limit)"}
    lines = [
        f"===== {titles.get(report.target, report.target)} — {report.interval} "
        f"({report.n_folds} folds, {report.n_usable} usable rows, gap={report.gap}) =====",
        f"POOLED over {p['n']} out-of-sample predictions "
        f"(base rate {100 * p['base_rate']:.1f}%, majority '{p['majority_class']}'):",
    ]
    for name in ("logistic_regression", "lightgbm"):
        m = p["models"][name]
        lines.append(
            f"  {name:<20} {100 * m['accuracy']:.1f}% "
            f"[{100 * m['ci_low']:.1f}%, {100 * m['ci_high']:.1f}%]  "
            f"edge {m['edge_pp']:+.1f}pp  →  {m['verdict']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Meta-labelling (two-stage)
# ---------------------------------------------------------------------------


@dataclass
class MetaCVReport:
    """Pooled meta-labelling result."""

    interval: str
    n_folds: int
    n_usable: int
    gap: int
    meta_threshold: float
    pooled: dict[str, Any]


def run_meta_labeling_cv(
    interval: str,
    cfg: dict[str, Any],
    *,
    n_folds: int = 8,
    embargo: int = 1,
    initial_frac: float = 0.5,
    meta_threshold: float = 0.5,
    meta_fit_frac: float = 0.3,
) -> MetaCVReport:
    """Two-stage meta-labelling CV: a meta-model gates a primary direction call.

    Per fold, the training region is split so the primary model's sides are
    graded **out of sample** before the meta-model sees them:

    1. Fit the primary (LightGBM direction) on the first ``1-meta_fit_frac`` of
       the train region.
    2. Predict its side on the held-out ``meta_fit_frac`` tail, grade each call
       (``meta_label`` = was it right), and fit the meta-model on that tail.
    3. On the test block: the primary predicts a side, the meta-model predicts
       P(correct); **act** only when that probability ≥ ``meta_threshold``.

    Reports the primary's accuracy on *all* test rows vs the *acted* accuracy
    (primary correct among acted rows) plus coverage — the honest question is
    whether gating raises precision without collapsing coverage.

    Args:
        interval: Binance interval.
        cfg: Config dict.
        n_folds, embargo, initial_frac: Fold geometry.
        meta_threshold: Act when meta P(correct) ≥ this.
        meta_fit_frac: Tail fraction of each train region reserved for grading
            the primary and fitting the meta-model.

    Returns:
        A :class:`MetaCVReport`.
    """
    from src.models.train import train_lightgbm

    modeling = cfg["modeling"]
    horizon = int(modeling["horizon"])
    seed = int(modeling["random_state"])
    label_col = f"label_{horizon}"
    gap = horizon + max(0, int(embargo))

    merged, feature_cols = assemble_dataset(interval, cfg)
    candles = _load_raw_candles(interval, cfg)
    usable = (
        merged.dropna(subset=[*feature_cols, label_col])
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    folds = make_walk_forward_folds(
        len(usable), n_folds=n_folds, gap=gap, initial_frac=initial_frac
    )
    if not folds:
        raise ValueError(f"{interval}/meta: no usable folds after purging")

    lgb_params = modeling["lightgbm"]
    primary_correct_all: list[np.ndarray] = []  # 1 if primary right, per test row
    acted_all: list[np.ndarray] = []            # 1 if meta said act, per test row

    for fold in folds:
        train = usable.iloc[fold.train_start:fold.train_end]
        test = usable.iloc[fold.test_start:fold.test_end]

        n_meta = max(40, int(len(train) * meta_fit_frac))
        n_meta = min(n_meta, len(train) // 2)
        prim_train = train.iloc[:-n_meta]
        meta_train = train.iloc[-n_meta:]

        x_pt, y_pt = prim_train[feature_cols], prim_train[label_col].astype("int64")
        # Small val tail for the primary's early stopping.
        n_val = max(30, int(len(x_pt) * 0.15))
        n_val = min(n_val, len(x_pt) // 3)
        primary = train_lightgbm(
            x_pt.iloc[:-n_val], y_pt.iloc[:-n_val],
            x_pt.iloc[-n_val:], y_pt.iloc[-n_val:],
            params=lgb_params, random_state=seed,
        )

        # Grade primary out-of-sample on the meta-fit tail → meta labels.
        meta_side = np.where(
            primary.predict_proba(meta_train[feature_cols])[:, 1] >= 0.5, 1, -1
        )
        meta_lbl_df = compute_meta_labels(
            candles.merge(meta_train[["open_time"]], on="open_time", how="right"),
            meta_side, horizon=horizon, context=f"{interval} meta-fit",
        )
        graded = meta_lbl_df["meta_label"].notna().to_numpy()
        x_meta = meta_train[feature_cols].iloc[graded]
        y_meta = meta_lbl_df["meta_label"].to_numpy()[graded].astype("int64")
        if len(np.unique(y_meta)) < 2:
            # Degenerate fold: meta target single-valued → act on everything.
            side_test = np.where(
                primary.predict_proba(test[feature_cols])[:, 1] >= 0.5, 1, -1
            )
            act = np.ones(len(test), dtype=bool)
        else:
            n_mval = max(20, int(len(x_meta) * 0.15))
            n_mval = min(n_mval, len(x_meta) // 3)
            meta_model = train_lightgbm(
                x_meta.iloc[:-n_mval], pd.Series(y_meta[:-n_mval], index=x_meta.index[:-n_mval]),
                x_meta.iloc[-n_mval:], pd.Series(y_meta[-n_mval:], index=x_meta.index[-n_mval:]),
                params=lgb_params, random_state=seed,
            )
            side_test = np.where(
                primary.predict_proba(test[feature_cols])[:, 1] >= 0.5, 1, -1
            )
            meta_p = meta_model.predict_proba(test[feature_cols])[:, 1]
            act = meta_p >= meta_threshold

        # Grade primary on the test block (this is the honest OOS number).
        y_test = test[label_col].astype("int64").to_numpy()
        primary_up = side_test > 0
        correct = (primary_up == (y_test == 1)).astype("int64")
        primary_correct_all.append(correct)
        acted_all.append(act.astype("int64"))

    correct_pool = np.concatenate(primary_correct_all)
    acted_pool = np.concatenate(acted_all).astype(bool)
    n = len(correct_pool)
    n_acted = int(acted_pool.sum())

    prim_k = int(correct_pool.sum())
    prim_ci = wilson_interval(prim_k, n)
    if n_acted > 0:
        acted_k = int(correct_pool[acted_pool].sum())
        acted_ci = wilson_interval(acted_k, n_acted)
    else:
        acted_k, acted_ci = 0, wilson_interval(0, 0)

    # Base rate to beat = the primary's own all-rows accuracy.
    prim_acc = prim_k / n if n else float("nan")
    acted_acc = acted_k / n_acted if n_acted else float("nan")
    lift_sig = accuracy_vs_base_rate(acted_k, n_acted, prim_acc) if n_acted else None

    pooled: dict[str, Any] = {
        "n": n,
        "primary_accuracy": prim_acc,
        "primary_ci": [prim_ci.low, prim_ci.high],
        "n_acted": n_acted,
        "coverage": n_acted / n if n else 0.0,
        "acted_accuracy": acted_acc,
        "acted_ci": [acted_ci.low, acted_ci.high],
        "acted_verdict": lift_sig.verdict if lift_sig else "no acted rows",
        "acted_beats_primary": bool(lift_sig.beats_base_rate) if lift_sig else False,
    }
    return MetaCVReport(
        interval=interval, n_folds=len(folds), n_usable=len(usable),
        gap=gap, meta_threshold=meta_threshold, pooled=pooled,
    )


def format_meta_report(report: MetaCVReport) -> str:
    """Render a meta-labelling report."""
    p = report.pooled
    lines = [
        f"===== Meta-labelling (primary=LGB direction, meta gates the bet) — "
        f"{report.interval} ({report.n_folds} folds, {report.n_usable} usable "
        f"rows, gap={report.gap}, act@{report.meta_threshold:.2f}) =====",
        f"Primary on ALL {p['n']} OOS rows: {100 * p['primary_accuracy']:.1f}% "
        f"[{100 * p['primary_ci'][0]:.1f}%, {100 * p['primary_ci'][1]:.1f}%]",
        f"Meta ACTED on {p['n_acted']} rows (coverage {100 * p['coverage']:.1f}%): "
        + (
            f"{100 * p['acted_accuracy']:.1f}% "
            f"[{100 * p['acted_ci'][0]:.1f}%, {100 * p['acted_ci'][1]:.1f}%]  →  "
            f"vs primary: {p['acted_verdict']}"
            if p["n_acted"] else "no rows cleared the meta gate"
        ),
        "\nMeta-labelling helps only if acted accuracy clears the primary's "
        "all-rows accuracy with meaningful coverage; a higher point estimate on "
        "a handful of acted rows is not evidence.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Evaluate one or more reframed targets for the configured intervals."""
    parser = argparse.ArgumentParser(
        prog="python -m src.models.reframe",
        description="Honest walk-forward evaluation of reframed targets (Task 3).",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--intervals", nargs="+", metavar="INTERVAL")
    parser.add_argument("--target", default="all",
                        choices=["voldir", "triple", "meta", "all"])
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--embargo", type=int, default=1)
    parser.add_argument("--initial-frac", type=float, default=0.5)
    parser.add_argument("--meta-threshold", type=float, default=0.5)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    cfg = load_modeling_config(args.config)
    intervals: list[str] = args.intervals or cfg["modeling"]["intervals"]
    targets = ["voldir", "triple", "meta"] if args.target == "all" else [args.target]
    failures: list[str] = []
    for interval in intervals:
        for target in targets:
            try:
                if target == "meta":
                    rpt = run_meta_labeling_cv(
                        interval, cfg, n_folds=args.folds, embargo=args.embargo,
                        initial_frac=args.initial_frac,
                        meta_threshold=args.meta_threshold,
                    )
                    print("\n" + format_meta_report(rpt) + "\n")
                else:
                    rpt = run_target_cv(
                        interval, cfg, target=target, n_folds=args.folds,
                        embargo=args.embargo, initial_frac=args.initial_frac,
                    )
                    print("\n" + format_target_report(rpt) + "\n")
            except Exception:
                logger.exception("%s/%s: reframe evaluation failed", interval, target)
                failures.append(f"{interval}/{target}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
