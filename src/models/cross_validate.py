"""Purged, embargoed walk-forward cross-validation (Task 1).

A single train/val/test split (Phase 4) gives **one** out-of-sample number from
**one** market regime — here, a 12-month bear.  That estimate is high-variance:
with ~330 test rows the 95% CI on accuracy is roughly ±5pp, so "45% vs 52% base
rate" could be a real deficit or just noise, and it says nothing about how the
model would do in other regimes.

This module answers the accuracy question more honestly with **walk-forward CV**:

* Chronological folds — always train on the past, test on the next block.
* A **purge + embargo gap** between train and test so the last training label
  (which peeks ``horizon`` candles ahead) cannot overlap the test block.
* Predictions pooled across folds → one accuracy over many regimes, reported
  with a **Wilson confidence interval** and a significance test vs the base rate
  (:mod:`src.models.stats`).

It deliberately uses the **full (base) feature set** — no candlestick pruning —
so that no feature-selection decision can leak across folds.  LightGBM early-
stops on a small validation tail carved from each fold's training region.

Run from the repo root::

    python -m src.models.cross_validate
    python -m src.models.cross_validate --intervals 1d --folds 8 --embargo 2
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.models.evaluate import base_rate
from src.models.stats import accuracy_vs_base_rate, wilson_interval
from src.models.train import (
    assemble_dataset,
    fit_scaler_on_train,
    load_modeling_config,
    train_lightgbm,
    train_logistic_regression,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fold geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold on the usable (NaN-dropped) row index."""

    index: int
    train_start: int
    train_end: int   # exclusive; excludes the purge+embargo gap
    test_start: int
    test_end: int     # exclusive


def make_walk_forward_folds(
    n_rows: int,
    *,
    n_folds: int,
    gap: int,
    initial_frac: float = 0.5,
    min_train: int = 100,
) -> list[Fold]:
    """Build expanding-window walk-forward folds with a purge+embargo gap.

    The first ``initial_frac`` of rows seeds the initial training set; the
    remainder is split into ``n_folds`` equal, contiguous test blocks.  For each
    block, training is everything up to ``gap`` rows before the block start.

    Args:
        n_rows: Number of usable rows (chronologically ordered).
        n_folds: Number of test blocks.
        gap: Rows skipped between train end and test start (>= horizon).
        initial_frac: Fraction of rows reserved for the first training window.
        min_train: Folds whose training region is smaller than this are dropped.

    Returns:
        A list of :class:`Fold` (may be shorter than ``n_folds`` if early folds
        lack enough training data).

    Raises:
        ValueError: If inputs are degenerate.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if not 0.0 < initial_frac < 1.0:
        raise ValueError(f"initial_frac must be in (0, 1), got {initial_frac}")
    test_region_start = int(n_rows * initial_frac)
    test_rows = n_rows - test_region_start
    if test_rows < n_folds:
        raise ValueError(
            f"not enough rows ({n_rows}) for {n_folds} folds after reserving "
            f"{initial_frac:.0%} for initial training"
        )
    block = test_rows // n_folds
    folds: list[Fold] = []
    for i in range(n_folds):
        ts = test_region_start + i * block
        te = n_rows if i == n_folds - 1 else ts + block
        train_end = ts - gap
        if train_end < min_train:
            continue
        folds.append(
            Fold(index=len(folds), train_start=0, train_end=train_end,
                 test_start=ts, test_end=te)
        )
    return folds


# ---------------------------------------------------------------------------
# Per-fold scoring
# ---------------------------------------------------------------------------


@dataclass
class FoldScore:
    """Accuracy of both models on one fold's test block."""

    fold: int
    n_train: int
    n_test: int
    lr_correct: int
    lgb_correct: int
    base_rate: float


def _score_fold(
    usable: pd.DataFrame,
    fold: Fold,
    *,
    feature_cols: list[str],
    label_col: str,
    seed: int,
    lr_params: dict[str, Any],
    lgb_params: dict[str, Any],
    val_frac: float = 0.15,
) -> tuple[FoldScore, np.ndarray, np.ndarray, np.ndarray]:
    """Fit LR + LGB on a fold's training region and score the test block.

    Returns the :class:`FoldScore` plus the raw ``(y_true, lr_pred, lgb_pred)``
    arrays for that fold so predictions can be pooled across folds.
    """
    train = usable.iloc[fold.train_start:fold.train_end]
    test = usable.iloc[fold.test_start:fold.test_end]

    # Carve an early-stopping validation tail from the END of the train region
    # (still strictly before the test block — no leakage).
    n_val = max(30, int(len(train) * val_frac))
    n_val = min(n_val, len(train) // 3)
    core = train.iloc[:-n_val] if n_val > 0 else train
    val = train.iloc[-n_val:] if n_val > 0 else train.iloc[-30:]

    x_core = core[feature_cols]
    y_core = core[label_col].astype("int64")
    x_val = val[feature_cols]
    y_val = val[label_col].astype("int64")
    x_test = test[feature_cols]
    y_test = test[label_col].astype("int64").to_numpy()

    scaler = fit_scaler_on_train(x_core)
    x_core_scaled = pd.DataFrame(
        scaler.transform(x_core), columns=feature_cols, index=x_core.index
    )
    lr = train_logistic_regression(
        x_core_scaled, y_core, params=lr_params, random_state=seed
    )
    lgb = train_lightgbm(
        x_core, y_core, x_val, y_val, params=lgb_params, random_state=seed
    )

    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test), columns=feature_cols, index=x_test.index
    )
    lr_pred = (lr.predict_proba(x_test_scaled)[:, 1] >= 0.5).astype("int64")
    lgb_pred = (lgb.predict_proba(x_test)[:, 1] >= 0.5).astype("int64")

    score = FoldScore(
        fold=fold.index,
        n_train=len(core),
        n_test=len(test),
        lr_correct=int((lr_pred == y_test).sum()),
        lgb_correct=int((lgb_pred == y_test).sum()),
        base_rate=base_rate(y_test)[0],
    )
    return score, y_test, lr_pred, lgb_pred


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class CVReport:
    """Pooled walk-forward CV result for one interval."""

    interval: str
    n_folds: int
    n_usable: int
    gap: int
    fold_scores: list[FoldScore]
    pooled: dict[str, Any]


def run_cross_validation(
    interval: str,
    cfg: dict[str, Any],
    *,
    n_folds: int = 8,
    embargo: int = 1,
    initial_frac: float = 0.5,
    extra_features: pd.DataFrame | None = None,
    use_extra: bool = False,
) -> CVReport:
    """Run purged walk-forward CV for both models on one interval.

    Args:
        interval: Binance interval, e.g. ``"1d"``.
        cfg: Config dict from :func:`src.models.train.load_modeling_config`.
        n_folds: Number of chronological test blocks.
        embargo: Extra rows (on top of the horizon) purged between train and test.
        initial_frac: Fraction of rows reserved for the first training window.
        extra_features: Optional frame with ``open_time`` + extra feature columns
            (e.g. ``deriv_*``).  When given, its columns are **always** part of
            the NaN-drop so the usable row set is identical whether or not the
            extra columns are actually fed to the models — that makes the
            base-vs-extra comparison apples-to-apples.
        use_extra: When ``True``, the extra columns are added to the model
            feature matrix; when ``False`` they only constrain the row set.

    Returns:
        A :class:`CVReport` with per-fold and pooled metrics.
    """
    modeling = cfg["modeling"]
    horizon = int(modeling["horizon"])
    label_col = f"label_{horizon}"
    seed = int(modeling["random_state"])
    gap = horizon + max(0, int(embargo))

    merged, feature_cols = assemble_dataset(interval, cfg)
    extra_cols: list[str] = []
    if extra_features is not None:
        extra_cols = [c for c in extra_features.columns if c != "open_time"]
        merged = merged.merge(extra_features, on="open_time", how="left")

    usable = (
        merged.dropna(subset=[*feature_cols, label_col, *extra_cols])
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    model_features = [*feature_cols, *extra_cols] if use_extra else feature_cols
    folds = make_walk_forward_folds(
        len(usable), n_folds=n_folds, gap=gap, initial_frac=initial_frac
    )
    if not folds:
        raise ValueError(f"{interval}: no usable folds after purging")

    fold_scores: list[FoldScore] = []
    y_all: list[np.ndarray] = []
    lr_all: list[np.ndarray] = []
    lgb_all: list[np.ndarray] = []
    for fold in folds:
        score, y_true, lr_pred, lgb_pred = _score_fold(
            usable, fold,
            feature_cols=model_features, label_col=label_col, seed=seed,
            lr_params=modeling["logistic_regression"],
            lgb_params=modeling["lightgbm"],
        )
        fold_scores.append(score)
        y_all.append(y_true)
        lr_all.append(lr_pred)
        lgb_all.append(lgb_pred)
        logger.info(
            "fold %d: train=%d test=%d | LR %.1f%% LGB %.1f%% (base %.1f%%)",
            fold.index, score.n_train, score.n_test,
            100 * score.lr_correct / score.n_test,
            100 * score.lgb_correct / score.n_test,
            100 * score.base_rate,
        )

    y_pool = np.concatenate(y_all)
    lr_pool = np.concatenate(lr_all)
    lgb_pool = np.concatenate(lgb_all)
    pooled_base, pooled_majority = base_rate(y_pool)

    pooled: dict[str, Any] = {"n": int(len(y_pool)), "base_rate": pooled_base,
                              "majority_class": pooled_majority,
                              "n_features": len(model_features), "models": {}}
    for name, pred in (("logistic_regression", lr_pool), ("lightgbm", lgb_pool)):
        k = int((pred == y_pool).sum())
        ci = wilson_interval(k, len(y_pool))
        sig = accuracy_vs_base_rate(k, len(y_pool), pooled_base)
        pooled["models"][name] = {
            "accuracy": float(accuracy_score(y_pool, pred)),
            "ci_low": ci.low, "ci_high": ci.high,
            "edge_pp": sig.edge_pp, "verdict": sig.verdict,
            "beats_base_rate": sig.beats_base_rate,
        }
    return CVReport(
        interval=interval, n_folds=len(folds), n_usable=len(usable),
        gap=gap, fold_scores=fold_scores, pooled=pooled,
    )


def format_cv_report(report: CVReport) -> str:
    """Render a CV report as a human-readable block."""
    lines = [
        f"===== Purged walk-forward CV — {report.interval} "
        f"({report.n_folds} folds, {report.n_usable} usable rows, "
        f"purge+embargo gap={report.gap}) =====",
        f"{'fold':>4} {'train':>7} {'test':>6} {'LR acc':>8} {'LGB acc':>8} {'base':>7}",
    ]
    for s in report.fold_scores:
        lines.append(
            f"{s.fold:>4} {s.n_train:>7} {s.n_test:>6} "
            f"{100 * s.lr_correct / s.n_test:>7.1f}% "
            f"{100 * s.lgb_correct / s.n_test:>7.1f}% "
            f"{100 * s.base_rate:>6.1f}%"
        )
    p = report.pooled
    lines.append(
        f"\nPOOLED over {p['n']} out-of-sample predictions "
        f"(base rate {100 * p['base_rate']:.1f}%, majority '{p['majority_class']}'):"
    )
    for name in ("logistic_regression", "lightgbm"):
        m = p["models"][name]
        lines.append(
            f"  {name:<20} {100 * m['accuracy']:.1f}% "
            f"[{100 * m['ci_low']:.1f}%, {100 * m['ci_high']:.1f}%]  "
            f"edge {m['edge_pp']:+.1f}pp  →  {m['verdict']}"
        )
    lines.append(
        "\nInterpretation: if the base rate sits inside a model's CI, its edge is "
        "not distinguishable from noise. Regime-averaged CV is a fairer, lower-"
        "variance estimate than the single Phase 5 test window."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base-vs-derivatives comparison (Task 2)
# ---------------------------------------------------------------------------


def compare_with_derivatives(
    interval: str,
    cfg: dict[str, Any],
    *,
    derivatives_path: str | None = None,
    n_folds: int = 8,
    embargo: int = 1,
    initial_frac: float = 0.5,
) -> tuple[CVReport, CVReport]:
    """Run walk-forward CV with and without the ``deriv_*`` features.

    Both runs use the **identical** usable row set (the derivatives columns
    always constrain the NaN-drop), so any accuracy difference is attributable
    to the features, not to a different sample.

    Args:
        interval: Binance interval.
        cfg: Config dict from :func:`src.models.train.load_modeling_config`.
        derivatives_path: Override for the derivatives parquet path.
        n_folds, embargo, initial_frac: Passed to :func:`run_cross_validation`.

    Returns:
        ``(base_report, derivatives_report)``.

    Raises:
        FileNotFoundError: If the derivatives parquet is missing.
    """
    from pathlib import Path

    from src.features.derivatives import build_derivatives_features

    deriv_path = Path(
        derivatives_path
        or Path(cfg["raw_dir"]) / f"derivatives_{interval}.parquet"
    )
    if not deriv_path.is_file():
        raise FileNotFoundError(
            f"{interval}: derivatives data not found at {deriv_path} — "
            "run `python -m src.data.fetch_derivatives` first"
        )

    merged, _ = assemble_dataset(interval, cfg)
    deriv = build_derivatives_features(
        merged[["open_time"]], deriv_path, context=f"{cfg['symbol']} {interval}"
    )

    base = run_cross_validation(
        interval, cfg, n_folds=n_folds, embargo=embargo, initial_frac=initial_frac,
        extra_features=deriv, use_extra=False,
    )
    with_deriv = run_cross_validation(
        interval, cfg, n_folds=n_folds, embargo=embargo, initial_frac=initial_frac,
        extra_features=deriv, use_extra=True,
    )
    return base, with_deriv


def format_comparison(base: CVReport, with_deriv: CVReport) -> str:
    """Render a base-vs-derivatives accuracy comparison with CIs."""
    bp, dp = base.pooled, with_deriv.pooled
    lines = [
        f"===== Base vs +derivatives — {base.interval} "
        f"(identical {bp['n']} OOS predictions, base rate "
        f"{100 * bp['base_rate']:.1f}%) =====",
        f"base features: {bp['n_features']}   "
        f"+derivatives: {dp['n_features']}  "
        f"(+{dp['n_features'] - bp['n_features']} deriv_ columns)",
    ]
    for name in ("logistic_regression", "lightgbm"):
        b = bp["models"][name]
        d = dp["models"][name]
        delta = 100 * (d["accuracy"] - b["accuracy"])
        lines += [
            f"\n--- {name} ---",
            f"  base        {100 * b['accuracy']:.1f}% "
            f"[{100 * b['ci_low']:.1f}%, {100 * b['ci_high']:.1f}%]  "
            f"edge {b['edge_pp']:+.1f}pp  →  {b['verdict']}",
            f"  +derivatives {100 * d['accuracy']:.1f}% "
            f"[{100 * d['ci_low']:.1f}%, {100 * d['ci_high']:.1f}%]  "
            f"edge {d['edge_pp']:+.1f}pp  →  {d['verdict']}",
            f"  Δ accuracy  {delta:+.1f}pp",
        ]
    lines.append(
        "\nInterpretation: the derivatives features help only if +derivatives' "
        "CI clears the base rate where base's does not — a Δ within the CI width "
        "is noise, not signal."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run purged walk-forward CV for the configured modeling intervals."""
    parser = argparse.ArgumentParser(
        prog="python -m src.models.cross_validate",
        description="Purged, embargoed walk-forward cross-validation with CIs.",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--intervals", nargs="+", metavar="INTERVAL")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--embargo", type=int, default=1,
                        help="extra rows purged between train and test (on top of horizon)")
    parser.add_argument("--initial-frac", type=float, default=0.5)
    parser.add_argument("--derivatives", action="store_true",
                        help="compare base vs base+derivatives features (Task 2)")
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
    failures: list[str] = []
    for interval in intervals:
        try:
            if args.derivatives:
                base, with_deriv = compare_with_derivatives(
                    interval, cfg, n_folds=args.folds, embargo=args.embargo,
                    initial_frac=args.initial_frac,
                )
                print("\n" + format_comparison(base, with_deriv) + "\n")
            else:
                report = run_cross_validation(
                    interval, cfg, n_folds=args.folds, embargo=args.embargo,
                    initial_frac=args.initial_frac,
                )
                print("\n" + format_cv_report(report) + "\n")
        except Exception:
            logger.exception("%s: cross-validation failed", interval)
            failures.append(interval)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
