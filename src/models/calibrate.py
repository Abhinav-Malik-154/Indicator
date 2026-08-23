"""Probability calibration for the confidence gate (Task 1).

The dashboard's signal only fires when a model's ``P(up)`` clears the 0.60
confidence gate.  That threshold is only meaningful if the probabilities are
**calibrated** — i.e. of the days the model says "70% up", roughly 70% actually
go up.  Tree ensembles and, to a lesser degree, logistic regression are often
mis-calibrated, which makes a fixed 0.60 gate fire at the wrong times.

This module fits an **isotonic** calibrator on the validation split (never the
test split) and measures the effect on the test split with:

* **Brier score** (mean squared error of the probability) — lower is better;
* **gated accuracy** at the confidence threshold — the number the indicator
  actually uses.

Calibration cannot manufacture edge (it is a monotonic re-mapping of the same
ranking), but it can make the *gated* signal honest and better-timed.

Run from the repo root::

    python -m src.models.calibrate --pruned --intervals 1d
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.models.evaluate import gated_report_dict, load_artifacts
from src.models.train import (
    assemble_dataset,
    load_modeling_config,
    make_split_bounds,
    split_dataset,
)

logger = logging.getLogger(__name__)


def brier_score(prob: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between predicted P(up) and the binary outcome."""
    prob = np.asarray(prob, dtype="float64")
    y = np.asarray(y, dtype="float64")
    return float(np.mean((prob - y) ** 2))


def fit_isotonic(val_prob: np.ndarray, val_y: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic calibrator mapping raw P(up) → calibrated P(up).

    Args:
        val_prob: Raw validation-split P(up).
        val_y: Validation-split binary outcomes.

    Returns:
        A fitted :class:`~sklearn.isotonic.IsotonicRegression` (clipped to [0,1]).
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(val_prob, dtype="float64"), np.asarray(val_y, dtype="float64"))
    return iso


def _model_probs(
    artifacts: dict[str, Any],
    frame: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> np.ndarray:
    """Predict P(up) for one model on a split, scaling only for logistic reg."""
    x = frame[feature_cols]
    if model_name == "logistic_regression":
        x = pd.DataFrame(
            artifacts["scaler"].transform(x), columns=feature_cols, index=x.index
        )
    return artifacts[model_name].predict_proba(x)[:, 1]


def calibrate_interval(
    interval: str,
    cfg: dict[str, Any],
    *,
    model_variant: str = "pruned",
) -> dict[str, Any]:
    """Fit calibration on validation and report its effect on the test split.

    Args:
        interval: Binance interval.
        cfg: Config dict from :func:`src.models.train.load_modeling_config`.
        model_variant: Which trained model directory to load.

    Returns:
        Dict keyed by model name, each with raw/calibrated Brier scores and
        raw/calibrated gated-accuracy reports.
    """
    modeling = cfg["modeling"]
    threshold = float(modeling["confidence_threshold"])
    artifacts = load_artifacts(interval, cfg, model_variant=model_variant)
    manifest = artifacts["manifest"]

    include_onchain = model_variant == "onchain"
    merged, all_cols = assemble_dataset(interval, cfg, include_onchain=include_onchain)
    feature_cols = manifest["feature_cols"] if model_variant else all_cols
    bounds = make_split_bounds(
        len(merged),
        train_frac=modeling["split"]["train_frac"],
        val_frac=modeling["split"]["val_frac"],
        gap_candles=modeling["split"]["gap_candles"],
        context=f"{cfg['symbol']} {interval}",
    )
    if asdict(bounds) != manifest["split_bounds"]:
        raise ValueError(f"{interval}: split bounds changed since training — retrain first")
    label_col = manifest["label_col"]
    splits = split_dataset(
        merged, bounds, feature_cols=feature_cols, label_col=label_col,
        context=f"{cfg['symbol']} {interval}",
    )
    val, test = splits["val"], splits["test"]
    val_y = val[label_col].astype("int64").to_numpy()
    test_y = test[label_col].astype("int64").to_numpy()

    result: dict[str, Any] = {}
    for model_name in ("logistic_regression", "lightgbm"):
        val_prob = _model_probs(artifacts, val, feature_cols, model_name)
        test_prob = _model_probs(artifacts, test, feature_cols, model_name)

        iso = fit_isotonic(val_prob, val_y)
        test_prob_cal = iso.predict(test_prob)

        result[model_name] = {
            "brier_raw": brier_score(test_prob, test_y),
            "brier_calibrated": brier_score(test_prob_cal, test_y),
            "gated_raw": gated_report_dict(test_y, test_prob, threshold=threshold),
            "gated_calibrated": gated_report_dict(test_y, test_prob_cal, threshold=threshold),
        }
    return result


def format_calibration_report(interval: str, result: dict[str, Any]) -> str:
    """Render the calibration comparison as a text block."""
    lines = [f"===== Probability calibration (isotonic, fit on val) — {interval} ====="]
    for model_name, r in result.items():
        gr, gc = r["gated_raw"], r["gated_calibrated"]

        def _acc(g: dict[str, Any]) -> str:
            n = g["n_signals"]
            if n == 0:
                return "no signals fired"
            return f"{100 * g['accuracy_signals']:.1f}% on {n} signals"

        lines += [
            f"\n--- {model_name} ---",
            f"Brier score:  raw {r['brier_raw']:.4f}  →  calibrated "
            f"{r['brier_calibrated']:.4f}  "
            f"({'better' if r['brier_calibrated'] < r['brier_raw'] else 'no better'})",
            f"Gated @ threshold: raw {_acc(gr)}  →  calibrated {_acc(gc)}",
        ]
    lines.append(
        "\nCalibration re-maps probabilities monotonically; it cannot create "
        "directional edge, only make the confidence gate fire more honestly."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Report calibration effect for the configured modeling intervals."""
    parser = argparse.ArgumentParser(
        prog="python -m src.models.calibrate",
        description="Isotonic probability calibration report (fit on val, tested on test).",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--intervals", nargs="+", metavar="INTERVAL")
    parser.add_argument("--pruned", action="store_true", help="use models/{interval}_pruned/")
    parser.add_argument("--onchain", action="store_true", help="use models/{interval}_onchain/")
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
    variant = "onchain" if args.onchain else ("pruned" if args.pruned else "")
    failures: list[str] = []
    for interval in intervals:
        try:
            result = calibrate_interval(interval, cfg, model_variant=variant)
            print("\n" + format_calibration_report(interval, result) + "\n")
        except Exception:
            logger.exception("%s: calibration failed", interval)
            failures.append(interval)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
