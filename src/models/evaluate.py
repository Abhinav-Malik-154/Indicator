"""Evaluate trained models honestly against the base rate (Phase 4).

Reporting rules enforced here, not left to discipline:

* Accuracy is **always** reported next to the split's base rate (the
  majority-class share of that split) — an accuracy number without its base
  rate is meaningless and never appears alone.
* Per-class precision/recall/F1 and the confusion matrix are shown for the
  "up" and "down" classes separately.
* Feature importances (logistic-regression coefficients, LightGBM gain) are
  printed for a human sanity check, and a single feature dominating the
  gain is flagged automatically as suspicious.
* If out-of-sample accuracy exceeds ``modeling.leak_alert_accuracy``
  (default 65%), the report is stamped PROBABLE LEAK and the process exits
  non-zero — an unusually good number is a bug until proven otherwise.
* Confidence gating: accuracy is reported separately for "all predictions"
  and "signals fired" (probability beyond the configured threshold), with
  coverage, because the gated number is what the indicator design uses.

Evaluation re-assembles the dataset through the exact training code path
(join + alignment guard + boundaries before NaN drop) and cross-checks the
training manifest (input checksums, feature list, boundaries) so it scores
precisely what was trained.

Run from the repo root::

    python -m src.models.evaluate
    python -m src.models.evaluate --intervals 1d
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.fetch_binance import sha256_of
from src.models.train import (
    assemble_dataset,
    load_modeling_config,
    make_split_bounds,
    split_dataset,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "down", 1: "up"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def base_rate(y_true: np.ndarray) -> tuple[float, str]:
    """Majority-class share of a label vector — the number accuracy must beat.

    Args:
        y_true: Binary labels (1 = up, 0 = down).

    Returns:
        Tuple of (majority-class fraction, majority-class name).
    """
    n_up = int((y_true == 1).sum())
    n_down = int((y_true == 0).sum())
    if n_up >= n_down:
        return n_up / len(y_true), CLASS_NAMES[1]
    return n_down / len(y_true), CLASS_NAMES[0]


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    """Accuracy vs base rate, per-class precision/recall/F1, confusion matrix.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Dict with ``accuracy``, ``base_rate``, ``majority_class``,
        ``edge_pp`` (accuracy minus base rate, percentage points),
        ``per_class`` stats, and the ``confusion`` matrix
        (rows = actual [down, up], cols = predicted [down, up]).
    """
    rate, majority = base_rate(y_true)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0.0
    )
    per_class = {
        CLASS_NAMES[cls]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate([0, 1])
    }
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy,
        "base_rate": rate,
        "majority_class": majority,
        "edge_pp": (accuracy - rate) * 100.0,
        "per_class": per_class,
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def gated_report_dict(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Accuracy for "signals fired" vs "all predictions" under gating.

    A signal fires only when the model is confident: predicted P(up) above
    ``threshold`` (signal: up) or below ``1 - threshold`` (signal: down).
    Everything in between is silence.

    Args:
        y_true: True binary labels.
        prob_up: Predicted P(up) per row.
        threshold: Confidence threshold in [0.5, 1).

    Returns:
        Dict with ungated accuracy, signal count/coverage, gated accuracy
        (NaN when no signal fires), and the per-direction signal counts.
    """
    y_pred_all = (prob_up >= 0.5).astype("int64")
    fired = (prob_up > threshold) | (prob_up < 1.0 - threshold)
    n_fired = int(fired.sum())
    result: dict[str, Any] = {
        "threshold": threshold,
        "n_total": int(len(y_true)),
        "accuracy_all": float(accuracy_score(y_true, y_pred_all)),
        "n_signals": n_fired,
        "coverage": n_fired / len(y_true) if len(y_true) else 0.0,
        "n_signals_up": int((prob_up > threshold).sum()),
        "n_signals_down": int((prob_up < 1.0 - threshold).sum()),
        "accuracy_signals": float("nan"),
        "base_rate_signals": float("nan"),
    }
    if n_fired > 0:
        gated_true = y_true[fired]
        gated_pred = (prob_up[fired] > threshold).astype("int64")
        result["accuracy_signals"] = float(accuracy_score(gated_true, gated_pred))
        result["base_rate_signals"] = base_rate(gated_true)[0]
    return result


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------


def logreg_importances(model: Any, feature_cols: list[str]) -> pd.DataFrame:
    """Logistic-regression coefficients, sorted by absolute value.

    Coefficients are in scaled-feature space (the model is fit on
    standardised features), so magnitudes are comparable across features.
    Positive pushes towards "up", negative towards "down".

    Args:
        model: Fitted ``LogisticRegression``.
        feature_cols: Feature names in training order.

    Returns:
        DataFrame with ``feature`` and ``coefficient``, most influential first.
    """
    coefs = model.coef_.ravel()
    frame = pd.DataFrame({"feature": feature_cols, "coefficient": coefs})
    return frame.reindex(
        frame["coefficient"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def lightgbm_importances(model: Any, feature_cols: list[str]) -> pd.DataFrame:
    """LightGBM gain-based importances, sorted descending.

    Args:
        model: Fitted ``LGBMClassifier``.
        feature_cols: Feature names in training order.

    Returns:
        DataFrame with ``feature``, ``gain`` and ``gain_share`` (fraction of
        total gain), most important first.
    """
    gain = model.booster_.feature_importance(importance_type="gain")
    total = float(gain.sum()) or 1.0
    frame = pd.DataFrame(
        {"feature": feature_cols, "gain": gain, "gain_share": gain / total}
    )
    return frame.sort_values("gain", ascending=False).reset_index(drop=True)


def flag_suspicious_importances(
    importances: pd.DataFrame, *, share_col: str, context: str = ""
) -> list[str]:
    """Return human-readable warnings for suspicious importance patterns.

    A single feature carrying most of the model is the classic signature of
    a leaked column — it deserves a loud flag, not silent acceptance.

    Args:
        importances: Importance table with a ``feature`` column.
        share_col: Column holding each feature's share of total importance.
        context: Label for the warning messages.

    Returns:
        List of warning strings (empty when nothing looks suspicious).
    """
    warnings: list[str] = []
    top = importances.iloc[0]
    if top[share_col] > 0.5:
        warnings.append(
            f"{context}: feature '{top['feature']}' carries "
            f"{top[share_col]:.0%} of total importance — a single dominant "
            "feature is the classic leak signature; investigate before trusting"
        )
    return warnings


# ---------------------------------------------------------------------------
# Artifact loading and manifest cross-check
# ---------------------------------------------------------------------------


def load_artifacts(
    interval: str,
    cfg: dict[str, Any],
    *,
    model_variant: str = "",
) -> dict[str, Any]:
    """Load persisted models, scaler, and training manifest for one interval.

    Cross-checks the manifest against the current config and data files:
    input checksums must match (otherwise the parquets changed since
    training) and the modeled horizon must agree.

    Args:
        interval: Binance interval string, e.g. ``"1d"``.
        cfg: Config dict from :func:`load_modeling_config`.
        model_variant: Optional variant suffix, e.g. ``"pruned"`` loads from
            ``models/{interval}_pruned/`` instead of ``models/{interval}/``.

    Returns:
        Dict with ``scaler``, ``logistic_regression``, ``lightgbm`` and
        ``manifest``.

    Raises:
        FileNotFoundError: If artifacts are missing.
        ValueError: If the manifest disagrees with the current data/config.
    """
    dir_name = f"{interval}_{model_variant}" if model_variant else interval
    context = f"{cfg['symbol']} {interval}" + (f" ({model_variant})" if model_variant else "")
    out_dir = Path(cfg["models_dir"]) / dir_name
    manifest_path = out_dir / "training_manifest.json"
    if not manifest_path.is_file():
        flag = " --pruned" if model_variant == "pruned" else ""
        raise FileNotFoundError(
            f"{context}: {manifest_path} not found — run "
            f"`python -m src.models.train{flag}` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for name in ("features", "labels"):
        recorded = manifest["inputs"][name]
        current_sha = sha256_of(Path(recorded["path"]))
        if current_sha != recorded["sha256"]:
            raise ValueError(
                f"{context}: {name} parquet changed since training "
                f"({recorded['path']}) — retrain before evaluating"
            )
    if manifest["horizon"] != cfg["modeling"]["horizon"]:
        raise ValueError(
            f"{context}: manifest horizon {manifest['horizon']} != config "
            f"horizon {cfg['modeling']['horizon']} — retrain before evaluating"
        )

    artifacts: dict[str, Any] = {"manifest": manifest}
    for name in ("scaler", "logistic_regression", "lightgbm"):
        path = out_dir / f"{name}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"{context}: missing artifact {path}")
        artifacts[name] = joblib.load(path)
    return artifacts


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_pct(value: float) -> str:
    """Format a fraction as a percentage string, NaN-safe."""
    return "n/a" if np.isnan(value) else f"{value * 100.0:.1f}%"


def format_split_report(
    model_name: str,
    split_name: str,
    metrics: dict[str, Any],
    gated: dict[str, Any],
) -> str:
    """Render one model/split evaluation block as text.

    Args:
        model_name: e.g. ``"logistic_regression"``.
        split_name: e.g. ``"test"``.
        metrics: Output of :func:`classification_report_dict`.
        gated: Output of :func:`gated_report_dict`.

    Returns:
        Multi-line report string.
    """
    lines = [
        f"--- {model_name} on {split_name} ({metrics['n']} usable rows) ---",
        (
            f"accuracy {_fmt_pct(metrics['accuracy'])} vs base rate "
            f"{_fmt_pct(metrics['base_rate'])} (majority: {metrics['majority_class']})"
            f" -> edge {metrics['edge_pp']:+.1f}pp"
        ),
        f"{'class':<6} {'precision':>9} {'recall':>7} {'f1':>6} {'support':>8}",
    ]
    for cls in ("down", "up"):
        stats = metrics["per_class"][cls]
        lines.append(
            f"{cls:<6} {stats['precision']:>9.3f} {stats['recall']:>7.3f} "
            f"{stats['f1']:>6.3f} {stats['support']:>8d}"
        )
    (tn, fp), (fn, tp) = metrics["confusion"]
    lines += [
        "confusion matrix (rows=actual, cols=predicted):",
        f"{'':>12} {'pred down':>10} {'pred up':>8}",
        f"{'actual down':>12} {tn:>10d} {fp:>8d}",
        f"{'actual up':>12} {fn:>10d} {tp:>8d}",
        (
            f"confidence gating @ {gated['threshold']:.2f}: "
            f"all {gated['n_total']} rows -> accuracy {_fmt_pct(gated['accuracy_all'])}; "
            f"signals fired {gated['n_signals']} "
            f"({_fmt_pct(gated['coverage'])} coverage, "
            f"{gated['n_signals_up']} up / {gated['n_signals_down']} down) "
            f"-> accuracy {_fmt_pct(gated['accuracy_signals'])} "
            f"vs signal base rate {_fmt_pct(gated['base_rate_signals'])}"
        ),
    ]
    return "\n".join(lines)


def evaluate_interval(
    interval: str,
    cfg: dict[str, Any],
    *,
    model_variant: str = "",
) -> dict[str, Any]:
    """Evaluate both trained models on the validation and test splits.

    Re-assembles the dataset through the training code path, re-derives the
    split boundaries, cross-checks them against the training manifest, and
    scores each model.  Prints the full report to stdout.

    Args:
        interval: Binance interval string, e.g. ``"1d"``.
        cfg: Config dict from :func:`load_modeling_config`.
        model_variant: Optional variant suffix.  ``"pruned"`` loads from
            ``models/{interval}_pruned/`` and uses the manifest's (pruned)
            feature list rather than requiring an exact match with the full
            feature set.

    Returns:
        Dict with all metric dicts plus ``leak_alerts`` and
        ``importance_warnings`` lists (empty means nothing suspicious).
    """
    context = (
        f"{cfg['symbol']} {interval}"
        + (f" ({model_variant})" if model_variant else "")
    )
    modeling = cfg["modeling"]
    artifacts = load_artifacts(interval, cfg, model_variant=model_variant)
    manifest = artifacts["manifest"]

    merged, all_feature_cols = assemble_dataset(interval, cfg)
    if model_variant:
        # Pruned variant: manifest records the subset of features the models
        # were trained on.  Verify they are all still available in the data.
        feature_cols: list[str] = manifest["feature_cols"]
        missing = [c for c in feature_cols if c not in all_feature_cols]
        if missing:
            raise ValueError(
                f"{context}: manifest feature cols missing from data: {missing}"
            )
    else:
        feature_cols = all_feature_cols
        if feature_cols != manifest["feature_cols"]:
            raise ValueError(
                f"{context}: feature columns changed since training — retrain first"
            )
    bounds = make_split_bounds(
        len(merged),
        train_frac=modeling["split"]["train_frac"],
        val_frac=modeling["split"]["val_frac"],
        gap_candles=modeling["split"]["gap_candles"],
        context=context,
    )
    if asdict(bounds) != manifest["split_bounds"]:
        raise ValueError(
            f"{context}: split boundaries changed since training — retrain first"
        )
    label_col: str = manifest["label_col"]
    splits = split_dataset(
        merged, bounds, feature_cols=feature_cols, label_col=label_col,
        context=context,
    )

    scaler = artifacts["scaler"]
    threshold: float = modeling["confidence_threshold"]
    leak_alert_accuracy: float = modeling["leak_alert_accuracy"]

    report: dict[str, Any] = {
        "interval": interval,
        "models": {},
        "leak_alerts": [],
        "importance_warnings": [],
    }
    blocks: list[str] = [
        f"===== {context} — horizon {manifest['horizon']} "
        f"(label: close[T+{manifest['horizon']}] direction, dead zone "
        f"±{cfg['labels']['dead_zone_pct']}%) =====",
        (
            "splits (usable rows after NaN drop): "
            + ", ".join(f"{name}={len(df)}" for name, df in splits.items())
        ),
    ]

    for model_name in ("logistic_regression", "lightgbm"):
        model = artifacts[model_name]
        report["models"][model_name] = {}
        for split_name in ("val", "test"):
            frame = splits[split_name]
            x = frame[feature_cols]
            y = frame[label_col].astype("int64").to_numpy()
            if model_name == "logistic_regression":
                x = pd.DataFrame(
                    scaler.transform(x), columns=feature_cols, index=x.index
                )
            prob_up = model.predict_proba(x)[:, 1]
            y_pred = (prob_up >= 0.5).astype("int64")

            metrics = classification_report_dict(y, y_pred)
            gated = gated_report_dict(y, prob_up, threshold=threshold)
            report["models"][model_name][split_name] = {
                "metrics": metrics,
                "gated": gated,
            }
            blocks.append(format_split_report(model_name, split_name, metrics, gated))

            if metrics["accuracy"] > leak_alert_accuracy:
                alert = (
                    f"{context}: {model_name} {split_name} accuracy "
                    f"{_fmt_pct(metrics['accuracy'])} exceeds the leak alert "
                    f"threshold {_fmt_pct(leak_alert_accuracy)} — PROBABLE "
                    "LEAK, do not report this as a result; investigate"
                )
                report["leak_alerts"].append(alert)
                logger.error(alert)

    lr_imp = logreg_importances(artifacts["logistic_regression"], feature_cols)
    lgb_imp = lightgbm_importances(artifacts["lightgbm"], feature_cols)
    report["logreg_importances"] = lr_imp
    report["lightgbm_importances"] = lgb_imp

    lr_share = lr_imp.assign(
        share=lr_imp["coefficient"].abs() / (lr_imp["coefficient"].abs().sum() or 1.0)
    )
    report["importance_warnings"] += flag_suspicious_importances(
        lr_share, share_col="share", context=f"{context} logistic_regression"
    )
    report["importance_warnings"] += flag_suspicious_importances(
        lgb_imp, share_col="gain_share", context=f"{context} lightgbm"
    )

    top_n = 15
    blocks.append(
        f"--- logistic_regression top {top_n} coefficients (scaled space) ---\n"
        + "\n".join(
            f"{row.feature:<24} {row.coefficient:+.4f}"
            for row in lr_imp.head(top_n).itertuples()
        )
    )
    blocks.append(
        f"--- lightgbm top {top_n} features by gain ---\n"
        + "\n".join(
            f"{row.feature:<24} {row.gain:>12.1f} ({row.gain_share:.1%})"
            for row in lgb_imp.head(top_n).itertuples()
        )
    )

    for warning in report["importance_warnings"]:
        logger.warning(warning)
        blocks.append(f"SUSPICIOUS: {warning}")
    if report["leak_alerts"]:
        blocks.append(
            "*** PROBABLE LEAK — accuracy above alert threshold; the numbers "
            "above are NOT results until the leak hunt comes back clean ***"
        )

    print("\n\n".join(blocks))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Evaluate trained models for every configured modeling interval.

    Args:
        argv: CLI arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if any leak alert
        fired (results must not be trusted).
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.models.evaluate",
        description="Evaluate trained models honestly against the base rate.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="path to the pipeline config (default: %(default)s)",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        metavar="INTERVAL",
        help="override modeling.intervals from the config",
    )
    parser.add_argument(
        "--pruned",
        action="store_true",
        help=(
            "evaluate the candlestick-pruned models from models/{interval}_pruned/ "
            "instead of the base models"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    cfg = load_modeling_config(args.config)
    intervals: list[str] = args.intervals or cfg["modeling"]["intervals"]
    model_variant = "pruned" if args.pruned else ""

    failures: list[str] = []
    leak_alerts: list[str] = []
    for interval in intervals:
        try:
            report = evaluate_interval(interval, cfg, model_variant=model_variant)
            leak_alerts += report["leak_alerts"]
        except Exception:
            logger.exception("%s %s: evaluation failed", cfg["symbol"], interval)
            failures.append(interval)

    if failures:
        logger.error("evaluation failed for interval(s): %s", ", ".join(failures))
        return 1
    if leak_alerts:
        logger.error("leak alert(s) fired — do not trust these results")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
