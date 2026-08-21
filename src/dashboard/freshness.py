"""Model freshness check + validated retrain-and-promote (Phase 9).

The dashboard should never quietly serve a model trained on stale data, and it
must never swap in a freshly trained model that has not passed the same
validation gate as the original.  This module enforces both:

* :func:`check_freshness` reads the training manifest's ``trained_at_utc`` and
  reports whether it is older than a threshold (default 24h).
* :func:`retrain_with_validation` retrains into a **staging directory**, runs
  the existing leak-tripwire gate (:func:`src.models.evaluate.evaluate_interval`)
  against the staged model, and **only promotes it atomically if the gate
  passes**.  On any failure the live model is left exactly as it was.
* Every attempt — pass or fail — is appended to a permanent audit file with its
  timestamp, outcome, and measured accuracy.  A retrain that fails silently is
  the exact failure mode this guards against.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.evaluate import evaluate_interval
from src.models.train import train_pruned_interval

logger = logging.getLogger(__name__)

MAX_AGE_HOURS = 24.0
RETRAIN_AUDIT_PATH = Path("data/signal_log/retrain_audit.csv")

AUDIT_COLUMNS: list[str] = [
    "timestamp_utc",
    "interval",
    "variant",
    "passed",
    "promoted",
    "lr_test_accuracy_pct",
    "lgb_test_accuracy_pct",
    "n_leak_alerts",
    "note",
]


# ---------------------------------------------------------------------------
# Freshness check
# ---------------------------------------------------------------------------


def check_freshness(
    manifest_path: str | Path,
    *,
    now: pd.Timestamp | None = None,
    max_age_hours: float = MAX_AGE_HOURS,
) -> dict[str, Any]:
    """Report whether a training manifest is older than ``max_age_hours``.

    Args:
        manifest_path: Path to a ``training_manifest.json``.
        now: Injectable current time (defaults to UTC now).
        max_age_hours: Age above which the model is considered stale.

    Returns:
        Dict with ``exists`` (bool), ``trained_at`` (str or None),
        ``age_hours`` (float or None), ``is_stale`` (bool), ``max_age_hours``,
        and a human ``message``.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    path = Path(manifest_path)
    if not path.is_file():
        return {
            "exists": False,
            "trained_at": None,
            "age_hours": None,
            "is_stale": True,
            "max_age_hours": max_age_hours,
            "message": f"No training manifest at {path} — train the model first.",
        }

    manifest = json.loads(path.read_text(encoding="utf-8"))
    trained_at = pd.Timestamp(manifest["trained_at_utc"])
    if trained_at.tzinfo is None:
        trained_at = trained_at.tz_localize("UTC")
    age_hours = (now - trained_at).total_seconds() / 3600.0
    is_stale = age_hours > max_age_hours

    if is_stale:
        message = (
            f"Model last trained {trained_at.date()} "
            f"({age_hours:.1f}h ago) — older than {max_age_hours:.0f}h. "
            "Consider retraining."
        )
    else:
        message = (
            f"Model is fresh — trained {age_hours:.1f}h ago "
            f"(threshold {max_age_hours:.0f}h)."
        )
    return {
        "exists": True,
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "age_hours": age_hours,
        "is_stale": is_stale,
        "max_age_hours": max_age_hours,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def log_retrain_attempt(
    record: dict[str, Any],
    *,
    audit_path: str | Path = RETRAIN_AUDIT_PATH,
) -> None:
    """Append one retrain attempt to the permanent audit log (append-only)."""
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    row = {col: record.get(col, "") for col in AUDIT_COLUMNS}
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info("retrain audit: %s", row)


# ---------------------------------------------------------------------------
# Validated retrain + atomic promotion
# ---------------------------------------------------------------------------


def _extract_test_accuracy(report: dict[str, Any], model: str) -> float | None:
    """Pull test-split accuracy (as a percentage) from an evaluate report."""
    try:
        return report["models"][model]["test"]["metrics"]["accuracy"] * 100.0
    except (KeyError, TypeError):
        return None


def retrain_with_validation(
    interval: str,
    cfg: dict[str, Any],
    *,
    model_variant: str = "pruned",
    now: pd.Timestamp | None = None,
    audit_path: str | Path = RETRAIN_AUDIT_PATH,
    train_fn: Callable[[str, dict[str, Any]], Any] = train_pruned_interval,
    evaluate_fn: Callable[..., dict[str, Any]] = evaluate_interval,
) -> dict[str, Any]:
    """Retrain into staging, run the validation gate, promote only on pass.

    The model is trained into a staging directory on the same filesystem as the
    live model.  The existing leak-tripwire gate is then run against the staged
    model; **only if it reports no leak alerts** is the staged model promoted to
    the live path with an atomic directory replace.  On any failure or exception
    the live model is untouched.

    Args:
        interval: Binance interval, e.g. ``"1d"``.
        cfg: Config dict from :func:`src.models.train.load_modeling_config`.
        model_variant: Model variant to retrain (only ``"pruned"`` supported by
            the default ``train_fn``).
        now: Injectable timestamp for the audit record.
        audit_path: Path to the append-only retrain audit log.
        train_fn: Training function ``(interval, cfg) -> manifest``; injectable
            for testing.
        evaluate_fn: Evaluation function returning a report with ``leak_alerts``
            and ``models``; injectable for testing.

    Returns:
        Dict with ``passed``, ``promoted``, ``lr_test_accuracy_pct``,
        ``lgb_test_accuracy_pct``, ``n_leak_alerts``, ``note``, and ``message``.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    models_dir = Path(cfg["models_dir"])
    stamp = now.strftime("%Y%m%d_%H%M%S")
    staging_root = models_dir / f".staging_{interval}_{model_variant}_{stamp}"
    live_dir = models_dir / f"{interval}_{model_variant}"

    record: dict[str, Any] = {
        "timestamp_utc": now.isoformat(timespec="seconds"),
        "interval": interval,
        "variant": model_variant,
        "passed": False,
        "promoted": False,
        "lr_test_accuracy_pct": "",
        "lgb_test_accuracy_pct": "",
        "n_leak_alerts": "",
        "note": "",
    }
    outcome: dict[str, Any] = {
        "passed": False,
        "promoted": False,
        "lr_test_accuracy_pct": None,
        "lgb_test_accuracy_pct": None,
        "n_leak_alerts": None,
        "note": "",
        "message": "",
    }

    try:
        staging_root.mkdir(parents=True, exist_ok=True)
        staged_cfg = {**cfg, "models_dir": str(staging_root)}

        logger.info(
            "retrain: training %s (%s) into staging %s",
            interval, model_variant, staging_root,
        )
        train_fn(interval, staged_cfg)

        logger.info("retrain: running validation gate on staged model")
        report = evaluate_fn(interval, staged_cfg, model_variant=model_variant)

        leak_alerts = report.get("leak_alerts", [])
        n_alerts = len(leak_alerts)
        lr_acc = _extract_test_accuracy(report, "logistic_regression")
        lgb_acc = _extract_test_accuracy(report, "lightgbm")
        passed = n_alerts == 0

        record["n_leak_alerts"] = n_alerts
        record["lr_test_accuracy_pct"] = "" if lr_acc is None else round(lr_acc, 2)
        record["lgb_test_accuracy_pct"] = "" if lgb_acc is None else round(lgb_acc, 2)
        outcome.update(
            passed=passed,
            lr_test_accuracy_pct=lr_acc,
            lgb_test_accuracy_pct=lgb_acc,
            n_leak_alerts=n_alerts,
        )

        if passed:
            staged_model_dir = staging_root / f"{interval}_{model_variant}"
            _atomic_promote(staged_model_dir, live_dir, models_dir, stamp)
            record["passed"] = True
            record["promoted"] = True
            record["note"] = "validation passed; promoted to live"
            outcome.update(
                promoted=True,
                note=record["note"],
                message=(
                    f"Retrain passed the validation gate (LR test "
                    f"{lr_acc:.1f}%, LGB test {lgb_acc:.1f}%) and is now live."
                ),
            )
        else:
            record["note"] = f"validation FAILED ({n_alerts} leak alert(s)); kept previous model"
            outcome.update(
                note=record["note"],
                message=(
                    f"Retrain FAILED the validation gate ({n_alerts} leak "
                    "alert(s)). The previous model is still live — no swap made."
                ),
            )
            logger.error("retrain: %s", record["note"])
    except Exception as exc:  # keep previous model live on any error
        record["note"] = f"error: {exc}"
        outcome.update(
            passed=False,
            promoted=False,
            note=record["note"],
            message=(
                f"Retrain errored ({exc}). The previous model is still live — "
                "no swap made."
            ),
        )
        logger.exception("retrain: failed with an exception; previous model kept")
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        log_retrain_attempt(record, audit_path=audit_path)

    return outcome


def _atomic_promote(
    staged_model_dir: Path,
    live_dir: Path,
    models_dir: Path,
    stamp: str,
) -> None:
    """Atomically replace ``live_dir`` with ``staged_model_dir``.

    Both directories are on the same filesystem (staging lives under
    ``models_dir``), so ``os.replace`` on the directories is atomic.  The old
    live directory is moved aside first and only removed after the new one is
    in place, so a crash mid-promotion can never leave the model half-swapped.
    """
    backup = models_dir / f".backup_{live_dir.name}_{stamp}"
    if live_dir.exists():
        os.replace(live_dir, backup)
    try:
        os.replace(staged_model_dir, live_dir)
    except Exception:
        # Roll back: restore the previous model if the swap failed.
        if backup.exists() and not live_dir.exists():
            os.replace(backup, live_dir)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)
