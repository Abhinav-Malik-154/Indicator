"""Tests for the freshness check and validated retrain-and-promote (Phase 9).

The safety-critical guarantee under test: a retrain is promoted to the live
model **only** when it passes the validation gate; on failure or error the
previous model is left byte-for-byte untouched, and every attempt is audited.

Train/evaluate are injected as fakes so these tests never train a real model.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.dashboard.freshness import (
    check_freshness,
    retrain_with_validation,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _write_manifest(path, trained_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trained_at_utc": trained_at}), encoding="utf-8")


def _make_cfg(models_dir) -> dict:
    return {"models_dir": str(models_dir)}


def _fake_report(*, leak: bool) -> dict:
    alerts = ["PROBABLE LEAK: lr test 0.92"] if leak else []
    acc_lr = 0.92 if leak else 0.45
    acc_lgb = 0.90 if leak else 0.52
    return {
        "leak_alerts": alerts,
        "models": {
            "logistic_regression": {"test": {"metrics": {"accuracy": acc_lr}}},
            "lightgbm": {"test": {"metrics": {"accuracy": acc_lgb}}},
        },
    }


def _make_train_fn(marker: str):
    """A fake train_fn that writes a staged pruned model dir with a marker file."""
    def _train(interval, staged_cfg):
        from pathlib import Path

        d = Path(staged_cfg["models_dir"]) / f"{interval}_pruned"
        d.mkdir(parents=True, exist_ok=True)
        (d / "marker.txt").write_text(marker, encoding="utf-8")
    return _train


# ── check_freshness ────────────────────────────────────────────────────────


class TestCheckFreshness:
    def test_missing_manifest_is_stale(self, tmp_path):
        r = check_freshness(tmp_path / "nope.json")
        assert r["exists"] is False
        assert r["is_stale"] is True

    def test_recent_model_is_fresh(self, tmp_path):
        m = tmp_path / "training_manifest.json"
        _write_manifest(m, "2026-08-20T00:00:00+00:00")
        r = check_freshness(m, now=pd.Timestamp("2026-08-20T06:00:00Z"))
        assert r["is_stale"] is False
        assert r["age_hours"] == pytest.approx(6.0, abs=0.1)

    def test_old_model_is_stale(self, tmp_path):
        m = tmp_path / "training_manifest.json"
        _write_manifest(m, "2026-08-20T00:00:00+00:00")
        r = check_freshness(m, now=pd.Timestamp("2026-08-22T00:00:00Z"))
        assert r["is_stale"] is True
        assert r["age_hours"] == pytest.approx(48.0, abs=0.1)

    def test_naive_timestamp_treated_as_utc(self, tmp_path):
        m = tmp_path / "training_manifest.json"
        _write_manifest(m, "2026-08-20T00:00:00")  # no tz
        r = check_freshness(m, now=pd.Timestamp("2026-08-20T01:00:00Z"))
        assert r["is_stale"] is False


# ── retrain_with_validation ────────────────────────────────────────────────


class TestRetrainPromotion:
    def test_pass_promotes_new_model(self, tmp_path):
        models = tmp_path / "models"
        live = models / "1d_pruned"
        live.mkdir(parents=True)
        (live / "marker.txt").write_text("ORIGINAL", encoding="utf-8")
        audit = tmp_path / "audit.csv"

        out = retrain_with_validation(
            "1d", _make_cfg(models),
            now=pd.Timestamp("2026-08-21T10:00:00Z"),
            audit_path=audit,
            train_fn=_make_train_fn("RETRAINED"),
            evaluate_fn=lambda i, c, model_variant="pruned": _fake_report(leak=False),
        )
        assert out["passed"] is True
        assert out["promoted"] is True
        assert (live / "marker.txt").read_text() == "RETRAINED"

    def test_fail_keeps_previous_model(self, tmp_path):
        models = tmp_path / "models"
        live = models / "1d_pruned"
        live.mkdir(parents=True)
        (live / "marker.txt").write_text("ORIGINAL", encoding="utf-8")
        audit = tmp_path / "audit.csv"

        out = retrain_with_validation(
            "1d", _make_cfg(models),
            now=pd.Timestamp("2026-08-21T10:00:00Z"),
            audit_path=audit,
            train_fn=_make_train_fn("SHOULD_NOT_PROMOTE"),
            evaluate_fn=lambda i, c, model_variant="pruned": _fake_report(leak=True),
        )
        assert out["passed"] is False
        assert out["promoted"] is False
        # Previous model untouched.
        assert (live / "marker.txt").read_text() == "ORIGINAL"

    def test_train_error_keeps_previous_model(self, tmp_path):
        models = tmp_path / "models"
        live = models / "1d_pruned"
        live.mkdir(parents=True)
        (live / "marker.txt").write_text("ORIGINAL", encoding="utf-8")
        audit = tmp_path / "audit.csv"

        def _boom(interval, staged_cfg):
            raise RuntimeError("training blew up")

        out = retrain_with_validation(
            "1d", _make_cfg(models),
            now=pd.Timestamp("2026-08-21T10:00:00Z"),
            audit_path=audit,
            train_fn=_boom,
            evaluate_fn=lambda i, c, model_variant="pruned": _fake_report(leak=False),
        )
        assert out["promoted"] is False
        assert "error" in out["note"]
        assert (live / "marker.txt").read_text() == "ORIGINAL"

    def test_no_staging_or_backup_dirs_left_behind(self, tmp_path):
        models = tmp_path / "models"
        (models / "1d_pruned").mkdir(parents=True)
        audit = tmp_path / "audit.csv"

        retrain_with_validation(
            "1d", _make_cfg(models),
            now=pd.Timestamp("2026-08-21T10:00:00Z"),
            audit_path=audit,
            train_fn=_make_train_fn("X"),
            evaluate_fn=lambda i, c, model_variant="pruned": _fake_report(leak=False),
        )
        leftovers = [
            p.name for p in models.iterdir()
            if p.name.startswith((".staging", ".backup"))
        ]
        assert leftovers == []

    def test_every_attempt_is_audited(self, tmp_path):
        models = tmp_path / "models"
        (models / "1d_pruned").mkdir(parents=True)
        audit = tmp_path / "audit.csv"

        # One pass, one fail.
        retrain_with_validation(
            "1d", _make_cfg(models), now=pd.Timestamp("2026-08-21T10:00:00Z"),
            audit_path=audit, train_fn=_make_train_fn("A"),
            evaluate_fn=lambda i, c, model_variant="pruned": _fake_report(leak=False),
        )
        retrain_with_validation(
            "1d", _make_cfg(models), now=pd.Timestamp("2026-08-21T11:00:00Z"),
            audit_path=audit, train_fn=_make_train_fn("B"),
            evaluate_fn=lambda i, c, model_variant="pruned": _fake_report(leak=True),
        )
        rows = pd.read_csv(audit)
        assert len(rows) == 2
        assert bool(rows.iloc[0]["passed"]) is True
        assert bool(rows.iloc[1]["passed"]) is False
