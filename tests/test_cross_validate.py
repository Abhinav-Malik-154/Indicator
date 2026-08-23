"""Tests for Task 1: honest measurement (stats, CV folds, calibration).

Covers the pure, fast pieces:
- Wilson CI + significance vs base rate (known values + edge cases).
- Walk-forward fold geometry: chronological, non-overlapping, purge gap
  respected, early folds dropped when the training region is too small.
- Brier score and isotonic calibration behaviour.

The full retraining path (`run_cross_validation`, `calibrate_interval`) is
exercised as an integration run from the CLI, not in these unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.calibrate import brier_score, fit_isotonic
from src.models.cross_validate import Fold, make_walk_forward_folds
from src.models.stats import (
    accuracy_vs_base_rate,
    format_ci,
    wilson_interval,
)

# ── Wilson interval ────────────────────────────────────────────────────────

class TestWilsonInterval:
    def test_half_of_hundred(self):
        ci = wilson_interval(50, 100)
        assert ci.point == pytest.approx(0.5)
        # Known Wilson 95% CI for 50/100 ≈ [0.404, 0.596]
        assert ci.low == pytest.approx(0.404, abs=0.005)
        assert ci.high == pytest.approx(0.596, abs=0.005)

    def test_bounds_are_ordered_and_clamped(self):
        ci = wilson_interval(99, 100)
        assert 0.0 <= ci.low <= ci.point <= ci.high <= 1.0

    def test_all_successes(self):
        ci = wilson_interval(20, 20)
        assert ci.point == 1.0
        assert ci.high <= 1.0
        assert ci.low < 1.0

    def test_zero_successes(self):
        ci = wilson_interval(0, 20)
        assert ci.point == 0.0
        assert ci.low >= 0.0

    def test_empty_is_full_interval(self):
        ci = wilson_interval(0, 0)
        assert ci.low == 0.0 and ci.high == 1.0

    def test_narrows_with_more_data(self):
        narrow = wilson_interval(500, 1000)
        wide = wilson_interval(5, 10)
        assert (narrow.high - narrow.low) < (wide.high - wide.low)

    def test_invalid_k_gt_n(self):
        with pytest.raises(ValueError):
            wilson_interval(11, 10)

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            wilson_interval(-1, 10)


# ── Significance vs base rate ──────────────────────────────────────────────

class TestAccuracyVsBaseRate:
    def test_clearly_beats(self):
        # 700/1000 = 70% vs 50% base → CI well above 50%
        r = accuracy_vs_base_rate(700, 1000, 0.50)
        assert r.beats_base_rate is True
        assert "beats" in r.verdict

    def test_clearly_worse(self):
        r = accuracy_vs_base_rate(300, 1000, 0.50)
        assert r.worse_than_base is True
        assert "worse" in r.verdict

    def test_indistinguishable(self):
        # 51% vs 50.8% on ~1100 rows → base rate inside CI
        r = accuracy_vs_base_rate(565, 1108, 0.508)
        assert r.beats_base_rate is False
        assert r.worse_than_base is False
        assert "noise" in r.verdict

    def test_no_data(self):
        r = accuracy_vs_base_rate(0, 0, 0.5)
        assert r.verdict == "no data"

    def test_edge_pp_sign(self):
        r = accuracy_vs_base_rate(600, 1000, 0.50)
        assert r.edge_pp == pytest.approx(10.0, abs=0.1)

    def test_format_ci_percent(self):
        s = format_ci(wilson_interval(50, 100))
        assert "%" in s and "[" in s


# ── Walk-forward fold geometry ─────────────────────────────────────────────

class TestWalkForwardFolds:
    def test_count_and_coverage(self):
        folds = make_walk_forward_folds(1000, n_folds=5, gap=2, initial_frac=0.5)
        assert len(folds) == 5
        # Test blocks are contiguous and cover through the end.
        assert folds[0].test_start == 500
        assert folds[-1].test_end == 1000
        for a, b in zip(folds, folds[1:], strict=False):
            assert a.test_end == b.test_start

    def test_purge_gap_respected(self):
        folds = make_walk_forward_folds(1000, n_folds=5, gap=3, initial_frac=0.5)
        for f in folds:
            assert f.train_end == f.test_start - 3

    def test_train_is_strictly_before_test(self):
        folds = make_walk_forward_folds(2000, n_folds=8, gap=2, initial_frac=0.5)
        for f in folds:
            assert f.train_start == 0
            assert f.train_end < f.test_start
            assert f.test_start < f.test_end

    def test_early_folds_dropped_when_train_too_small(self):
        # initial_frac tiny → first test blocks would have < min_train training
        folds = make_walk_forward_folds(1000, n_folds=10, gap=2, initial_frac=0.02)
        assert all(f.train_end >= 100 for f in folds)
        # Fold indices are renumbered from 0 after dropping.
        assert [f.index for f in folds] == list(range(len(folds)))

    def test_fold_is_frozen(self):
        import dataclasses

        f = make_walk_forward_folds(1000, n_folds=5, gap=2)[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.index = 99  # type: ignore[misc]

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError):
            make_walk_forward_folds(10, n_folds=20, gap=1, initial_frac=0.5)

    def test_invalid_initial_frac(self):
        with pytest.raises(ValueError):
            make_walk_forward_folds(1000, n_folds=5, gap=1, initial_frac=1.5)

    def test_returns_fold_objects(self):
        folds = make_walk_forward_folds(1000, n_folds=3, gap=1)
        assert all(isinstance(f, Fold) for f in folds)


# ── Brier score + isotonic calibration ─────────────────────────────────────

class TestCalibrationHelpers:
    def test_brier_perfect(self):
        assert brier_score(np.array([1.0, 0.0, 1.0]), np.array([1, 0, 1])) == 0.0

    def test_brier_worst(self):
        assert brier_score(np.array([0.0, 1.0]), np.array([1, 0])) == pytest.approx(1.0)

    def test_brier_half(self):
        assert brier_score(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)

    def test_isotonic_is_monotonic(self):
        rng = np.random.default_rng(0)
        prob = rng.uniform(0, 1, 500)
        # Outcomes correlated with prob so isotonic learns an increasing map.
        y = (rng.uniform(0, 1, 500) < prob).astype(int)
        iso = fit_isotonic(prob, y)
        grid = np.linspace(0, 1, 50)
        mapped = iso.predict(grid)
        assert np.all(np.diff(mapped) >= -1e-9)  # non-decreasing

    def test_isotonic_output_in_unit_interval(self):
        rng = np.random.default_rng(1)
        prob = rng.uniform(0, 1, 200)
        y = (rng.uniform(0, 1, 200) < prob).astype(int)
        iso = fit_isotonic(prob, y)
        out = iso.predict(np.array([-0.5, 0.0, 0.5, 1.0, 1.5]))  # clipped
        assert out.min() >= 0.0 and out.max() <= 1.0
