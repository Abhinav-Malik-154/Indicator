"""Tests for the signal ledger + prediction scorecard tables."""

from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard.ledger import build_scorecard, build_signal_ledger


@pytest.fixture
def markers():
    return pd.DataFrame({
        "date": pd.to_datetime(
            ["2026-01-01", "2026-01-05", "2026-01-10"], utc=True
        ),
        "signal": ["BUY", "SELL", "BUY"],
        "price": [42000.0, 45000.5, 47230.9],
        "realized": ["up", "up", "down"],
        "correct": [True, False, False],
    })


class TestSignalLedger:
    def test_columns_and_order(self, markers):
        led = build_signal_ledger(markers)
        assert list(led.columns) == ["Date", "Signal", "Entry price"]
        # Most recent first.
        assert led["Date"].iloc[0] == "2026-01-10"
        assert led["Entry price"].iloc[0] == "$47,231"

    def test_limit(self, markers):
        assert len(build_signal_ledger(markers, limit=2)) == 2

    def test_empty(self):
        led = build_signal_ledger(pd.DataFrame())
        assert led.empty
        assert list(led.columns) == ["Date", "Signal", "Entry price"]


class TestScorecard:
    def test_summary_counts(self, markers):
        _, summ = build_scorecard(markers)
        assert summ["n_total"] == 3
        assert summ["n_correct"] == 1
        assert summ["n_wrong"] == 2
        assert summ["accuracy_pct"] == pytest.approx(100 * 1 / 3)

    def test_table_columns_and_result_glyphs(self, markers):
        tbl, _ = build_scorecard(markers)
        assert list(tbl.columns) == ["Date", "Signal", "Entry", "Outcome", "Result"]
        assert tbl["Result"].iloc[0] in {"correct", "wrong"}
        assert set(tbl["Outcome"]) <= {"▲ up", "▼ down"}

    def test_summary_reflects_all_rows_even_when_table_limited(self, markers):
        tbl, summ = build_scorecard(markers, limit=1)
        assert len(tbl) == 1
        assert summ["n_total"] == 3  # summary is over all rows

    def test_empty(self):
        tbl, summ = build_scorecard(pd.DataFrame())
        assert tbl.empty
        assert summ["accuracy_pct"] is None and summ["n_total"] == 0
