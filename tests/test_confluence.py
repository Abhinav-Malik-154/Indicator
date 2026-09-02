"""Tests for multi-timeframe confluence + volatility-regime gating.

Pure logic: the weighted directional vote, the MIXED dead-zone, higher-timeframe
weighting, and the A/B/C/No-setup grading against the volatility regime.  The
live fetch is injected so nothing touches the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.dashboard.confluence import (
    CONFLUENCE_TIMEFRAMES,
    TFCall,
    confluence,
    gather_timeframe_calls,
    setup_verdict,
)


def _tf(tf, predicted, score, weight, regime="range"):
    return TFCall(timeframe=tf, predicted=predicted, score=score,
                  regime=regime, weight=weight)


# ── Confluence vote ────────────────────────────────────────────────────────


class TestConfluence:
    def test_all_agree_up(self):
        calls = [_tf("1m", "UP", 0.5, 1.0), _tf("1h", "UP", 0.7, 3.0)]
        c = confluence(calls)
        assert c["direction"] == "UP"
        assert c["net"] == 1.0 and c["strength"] == 1.0
        assert c["agree"] == 2 and c["n_up"] == 2

    def test_all_agree_down(self):
        c = confluence([_tf("5m", "DOWN", -0.4, 1.5), _tf("15m", "DOWN", -0.6, 2.0)])
        assert c["direction"] == "DOWN"
        assert c["net"] == -1.0

    def test_conflict_is_mixed(self):
        # 1m UP (weight 1) vs 1h DOWN (weight 3) → net strongly negative, not mixed;
        # use equal weights to force a genuine tie → MIXED.
        c = confluence([_tf("1m", "UP", 0.5, 1.0), _tf("5m", "DOWN", -0.5, 1.0)])
        assert c["direction"] == "MIXED"
        assert c["net"] == 0.0

    def test_higher_timeframe_dominates(self):
        # Three low TFs UP, one high TF DOWN — the heavy 1h should pull it down.
        calls = [
            _tf("1m", "UP", 0.3, 1.0),
            _tf("5m", "UP", 0.3, 1.5),
            _tf("15m", "UP", 0.3, 2.0),
            _tf("1h", "DOWN", -0.8, 3.0),
        ]
        c = confluence(calls)
        # weighted net = (1+1.5+2-3)/7.5 = 1.5/7.5 = 0.2 → still UP but weak
        assert c["net"] == 0.2
        assert c["direction"] == "UP"

    def test_neutral_dilutes(self):
        c = confluence([_tf("1m", "UP", 0.5, 1.0), _tf("5m", "NEUTRAL", 0.0, 1.0)])
        # net = 1/2 = 0.5 UP, but only one directional vote
        assert c["direction"] == "UP"
        assert c["n_neutral"] == 1 and c["n_up"] == 1

    def test_empty_is_mixed(self):
        c = confluence([])
        assert c["direction"] == "MIXED" and c["n_tf"] == 0


# ── Setup grading ──────────────────────────────────────────────────────────


class TestSetupVerdict:
    def _strong_up(self):
        return confluence([_tf("15m", "UP", 0.7, 2.0), _tf("1h", "UP", 0.8, 3.0)])

    def test_grade_a_aligned_and_expanding(self):
        v = setup_verdict(self._strong_up(), {"regime": "EXPAND"})
        assert v["grade"].startswith("A")
        assert v["direction"] == "UP" and v["expanding"] is True

    def test_grade_b_aligned_but_contracting(self):
        v = setup_verdict(self._strong_up(), {"regime": "CONTRACT"})
        assert v["grade"].startswith("B")
        assert v["strong_align"] is True and v["expanding"] is False

    def test_no_setup_when_mixed(self):
        mixed = confluence([_tf("1m", "UP", 0.5, 1.0), _tf("5m", "DOWN", -0.5, 1.0)])
        v = setup_verdict(mixed, {"regime": "EXPAND"})
        assert v["grade"] == "No setup"

    def test_weak_lean_without_support(self):
        weak = confluence([_tf("1m", "UP", 0.2, 1.0), _tf("5m", "NEUTRAL", 0.0, 1.0)])
        # net = 0.5 → not strong_align (<0.6); no vol regime → C
        v = setup_verdict(weak, None)
        assert v["grade"].startswith("C") or v["grade"] == "No setup"

    def test_missing_vol_regime_never_grade_a(self):
        v = setup_verdict(self._strong_up(), None)
        assert not v["grade"].startswith("A")


# ── Gathering (injected poll, no network) ──────────────────────────────────


class TestGather:
    @staticmethod
    def _trend(n, slope):
        close = np.linspace(100, 100 + slope * n, n)
        op = np.r_[close[0], close[:-1]]
        return pd.DataFrame({
            "open_time": pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"),
            "open": op, "high": np.maximum(op, close) + 0.2,
            "low": np.minimum(op, close) - 0.2, "close": close,
        })

    def test_gathers_all_timeframes(self):
        calls = gather_timeframe_calls(
            "BTCUSDT", poll=lambda sym, tf: self._trend(90, 0.5)
        )
        assert len(calls) == len(CONFLUENCE_TIMEFRAMES)
        assert all(c.predicted == "UP" for c in calls)  # every TF sees the uptrend
        assert confluence(calls)["direction"] == "UP"

    def test_skips_failing_timeframe(self):
        def poll(sym, tf):
            if tf == "1h":
                raise RuntimeError("fetch failed")
            return self._trend(90, 0.5)

        calls = gather_timeframe_calls("BTCUSDT", poll=poll)
        assert len(calls) == len(CONFLUENCE_TIMEFRAMES) - 1
        assert "1h" not in {c.timeframe for c in calls}

    def test_weights_follow_timeframe(self):
        calls = gather_timeframe_calls(
            "BTCUSDT", timeframes=("1m", "1h"),
            poll=lambda sym, tf: self._trend(90, 0.5),
        )
        by_tf = {c.timeframe: c.weight for c in calls}
        assert by_tf["1h"] > by_tf["1m"]  # higher timeframe weighted more
