"""Tests for the paper-trading engine (pure portfolio logic + persistence)."""

from __future__ import annotations

import pytest

from src.dashboard.paper_trader import (
    FEE_RATE,
    STARTING_CAPITAL,
    apply_decision,
    buy_and_hold,
    load_portfolio,
    max_drawdown,
    new_portfolio,
    paper_path,
    portfolio_summary,
    reset_portfolio,
    save_portfolio,
    trade_stats,
)


def _buy(state, price, *, candle, fee=FEE_RATE):
    return apply_decision(state, price, "BUY", now="t", candle_open=candle, fee_rate=fee)


def _sell(state, price, *, candle, fee=FEE_RATE):
    return apply_decision(state, price, "SELL", now="t", candle_open=candle, fee_rate=fee)


class TestPortfolioMechanics:
    def test_fresh_portfolio(self):
        s = new_portfolio()
        assert s["cash"] == STARTING_CAPITAL and s["btc"] == 0.0
        assert s["trades"] == []

    def test_buy_converts_cash_to_btc_minus_fee(self):
        s = _buy(new_portfolio(), 100.0, candle="c1")
        assert s["cash"] == 0.0
        assert s["btc"] == (STARTING_CAPITAL * (1 - FEE_RATE)) / 100.0
        assert s["fees_paid"] == STARTING_CAPITAL * FEE_RATE
        assert s["trades"][-1]["side"] == "BUY"

    def test_sell_converts_btc_to_cash(self):
        s = _buy(new_portfolio(), 100.0, candle="c1")
        s = _sell(s, 100.0, candle="c2")
        assert s["btc"] == 0.0
        # Round-trip at the same price loses two fees → cash < starting capital.
        assert s["cash"] < STARTING_CAPITAL
        assert s["trades"][-1]["side"] == "SELL"

    def test_profit_when_price_rises(self):
        s = _buy(new_portfolio(), 100.0, candle="c1")
        s = _sell(s, 110.0, candle="c2")
        # +10% move easily beats the ~0.2% round-trip fee.
        assert s["cash"] > STARTING_CAPITAL

    def test_loss_when_price_falls(self):
        s = _buy(new_portfolio(), 100.0, candle="c1")
        s = _sell(s, 90.0, candle="c2")
        assert s["cash"] < STARTING_CAPITAL

    def test_one_trade_per_candle(self):
        s = new_portfolio()
        s = _buy(s, 100.0, candle="c1")
        s = _sell(s, 100.0, candle="c1")  # same candle → ignored
        assert len(s["trades"]) == 1  # only the BUY fired

    def test_cannot_buy_twice_without_selling(self):
        s = _buy(new_portfolio(), 100.0, candle="c1")
        s = _buy(s, 105.0, candle="c2")  # already long → no-op
        assert len([t for t in s["trades"] if t["side"] == "BUY"]) == 1

    def test_cannot_sell_when_flat(self):
        s = _sell(new_portfolio(), 100.0, candle="c1")
        assert s["trades"] == [] and s["btc"] == 0.0

    def test_hold_does_nothing(self):
        s = apply_decision(new_portfolio(), 100.0, "HOLD", now="t", candle_open="c1")
        assert s["trades"] == [] and s["cash"] == STARTING_CAPITAL

    def test_equity_curve_grows(self):
        s = new_portfolio()
        s = _buy(s, 100.0, candle="c1")
        s = apply_decision(s, 110.0, "HOLD", now="t2", candle_open="c2")
        assert len(s["equity_curve"]) >= 2


class TestSummary:
    def test_summary_flat(self):
        summ = portfolio_summary(new_portfolio(), 100.0)
        assert summ["equity"] == STARTING_CAPITAL
        assert summ["pnl"] == 0.0 and summ["position"] == "cash (flat)"

    def test_summary_reports_unrealized(self):
        s = _buy(new_portfolio(), 100.0, candle="c1")
        summ = portfolio_summary(s, 120.0)
        assert summ["holding"] is True
        assert summ["unrealized"] > 0
        assert summ["pnl_pct"] > 0


class TestAnalytics:
    def test_buy_and_hold_tracks_price(self):
        s = new_portfolio()
        s["first_price"] = 100.0
        bh = buy_and_hold(s, 120.0)
        # ₹10,000 at 100 (minus one fee), marked at 120 → ~+20% minus the fee.
        assert bh["pnl_pct"] > 18.0
        assert bh["value"] > STARTING_CAPITAL

    def test_buy_and_hold_without_anchor(self):
        bh = buy_and_hold(new_portfolio(), 120.0)
        assert bh["pnl"] == 0.0 and bh["first_price"] is None

    def test_trade_stats_win_loss(self):
        s = new_portfolio()
        s["trades"] = [
            {"side": "BUY", "price": 100, "qty": 1, "fee": 1},
            {"side": "SELL", "price": 110, "qty": 1, "fee": 1, "realized": 90.0},
            {"side": "BUY", "price": 110, "qty": 1, "fee": 1},
            {"side": "SELL", "price": 105, "qty": 1, "fee": 1, "realized": -55.0},
        ]
        st = trade_stats(s)
        assert st["n_closed"] == 2
        assert st["n_wins"] == 1 and st["n_losses"] == 1
        assert st["win_rate"] == 50.0
        assert st["best"] == 90.0 and st["worst"] == -55.0

    def test_trade_stats_empty(self):
        st = trade_stats(new_portfolio())
        assert st["n_closed"] == 0 and st["win_rate"] is None

    def test_max_drawdown(self):
        s = new_portfolio()
        s["equity_curve"] = [
            {"time": "t", "equity": v, "price": 1}
            for v in (10000, 11000, 9900, 10500)  # peak 11000 → trough 9900
        ]
        # drawdown = (9900 - 11000) / 11000 = -10%
        assert max_drawdown(s) == pytest.approx(-10.0)

    def test_max_drawdown_flat(self):
        assert max_drawdown(new_portfolio()) == 0.0


class TestPersistence:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "paper.json"
        s = _buy(new_portfolio(), 100.0, candle="c1")
        save_portfolio(s, path)
        loaded = load_portfolio(path)
        assert loaded["btc"] == s["btc"]
        assert loaded["cash"] == s["cash"]
        assert loaded["trades"][-1]["side"] == "BUY"

    def test_missing_file_returns_fresh(self, tmp_path):
        s = load_portfolio(tmp_path / "nope.json")
        assert s["cash"] == STARTING_CAPITAL and s["trades"] == []

    def test_reset(self, tmp_path):
        path = tmp_path / "paper.json"
        save_portfolio(_buy(new_portfolio(), 100.0, candle="c1"), path)
        s = reset_portfolio(path)
        assert s["cash"] == STARTING_CAPITAL and s["btc"] == 0.0
        assert load_portfolio(path)["trades"] == []

    def test_path_per_symbol_interval(self):
        assert paper_path("BTCUSDT", "5m") != paper_path("BTCUSDT", "15m")
        assert paper_path("BTCUSDT", "5m").suffix == ".json"
