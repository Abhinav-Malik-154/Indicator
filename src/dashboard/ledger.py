"""Signal ledger + prediction scorecard tables for the dashboard.

Two views of the model's actual directional calls (built from
:func:`src.dashboard.chart.compute_historical_markers`, i.e. the pruned
logistic-regression signals on the out-of-sample test split the model never
trained on):

* :func:`build_signal_ledger` — **when & where**: each BUY/SELL call with its
  date and entry price ("here is where the model said act").
* :func:`build_scorecard` — **right vs wrong**: the same calls annotated with the
  realized outcome N days later and a ✅/❌, plus a summary tally.

These are honest, leakage-free records — every marked row already has a known
outcome and the model never saw the test split.  They are *not* a promise of
future performance (measured edge is ~0); they show what the calls actually did.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_LEDGER_COLS = ["Date", "Signal", "Entry price"]
_SCORE_COLS = ["Date", "Signal", "Entry", "Outcome", "Result"]


def build_signal_ledger(
    markers: pd.DataFrame, *, limit: int | None = None,
) -> pd.DataFrame:
    """Table 1 — the model's BUY/SELL calls with date and entry price.

    Args:
        markers: Output of :func:`src.dashboard.chart.compute_historical_markers`
            (columns ``date``, ``signal``, ``price``, ``realized``, ``correct``).
        limit: Keep only the most recent ``limit`` rows (``None`` = all).

    Returns:
        DataFrame with ``Date`` / ``Signal`` / ``Entry price`` (most recent first).
    """
    if markers is None or markers.empty:
        return pd.DataFrame(columns=_LEDGER_COLS)
    df = markers.sort_values("date", ascending=False)
    out = pd.DataFrame({
        "Date": pd.to_datetime(df["date"]).dt.date.astype("string"),
        "Signal": df["signal"].astype("string"),
        "Entry price": df["price"].map(lambda p: f"${p:,.0f}"),
    })
    return out.head(limit).reset_index(drop=True) if limit else out.reset_index(drop=True)


def build_scorecard(
    markers: pd.DataFrame, *, limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Table 2 — the same calls scored correct/wrong, plus a summary tally.

    Args:
        markers: Output of :func:`src.dashboard.chart.compute_historical_markers`.
        limit: Keep only the most recent ``limit`` rows in the table (the summary
            always reflects **all** rows).

    Returns:
        ``(table, summary)`` where ``table`` has ``Date`` / ``Signal`` / ``Entry``
        / ``Outcome`` / ``Result`` and ``summary`` has ``n_total`` / ``n_correct``
        / ``n_wrong`` / ``accuracy_pct`` (``None`` when empty).
    """
    if markers is None or markers.empty:
        return pd.DataFrame(columns=_SCORE_COLS), {
            "n_total": 0, "n_correct": 0, "n_wrong": 0, "accuracy_pct": None,
        }
    df = markers.sort_values("date", ascending=False)
    table = pd.DataFrame({
        "Date": pd.to_datetime(df["date"]).dt.date.astype("string"),
        "Signal": df["signal"].astype("string"),
        "Entry": df["price"].map(lambda p: f"${p:,.0f}"),
        "Outcome": df["realized"].map({"up": "▲ up", "down": "▼ down"}).astype("string"),
        "Result": df["correct"].map(lambda c: "✅ correct" if c else "❌ wrong"),
    })
    n_total = len(df)
    n_correct = int(df["correct"].sum())
    summary = {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_wrong": n_total - n_correct,
        "accuracy_pct": (100.0 * n_correct / n_total) if n_total else None,
    }
    if limit:
        table = table.head(limit)
    return table.reset_index(drop=True), summary
