"""Alternative prediction targets for the accuracy experiment (Task 3).

Tasks 1-2 established that next-day **price direction** is, on this data, close
to a coin flip — no feature set clears the base rate with any confidence.  A
standard response in the quant literature (López de Prado, *Advances in
Financial Machine Learning*) is to stop fighting an unpredictable target and
**reframe** the problem into one with more structure:

* :func:`compute_volatility_direction_labels` — predict whether realized
  volatility will **expand or contract** next.  Volatility *clusters*
  (GARCH-style autocorrelation), so its direction is far more forecastable than
  price direction, even when price itself is a martingale.

* :func:`compute_triple_barrier_labels` — the triple-barrier method.  Instead of
  a fixed-horizon return, label a trade by which of three barriers it hits first:
  a volatility-scaled profit-take (upper), stop-loss (lower), or a time limit
  (vertical).  This yields path-dependent labels that match how a position is
  actually managed.

* :func:`compute_meta_labels` — meta-labeling.  Take a primary directional
  *side* and label whether that call **would have been correct**.  A secondary
  model then predicts the primary's reliability and decides *whether to act*
  (bet size 0/1), which can lift precision even when the primary side is weak.

Every function here is a **label** builder: it deliberately looks forward.  The
feature matrix stays strictly ``≤ T``; the walk-forward evaluator
(:mod:`src.models.reframe`) purges a gap equal to each target's forward span so
no future information leaks into training.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Volatility-direction target
# ---------------------------------------------------------------------------


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Backward-looking realized volatility: rolling std of log returns.

    Value at T uses returns through T only (no look-ahead).

    Args:
        close: Close-price series.
        window: Rolling window length in candles.

    Returns:
        Float Series; the first ``window`` rows are NaN (warm-up).
    """
    log_ret = np.log(close.astype("float64")).diff()
    return log_ret.rolling(window, min_periods=window).std()


def compute_volatility_direction_labels(
    df: pd.DataFrame,
    *,
    vol_window: int = 7,
    context: str = "",
) -> pd.DataFrame:
    """Label whether realized volatility expands or contracts next window.

    For each row T:

    * ``current_vol``  = realized vol of returns over ``(T-vol_window, T]``;
    * ``forward_vol``  = realized vol of returns over ``(T, T+vol_window]``;
    * ``label_voldir`` = 1 if ``forward_vol > current_vol`` (expansion) else 0.

    The last ``vol_window`` rows have no forward window and are NaN.

    Args:
        df: Candle DataFrame with ``open_time`` and ``close`` (sorted).
        vol_window: Window length for both the trailing and forward vol.
        context: Label for log messages.

    Returns:
        DataFrame with ``open_time``, ``current_vol``, ``forward_vol`` and
        ``label_voldir``.  ``forward_span`` (candles of look-ahead) is
        ``vol_window``.
    """
    label = context or "voldir"
    close = df["close"].astype("float64")
    current_vol = realized_volatility(close, vol_window)
    # forward_vol[T] = realized vol over (T, T+vol_window] = current_vol shifted back.
    forward_vol = current_vol.shift(-vol_window)

    lbl = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = current_vol.notna() & forward_vol.notna()
    lbl[valid] = (forward_vol[valid] > current_vol[valid]).astype("float64")

    n_up = int((lbl == 1).sum())
    n_down = int((lbl == 0).sum())
    logger.info(
        "%s: vol_window=%d, expansion=%d, contraction=%d, unlabelled=%d (of %d)",
        label, vol_window, n_up, n_down, len(df) - n_up - n_down, len(df),
    )
    out = df[["open_time"]].copy()
    out["current_vol"] = current_vol.to_numpy()
    out["forward_vol"] = forward_vol.to_numpy()
    out["label_voldir"] = lbl.to_numpy()
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Triple-barrier target
# ---------------------------------------------------------------------------


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    *,
    vol_window: int = 20,
    upper_mult: float = 1.5,
    lower_mult: float = 1.5,
    max_horizon: int = 10,
    context: str = "",
) -> pd.DataFrame:
    """Triple-barrier labels with volatility-scaled profit-take / stop-loss.

    For each entry T (using ``high``/``low`` to detect intrabar touches):

    * upper barrier = ``close[T] * (1 + upper_mult * σ[T])``
    * lower barrier = ``close[T] * (1 - lower_mult * σ[T])``
    * vertical barrier = ``max_horizon`` candles

    where ``σ[T]`` is the trailing realized volatility (``vol_window``).  We
    scan candles ``T+1 .. T+max_horizon`` and record which barrier is touched
    first.  ``tb_label`` = 1 if the **upper** barrier is hit first, else 0
    (lower hit, or vertical timeout that closed down).  A vertical timeout that
    closed up is labelled 1.

    Args:
        df: Candle DataFrame with ``open_time``, ``high``, ``low``, ``close``.
        vol_window: Trailing window for the volatility that scales the barriers.
        upper_mult: Upper-barrier width in units of σ.
        lower_mult: Lower-barrier width in units of σ.
        max_horizon: Vertical-barrier length in candles.
        context: Label for log messages.

    Returns:
        DataFrame with ``open_time``, ``tb_label`` (1/0), ``tb_barrier``
        ('up' / 'down' / 'vertical'), ``tb_bars`` (candles to touch) and
        ``tb_ret`` (return at the touch).  ``forward_span`` is ``max_horizon``.
    """
    label = context or "triple_barrier"
    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    sigma = realized_volatility(df["close"], vol_window).to_numpy(dtype="float64")
    n = len(df)

    tb_label = np.full(n, np.nan)
    tb_barrier = np.array([None] * n, dtype=object)
    tb_bars = np.full(n, np.nan)
    tb_ret = np.full(n, np.nan)

    for t in range(n):
        s = sigma[t]
        if np.isnan(s) or s == 0.0 or t + 1 >= n:
            continue
        entry = close[t]
        up = entry * (1.0 + upper_mult * s)
        dn = entry * (1.0 - lower_mult * s)
        last = min(t + max_horizon, n - 1)
        hit_barrier = "vertical"
        hit_j = last
        for j in range(t + 1, last + 1):
            touched_up = high[j] >= up
            touched_dn = low[j] <= dn
            if touched_up and touched_dn:
                # Both barriers within one candle: conservative → assume the
                # adverse (lower) barrier hit first.
                hit_barrier, hit_j = "down", j
                break
            if touched_up:
                hit_barrier, hit_j = "up", j
                break
            if touched_dn:
                hit_barrier, hit_j = "down", j
                break
        ret = close[hit_j] / entry - 1.0
        if hit_barrier == "up":
            lbl = 1.0
        elif hit_barrier == "down":
            lbl = 0.0
        else:  # vertical timeout → sign of the realised return
            lbl = 1.0 if ret > 0.0 else 0.0
        tb_label[t] = lbl
        tb_barrier[t] = hit_barrier
        tb_bars[t] = hit_j - t
        tb_ret[t] = ret

    n_up = int((tb_label == 1).sum())
    n_dn = int((tb_label == 0).sum())
    barr = pd.Series(tb_barrier)
    logger.info(
        "%s: vol_window=%d ±(%.1f/%.1f)σ maxH=%d | up=%d down=%d | "
        "touches up=%d down=%d vertical=%d",
        label, vol_window, upper_mult, lower_mult, max_horizon, n_up, n_dn,
        int((barr == "up").sum()), int((barr == "down").sum()),
        int((barr == "vertical").sum()),
    )
    out = df[["open_time"]].copy()
    out["tb_label"] = tb_label
    out["tb_barrier"] = tb_barrier
    out["tb_bars"] = tb_bars
    out["tb_ret"] = tb_ret
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Meta-labelling target
# ---------------------------------------------------------------------------


def compute_meta_labels(
    df: pd.DataFrame,
    primary_side: np.ndarray | pd.Series,
    *,
    horizon: int = 1,
    dead_zone_pct: float = 0.0,
    context: str = "",
) -> pd.DataFrame:
    """Label whether a primary directional *side* would have been correct.

    Meta-labelling separates the *direction* decision (the primary side, taken
    as given) from the *act / don't-act* decision.  The meta-label at T is 1 if
    the primary side matched the realised forward move, else 0.  A secondary
    model trained on these labels predicts the primary's reliability.

    Args:
        df: Candle DataFrame with ``open_time`` and ``close``.
        primary_side: Per-row primary call — ``+1`` (long), ``-1`` (short) or
            ``0`` (no position → meta-label NaN).
        horizon: Forward horizon for judging correctness.
        dead_zone_pct: Moves within ±this percent count as no-move → NaN
            (the primary can't be graded).
        context: Label for log messages.

    Returns:
        DataFrame with ``open_time``, ``primary_side``, ``fwd_return`` and
        ``meta_label`` (1 correct / 0 wrong / NaN ungraded).
    """
    label = context or "meta"
    side = np.asarray(primary_side, dtype="float64")
    if len(side) != len(df):
        raise ValueError(
            f"{label}: primary_side length {len(side)} != df length {len(df)}"
        )
    close = df["close"].astype("float64")
    fwd_return = (close.shift(-horizon) / close - 1.0).to_numpy()
    threshold = dead_zone_pct / 100.0

    meta = np.full(len(df), np.nan)
    graded = (~np.isnan(fwd_return)) & (side != 0) & (np.abs(fwd_return) > threshold)
    actual_up = fwd_return > 0
    primary_up = side > 0
    meta[graded] = (primary_up[graded] == actual_up[graded]).astype("float64")

    n_right = int(np.nansum(meta == 1))
    n_wrong = int(np.nansum(meta == 0))
    logger.info(
        "%s: horizon=%d, graded=%d, primary_correct=%d (%.1f%%), wrong=%d",
        label, horizon, n_right + n_wrong, n_right,
        100.0 * n_right / (n_right + n_wrong) if (n_right + n_wrong) else 0.0,
        n_wrong,
    )
    out = df[["open_time"]].copy()
    out["primary_side"] = side
    out["fwd_return"] = fwd_return
    out["meta_label"] = meta
    return out.reset_index(drop=True)
