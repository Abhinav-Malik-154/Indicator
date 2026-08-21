"""In-app + browser signal alerts, always carrying accuracy context (Phase 9).

When a model emits a non-silent signal the dashboard raises a visible banner and
(if the user granted permission) a browser notification.  The single hard rule
here: **an alert never states a bare directional call.**  Every alert inlines
the model's measured historical hit rate, so "BUY" can never appear on screen
without "...historically right 38.6% of the time" right next to it.  A signal
without its track record is exactly the kind of overconfident nudge this project
exists to avoid.

Alerts only fire while the dashboard tab is open — browser notifications are a
client-side feature of the open page, and the banner is re-rendered on each run.
For a signal record that persists whether or not anyone is watching, see
:mod:`src.monitor.record_signal`.
"""

from __future__ import annotations

import html
import json
import math
from typing import Any

from src.dashboard.signals import HISTORICAL_ACCURACY

_MODEL_LABEL = {"lr": "LR", "lgb": "LightGBM"}


# ---------------------------------------------------------------------------
# Alert text (pure — the accuracy context is mandatory and always present)
# ---------------------------------------------------------------------------


def build_alert_text(
    model: str,
    signal: str,
    prob_up: float,
    *,
    hist: dict[str, Any] = HISTORICAL_ACCURACY,
) -> str:
    """Compose an alert string that always includes historical accuracy context.

    Args:
        model: ``"lr"`` or ``"lgb"``.
        signal: ``"BUY"``, ``"SELL"`` or ``"SILENT"``.
        prob_up: Live P(up) from the model.
        hist: Historical accuracy dict (defaults to the project's measured
            numbers).

    Returns:
        A single line such as::

            LR: BUY (confidence 0.63) — historically right 38.6% of the time.

        For SILENT it states no position is implied but still cites the track
        record.  The accuracy clause is never omitted.
    """
    label = _MODEL_LABEL.get(model, model.upper())
    # Directional confidence: for a SELL the model's conviction is in "down".
    if signal == "SELL":
        confidence = 1.0 - prob_up
    else:
        confidence = prob_up

    win_rate = hist.get(model, {}).get("win_rate_pct")
    if win_rate is None or (isinstance(win_rate, float) and math.isnan(win_rate)):
        context = (
            "this model has historically fired no signals on out-of-sample data, "
            "so it has no measured hit rate — treat with extra caution"
        )
    else:
        context = f"historically right {win_rate:.1f}% of the time"

    if signal == "SILENT":
        return (
            f"{label}: SILENT (P(up) {prob_up:.2f}) — no position implied; "
            f"when it does fire it is {context}."
        )
    return f"{label}: {signal} (confidence {confidence:.2f}) — {context}."


def alerts_for_result(
    result: dict[str, Any],
    *,
    hist: dict[str, Any] = HISTORICAL_ACCURACY,
) -> list[dict[str, Any]]:
    """Return one alert descriptor per model that is currently NOT silent.

    Args:
        result: Output of :func:`src.dashboard.signals.compute_live_signal`.
        hist: Historical accuracy dict.

    Returns:
        List of dicts with ``model``, ``signal``, ``prob_up``, ``text``,
        ``level`` (``"warning"``).  Empty when every model is silent.
    """
    alerts: list[dict[str, Any]] = []
    for model in ("lr", "lgb"):
        signal = result[f"signal_{model}"]
        if signal == "SILENT":
            continue
        prob = float(result[f"prob_{model}"])
        alerts.append(
            {
                "model": model,
                "signal": signal,
                "prob_up": prob,
                "text": build_alert_text(model, signal, prob, hist=hist),
                "level": "warning",
            }
        )
    return alerts


# ---------------------------------------------------------------------------
# Browser notification (client-side JS; only works while the tab is open)
# ---------------------------------------------------------------------------


def browser_notification_html(texts: list[str], *, title: str = "BTC signal") -> str:
    """Return an HTML/JS snippet that raises browser notifications for ``texts``.

    Requests notification permission on first use and shows one notification per
    alert.  Rendered via ``streamlit.components.v1.html``.  If the user denies
    permission or the browser blocks it, nothing happens — the in-app banner is
    always the primary channel.
    """
    payload = json.dumps(texts)
    safe_title = json.dumps(title)
    return f"""
    <script>
    (function() {{
        const texts = {payload};
        const title = {safe_title};
        if (!("Notification" in window) || texts.length === 0) return;
        function fire() {{
            texts.forEach(function(t) {{ new Notification(title, {{ body: t }}); }});
        }}
        if (Notification.permission === "granted") {{
            fire();
        }} else if (Notification.permission !== "denied") {{
            Notification.requestPermission().then(function(p) {{
                if (p === "granted") fire();
            }});
        }}
    }})();
    </script>
    """


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


def render_alerts(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Render banners + a browser notification for any non-silent signal.

    Args:
        result: Output of :func:`src.dashboard.signals.compute_live_signal`.

    Returns:
        The list of alerts that fired (also useful for tests).
    """
    import streamlit as st
    import streamlit.components.v1 as components

    alerts = alerts_for_result(result)
    if not alerts:
        st.success(
            "No active signal — both models are SILENT at the confidence "
            "threshold. Nothing to act on (which is usually the honest state)."
        )
        return alerts

    for alert in alerts:
        st.warning(f"⚠️ {html.unescape(alert['text'])}")

    components.html(
        browser_notification_html([a["text"] for a in alerts]),
        height=0,
    )
    st.caption(
        "Alerts fire only while this tab is open. The permanent, always-on "
        "record is written by `python -m src.monitor.record_signal` (see "
        "MONITORING.md)."
    )
    return alerts
