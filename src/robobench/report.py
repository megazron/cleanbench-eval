"""Render a scorecard as text, JSON, or a single self-contained HTML file with no
dependencies, so one run produces a shareable artifact."""
from __future__ import annotations

import html

from .score import Scorecard, wilson_interval


def to_text(card: Scorecard) -> str:
    return card.table()


def to_html(card: Scorecard) -> str:
    gate_clean = card.contamination_clean and card.leakage_clean
    gate_txt = "CONTAMINATION-CLEAN" if gate_clean else "NOT CLEAN"
    gate_col = "#2b8a3e" if gate_clean else "#b02a37"
    lo, hi = card.overall_ci
    rows = []
    for t in card.tasks:
        tlo, thi = wilson_interval(t["successes"], t["trials"])
        w = 100.0 * t["successes"] / t["trials"] if t["trials"] else 0.0
        flag = "" if t["embodiment_ok"] else " <span class='x'>x-embodiment</span>"
        rows.append(
            "<tr><td>%s%s</td><td>%s</td><td>%d/%d</td>"
            "<td><div class='bar'><div style='width:%.0f%%'></div></div></td>"
            "<td>[%.2f, %.2f]</td></tr>" % (
                html.escape(t["task_id"]), flag, html.escape(t["split"]),
                t["successes"], t["trials"], w, tlo, thi))
    warn = ""
    if card.warnings:
        warn = "<ul class='warn'>" + "".join(
            "<li>%s</li>" % html.escape(x) for x in card.warnings) + "</ul>"
    perts = ", ".join(p["name"] for p in card.perturbations) or "none"
    return _TEMPLATE % {
        "policy": html.escape(card.policy),
        "gate_txt": gate_txt, "gate_col": gate_col,
        "suite": html.escape(card.suite_hash[:23]),
        "n": card.n_tasks, "trials": card.trials,
        "overall": card.overall_success, "lo": lo, "hi": hi,
        "perts": html.escape(perts),
        "rows": "".join(rows), "warn": warn,
    }


def write_html(card: Scorecard, path: str) -> str:
    s = to_html(card)
    with open(path, "w") as f:
        f.write(s)
    return path


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>robobench scorecard: %(policy)s</title><style>
:root{color-scheme:light dark}
body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;max-width:820px;margin:2rem auto;
padding:0 1rem;color:#1f2933;background:#fff;line-height:1.5}
h1{font-size:1.4rem;margin-bottom:.2rem}
.gate{display:inline-block;padding:.2rem .6rem;border-radius:6px;color:#fff;font-weight:700;
font-size:.8rem;letter-spacing:.5px;background:%(gate_col)s}
.meta{color:#6b7580;font-size:.85rem;margin:.4rem 0 1rem}
.big{font-size:2.2rem;font-weight:800;color:#0b7285}
.ci{color:#6b7580;font-size:.9rem}
table{border-collapse:collapse;width:100%%;margin-top:1rem;font-size:.9rem}
th,td{text-align:left;padding:.4rem .5rem;border-bottom:1px solid #e3e8ee}
th{color:#6b7580;font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.4px}
.bar{background:#e3f2f4;border-radius:4px;height:14px;width:120px}
.bar div{background:#0b7285;height:14px;border-radius:4px}
.x{color:#b25a00;font-size:.72rem;font-weight:700}
.warn{background:#fbe9eb;border-left:3px solid #b02a37;padding:.5rem 1rem;border-radius:4px;
color:#7a1620;font-size:.85rem}
code{font-family:'Cascadia Code',Consolas,monospace}
</style></head><body>
<h1>robobench scorecard &mdash; %(policy)s <span class="gate">%(gate_txt)s</span></h1>
<div class="meta">suite <code>%(suite)s</code> &middot; %(n)d tasks &times; %(trials)d trials
&middot; perturbations: %(perts)s</div>
<div class="big">%(overall).2f</div>
<div class="ci">in-distribution success, 95%% CI [%(lo).2f, %(hi).2f]</div>
%(warn)s
<table><thead><tr><th>task</th><th>split</th><th>succ</th><th>rate</th><th>95%% CI</th></tr></thead>
<tbody>%(rows)s</tbody></table>
<p class="meta">A rate with no interval is not a result. Cross-embodiment tasks are
listed but excluded from the headline number.</p>
</body></html>"""
