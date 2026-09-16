#!/usr/bin/env python3
"""Regenerate the figures in docs/img/.

    python3 docs/make_figures.py

The pipeline diagram is hand-built SVG (stdlib only); the two charts use matplotlib
and the kit's own Wilson interval and contamination fingerprints, so they show real
behaviour. Every figure is captioned in README.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
os.makedirs(IMG, exist_ok=True)

INK, MUTE, LINE = "#1f2933", "#6b7580", "#c3cbd3"
ACCENT, ACCENT_BG = "#0b7285", "#e3f2f4"
OK, WARN, FAIL = "#2b8a3e", "#b25a00", "#b02a37"
FAIL_BG = "#fbe9eb"
FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "'Cascadia Code',Consolas,monospace"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pipeline_svg():
    W, H = 1000, 340
    b = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" '
         f'orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="{MUTE}"/></marker></defs>']
    b.append(f'<text x="30" y="40" font-size="22" font-weight="700" fill="{INK}">'
             f'What robobench puts between a policy and a number</text>')
    stages = [
        ("task suite", "YAML tasks, each\ncontent-hashed and\nsplit-labelled", ACCENT),
        ("manifest", "sha256 per task,\none suite hash,\nsigned + shipped", ACCENT),
        ("contamination\ngate", "refuses a 'clean'\nscore if holdout is\nin the training set", FAIL),
        ("runner", "your env + policy,\nembodiment checked,\nperturbations declared", ACCENT),
        ("scorecard", "Wilson CI, per split,\nper embodiment,\nclean/not-clean", OK),
    ]
    x, y, bw, bh, gap = 30, 90, 165, 150, 27
    for i, (title, body, col) in enumerate(stages):
        cx = x + i * (bw + gap)
        b.append(f'<rect x="{cx}" y="{y}" width="{bw}" height="{bh}" rx="10" '
                 f'fill="{ACCENT_BG if col!=FAIL else FAIL_BG}" stroke="{col}" stroke-width="2"/>')
        for j, ln in enumerate(title.split("\n")):
            b.append(f'<text x="{cx+bw/2}" y="{y+28+j*20}" font-size="15" '
                     f'font-weight="700" fill="{col if col!=ACCENT else INK}" '
                     f'text-anchor="middle">{esc(ln)}</text>')
        yb = y + 28 + len(title.split("\n")) * 20 + 6
        for j, ln in enumerate(body.split("\n")):
            b.append(f'<text x="{cx+bw/2}" y="{yb+j*17}" font-size="11.5" '
                     f'fill="{MUTE}" text-anchor="middle">{esc(ln)}</text>')
        if i < len(stages) - 1:
            ax = cx + bw
            b.append(f'<line x1="{ax}" y1="{y+bh/2}" x2="{ax+gap}" y2="{y+bh/2}" '
                     f'stroke="{MUTE}" stroke-width="1.8" marker-end="url(#a)"/>')
    b.append(f'<text x="30" y="{H-18}" font-size="12.5" fill="{MUTE}" font-style="italic">'
             f'The gate is the point: a score is withheld, not footnoted, when the '
             f'holdout was trained on.</text>')
    b.append("</svg>")
    open(os.path.join(IMG, "pipeline.svg"), "w").write("\n".join(b))
    print("wrote pipeline.svg")


def wilson_fig():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from robobench.score import wilson_interval

    trials = 10
    xs = list(range(trials + 1))
    means = [k / trials for k in xs]
    los, his = zip(*[wilson_interval(k, trials) for k in xs])
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    ax.plot(xs, means, "-o", color=MUTE, label="bare mean k/10", zorder=3, ms=5)
    ax.fill_between(xs, los, his, color=ACCENT, alpha=0.18,
                    label="95% Wilson interval")
    ax.plot(xs, los, color=ACCENT, lw=1)
    ax.plot(xs, his, color=ACCENT, lw=1)
    k = 7
    lo, hi = wilson_interval(k, trials)
    ax.annotate("7/10 is 0.70,\nbut the interval is [%.2f, %.2f]" % (lo, hi),
                xy=(k, 0.7), xytext=(2.4, 0.86), fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.set_xlabel("successes out of 10 trials")
    ax.set_ylabel("success rate")
    ax.set_title("A rate over ten trials is not a number without its interval")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "wilson.png"), facecolor="white")
    plt.close(fig)
    print("wrote wilson.png")


def leakage_fig():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from robobench.task import load_suite

    suite = os.path.join(os.path.dirname(HERE), "examples", "suite")
    tasks = load_suite(suite)
    # A genuine near-duplicate: the holdout task copied to dev with a new id and a
    # reworded description. Same tested content, so the same fingerprint -- a leak.
    from robobench.task import TaskSpec
    holdout = next(t for t in tasks if t.split == "holdout")
    d = holdout.to_dict()
    d["id"] = "lift_high_copy"
    d["description"] = "reworded copy of the holdout task, dropped into dev"
    d["split"] = "dev"
    dup = TaskSpec.from_dict(d)
    allt = tasks + [dup]
    fps = [t.fingerprint() for t in allt]
    n = len(allt)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = 1.0 if fps[i] == fps[j] else 0.0
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    ax.imshow(M, cmap="BuGn", vmin=0, vmax=1)
    labels = ["%s\n(%s)" % (t.id, t.split) for t in allt]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            if M[i, j] and i != j:
                ax.text(j, i, "leak" if allt[i].split != allt[j].split else "=",
                        ha="center", va="center", fontsize=8,
                        color=FAIL if allt[i].split != allt[j].split else MUTE,
                        fontweight="bold")
    ax.set_title("Task-shape fingerprint matrix:\na holdout copied into dev lights up as a leak",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "leakage.png"), facecolor="white")
    plt.close(fig)
    print("wrote leakage.png")


if __name__ == "__main__":
    pipeline_svg()
    wilson_fig()
    leakage_fig()
