"""Publication figures. Run from the repo root:

    uv run python paper/figures/make_figures.py

Numbers are transcribed from the analyses recorded in the repo history
(commits 32417fb, 842bd70, and the iteration 5 sensitivity runs); each
figure's data block cites its source so the figure is auditable.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
INK = "#1a1e24"
MUTED = "#6b7280"
ACCENT = "#2563a8"
CONFIRM = "#b0413e"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="x", color="#d9dde3", linewidth=0.6)
    ax.set_axisbelow(True)


def artifact_arc():
    """Delta AUC as identity attribution was progressively cleaned.

    Stages 1 to 3: two-session in-sample estimates (is_weak, face+pose,
    n=218/213/186). Stage 4: the preregistered four-session test (n=292).
    """
    stages = [
        ("Polluted attribution\n(no identity checks)", 0.044, -0.028, 0.117, False),
        ("Distractors added\n(wrong positive ref remains)", 0.096, 0.005, 0.181, False),
        ("Verified enrollment\n(fully cleaned)", 0.009, -0.058, 0.074, False),
        ("Preregistered test\n(4 sessions, held out)", -0.075, -0.134, -0.020, True),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ys = range(len(stages), 0, -1)
    for y, (label, d, lo, hi, confirm) in zip(ys, stages):
        color = CONFIRM if confirm else ACCENT
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(d, y, "o", color=color, markersize=7, zorder=3)
        ax.annotate(
            f"{d:+.3f}", (d, y), textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=9, color=INK,
        )
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_yticks(list(ys))
    ax.set_yticklabels([s[0] for s in stages], fontsize=9)
    ax.set_xlabel("delta AUC, face and pose features over betting baseline (95% CI)", fontsize=9)
    style(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"artifact_arc.{ext}", dpi=200)
    plt.close(fig)


def sensitivity_forest():
    """Every arm of the iteration 5 sensitivity analysis (is_weak)."""
    arms = [
        ("Preregistered: L2, C=1.0 (n=292)", -0.075, -0.134, -0.020, True),
        ("L2, C=0.1", -0.066, -0.124, -0.014, False),
        ("L2, C=0.01", -0.115, -0.177, -0.051, False),
        ("L1, C=0.1", -0.154, -0.228, -0.084, False),
        ("L1, C=0.05", -0.147, -0.250, -0.041, False),
        ("Per player-session z-scores", -0.056, -0.115, -0.004, False),
        ("Within-person (Airball, n=280)", -0.069, -0.128, -0.009, False),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ys = range(len(arms), 0, -1)
    for y, (label, d, lo, hi, confirm) in zip(ys, arms):
        color = CONFIRM if confirm else ACCENT
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(d, y, "o", color=color, markersize=6, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_yticks(list(ys))
    ax.set_yticklabels([a[0] for a in arms], fontsize=9)
    ax.set_xlabel(
        "delta AUC over betting baseline, leave-one-session-out (95% CI)", fontsize=9
    )
    style(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"sensitivity_forest.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    artifact_arc()
    sensitivity_forest()
    print("wrote", sorted(p.name for p in OUT.glob("*.p*")))
