"""Figure 4: Causal chain diagram of MVP mechanism failure."""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SUFFIX = __import__("os").environ.get("FIG_SUFFIX", "")
OUT_DIR = "/data/wendiyu/projects/iPLAN/results/figures"
plt.style.use("seaborn-paper")
plt.rcParams.update({"font.size": 10, "figure.dpi": 120})


def box(ax, xy, w, h, text, fc, ec="black", fontsize=9, fontweight="normal"):
    cx, cy = xy
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.03",
        facecolor=fc, edgecolor=ec, linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, wrap=True)


def arrow(ax, start, end, color="black", style="->", lw=1.2):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14,
                            color=color, linewidth=lw,
                            connectionstyle="arc3,rad=0")
    ax.add_patch(patch)


fig, ax = plt.subplots(figsize=(11, 7.5))

# Colors
C_CAUSE = "#fde0dc"    # pink - root cause
C_MECH  = "#dceefc"    # light blue - intermediate
C_OUT   = "#e8d8f0"    # light purple - output/effect
C_EMPIRICAL = "#555555"

# --- Layer 1: Two root causes ---
box(ax, (2.5, 9.0), 3.6, 1.1,
    "Root cause 1\n"
    r"IB weight $\beta = 0.01$ too small"
    "\n(KL contribution ~0.05% of loss)",
    fc=C_CAUSE, fontsize=9, fontweight="bold")

box(ax, (8.5, 9.0), 3.6, 1.1,
    "Root cause 2\n"
    "Behavior loss is L1 reconstruction\n"
    r"$\partial \mathrm{loss} / \partial \sigma \approx 0$",
    fc=C_CAUSE, fontsize=9, fontweight="bold")

# --- Layer 2: sigma stays at init ---
box(ax, (5.5, 7.2), 5.8, 1.0,
    r"$\sigma$ stays near init (~0.35)"
    "  ·  specialization gap " + r"$\sigma_{p90}-\sigma_{p10}\approx 0.05$"
    "\n" + r"(empirical: Fig 2a, 2b)",
    fc=C_MECH, fontsize=9)

# --- Layer 3: Two downstream mechanisms ---
box(ax, (2.3, 5.3), 4.2, 1.1,
    "logvar near-constant across opponents\n"
    r"$\Rightarrow$ eta_mlp(logvar) output is flat"
    "\n" r"(empirical: eta_std $\approx 7\mathrm{e}{-}5$)",
    fc=C_MECH, fontsize=9)

box(ax, (8.7, 5.3), 4.2, 1.1,
    "Reparameterized z degenerates to\n"
    r"$\mu + 0.35\cdot\varepsilon$ (fixed noise)"
    "\n" + "(no opponent-specific uncertainty)",
    fc=C_MECH, fontsize=9)

# --- Layer 4: Mechanism collapse ---
box(ax, (5.5, 3.3), 6.5, 1.1,
    "Adaptive Kalman gain "
    r"$\eta=\sigma(\mathrm{MLP}(\mathrm{logvar}))$ degenerates"
    "\n" + r"into a constant $\approx 0.077$ "
    "(empirical: Fig 2c)",
    fc=C_MECH, fontsize=9)

# --- Layer 5: Outcome ---
box(ax, (5.5, 1.3), 7.2, 1.1,
    "All 4 MVP variants (C1-C4) "
    r"$\approx$" + " C0 baseline\n"
    "(|" + r"$\Delta$" + "reward| within seed variance — Figs 1, 3)",
    fc=C_OUT, fontsize=10, fontweight="bold")


# --- Arrows ---
# Layer 1 -> Layer 2
arrow(ax, (2.5, 8.45), (4.8, 7.78))
arrow(ax, (8.5, 8.45), (6.5, 7.78))

# Layer 2 -> Layer 3
arrow(ax, (4.5, 6.7), (2.8, 5.90))
arrow(ax, (6.5, 6.7), (8.2, 5.90))

# Layer 3 -> Layer 4
arrow(ax, (2.8, 4.75), (4.5, 3.90))
arrow(ax, (8.2, 4.75), (6.5, 3.90))

# Layer 4 -> Layer 5
arrow(ax, (5.5, 2.75), (5.5, 1.88), lw=1.6)

# Right-side annotation for Laplace NLL corrective attempt
ax.text(12.2, 9.0,
        "Corrective\nexperiment (C5)",
        fontsize=9, ha="center", va="center", fontweight="bold",
        color="#bb6000")
rect = mpatches.FancyBboxPatch((10.8, 4.3), 2.8, 3.5,
                                boxstyle="round,pad=0.03",
                                facecolor="#fff3e0",
                                edgecolor="#bb6000",
                                linewidth=1.0, linestyle="--")
ax.add_patch(rect)
ax.text(12.2, 7.3,
        "C5: Replace L1\nwith Laplace NLL\n"
        "+ " + r"$\beta$" + " = 0.1\n\n"
        "Attempts to fix\nRoot cause 2",
        fontsize=8.5, ha="center", va="center")
ax.text(12.2, 5.5,
        "Observed:\n"
        "NLL drives " + r"$\sigma$" + " to\n"
        "clamp floor;\n"
        "behavior_loss\n"
        "goes negative",
        fontsize=8, ha="center", va="center", style="italic",
        color="#5a2900")

# Title / layout
ax.set_xlim(0, 14)
ax.set_ylim(0, 10.2)
ax.axis("off")
fig.suptitle("Figure 4: Causal Chain of MVP Mechanism Failure",
             fontsize=12, weight="bold", y=0.99)

plt.tight_layout()
out_pdf = os.path.join(OUT_DIR, f"fig4_causal_chain{SUFFIX}.pdf")
out_png = os.path.join(OUT_DIR, f"fig4_causal_chain{SUFFIX}.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"saved {out_pdf}")
print(f"saved {out_png}")
