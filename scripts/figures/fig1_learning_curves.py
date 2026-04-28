"""Figure 1: Learning curves for all 5 configs over env steps."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (
    CONFIGS, CONFIG_LABELS, CONFIG_COLORS, load_all_logs, rolling_mean,
)

SUFFIX = __import__("os").environ.get("FIG_SUFFIX", "")
OUT_DIR = "/data/wendiyu/projects/iPLAN/results/figures"
plt.style.use("seaborn-paper")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
})

WINDOW = 15


def plot_traj(ax, cfg, metric_key, label_shown):
    seeds = sorted(data[cfg].keys())
    color = CONFIG_COLORS[cfg]
    if len(seeds) >= 2:
        # Mean + shaded range of two seeds (align on common step grid)
        traj_list = []
        for seed in seeds:
            t = data[cfg][seed]
            y = rolling_mean(t[metric_key], WINDOW)
            traj_list.append((t["step"], y))
        # Interpolate to common grid
        lo = max(tr[0][0] for tr in traj_list)
        hi = min(tr[0][-1] for tr in traj_list)
        grid = np.linspace(lo, hi, 400)
        ys = np.stack([np.interp(grid, tr[0], tr[1]) for tr in traj_list])
        mean_y = ys.mean(axis=0)
        min_y = ys.min(axis=0)
        max_y = ys.max(axis=0)
        ax.fill_between(grid, min_y, max_y, color=color, alpha=0.18, linewidth=0)
        ax.plot(grid, mean_y, color=color, linewidth=1.6,
                label=CONFIG_LABELS[cfg] if label_shown else None)
    else:
        seed = seeds[0]
        t = data[cfg][seed]
        y = rolling_mean(t[metric_key], WINDOW)
        ax.plot(t["step"], y, color=color, linewidth=1.5,
                label=CONFIG_LABELS[cfg] if label_shown else None)


data = load_all_logs()

fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
axes = axes.flatten()

panels = [
    ("reward", "Episodic reward (5 agents summed)", "(a) Episodic Reward", axes[0]),
    ("win",    "Avg win count / 5 agents",           "(b) Win count (success proxy)", axes[1]),
    ("len",    "Avg episode length (max=90)",        "(c) Episode length",     axes[2]),
    ("b_loss", "Behavior reconstruction loss (L1)",  "(d) Behavior loss",      axes[3]),
]

for metric_key, ylabel, title, ax in panels:
    for i, cfg in enumerate(CONFIGS):
        plot_traj(ax, cfg, metric_key, label_shown=(metric_key == "reward"))
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

for ax in axes[2:]:
    ax.set_xlabel("Environment step")

# Shared legend above the figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.02), frameon=False)

fig.suptitle("Figure 1: Learning Curves — 5 Ablation Configurations in Chaotic Heterogeneous Highway",
             y=1.06, fontsize=11, weight="bold")

plt.tight_layout()
out_pdf = os.path.join(OUT_DIR, f"fig1_learning_curves{SUFFIX}.pdf")
out_png = os.path.join(OUT_DIR, f"fig1_learning_curves{SUFFIX}.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"saved {out_pdf}")
print(f"saved {out_png}")
