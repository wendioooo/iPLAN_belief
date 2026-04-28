"""Figure 3: Final-state bar chart comparing all configs on navigation metrics."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import CONFIGS, CONFIG_LABELS, CONFIG_COLORS, load_all_logs

SUFFIX = __import__("os").environ.get("FIG_SUFFIX", "")
OUT_DIR = "/data/wendiyu/projects/iPLAN/results/figures"
plt.style.use("seaborn-paper")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 120,
})

# Final-bucket fraction (use last 25% of each run for smoother means)
BUCKET_FRAC = 0.25


def final_stats(traj, frac=BUCKET_FRAC):
    """Return means of (reward, win, len) over the final fraction of episodes."""
    n = len(traj["step"])
    cut = int(n * (1 - frac))
    return (
        float(np.mean(traj["reward"][cut:])),
        float(np.mean(traj["win"][cut:])),
        float(np.mean(traj["len"][cut:])),
    )


data = load_all_logs()

# Collect mean per (cfg, seed), then aggregate across seeds per cfg
per_cfg = {}
for cfg in CONFIGS:
    seed_vals = []
    for seed, traj in data[cfg].items():
        seed_vals.append(final_stats(traj))
    arr = np.array(seed_vals)  # shape: (n_seeds, 3)
    per_cfg[cfg] = {
        "reward_mean": arr[:, 0].mean(),
        "reward_std":  arr[:, 0].std(ddof=0),
        "win_mean":    arr[:, 1].mean(),
        "win_std":     arr[:, 1].std(ddof=0),
        "len_mean":    arr[:, 2].mean(),
        "len_std":     arr[:, 2].std(ddof=0),
        "n_seeds":     len(seed_vals),
    }

# Print for the record
print("Final-state stats (final 25% of each run):")
for cfg in CONFIGS:
    s = per_cfg[cfg]
    print(f"  {cfg:<22} n={s['n_seeds']}  reward={s['reward_mean']:.1f}±{s['reward_std']:.1f}  "
          f"win/5={s['win_mean']:.3f}±{s['win_std']:.3f}  len={s['len_mean']:.1f}±{s['len_std']:.1f}")
print()

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

metrics = [
    ("reward_mean", "reward_std", "Episodic reward (5 agents)", axes[0]),
    ("win_mean",    "win_std",    "Avg win / 5 agents",          axes[1]),
    ("len_mean",    "len_std",    "Avg episode length",          axes[2]),
]

x = np.arange(len(CONFIGS))
labels_short = ["C0\nbaseline", "C1\ngauss.", "C2\neta", "C3\nIB", "C4\nfull"]

for mean_key, std_key, ylabel, ax in metrics:
    means = [per_cfg[c][mean_key] for c in CONFIGS]
    stds = [per_cfg[c][std_key] for c in CONFIGS]
    colors = [CONFIG_COLORS[c] for c in CONFIGS]
    bars = ax.bar(x, means, yerr=stds, color=colors,
                  alpha=0.85, edgecolor="black", linewidth=0.7,
                  capsize=4, error_kw={"linewidth": 1.0})
    # Horizontal dashed baseline reference
    base = per_cfg["C0_baseline"][mean_key]
    ax.axhline(base, color="#444444", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate delta on top of each non-baseline bar
    for i, (c, m) in enumerate(zip(CONFIGS, means)):
        if c == "C0_baseline":
            continue
        delta = m - base
        sign = "+" if delta >= 0 else ""
        ax.text(i, m, f"{sign}{delta:.1f}",
                ha="center", va="bottom", fontsize=8,
                color="#222222")

# Scale: zoom y-axes so deltas are visible
axes[0].set_ylim(180, max(per_cfg[c]["reward_mean"] for c in CONFIGS) * 1.08)
axes[1].set_ylim(0, max(per_cfg[c]["win_mean"] for c in CONFIGS) * 1.15)
axes[2].set_ylim(50, max(per_cfg[c]["len_mean"] for c in CONFIGS) * 1.08)

axes[0].set_title("(a) Reward", loc="left")
axes[1].set_title("(b) Win count", loc="left")
axes[2].set_title("(c) Episode length", loc="left")

fig.suptitle("Figure 3: Final-State Navigation Metrics "
             "(mean over final 25% of training; error bars = seed std where available)",
             y=1.03, fontsize=11, weight="bold")

plt.tight_layout()
out_pdf = os.path.join(OUT_DIR, f"fig3_final_bar{SUFFIX}.pdf")
out_png = os.path.join(OUT_DIR, f"fig3_final_bar{SUFFIX}.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"saved {out_pdf}")
print(f"saved {out_png}")
