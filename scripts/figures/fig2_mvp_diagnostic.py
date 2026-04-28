"""Figure 2: MVP diagnostic metric trajectories (sigma/eta/KL over time).

These are the internal mechanism metrics that reveal WHY the MVP modifications
did not produce performance improvements in Figure 1.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import CONFIGS, CONFIG_LABELS, CONFIG_COLORS, load_tb

SUFFIX = __import__("os").environ.get("FIG_SUFFIX", "")
OUT_DIR = "/data/wendiyu/projects/iPLAN/results/figures"
plt.style.use("seaborn-paper")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.dpi": 120,
})

# Only MVP-enabled configs have these metrics
MVP_CONFIGS = [c for c in CONFIGS if c != "C0_baseline"]
ETA_CONFIGS = ["C2_eta_only", "C4_mvp_full"]  # only these have adaptive eta


def get_tb(cfg):
    return load_tb(cfg, 42)


tb_data = {cfg: get_tb(cfg) for cfg in MVP_CONFIGS}

fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
ax_sigma, ax_gap, ax_eta, ax_kl = axes.flatten()


# (a) sigma_mean
for cfg in MVP_CONFIGS:
    d = tb_data[cfg]
    if "mvp_sigma_mean" not in d:
        continue
    s, v = d["mvp_sigma_mean"]
    ax_sigma.plot(s, v, color=CONFIG_COLORS[cfg], linewidth=1.4,
                  label=CONFIG_LABELS[cfg])
ax_sigma.axhline(0.367, color="k", linestyle=":", linewidth=0.9, alpha=0.5)
ax_sigma.text(22000, 0.375, r"init $\sigma \approx 0.367$", fontsize=8, alpha=0.7)
ax_sigma.set_title("(a) Posterior std mean — "
                   r"$\sigma$ barely moves from init", loc="left")
ax_sigma.set_ylabel(r"$\sigma$ (mean over batch)")
ax_sigma.set_ylim(0.25, 0.42)
ax_sigma.grid(True, alpha=0.3)


# (b) sigma_p90 - sigma_p10 (specialization gap)
for cfg in MVP_CONFIGS:
    d = tb_data[cfg]
    if "mvp_sigma_p90" not in d or "mvp_sigma_p10" not in d:
        continue
    s90, v90 = d["mvp_sigma_p90"]
    s10, v10 = d["mvp_sigma_p10"]
    # Align (both should be same steps)
    n = min(len(v90), len(v10))
    gap = v90[:n] - v10[:n]
    ax_gap.plot(s90[:n], gap, color=CONFIG_COLORS[cfg], linewidth=1.4,
                label=CONFIG_LABELS[cfg])
ax_gap.axhline(0.30, color="green", linestyle="--", linewidth=0.9, alpha=0.6)
ax_gap.text(22000, 0.31, "healthy specialization (>0.30)",
            fontsize=8, color="green", alpha=0.7)
ax_gap.set_title("(b) Specialization gap " + r"$\sigma_{p90}-\sigma_{p10}$"
                 + " — no opponent-specific variance", loc="left")
ax_gap.set_ylabel(r"$\sigma_{p90} - \sigma_{p10}$")
ax_gap.set_ylim(0, 0.4)
ax_gap.grid(True, alpha=0.3)


# (c) eta_mean + eta_std (twinx for std)
for cfg in ETA_CONFIGS:
    d = tb_data[cfg]
    if "mvp_eta_mean" not in d:
        continue
    s, v = d["mvp_eta_mean"]
    ax_eta.plot(s, v, color=CONFIG_COLORS[cfg], linewidth=1.4,
                label=CONFIG_LABELS[cfg] + r" $\eta_\mathrm{mean}$")
ax_eta.axhline(0.10, color="k", linestyle=":", linewidth=0.9, alpha=0.5)
ax_eta.text(22000, 0.104, r"init $\eta = 0.10$", fontsize=8, alpha=0.7)

# Twin axis for eta_std
ax_eta_std = ax_eta.twinx()
for cfg in ETA_CONFIGS:
    d = tb_data[cfg]
    if "mvp_eta_std" not in d:
        continue
    s, v = d["mvp_eta_std"]
    ax_eta_std.plot(s, v, color=CONFIG_COLORS[cfg], linewidth=1.0,
                    linestyle="--", alpha=0.8)
ax_eta_std.set_ylabel(r"$\eta_\mathrm{std}$ (dashed)", color="#666666")
ax_eta_std.tick_params(axis="y", labelcolor="#666666")
ax_eta_std.set_ylim(0, 0.0005)

ax_eta.set_title("(c) Adaptive gain "
                 r"$\eta$ — mean drifts, std $\approx 0$ (no specialization)",
                 loc="left")
ax_eta.set_ylabel(r"$\eta_\mathrm{mean}$ (solid)")
ax_eta.set_ylim(0.0, 0.15)
ax_eta.grid(True, alpha=0.3)


# (d) kl_ib (times warmup for effective contribution)
for cfg in MVP_CONFIGS:
    d = tb_data[cfg]
    if "mvp_kl_ib" not in d:
        continue
    s, v = d["mvp_kl_ib"]
    ax_kl.plot(s, v, color=CONFIG_COLORS[cfg], linewidth=1.4,
               label=CONFIG_LABELS[cfg])
ax_kl.set_title(r"(d) IB KL divergence — all variants $\approx 5$"
                + ", " + r"$\beta$" + " too small to pull", loc="left")
ax_kl.set_ylabel(r"$\mathrm{KL}(q(z|x)\ \|\ \mathcal{N}(0,I))$")
ax_kl.set_ylim(4.4, 6.0)
ax_kl.grid(True, alpha=0.3)


for ax in (ax_eta, ax_kl):
    ax.set_xlabel("Environment step")

handles_sigma, labels_sigma = ax_sigma.get_legend_handles_labels()
fig.legend(handles_sigma, labels_sigma, loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 1.02), frameon=False)
fig.suptitle("Figure 2: MVP Diagnostic Metrics — "
             "Mechanism Failure Across All Variants",
             y=1.07, fontsize=11, weight="bold")

plt.tight_layout()
out_pdf = os.path.join(OUT_DIR, f"fig2_mvp_diagnostic{SUFFIX}.pdf")
out_png = os.path.join(OUT_DIR, f"fig2_mvp_diagnostic{SUFFIX}.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"saved {out_pdf}")
print(f"saved {out_png}")
