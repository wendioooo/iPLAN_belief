"""Shared data loader for figure scripts.
Pulls per-config trajectories from phase1 logs and TB events.
"""
import os
import re
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LOG_DIR = "/data/wendiyu/projects/iPLAN/results/phase1_logs"
TB_DIR = "/data/wendiyu/projects/iPLAN/results/tb_logs"

CONFIGS = ["C0_baseline", "C1_gaussian_only", "C2_eta_only", "C3_ib_only", "C4_mvp_full"]
CONFIG_LABELS = {
    "C0_baseline":      "C0  baseline (vanilla iPLAN)",
    "C1_gaussian_only": "C1  Gaussian encoder only",
    "C2_eta_only":      "C2  + adaptive Kalman (A)",
    "C3_ib_only":       "C3  + IB regularization (B)",
    "C4_mvp_full":      "C4  full MVP (A + B)",
}
CONFIG_COLORS = {
    "C0_baseline":      "#444444",
    "C1_gaussian_only": "#1f77b4",
    "C2_eta_only":      "#2ca02c",
    "C3_ib_only":       "#d62728",
    "C4_mvp_full":      "#9467bd",
}

EP_RE = re.compile(
    r"Episode # (\d+)\s+\|\s+Current time step:\s+(\d+)\s+\|\s+Average Episode Win Num:\s+([\d.]+)"
    r"\s+\|\s+Average Episode Reward:\s+([\d.\-]+)\s+\|\s+Average Episode Length:\s+([\d.]+)"
)
BL_RE = re.compile(r"Behavior Loss ([\d.\-]+)\s+\|\s+Stability Loss ([\d.\-]+)")
PL_RE = re.compile(r"Prediction Loss ([\d.\-]+)")


def load_log(path):
    """Returns dict with arrays for step / reward / win / len / b_loss / p_loss."""
    steps, rewards, wins, lens, blosses, plosses = [], [], [], [], [], []
    prev_bl = prev_pl = None
    with open(path) as fh:
        for line in fh:
            m_ep = EP_RE.search(line)
            if m_ep:
                steps.append(int(m_ep.group(2)))
                wins.append(float(m_ep.group(3)))
                rewards.append(float(m_ep.group(4)))
                lens.append(float(m_ep.group(5)))
                blosses.append(prev_bl if prev_bl is not None else np.nan)
                plosses.append(prev_pl if prev_pl is not None else np.nan)
                prev_bl = prev_pl = None
                continue
            m_bl = BL_RE.search(line)
            if m_bl:
                prev_bl = float(m_bl.group(1))
            m_pl = PL_RE.search(line)
            if m_pl:
                prev_pl = float(m_pl.group(1))
    return dict(
        step=np.array(steps),
        reward=np.array(rewards),
        win=np.array(wins),
        len=np.array(lens),
        b_loss=np.array(blosses, dtype=float),
        p_loss=np.array(plosses, dtype=float),
    )


def load_all_logs():
    """Returns nested dict: data[cfg] = {seed: trajectory_dict, ...}."""
    out = {cfg: {} for cfg in CONFIGS}
    for f in sorted(glob.glob(os.path.join(LOG_DIR, "C*.log"))):
        name = os.path.basename(f).replace(".log", "")
        if "C5" in name:
            continue
        for cfg in CONFIGS:
            if name.startswith(cfg + "-seed"):
                seed = int(name.split("-seed")[1])
                out[cfg][seed] = load_log(f)
                break
    return out


def load_tb(cfg, seed=42):
    """Returns dict tag -> (steps_array, values_array) for one config/seed."""
    pattern = os.path.join(TB_DIR, f"ippo_highway_{cfg.replace('_','-')}-seed{seed}_*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        return {}
    # Use the most recent TB dir for this run
    d = dirs[-1]
    ea = EventAccumulator(d)
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        evs = ea.Scalars(tag)
        if not evs:
            continue
        steps = np.array([e.step for e in evs])
        vals = np.array([e.value for e in evs])
        # Strip prefix
        short = tag
        for prefix in ["ippo_GAT_behavior_stable_H_", "ippo_"]:
            if short.startswith(prefix):
                short = short[len(prefix):]
        out[short] = (steps, vals)
    return out


def rolling_mean(x, w=15):
    """Causal rolling mean with edge padding."""
    if len(x) == 0:
        return x
    x = np.asarray(x, dtype=float)
    pad = np.full(w - 1, x[0])
    padded = np.concatenate([pad, x])
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")


if __name__ == "__main__":
    data = load_all_logs()
    print("=== Log data loaded ===")
    for cfg in CONFIGS:
        for seed, traj in data[cfg].items():
            n = len(traj["step"])
            last = traj["step"][-1] if n else 0
            print(f"  {cfg} seed={seed}: {n} episodes, latest step={last}")
    print()
    print("=== Sample TB metrics for C2_eta_only seed=42 ===")
    tb = load_tb("C2_eta_only", 42)
    for k in sorted(tb):
        if "mvp" in k:
            steps, vals = tb[k]
            print(f"  {k}: {len(vals)} pts, last={vals[-1]:.4f}")
