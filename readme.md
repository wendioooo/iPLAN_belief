# iPLAN — Belief Representation & Belif Usage
This repository contains our CSCE 635 final project on belief integration in a MAPPO-style framework for heterogeneous highway driving.
This fork extends [**iPLAN**](https://arxiv.org/abs/2306.06236) (Wu et al., CoRL 2023).

This branch documents the **belief representation** side — i.e. how the latent is produced.
Downstream **belief utilization** (how the GAT, predictor, and policy heads consume the latent)

## Branch Guide

- `master`  
  Original iPLAN starting point.

- `mappo-base`  
  MAPPO baseline without belief integration.  
  
- `mappo-instant`  
  MAPPO with critic-side instant belief integration.  

- `mappo-st-instant`  
  MAPPO with ST-based instant belief integration.  
---


## Running

Full environment setup, GPU memory tuning, and common pitfalls live in
[MIGRATION.md](MIGRATION.md). Quick path:

```bash
conda create -n iplan python=3.9 -y && conda activate iplan
pip install torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install ./third_party/highway_env_fork
pip install -e .
```

Sanity check that the env registers:

```bash
python -c "import highway_env, gym; gym.make('highway-hetero-H-v0'); print('OK')"
```

100-step smoke run (~1 min on a small GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python main.py with \
  t_max=100 Behavior_warmup=50 GAT_warmup=50 seed=42 label=smoke
```

> The `env` and `difficulty` fields must be edited directly in
> [config/default.yaml](config/default.yaml). [main.py](main.py) reads them *before* sacred
> processes the CLI, so `with env=...` overrides do not switch the env.

4-way MVP ablation (writes to `results/sacred/`, `results/tb_logs/`, `results/models/` — all gitignored):

```bash
# Serial — recommended on shared GPUs (~1.8 GB peak)
GPU=0 BEHAVIOR_WARMUP=500 GAT_WARMUP=500 \
  bash scripts/run_mvp_ablation_serial.sh 42 highway chaotic 3000

# Parallel — needs 4 GPUs, each ≥4 GB free
GPUS="0 1 2 3" BEHAVIOR_WARMUP=500 GAT_WARMUP=500 \
  bash scripts/run_mvp_ablation.sh 42 highway chaotic 3000
```

---


## Acknowledgement

Forked from the authors' iPLAN repo, which itself builds on
[pymarl](https://github.com/oxwhirl/pymarl), [dm2](https://github.com/carolinewang01/dm2),
[epymarl](https://github.com/uoe-agents/epymarl) (MAPPO baseline),
[MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms) (G2ANet / GAT-RNN),
and [gin](https://github.com/usaywook/gin) (instant-incentive inference).

## Citation

```
@inproceedings{wu2023intent,
  title={Intent-Aware Planning in Heterogeneous Traffic via Distributed Multi-Agent Reinforcement Learning},
  author={Wu, Xiyang and Chandra, Rohan and Guan, Tianrui and Bedi, Amrit and Manocha, Dinesh},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```
