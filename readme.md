# iPLAN — Belief Representation & Belif Usage

This fork extends [**iPLAN**](https://arxiv.org/abs/2306.06236) (Wu et al., CoRL 2023).

This branch documents the **belief representation** side — i.e. how the latent is produced.
Downstream **belief usage** (how the GAT, predictor, and policy heads consume the latent)
lives on the `belief-usage` branch.

---

## What we modified

In iPLAN, the "belief" is the per-vehicle latent that an EncoderRNN distills from each
observed neighbor's recent history; iPLAN's Eq. (3) then EMA-updates this latent with a
constant gain. Our changes touch only the encoder / posterior / latent-update path:

| File | Change |
|------|--------|
| [nova/mvp_utils.py](nova/mvp_utils.py) | New file. `reparameterize`, `kl_standard_normal` (with free bits), and `AdaptiveEtaMLP` (η = sigmoid(MLP(log σ²))). |
| [nova/behavior_net.py](nova/behavior_net.py) | `EncoderRNN` gains an optional `gaussian=True` mode that emits (μ, log σ²) via a parallel `logvar_head` instead of a softmax categorical head. |
| [nova/stable_behavior_policy.py](nova/stable_behavior_policy.py) | `__init__`, `latent_update`, and `learn` gain an MVP branch: Gaussian encoder → reparameterized sample → IB KL term → adaptive Kalman gain replaces the constant-η EMA. |


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

## Belief usage

The downstream consumers of the belief latent — `GAT_Net`, the instant-incentive predictor,
and how each is wired into the controller / learner — are documented on the `belief-usage`
branch:

```bash
git checkout belief-usage
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
