# iPLAN 迁移与环境配置指南

在新服务器上从零复现本 repo 的可运行状态。包含作者原版论文代码、我们的 MVP 改造、和所有已踩过的坑的解决方案。

---

## 0 · 一句话 TL;DR

```bash
# 1. 拷贝整个 repo
rsync -av /path/to/iPLAN/ new-server:/path/to/iPLAN/

# 2. 在新服务器上
cd /path/to/iPLAN
conda create -n iplan python=3.9 -y
conda activate iplan
pip install -r requirements.txt
pip install torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu121  # 选合适的 cuda
pip install ./third_party/highway_env_fork   # 作者 fork，已捆绑进 repo
pip install -e .                              # iPLAN 本体
python -c "import highway_env, gym; gym.make('highway-hetero-H-v0'); print('OK')"
```

---

## 1 · 要复制什么

整个 `iPLAN/` 目录都带走。关键目录：

| 目录/文件 | 作用 | 必须 |
|---|---|---|
| `main.py`, `run_ippo.py` | 入口 + 训练主循环 | ✅ |
| `nova/` | behavior / GAT / MVP 新模块 | ✅ |
| `nova/mvp_utils.py` | **我加的** — MVP 工具函数 | ✅ |
| `config/` | 默认 + 环境 + 算法 yaml | ✅ |
| `controllers/`, `learners/`, `modules/`, `runners/`, `components/`, `utils/` | 原版 iPLAN 架构 | ✅ |
| `envs/` | iPLAN 自己的 MPE / 环境 wrapper（**不是** highway_env） | ✅ |
| `third_party/highway_env_fork/` | **我捆绑的** — 作者 GitHub fork + 修过的 setup.py | ✅ |
| `scripts/` | 我写的 ablation 启动脚本 | ✅ |
| `MIGRATION.md` (本文件) | 这个文档 | 建议 |
| `readme.md` | 作者原版 README | 建议 |
| `results/` | 训练输出（可选，大） | 可选 |
| `animation/` | 作者预录的示例 GIF（大） | 可选 |

**不用复制**:
- `__pycache__/`, `*.pyc` — Python 字节码缓存
- `results/sacred/_sources/` — Sacred 自动存的代码快照（巨大）
- `iPLAN.egg-info/` — 安装生成物
- 已训练的 checkpoint（除非你想继续训练）

推荐 rsync 命令：

```bash
rsync -av --progress \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='results/sacred/_sources' \
  --exclude='results/models' \
  --exclude='iPLAN.egg-info' \
  --exclude='.git' \
  /home/wendiyu/projects/iPLAN/ \
  user@new-server:/path/to/iPLAN/
```

如果要继续训练已有模型，去掉 `--exclude='results/models'`。

---

## 2 · 环境要求

### 系统
- Linux
- Python **3.9**（作者代码在 3.9 下开发；3.10+ 可能遇到废弃 API）
- CUDA 11.8+ 或 12.x（如果要 GPU）
- ≥16 GB 系统 RAM
- **单卡 ≥4 GB GPU 内存** ← 重要，下文有说明

### 核心 Python 依赖

直接装 `requirements.txt` 会缺 torch，单独装：

```bash
# 推荐顺序
conda create -n iplan python=3.9 -y
conda activate iplan

# 1. torch — 按你 CUDA 版本选
pip install torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu121
# 或 CPU-only:
# pip install torch==2.8.0

# 2. iPLAN 依赖
pip install -r requirements.txt

# 3. 作者 highway fork（已捆绑进 repo）
pip install ./third_party/highway_env_fork

# 4. iPLAN 本体（把项目注册到 site-packages）
pip install -e .
```

### 版本锁定（已验证能跑的组合）

```
python    = 3.9.x
torch     = 2.8.0+cu128
numpy     = 1.21.6
gym       = 0.26.2
sacred    = 0.8.2
pyyaml    = 6.0
tensorboard = 2.11.0
pygame    = 2.1.2
pandas    = 1.3.5
scipy     = 1.7.3
matplotlib = 3.5.3
```

---

## 3 · 作者 highway_env fork 的坑（最关键）

**这是最费时间的坑**。作者在 README 里说 `pip install Heterogeneous_Highway_Env`，**但 PyPI 上的这个包是坏的**。内部 import 写的是 `from highway_env.envs... import *` 但 PyPI 把它装成了 `Heterogeneous_Highway_Env` 目录下，导致 `ModuleNotFoundError: highway_env`。

### 我的解决方案

把作者的 GitHub fork 捆绑进了 `third_party/highway_env_fork/`，配了一个自动把文件 stage 到 `highway_env/` 目录下的 `setup.py`。**直接 pip install 就会正确注册 `highway_env` 模块和 gym env**。

### 验证安装成功

```bash
python -c "
import highway_env
print('highway_env file:', highway_env.__file__)
import gym
env = gym.make('highway-hetero-H-v0')
print('env created OK:', type(env).__name__)
from highway_env.envs.highway_env import HighwayEnvHetero_H
from highway_env.vehicle.behavior import AggressiveVehicle, DefensiveVehicle
print('hetero classes OK')
"
```

期望输出:
```
highway_env file: /.../site-packages/highway_env/__init__.py
env created OK: OrderEnforcing
hetero classes OK
```

### 如果还有老包残留

旧版安装可能在 `site-packages/` 里留下这些幽灵目录（必须清掉）：

```bash
SP=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "$SP/Heterogeneous_Highway_Env"       # 坏的 PyPI 包
rm -rf "$SP/Heteogeneous_Highway_Env"        # 错别字版本
rm -rf "$SP/HighwayEnv_iPLAN"                # 另一个变体
rm -rf "$SP/envs"                            # 遗留空 envs 包（会 shadow iPLAN 自己的 envs/）
pip uninstall -y Heterogeneous_Highway_Env HighwayEnv_iPLAN 2>/dev/null || true
```

**特别注意 `site-packages/envs/`**：旧版安装会把一个独立的 `envs/` 目录塞进 site-packages，它会把 iPLAN 项目内的本地 `envs/` 目录**完全遮蔽**，导致 `ModuleNotFoundError: No module named 'envs.env_wrappers'`。必须删掉。

---

## 4 · 运行前的快速验证

```bash
cd /path/to/iPLAN

# (a) 所有关键模块能 import
python -c "
import torch
import highway_env
import gym
from nova.mvp_utils import AdaptiveEtaMLP, kl_standard_normal, reparameterize
from nova.stable_behavior_policy import Behavior_policy
from nova.behavior_net import EncoderRNN
from envs.env_wrappers import SubprocVecEnv
from envs.mpe.MPE_env import MPEEnv
print('all imports OK')
"

# (b) gym env 能创建
python -c "import gym; gym.make('highway-hetero-H-v0'); print('gym env OK')"

# (c) 跑一个 100 步的 smoke run
CUDA_VISIBLE_DEVICES=0 python main.py with env=highway difficulty=chaotic \
  t_max=100 Behavior_warmup=50 GAT_warmup=50 seed=42 \
  label=install_test
```

如果 (c) 能跑完，打印 `Finished Training`，说明环境就位。

---

## 5 · GPU 内存要求（重要）

iPLAN 训练在 highway chaotic 环境下**单进程峰值 ~3.1 GB GPU 内存**（PyTorch 2.8 + 默认 `batch_size_run=8`）。

### 如果 GPU 内存紧张

在共享 GPU 上常见单卡只剩 < 3 GB 可用。解决方法：

**方法 1（已在 `scripts/run_mvp_ablation_serial.sh` 里默认启用）**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py with batch_size_run=4 num_test_episodes=4 ...
```
- `expandable_segments` 减少内存碎片
- `batch_size_run=4` 把并发环境 worker 砍半（默认 8）
- 峰值内存降到 ~1.8 GB

**方法 2（极限节省，性能代价大）**:
```bash
python main.py with batch_size_run=2 num_test_episodes=2 ...
```

**方法 3（CPU 回退）**:
在 [config/default.yaml](config/default.yaml) 里 `use_cuda: False`。慢很多，但无内存限制。

---

## 6 · 常见陷阱对照表

| 症状 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: highway_env` | 没装 fork 包，或装了坏的 PyPI 版 | `pip install third_party/highway_env_fork/` |
| `NameNotFound: Environment highway-hetero-H doesn't exist` | 包装上但没触发 gym register | 确认 `import highway_env` 成功，检查 site-packages 有没有幽灵 `envs/` |
| `ModuleNotFoundError: envs.env_wrappers` | site-packages 里有 shadow `envs/` | 删除 site-packages 里的 `envs/` 目录 |
| `ModuleNotFoundError: numpy`（python3 执行时） | 调用的是系统 `python3`，不是 conda 的 `python` | 用 `python` 而非 `python3`；或 `source activate iplan` 后再跑 |
| `torch.OutOfMemoryError` | GPU 内存不够 | 见 §5 |
| Sacred `with` 语法失败 | list/dict 需单引号包裹整个字符串 | `checkpoint_paths='["path"]'` |
| pygame/SDL 报错（无头服务器） | 没有显示 | `export SDL_VIDEODRIVER=dummy` 或 `xvfb-run -a python main.py ...` |

---

## 7 · 我们的 MVP 改造（Direction A+B）

本 repo 在作者原版基础上加了 **Information Bottleneck + 自适应 Kalman 增益** 的增量改造。所有改动都被 `mvp_enable` flag 守卫：

- `mvp_enable=False`（默认）→ 代码路径与原 iPLAN **bit-identical**
- `mvp_enable=True` → 启用 Gaussian encoder + VIB KL + adaptive η

### 相关文件

| 文件 | 改动 |
|---|---|
| [nova/mvp_utils.py](nova/mvp_utils.py) | 新文件：`reparameterize`, `kl_standard_normal`, `AdaptiveEtaMLP` |
| [nova/behavior_net.py](nova/behavior_net.py) | `EncoderRNN` 增加可选 `gaussian=True` 模式 |
| [nova/stable_behavior_policy.py](nova/stable_behavior_policy.py) | `__init__` / `latent_update` / `learn` 增加 MVP 分支 |
| [config/default.yaml](config/default.yaml) | 新增 7 个 `mvp_*` flag（默认全关） |
| [scripts/run_mvp_ablation.sh](scripts/run_mvp_ablation.sh) | 4-way 并行 ablation |
| [scripts/run_mvp_ablation_serial.sh](scripts/run_mvp_ablation_serial.sh) | 4-way 串行 ablation（低 GPU 内存） |

### 配置 flag

```yaml
mvp_enable: False                # 总开关
mvp_ib_kl_weight: 0.01           # Tishby β
mvp_ib_kl_warmup_steps: 50000
mvp_free_bits: 0.05              # 防 posterior collapse
mvp_adaptive_eta: True           # True: eta = sigmoid(MLP(logvar)); False: 用 soft_update_coef
mvp_eta_mlp_hidden: 16
mvp_use_sample_rollout: False    # rollout 用 mean (False) 还是 sample (True)
```

### 4-way ablation

```bash
# 串行版（推荐，低 GPU 内存）
GPU=0 BEHAVIOR_WARMUP=500 GAT_WARMUP=500 \
  bash scripts/run_mvp_ablation_serial.sh 42 highway chaotic 3000

# 并行版（需要 4 张 GPU，每张 ≥4 GB 空闲）
GPUS="0 1 2 3" BEHAVIOR_WARMUP=500 GAT_WARMUP=500 \
  bash scripts/run_mvp_ablation.sh 42 highway chaotic 3000
```

4 个配置:
1. `baseline`：`mvp_enable=False`（原 iPLAN）
2. `ib_only`：Gaussian encoder + IB KL，保留常数 η
3. `eta_only`：Gaussian encoder + adaptive η，不加 IB
4. `mvp_full`：两者都开

---

## 8 · 默认 config 的改动

[config/default.yaml](config/default.yaml) 相对原版有两处修改:

1. `env: "highway"` (原版 `"MPE"`) — 方便直接跑 highway
2. `difficulty: "chaotic"` (原版 `"easy"`) — 默认跑最难场景

**重要**：`main.py` 第 86 行在 sacred CLI 解析**之前**就根据 `env` 字段决定加载哪个 env config 文件。所以**必须修改 default.yaml 的 `env` 字段**才能切换，`with env=X` 这种 CLI override **不起作用**。

如果要跑 MPE，改回 `env: "MPE"`。

---

## 9 · 迁移后的第一件事

```bash
# 1. 验证 import 链
python -c "
import highway_env, gym
from nova.mvp_utils import AdaptiveEtaMLP
from nova.stable_behavior_policy import Behavior_policy
print('imports OK')
gym.make('highway-hetero-H-v0')
print('gym env OK')
"

# 2. 跑 100 步 smoke test
CUDA_VISIBLE_DEVICES=0 python main.py with \
  t_max=100 Behavior_warmup=50 GAT_warmup=50 \
  seed=42 label=migration_test

# 3. 确认 results/ 和 tb_logs/ 生成
ls results/sacred/ results/tb_logs/ results/models/

# 4. 跑 4-way ablation smoke（串行版，~36 min）
GPU=0 BEHAVIOR_WARMUP=500 GAT_WARMUP=500 \
  bash scripts/run_mvp_ablation_serial.sh 42 highway chaotic 3000

# 5. 对比 TB 日志
tensorboard --logdir results/tb_logs --port 6006
```

如果 (4) 跑通，说明迁移完成且 MVP 代码路径正确。

---

## 10 · 联系/参考

- 原论文: https://arxiv.org/abs/2306.06236
- 作者原 repo: https://github.com/wuxiyang1996/iPLAN
- highway-env fork 源: https://github.com/wuxiyang1996/Heterogeneous_Highway_Env
- 本 repo 的所有 MVP 改造都在 `nova/mvp_utils.py`, `nova/behavior_net.py`, `nova/stable_behavior_policy.py`, `config/default.yaml` 的 `mvp_*` flag 之下
