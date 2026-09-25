# NeuroForge

> 深度强化学习（Deep RL）智能体训练系统 · 随机组合顶级开源 · 离线可跑 · 一键复现
> 作者：晨星 ｜ 版本：v0.1.0

NeuroForge 是一套**可实际运行、模块可独立验证**的强化学习训练框架。它优先复用业界标杆开源成果，仅在无 SOTA 依赖时降级为纯 numpy 自研 DQN，确保**干净环境克隆 → 一键安装 → demo 跑通零手工干预**。

## 随机选型（本回合）

| 维度 | 选择 |
|------|------|
| AI 域 | 深度强化学习（Deep RL）智能体训练 |
| 环境 | **Gymnasium**（Farama，MIT） |
| SOTA 算法 | **Stable-Baselines3**（DLR-RM，MIT，PPO/DQN + PyTorch） |
| 超参搜索 | **Optuna**（Preferred Networks，MIT，贝叶斯优化） |
| 自研兜底 | 纯 numpy DQN（无框架依赖，零下载） |

## 架构

```
            ┌─────────────────────────────────────────────┐
            │                 NeuroForge                  │
            │            (runner: train/eval/bench/hpo)    │
            └───────┬───────────────┬───────────┬─────────┘
                    │               │           │
        ┌───────────┴───┐   ┌───────┴──────┐ ┌──┴───────────┐
        │ Environments  │   │   Agents    │ │   HPO        │
        │ factory(make) │   │ DQN(numpy)  │ │ OptunaSearch│
        │ Gymnasium     │   │ SB3(PPO/DQN)│ └──────────────┘
        └───────┬───────┘   └──────┬───────┘
                │                  │
        ┌───────┴───────┐   ┌──────┴──────┐
        │    Trainer    │◀──│  Evaluator  │
        │ 编排 env+agent│   │ N 轮 rollout│
        └───────────────┘   └─────────────┘
```

模块与接口（见 `docs/architecture.md`）：

| 模块 | 接口 | 实现 |
|------|------|------|
| 环境 | `gymnasium.Env` | `environments/factory.make_env` |
| 智能体 | `Agent`（train/predict/save/load） | `agents/dqn.DQNAgent`（兜底）、`agents/sb3.SB3Agent`（SOTA） |
| 训练 | `Trainer.train → TrainResult` | `training/trainer.Trainer` |
| 评测 | `Evaluator.evaluate → EvalResult` | `eval/evaluator.Evaluator` |
| 超参搜索 | `HyperparamSearch.search` | `hpo/searcher.OptunaSearcher` |

错误码：`E100` 环境 / `E200` 智能体 / `E300` 训练 / `E400` 评测 / `E500` HPO。

## 一键复现

```bash
git clone https://github.com/CJX0712/neuroforge.git
cd neuroforge
make install        # 建 .venv + 装核心与 SOTA 依赖
make demo          # 训练 CartPole 并评测（默认优先 SB3，缺失则 numpy DQN）
make test          # pytest 全模块单测
make benchmark     # DQN vs SB3(PPO) 对标，落 benchmark_report.json
```

Docker：

```bash
docker build -t neuroforge . && docker run --rm neuroforge
```

## 用法

```python
from neuroforge import NeuroForge, RLConfig

cfg = RLConfig(env_id="CartPole-v1", algo="auto", total_timesteps=30000, seed=42)
nf = NeuroForge.from_config(cfg)
nf.train()
print(nf.evaluate(n_episodes=20))
# 对标多种后端
print(nf.benchmark())
# 超参搜索
print(nf.hpo(n_trials=20))
```

CLI：

```bash
python -m neuroforge.cli train   --env CartPole-v1 --algo ppo --steps 30000
python -m neuroforge.cli eval    --env CartPole-v1 --eval-episodes 20
python -m neuroforge.cli benchmark
python -m neuroforge.cli hpo     --hpo-trials 20
```

## 文档

- `docs/architecture.md` — 架构、模块职责、接口契约、选型依据
- `docs/deployment.md` — 本地 / Docker / 依赖锁定与复现
- `docs/usage.md` — 训练、评测、对标、超参搜索详细示例

## 已知限制

- SB3 / Optuna / torch 为可选后端；未安装时系统自动降级为纯 numpy DQN（离线可跑，但非 SOTA）。
- 纯 numpy DQN 仅作轻量基线，样本效率远低于 SB3（PPO/DQN）。
- 默认环境为经典控制任务（CartPole 等），需 Atari/连续控制请自行接入对应 Gymnasium 注册表与环境依赖。
