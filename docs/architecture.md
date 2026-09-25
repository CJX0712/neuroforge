# NeuroForge 架构文档

## 1. 设计哲学

- **禁止从零自研核心算法**：优先复用 Gymnasium / Stable-Baselines3 / Optuna。
- **单一职责 + 接口化**：每个模块仅依赖 Protocol 接口，实现可替换、可独立验证。
- **离线兜底**：SOTA 后端（SB3/Optuna）缺失时，自动降级为纯 numpy DQN，保证 demo 零下载可跑。

## 2. 模块划分与接口契约

| 模块 | 文件 | 接口 | 输入 | 输出 | 错误码 |
|------|------|------|------|------|--------|
| 环境工厂 | `environments/factory.py` | `make_env(env_id, seed)` | env_id 字符串 | `gym.Env` | `E100` |
| 智能体（兜底） | `agents/dqn.py` | `Agent.train/predict/save/load` | env + config | `TrainResult` | `E200` |
| 智能体（SOTA） | `agents/sb3.py` | 同上 | env + config | `TrainResult` | `E200` |
| 训练编排 | `training/trainer.py` | `Trainer.train(config)` | `RLConfig` | `TrainResult` | `E300` |
| 评测 | `eval/evaluator.py` | `Evaluator.evaluate(agent, env_id, n)` | agent + env_id | `EvalResult` | `E400` |
| 超参搜索 | `hpo/searcher.py` | `OptunaSearcher.search(config, n)` | `RLConfig` | dict | `E500` |
| 高层入口 | `runner.py` | `NeuroForge.train/evaluate/benchmark/hpo` | `RLConfig` | 结果对象 | 透传 |

数据类型：`EpisodeResult` / `TrainResult` / `EvalResult`（`core/types.py`）；错误码 `E100~E600`（`core/errors.py`）。

## 3. 调用关系（单向无环）

```
NeuroForge → Trainer → (make_env + Agent.train) → Evaluator.evaluate
                      ↘ OptunaSearcher（独立 trial 内复用 Trainer）
```

`Trainer` 负责：装配环境 → 按 config 选择 Agent（SB3 优先，否则 DQN）→ 训练 → 评测 → 存档（`<model_dir>/<env>_<agent>.zip`）。

## 4. 关键选型依据

| 维度 | 选定（理由） | 候选 | 说明 |
|------|--------------|------|------|
| 环境标准 | **Gymnasium**(MIT) | 旧版 Gym / 自研 | 社区事实标准、活跃维护、API 稳定 |
| SOTA 算法 | **Stable-Baselines3**(MIT, PyTorch) | RLlib / Tianshou | 文档完善、PPO/DQN 开箱即用、MIT |
| 超参搜索 | **Optuna**(MIT) | Ray Tune | 轻量、贝叶斯优化、易嵌入 |
| 兜底实现 | 纯 numpy DQN | — | 零依赖、零下载，保证离线可跑 |

## 5. 自研成分说明

仅 `agents/dqn.py`（纯 numpy DQN）为自研，理由：在 SB3/torch 不可用时提供**确定性、零下载、可单测**的兜底训练能力，保证 DoD。其接口与 SB3Agent 完全一致，可随时被 SOTA 后端替换。

## 6. 可独立验证

- `tests/test_dqn.py`：DQN 训练后奖励显著提升、预测动作合法、save/load 等价。
- `tests/test_sb3.py`：SB3(PPO) 训练后 final_eval_reward > 30（importorskip 隔离）。
- `tests/test_optuna.py`：Optuna 搜索返回 best_params（importorskip 隔离）。
- `tests/test_evaluator.py` / `test_trainer.py` / `test_env.py`：管线各段正确性。
