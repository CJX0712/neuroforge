# NeuroForge 使用文档

## 1. 最小可运行示例

```bash
python -m neuroforge.examples.run_demo
```

默认训练 `CartPole-v1`（20000 步），优先 SB3(PPO)，缺失则 numpy DQN，输出训练/评测结果。

## 2. Python API

```python
from neuroforge import NeuroForge, RLConfig

cfg = RLConfig(env_id="CartPole-v1", algo="auto", total_timesteps=30000, seed=42)

nf = NeuroForge.from_config(cfg)
train_res = nf.train()
print(train_res.algo, train_res.final_eval_reward, train_res.model_path)

eval_res = nf.evaluate(n_episodes=20)
print(eval_res.mean_reward, eval_res.solved_rate)

# 多后端对标
for algo, res in nf.benchmark().items():
    print(algo, res.final_eval_reward)

# 超参搜索（Optuna，轻量 DQN）
print(nf.hpo(n_trials=20))
```

## 3. 命令行

```bash
# 训练（指定算法 / 步数）
python -m neuroforge.cli train --env CartPole-v1 --algo ppo --steps 30000
# 仅用 numpy DQN（不调用 SB3）
python -m neuroforge.cli train --env CartPole-v1 --algo dqn-np --no-sb3
# 评测
python -m neuroforge.cli eval --eval-episodes 20
# 对标
python -m neuroforge.cli benchmark
# 超参搜索
python -m neuroforge.cli hpo --hpo-trials 20
```

## 4. 接入新环境

Gymnasium 注册的环境可直接使用：

```python
cfg = RLConfig(env_id="MountainCar-v0", algo="dqn-np", total_timesteps=50000)
NeuroForge.from_config(cfg).train()
```

> Atari / 连续控制 / 自定义环境请按 Gymnasium 规范注册，并安装对应依赖（如 `gymnasium[atari]`、`pybox2d` 等）。

## 5. 结果解读

- `TrainResult.reward_curve`：每 episode 累计奖励（用于绘制学习曲线）。
- `EvalResult`：`mean_reward` / `std_reward` / `solved_rate`（达 `SOLVE_THRESHOLDS` 比例）/ `mean_steps`。
- `benchmark_report.json`：各后端 final_eval_reward 与训练耗时，用于对标。
