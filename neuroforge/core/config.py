"""NeuroForge 运行配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# 各环境的“解题阈值”：单 episode 累计奖励达到该值即记为 solved。
SOLVE_THRESHOLDS = {
    "CartPole-v1": 195.0,
    "MountainCar-v0": -110.0,
    "Acrobot-v1": -100.0,
    "Pendulum-v1": -200.0,
}


@dataclass
class RLConfig:
    """强化学习训练配置（全部可锁定、可复现）。"""

    env_id: str = "CartPole-v1"
    algo: str = "auto"  # auto | ppo | dqn | dqn-np
    total_timesteps: int = 30000
    seed: int = 42
    use_sb3: bool = True  # 是否优先使用 Stable-Baselines3（SOTA 后端）
    hpo: bool = False
    hpo_trials: int = 20
    eval_episodes: int = 10
    model_dir: str = "models"
    log_dir: str = "logs"
    # 超参数
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    train_freq: int = 4
    target_update_interval: int = 500
    net_arch: List[int] = field(default_factory=lambda: [64, 64])
    n_steps: int = 1_024

    @property
    def solve_threshold(self) -> Optional[float]:
        return SOLVE_THRESHOLDS.get(self.env_id)
