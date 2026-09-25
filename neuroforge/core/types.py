"""NeuroForge 共享数据类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EpisodeResult:
    """单轮 episode 结果。"""

    episode: int
    total_reward: float
    steps: int
    solved: bool = False


@dataclass
class TrainResult:
    """训练结果。"""

    algo: str
    env_id: str
    total_timesteps: int
    reward_curve: List[float] = field(default_factory=list)  # 每 episode 累计奖励
    mean_episode_reward: float = 0.0
    final_eval_reward: float = 0.0
    train_seconds: float = 0.0
    model_path: str = ""


@dataclass
class EvalResult:
    """评测结果。"""

    env_id: str
    n_episodes: int
    mean_reward: float
    std_reward: float
    solved_rate: float
    mean_steps: float
