"""NeuroForge — 深度强化学习智能体训练系统（Deep RL training framework）。

随机组合顶级开源：Gymnasium（环境） + Stable-Baselines3（SOTA 算法） +
Optuna（超参搜索） + PyTorch（后端）。默认离线可跑（纯 numpy 自研 DQN 兜底），
SOTA 后端在依赖可用时自动启用。
"""

__version__ = "0.1.0"

from .core.config import RLConfig, SOLVE_THRESHOLDS
from .core.types import EpisodeResult, TrainResult, EvalResult
from .runner import NeuroForge

__all__ = [
    "RLConfig",
    "SOLVE_THRESHOLDS",
    "EpisodeResult",
    "TrainResult",
    "EvalResult",
    "NeuroForge",
    "__version__",
]
