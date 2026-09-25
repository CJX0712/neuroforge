"""环境工厂：基于 Gymnasium 创建与校验环境。"""

from __future__ import annotations

import gymnasium as gym

from ..core.errors import EnvError

# 无需额外依赖即可运行的环境白名单（经典控制任务）。
KNOWN_ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"]


def make_env(env_id: str, seed: int | None = None) -> gym.Env:
    """创建环境；失败统一抛 EnvError(E100)。"""
    try:
        env = gym.make(env_id)
    except Exception as exc:  # pragma: no cover - 依赖/注册表异常
        raise EnvError(f"无法创建环境 {env_id!r}: {exc}", code="E100")
    if seed is not None:
        try:
            env.reset(seed=seed)
        except Exception:  # pragma: no cover - 部分环境不支持 seed 透传
            pass
    return env


def available_envs() -> list[str]:
    """返回可用的经典控制环境列表。"""
    return list(KNOWN_ENVS)
