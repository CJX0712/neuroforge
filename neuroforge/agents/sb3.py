"""Stable-Baselines3 智能体封装（SOTA 后端）。

优先复用业界标杆 SB3（PPO / DQN + PyTorch）。若依赖未安装，则本模块整体不可用，
由 Trainer 自动降级为 DQNAgent（纯 numpy 兜底）。构造失败统一抛 AgentError(E200)。
"""

from __future__ import annotations

import time

from ..core.config import RLConfig
from ..core.errors import AgentError
from ..core.types import TrainResult
from .base import AgentBase

try:  # pragma: no cover - 依赖可选
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback

    _SB3_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖可选
    _SB3_AVAILABLE = False


if _SB3_AVAILABLE:

    class _RewardCurve(BaseCallback):
        """收集训练过程中的 episode 奖励曲线。"""

        def __init__(self) -> None:
            super().__init__()
            self.curve: list[float] = []

        def _on_step(self) -> bool:  # pragma: no cover - 运行时回调
            for info in self.locals.get("infos", []):
                if isinstance(info, dict) and "episode" in info:
                    self.curve.append(float(info["episode"]["r"]))
            return True

    class SB3Agent(AgentBase):
        """SB3（PPO/DQN）智能体。"""

        name = "sb3"

        def __init__(self, env_id: str, config: RLConfig) -> None:
            self.env_id = env_id
            self.cfg = config
            self.algo = "ppo" if config.algo in ("auto", "ppo") else "dqn"
            self.model = None

        def _algo_cls(self):
            return PPO if self.algo == "ppo" else DQN

        def train(self, env, config: RLConfig | None = None) -> TrainResult:
            cfg = config or self.cfg
            try:
                mon_env = Monitor(env)
            except Exception as exc:  # pragma: no cover - 包装异常
                raise AgentError(f"环境监控包装失败: {exc}", code="E200")
            t0 = time.time()
            cls = self._algo_cls()
            try:
                if self.algo == "ppo":
                    self.model = cls(
                        "MlpPolicy",
                        mon_env,
                        verbose=0,
                        n_steps=cfg.n_steps,
                        batch_size=cfg.batch_size,
                        learning_rate=cfg.learning_rate,
                        gamma=cfg.gamma,
                        seed=cfg.seed,
                    )
                else:
                    self.model = cls(
                        "MlpPolicy",
                        mon_env,
                        verbose=0,
                        learning_rate=cfg.learning_rate,
                        gamma=cfg.gamma,
                        buffer_size=cfg.buffer_size,
                        learning_starts=cfg.learning_starts,
                        batch_size=cfg.batch_size,
                        train_freq=cfg.train_freq,
                        target_update_interval=cfg.target_update_interval,
                        seed=cfg.seed,
                    )
                cb = _RewardCurve()
                self.model.learn(total_timesteps=cfg.total_timesteps, callback=cb)
            except Exception as exc:
                raise AgentError(f"SB3 训练失败: {exc}", code="E200")

            curve = list(cb.curve)
            mean_reward = float(sum(curve) / len(curve)) if curve else 0.0
            return TrainResult(
                algo=f"{self.algo}-sb3",
                env_id=cfg.env_id,
                total_timesteps=cfg.total_timesteps,
                reward_curve=curve,
                mean_episode_reward=mean_reward,
                train_seconds=time.time() - t0,
            )

        def predict(self, obs) -> int:
            if self.model is None:
                raise AgentError("模型未训练，无法预测", code="E200")
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)

        def save(self, path: str) -> None:
            if self.model is None:
                raise AgentError("模型未训练，无法保存", code="E200")
            self.model.save(path)

        def load(self, path: str) -> None:
            cls = self._algo_cls()
            try:
                self.model = cls.load(path)
            except Exception as exc:  # pragma: no cover - IO 异常
                raise AgentError(f"SB3 模型读取失败: {exc}", code="E200")

else:  # pragma: no cover - 依赖缺失分支

    class SB3Agent(AgentBase):  # type: ignore
        name = "sb3"

        def __init__(self, env_id: str, config: RLConfig) -> None:
            raise AgentError(
                "Stable-Baselines3 未安装，无法使用 SOTA 后端", code="E200"
            )
