"""从零实现的 DQN 智能体（纯 numpy，零外部依赖，作为离线兜底后端）。

设计说明：不依赖任何深度学习框架，仅用 numpy 实现单层隐藏层 MLP + 经验回放 +
目标网络 + ε-贪婪。用于在无 PyTorch/SB3 环境下保证系统可运行；同时也作为
轻量基线参与 benchmark 对比。接口与 SB3Agent 完全一致。
"""

from __future__ import annotations

import time

import numpy as np

from ..core.config import RLConfig
from ..core.errors import AgentError
from ..core.types import TrainResult
from .base import AgentBase


class DQNAgent(AgentBase):
    """纯 numpy 实现的 Deep Q-Network。"""

    name = "dqn-np"

    def __init__(self, obs_dim: int, n_actions: int, config: RLConfig) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.cfg = config
        # DQN 专属超参（覆盖 config 默认值以确保稳定学习）
        self.gamma = config.gamma
        self.lr = 1e-3
        self.batch_size = 32
        self.buffer_size = 10_000
        self.learning_starts = 500
        self.train_freq = 1
        self.target_update = 200
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        hidden = int(config.net_arch[0]) if config.net_arch else 32
        self.hidden = hidden
        self.rng = np.random.default_rng(config.seed)
        # 网络参数：obs -> hidden (relu) -> action
        scale = np.sqrt(2.0 / self.obs_dim)
        self.W0 = self.rng.standard_normal((self.obs_dim, hidden)) * scale
        self.b0 = np.zeros(hidden)
        self.W1 = self.rng.standard_normal((hidden, n_actions)) * np.sqrt(2.0 / hidden)
        self.b1 = np.zeros(n_actions)
        self.W0t, self.b0t, self.W1t, self.b1t = self._copy_target()
        self.buffer: list = []

    # ------------------------------------------------------------------
    def _copy_target(self):
        return self.W0.copy(), self.b0.copy(), self.W1.copy(), self.b1.copy()

    def _forward(self, W0, b0, W1, b1, x):
        h = np.maximum(0.0, x @ W0 + b0)
        return h @ W1 + b1

    def q(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        return self._forward(self.W0, self.b0, self.W1, self.b1, x)

    def predict(self, obs) -> int:
        return int(np.argmax(self.q(obs)))

    # ------------------------------------------------------------------
    def _train_batch(self) -> None:
        n = min(self.batch_size, len(self.buffer))
        idx = self.rng.choice(len(self.buffer), size=n, replace=False)
        s = np.array([self.buffer[i][0] for i in idx], dtype=np.float64)
        a = np.array([self.buffer[i][1] for i in idx], dtype=int)
        r = np.array([self.buffer[i][2] for i in idx], dtype=np.float64)
        s2 = np.array([self.buffer[i][3] for i in idx], dtype=np.float64)
        d = np.array([self.buffer[i][4] for i in idx], dtype=bool)

        q_next = np.max(self._forward(self.W0t, self.b0t, self.W1t, self.b1t, s2), axis=1)
        target = r + self.gamma * q_next * (~d)

        h = np.maximum(0.0, s @ self.W0 + self.b0)
        q = h @ self.W1 + self.b1

        dq = np.zeros_like(q)
        dq[np.arange(n), a] = q[np.arange(n), a] - target  # d(loss)/d q_a
        dW1 = h.T @ dq / n
        db1 = dq.sum(0) / n
        dh = (dq @ self.W1.T) * (h > 0)
        dW0 = s.T @ dh / n
        db0 = dh.sum(0) / n

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W0 -= self.lr * dW0
        self.b0 -= self.lr * db0

    def train(self, env, config: RLConfig | None = None) -> TrainResult:
        cfg = config or self.cfg
        try:
            obs, _ = env.reset(seed=cfg.seed)
        except Exception as exc:
            raise AgentError(f"环境 reset 失败: {exc}", code="E200")
        t0 = time.time()
        step = 0
        ep = 0
        ep_reward = 0.0
        reward_curve: list[float] = []
        done = False
        while step < cfg.total_timesteps:
            if self.rng.random() < self.eps:
                action = int(self.rng.integers(self.n_actions))
            else:
                action = self.predict(obs)
            try:
                nxt, reward, terminated, truncated, _ = env.step(action)
            except Exception as exc:
                raise AgentError(f"环境 step 失败: {exc}", code="E200")
            done = bool(terminated or truncated)
            self.buffer.append((obs, action, float(reward), nxt, done))
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
            obs = nxt
            ep_reward += float(reward)
            step += 1

            if len(self.buffer) >= self.learning_starts and step % self.train_freq == 0:
                self._train_batch()
            if step % self.target_update == 0:
                self.W0t, self.b0t, self.W1t, self.b1t = self._copy_target()

            if done:
                reward_curve.append(ep_reward)
                ep += 1
                self.eps = max(self.eps_min, self.eps * self.eps_decay)
                obs, _ = env.reset(seed=cfg.seed + ep)
                ep_reward = 0.0

        mean_reward = float(np.mean(reward_curve)) if reward_curve else 0.0
        return TrainResult(
            algo=self.name,
            env_id=cfg.env_id,
            total_timesteps=step,
            reward_curve=reward_curve,
            mean_episode_reward=mean_reward,
            train_seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------
    def _state_dict(self) -> dict:
        return {
            "W0": self.W0,
            "b0": self.b0,
            "W1": self.W1,
            "b1": self.b1,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "eps": self.eps,
        }

    def _load_state_dict(self, sd: dict) -> None:
        self.obs_dim = int(sd["obs_dim"])
        self.n_actions = int(sd["n_actions"])
        self.hidden = int(sd["hidden"])
        self.W0, self.b0, self.W1, self.b1 = sd["W0"], sd["b0"], sd["W1"], sd["b1"]
        self.W0t, self.b0t, self.W1t, self.b1t = self._copy_target()
        self.eps = float(sd.get("eps", self.eps_min))
