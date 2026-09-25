"""NeuroForge 模块接口契约（Protocol）。

每个模块仅依赖接口，实现可替换、可独立验证：
    Environment  -> gymnasium.Env（直接使用业界标准）
    Agent        -> 训练 / 预测 / 存档 / 读档
    Trainer      -> 按配置产出 TrainResult
    Evaluator    -> 按配置产出 EvalResult
    HyperparamSearch -> 按目标产出最优超参
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import gymnasium as gym

from .config import RLConfig
from .types import EvalResult, TrainResult


@runtime_checkable
class Agent(Protocol):
    """智能体接口。"""

    name: str

    def train(self, env: gym.Env, config: RLConfig) -> TrainResult: ...
    def predict(self, obs: Any) -> int: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...


@runtime_checkable
class Trainer(Protocol):
    """训练器接口。"""

    def train(self, config: RLConfig) -> TrainResult: ...


@runtime_checkable
class Evaluator(Protocol):
    """评测器接口。"""

    def evaluate(
        self, agent: Agent, env_id: str, n_episodes: int, seed: int | None = None
    ) -> EvalResult: ...


@runtime_checkable
class HyperparamSearch(Protocol):
    """超参搜索接口。"""

    def search(self, config: RLConfig, n_trials: int) -> Dict[str, Any]: ...
