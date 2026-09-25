"""NeuroForge 高层入口：串联训练 / 评测 / 对标 / 超参搜索。"""

from __future__ import annotations

from .agents.sb3 import _SB3_AVAILABLE
from .core.config import RLConfig
from .core.types import EvalResult, TrainResult
from .eval.evaluator import Evaluator
from .hpo.searcher import OptunaSearcher
from .training.trainer import Trainer


class NeuroForge:
    """深度强化学习训练系统入口。"""

    def __init__(self, config: RLConfig) -> None:
        self.config = config
        self.trainer = Trainer()
        self.train_result: TrainResult | None = None
        self.agent = None

    @classmethod
    def from_config(cls, config: RLConfig) -> "NeuroForge":
        return cls(config)

    def train(self) -> TrainResult:
        self.train_result = self.trainer.train(self.config)
        self.agent = getattr(self.trainer, "last_agent", None)
        return self.train_result

    def evaluate(self, n_episodes: int | None = None) -> EvalResult:
        if self.agent is None:
            self.train()
        ep = n_episodes or self.config.eval_episodes
        return Evaluator().evaluate(
            self.agent, self.config.env_id, ep, seed=self.config.seed + 999
        )

    def benchmark(self, algorithms: list[str] | None = None) -> dict[str, TrainResult]:
        algos = algorithms or (["dqn-np", "ppo"] if _SB3_AVAILABLE else ["dqn-np"])
        results: dict[str, TrainResult] = {}
        for algo in algos:
            cfg = RLConfig(
                env_id=self.config.env_id,
                algo=algo,
                total_timesteps=self.config.total_timesteps,
                seed=self.config.seed,
                use_sb3=(algo != "dqn-np"),
            )
            results[algo] = Trainer().train(cfg)
        return results

    def hpo(self, n_trials: int | None = None) -> dict:
        return OptunaSearcher().search(self.config, n_trials or self.config.hpo_trials)
