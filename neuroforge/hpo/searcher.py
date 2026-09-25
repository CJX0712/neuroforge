"""超参搜索：基于 Optuna 的贝叶斯搜索（可选后端）。"""

from __future__ import annotations

from ..core.config import RLConfig
from ..core.errors import HPOError
from ..training.trainer import Trainer

try:  # pragma: no cover - 依赖可选
    import optuna

    _OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖可选
    _OPTUNA_AVAILABLE = False


class OptunaSearcher:
    """对（轻量）DQN 智能体做超参搜索，最大化最终评测奖励。"""

    name = "optuna"

    def search(self, config: RLConfig, n_trials: int = 20) -> dict:
        if not _OPTUNA_AVAILABLE:
            raise HPOError("optuna 未安装，无法执行超参搜索", code="E500")

        def objective(trial):
            cfg = RLConfig(
                env_id=config.env_id,
                algo="dqn-np",
                use_sb3=False,
                total_timesteps=min(config.total_timesteps, 8000),
                seed=config.seed,
            )
            cfg.learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            cfg.gamma = trial.suggest_float("gamma", 0.9, 0.999)
            cfg.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            res = Trainer().train(cfg)
            return res.final_eval_reward

        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
        except Exception as exc:
            raise HPOError(f"超参搜索失败: {exc}", code="E500")

        return {"best_params": study.best_params, "best_value": study.best_value}
