"""评测器：对训练好的智能体做 N 轮 rollout，产出统计指标。"""

from __future__ import annotations

import numpy as np

from ..core.config import SOLVE_THRESHOLDS
from ..core.errors import EvalError
from ..core.types import EvalResult


class Evaluator:
    """在指定环境上对智能体做无训练评测。"""

    def evaluate(
        self, agent, env_id: str, n_episodes: int = 10, seed: int | None = None, max_steps: int = 2000
    ) -> EvalResult:
        # 延迟导入避免循环依赖
        from ..environments.factory import make_env

        try:
            env = make_env(env_id, seed=seed)
        except Exception as exc:
            raise EvalError(f"评测环境创建失败: {exc}", code="E400")

        threshold = SOLVE_THRESHOLDS.get(env_id)
        rewards: list[float] = []
        steps: list[int] = []
        solved = 0
        try:
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=(seed + ep) if seed is not None else None)
                done = False
                total = 0.0
                t = 0
                while not done and t < max_steps:
                    try:
                        action = agent.predict(obs)
                    except Exception as exc:
                        raise EvalError(f"预测失败: {exc}", code="E400")
                    obs, r, terminated, truncated, _ = env.step(action)
                    total += float(r)
                    t += 1
                    done = bool(terminated or truncated)
                rewards.append(total)
                steps.append(t)
                if threshold is not None and total >= threshold:
                    solved += 1
        finally:
            env.close()

        rewards_arr = np.asarray(rewards, dtype=float)
        steps_arr = np.asarray(steps, dtype=float)
        return EvalResult(
            env_id=env_id,
            n_episodes=n_episodes,
            mean_reward=float(rewards_arr.mean()),
            std_reward=float(rewards_arr.std()),
            solved_rate=float(solved / n_episodes),
            mean_steps=float(steps_arr.mean()),
        )
