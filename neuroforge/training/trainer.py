"""训练编排：按配置装配环境 + 智能体，训练、评测、存档。"""

from __future__ import annotations

import os

import numpy as np

from ..agents.dqn import DQNAgent
from ..agents.sb3 import SB3Agent, _SB3_AVAILABLE
from ..core.config import RLConfig
from ..core.errors import AgentError, TrainError
from ..core.types import TrainResult
from ..eval.evaluator import Evaluator
from ..environments.factory import make_env


class Trainer:
    """依据 RLConfig 训练一个智能体并返回 TrainResult。"""

    def train(self, config: RLConfig) -> TrainResult:
        try:
            env = make_env(config.env_id, seed=config.seed)
            obs_dim = int(env.observation_space.shape[0])
            n_actions = int(env.action_space.n)
        except Exception as exc:
            raise TrainError(f"环境装配失败: {exc}", code="E300")

        use_sb3 = config.use_sb3 and _SB3_AVAILABLE
        try:
            if config.algo in ("ppo", "dqn") and use_sb3:
                agent = SB3Agent(config.env_id, config)
            elif config.algo == "dqn-np":
                agent = DQNAgent(obs_dim, n_actions, config)
            elif use_sb3:
                agent = SB3Agent(config.env_id, config)
            else:
                agent = DQNAgent(obs_dim, n_actions, config)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"智能体装配失败: {exc}", code="E200")

        try:
            result = agent.train(env, config)
        except AgentError:
            raise
        except Exception as exc:
            raise TrainError(f"训练失败: {exc}", code="E300")
        finally:
            env.close()

        self.last_agent = agent

        # 评测
        try:
            evalr = Evaluator().evaluate(
                agent, config.env_id, config.eval_episodes, seed=config.seed + 999
            )
            result.final_eval_reward = evalr.mean_reward
            if not result.reward_curve:
                result.mean_episode_reward = evalr.mean_reward
            else:
                result.mean_episode_reward = float(np.mean(result.reward_curve))
        except Exception:  # pragma: no cover - 评测失败不阻断训练产出
            pass

        # 存档
        try:
            os.makedirs(config.model_dir, exist_ok=True)
            path = os.path.join(config.model_dir, f"{config.env_id}_{agent.name}.zip")
            agent.save(path)
            result.model_path = path
        except Exception:  # pragma: no cover - 存档失败不阻断
            pass

        return result
