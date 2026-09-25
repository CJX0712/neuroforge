import math

from neuroforge.agents.dqn import DQNAgent
from neuroforge.core.config import RLConfig
from neuroforge.environments.factory import make_env
from neuroforge.eval.evaluator import Evaluator


def test_evaluator_runs():
    cfg = RLConfig(env_id="CartPole-v1", algo="dqn-np", total_timesteps=3000, seed=8)
    env = make_env(cfg.env_id, seed=8)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)
    agent.train(env, cfg)
    res = Evaluator().evaluate(agent, "CartPole-v1", n_episodes=5, seed=9)
    env.close()

    assert res.n_episodes == 5
    assert math.isfinite(res.mean_reward)
    assert 0.0 <= res.solved_rate <= 1.0
    assert res.mean_steps > 0
