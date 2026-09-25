import numpy as np

from neuroforge.agents.dqn import DQNAgent
from neuroforge.core.config import RLConfig
from neuroforge.environments.factory import make_env


def _agent(cfg):
    env = make_env(cfg.env_id, seed=cfg.seed)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)
    env.close()
    return agent


def test_dqn_train_improves():
    cfg = RLConfig(env_id="CartPole-v1", algo="dqn-np", total_timesteps=8000, seed=1)
    env = make_env(cfg.env_id, seed=1)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)
    res = agent.train(env, cfg)
    env.close()

    assert res.algo == "dqn-np"
    assert len(res.reward_curve) > 0
    final = float(np.mean(res.reward_curve[-10:]))
    assert final > 30.0, f"DQN 未学到有效策略，末段均值={final:.2f}"


def test_dqn_predict_valid_action():
    cfg = RLConfig(env_id="CartPole-v1", algo="dqn-np", total_timesteps=2000, seed=2)
    env = make_env(cfg.env_id, seed=2)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)
    agent.train(env, cfg)
    obs, _ = env.reset(seed=3)
    act = agent.predict(obs)
    assert act in set(range(env.action_space.n))
    env.close()


def test_dqn_save_load_equivalence(tmp_path):
    cfg = RLConfig(env_id="CartPole-v1", algo="dqn-np", total_timesteps=2000, seed=4)
    env = make_env(cfg.env_id, seed=4)
    a1 = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)
    a1.train(env, cfg)
    path = tmp_path / "dqn.pkl"
    a1.save(str(path))

    a2 = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)
    a2.load(str(path))
    obs, _ = env.reset(seed=5)
    assert a1.predict(obs) == a2.predict(obs)
    env.close()
