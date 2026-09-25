import pytest

pytest.importorskip("stable_baselines3")

from neuroforge.core.config import RLConfig
from neuroforge.training.trainer import Trainer


def test_sb3_ppo_train():
    cfg = RLConfig(
        env_id="CartPole-v1",
        algo="ppo",
        use_sb3=True,
        total_timesteps=6000,
        seed=11,
        eval_episodes=5,
    )
    res = Trainer().train(cfg)
    assert "sb3" in res.algo
    assert res.total_timesteps > 0
    assert res.final_eval_reward > 30.0, f"PPO 未学到有效策略: {res.final_eval_reward:.2f}"
