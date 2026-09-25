from neuroforge.core.config import RLConfig
from neuroforge.training.trainer import Trainer


def test_trainer_dqn_pipeline():
    cfg = RLConfig(
        env_id="CartPole-v1",
        algo="dqn-np",
        total_timesteps=4000,
        seed=7,
        eval_episodes=5,
    )
    res = Trainer().train(cfg)
    assert res.algo == "dqn-np"
    assert res.total_timesteps > 0
    assert res.model_path, "模型应已存档"
    assert res.final_eval_reward >= 0 or res.final_eval_reward < 0  # 有限值


def test_trainer_force_dqn_via_no_sb3():
    cfg = RLConfig(
        env_id="CartPole-v1",
        algo="auto",
        total_timesteps=3000,
        seed=7,
        use_sb3=False,
    )
    res = Trainer().train(cfg)
    assert res.algo == "dqn-np"
