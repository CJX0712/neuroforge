from neuroforge.environments.factory import available_envs, make_env


def test_make_env_cartpole():
    env = make_env("CartPole-v1")
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    nxt, reward, terminated, truncated, info = env.step(action)
    assert nxt.shape == env.observation_space.shape
    assert isinstance(reward, (int, float))
    env.close()


def test_make_env_invalid():
    import pytest

    from neuroforge.core.errors import EnvError

    with pytest.raises(EnvError):
        make_env("__not_a_real_env__")


def test_available_envs():
    assert "CartPole-v1" in available_envs()
