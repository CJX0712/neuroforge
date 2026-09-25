from neuroforge.agents.dqn import DQNAgent
from neuroforge.core.config import RLConfig
from neuroforge.core.interfaces import Agent, Evaluator, HyperparamSearch, Trainer
from neuroforge.environments.factory import make_env
from neuroforge.training.trainer import Trainer as TrainerImpl


def test_dqn_satisfies_agent_protocol():
    # 运行期可检查性：DQNAgent 实现 Agent 接口方法
    cfg = RLConfig()
    env = make_env(cfg.env_id, seed=0)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)
    env.close()
    assert isinstance(agent, Agent) or hasattr(agent, "train")
    assert hasattr(agent, "predict") and hasattr(agent, "save") and hasattr(agent, "load")


def test_interfaces_importable():
    assert Trainer is not None
    assert Evaluator is not None
    assert HyperparamSearch is not None
    assert TrainerImpl is not None
