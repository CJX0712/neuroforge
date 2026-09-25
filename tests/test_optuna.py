import pytest

pytest.importorskip("optuna")

from neuroforge.core.config import RLConfig
from neuroforge.hpo.searcher import OptunaSearcher


def test_optuna_search():
    cfg = RLConfig(env_id="CartPole-v1", total_timesteps=3000, seed=12)
    out = OptunaSearcher().search(cfg, n_trials=2)
    assert "best_params" in out
    assert isinstance(out["best_params"], dict)
