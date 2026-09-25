"""NeuroForge 最小可运行示例：训练 CartPole 并评测。

离线可跑：默认优先 SB3（若安装），否则自动降级为纯 numpy DQN。
运行：python -m neuroforge.examples.run_demo
"""

from __future__ import annotations

from ..core.config import RLConfig
from ..runner import NeuroForge


def main() -> None:
    cfg = RLConfig(env_id="CartPole-v1", algo="auto", total_timesteps=20000, seed=42)
    print(f"[demo] 环境={cfg.env_id} 算法={cfg.algo} 步数={cfg.total_timesteps}")

    nf = NeuroForge.from_config(cfg)
    train_res = nf.train()
    print(f"[demo] 训练完成：algo={train_res.algo} 用时={train_res.train_seconds:.2f}s "
          f"最终评测奖励={train_res.final_eval_reward:.2f} 模型={train_res.model_path or '未保存'}")

    eval_res = nf.evaluate(n_episodes=10)
    print(f"[demo] 评测：均值奖励={eval_res.mean_reward:.2f}±{eval_res.std_reward:.2f} "
          f"解题率={eval_res.solved_rate:.0%} 平均步数={eval_res.mean_steps:.1f}")

    print("[demo] 完成 ✅ NeuroForge 离线链路运行正常。")


if __name__ == "__main__":
    main()
