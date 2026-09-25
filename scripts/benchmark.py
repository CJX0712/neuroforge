"""NeuroForge 基准脚本：在 CartPole 上对标 DQN(纯 numpy) 与 SB3(PPO)。

输出 benchmark_report.json，并打印各后端的最终评测奖励与训练耗时。
运行：python scripts/benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroforge.core.config import RLConfig
from neuroforge.runner import NeuroForge


def main() -> None:
    cfg = RLConfig(env_id="CartPole-v1", algo="auto", total_timesteps=20000, seed=42)
    t0 = time.time()
    out = NeuroForge.from_config(cfg).benchmark()
    elapsed = time.time() - t0

    report = {"env_id": cfg.env_id, "wall_seconds": round(elapsed, 2), "backends": {}}
    for key, res in out.items():
        report["backends"][key] = {
            "algo": res.algo,
            "total_timesteps": res.total_timesteps,
            "mean_episode_reward": round(res.mean_episode_reward, 3),
            "final_eval_reward": round(res.final_eval_reward, 3),
            "train_seconds": round(res.train_seconds, 3),
            "model_path": res.model_path,
        }

    with open("benchmark_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"[benchmark] 总耗时 {elapsed:.2f}s")
    for key, b in report["backends"].items():
        print(
            f"  {key:10s} algo={b['algo']:10s} "
            f"final_eval={b['final_eval_reward']:.2f} "
            f"train={b['train_seconds']:.2f}s"
        )
    print("[benchmark] 报告已写入 benchmark_report.json")


if __name__ == "__main__":
    main()
