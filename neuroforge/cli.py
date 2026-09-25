"""NeuroForge 命令行入口：train / eval / benchmark / hpo。"""

from __future__ import annotations

import argparse
import json
import sys

from .core.config import RLConfig
from .runner import NeuroForge


def _build_config(args) -> RLConfig:
    return RLConfig(
        env_id=args.env,
        algo=args.algo,
        total_timesteps=args.steps,
        seed=args.seed,
        use_sb3=not args.no_sb3,
        eval_episodes=args.eval_episodes,
        hpo_trials=args.hpo_trials,
    )


def _cmd_train(args) -> int:
    cfg = _build_config(args)
    res = NeuroForge.from_config(cfg).train()
    print(json.dumps(res.__dict__, default=list))
    return 0


def _cmd_eval(args) -> int:
    cfg = _build_config(args)
    nf = NeuroForge.from_config(cfg)
    res = nf.evaluate(n_episodes=args.eval_episodes)
    print(json.dumps(res.__dict__, default=list))
    return 0


def _cmd_benchmark(args) -> int:
    cfg = _build_config(args)
    out = NeuroForge.from_config(cfg).benchmark()
    print(json.dumps({k: v.__dict__ for k, v in out.items()}, default=list))
    return 0


def _cmd_hpo(args) -> int:
    cfg = _build_config(args)
    out = NeuroForge.from_config(cfg).hpo(n_trials=args.hpo_trials)
    print(json.dumps(out, default=list))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neuroforge", description="NeuroForge 深度强化学习训练系统")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "eval", "benchmark", "hpo"):
        p = sub.add_parser(name)
        p.add_argument("--env", default="CartPole-v1")
        p.add_argument("--algo", default="auto", choices=["auto", "ppo", "dqn", "dqn-np"])
        p.add_argument("--steps", type=int, default=30000)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--eval-episodes", type=int, default=10)
        p.add_argument("--hpo-trials", type=int, default=20)
        p.add_argument("--no-sb3", action="store_true", help="强制使用纯 numpy DQN（不调用 SB3）")
        p.set_defaults(func=getattr(sys.modules[__name__], f"_cmd_{name}"))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
