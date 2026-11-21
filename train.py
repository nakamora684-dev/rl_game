"""Train the PPO agent (headless, no display needed).

  uv run python train.py --run-name exp0 --opponent rule_based
  uv run python train.py --run-name smoke --total-timesteps 20000 --n-envs 4 --opponent chaser
  uv run python train.py --run-name exp0 --resume models/exp0/latest.pt

Outputs
  logs/<run>/        TensorBoard events + episodes.csv (undiscounted return per episode),
                     updates.csv (PPO diagnostics), evals.csv, config.json
  models/<run>/      latest.pt, best.pt (best periodic eval), final.pt, ckpt_<step>.pt

View curves:  uv run tensorboard --logdir logs
Every PPOConfig / NetworkConfig field is also a CLI flag (e.g. --lr 1e-4 --hidden 256).
Defaults come from configs/ppo_default.json; flags override the file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from agents.networks import ActorCritic, NetworkConfig, count_parameters
from agents.opponents import OPPONENT_NAMES
from agents.ppo import PPOConfig, PPOTrainer, load_checkpoint, resolve_device
from evaluation.runner import quick_eval_fn
from training.logger import TrainingLogger
from training.vec_env import EnvSpec, VecFootball, parse_reward_overrides

DEFAULT_CONFIG = Path("configs/ppo_default.json")
TRAIN_SEED_LIMIT = 100_000  # evaluation seeds start at 500_000 (periodic) / 1_000_000 (test.py)
PERIODIC_EVAL_SEED = 500_000


def _add_dataclass_args(parser, cls, defaults: dict):
    for f in fields(cls):
        default = defaults.get(f.name, f.default)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=type(f.default), default=default)


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = pre.parse_known_args(argv)
    file_cfg = {}
    if known.config and Path(known.config).is_file():
        file_cfg = json.loads(Path(known.config).read_text(encoding="utf-8"))

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=known.config, help="JSON defaults file")
    p.add_argument("--run-name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    p.add_argument("--opponent", default=file_cfg.get("env", {}).get("opponent", "rule_based"),
                   choices=OPPONENT_NAMES)
    p.add_argument("--start-jitter", type=float,
                   default=file_cfg.get("env", {}).get("start_jitter", 0.0),
                   help="+-units of seeded start-position jitter (0 = fixed start)")
    p.add_argument("--no-randomize-sides", action="store_true",
                   help="always play as player A attacking right (default: random each episode)")
    p.add_argument("--shaping-gamma", type=float, default=None,
                   help="override the env's potential-shaping gamma (default: env's own 1.0). "
                        "Decide with Member C whether to align it with --gamma.")
    p.add_argument("--reward", nargs="*", default=[], metavar="NAME=VALUE",
                   help="override RewardConfig fields, e.g. --reward steal=0 loose_recovery=0 "
                        "(see environment/reward.py)")
    p.add_argument("--eval-every", type=int, default=20, help="PPO updates between evaluations")
    p.add_argument("--eval-matches", type=int, default=8, help="matches per periodic evaluation")
    p.add_argument("--save-every", type=int, default=20, help="PPO updates between ckpt_*.pt")
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--no-tensorboard", action="store_true")
    _add_dataclass_args(p, PPOConfig, file_cfg.get("ppo", {}))
    _add_dataclass_args(p, NetworkConfig, file_cfg.get("network", {}))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ppo_cfg = PPOConfig(**{f.name: getattr(args, f.name) for f in fields(PPOConfig)})
    net_cfg = NetworkConfig(**{f.name: getattr(args, f.name) for f in fields(NetworkConfig)})
    if not 0 <= ppo_cfg.seed * 1000 + ppo_cfg.n_envs < TRAIN_SEED_LIMIT:
        raise SystemExit(f"env seeds (seed*1000 + env index) must stay below {TRAIN_SEED_LIMIT} "
                         "so evaluation seeds remain held-out; use --seed < 99.")

    np.random.seed(ppo_cfg.seed)
    torch.manual_seed(ppo_cfg.seed)
    device = resolve_device(ppo_cfg.device)

    overrides = parse_reward_overrides(args.reward)
    if args.shaping_gamma is not None:
        overrides["shaping_gamma"] = args.shaping_gamma
    spec = EnvSpec(opponent=args.opponent, start_jitter=args.start_jitter,
                   randomize_sides=not args.no_randomize_sides, reward_overrides=overrides)
    eval_spec = EnvSpec(opponent=args.opponent, start_jitter=args.start_jitter,
                        randomize_sides=False, reward_overrides=overrides)

    run_log, run_models = Path("logs") / args.run_name, Path("models") / args.run_name
    run_log.mkdir(parents=True, exist_ok=True)
    model = ActorCritic(net_cfg).to(device)
    ckpt = None
    if args.resume:
        model, ckpt = load_checkpoint(args.resume, device)
        model.train()
    (run_log / "config.json").write_text(
        json.dumps({"ppo": asdict(ppo_cfg), "network": asdict(net_cfg),
                    "env": asdict(spec), "args": vars(args)}, indent=2, default=str),
        encoding="utf-8")
    print(f"Run '{args.run_name}' on {device} | params={count_parameters(model):,} | "
          f"opponent={args.opponent} | batch={ppo_cfg.n_envs * ppo_cfg.n_steps}")

    vec = VecFootball(spec, ppo_cfg.n_envs, seed=ppo_cfg.seed * 1000, n_workers=ppo_cfg.n_workers)
    logger = TrainingLogger(run_log, tensorboard=not args.no_tensorboard)
    trainer = PPOTrainer(
        model, vec, ppo_cfg, logger, run_models,
        eval_fn=quick_eval_fn(eval_spec, args.eval_matches, PERIODIC_EVAL_SEED)
        if args.eval_matches > 0 else None,
        eval_every=args.eval_every, save_every=args.save_every,
    )
    if ckpt:
        trainer.load_state(ckpt)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Interrupted: saving latest checkpoint.")
        trainer._save("latest.pt")
    finally:
        vec.close()
        logger.close()
    print(f"Done. Checkpoints: {run_models}  Logs: {run_log}")
    return 0


if __name__ == "__main__":  # required: worker processes use the 'spawn' start method
    sys.exit(main())
