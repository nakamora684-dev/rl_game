"""Repeatable evaluation: agents vs a rule-based (or other) opponent -> CSV + summary.

Examples
--------
  uv run python test.py --agents models/run1/best.pt --opponent rule_based --matches 100
  uv run python test.py --agents models/run1/ckpt_500000.pt models/run1/final.pt random idle
  uv run python test.py --agents random idle --opponent chaser --matches 20   # no torch needed

Matches cycle through the 4 (player identity x attack direction) combinations and use
the same seeds for every agent (paired comparison). No weights are ever updated.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from agents.opponents import OPPONENT_NAMES
from evaluation.metrics import MATCH_COLUMNS, summarize
from evaluation.policies import load_policy
from evaluation.runner import evaluate
from training.vec_env import EnvSpec, parse_reward_overrides

HELD_OUT_SEED_BASE = 1_000_000  # training seeds are < 100_000 (see train.py)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--agents", nargs="+", required=True,
                   help="checkpoint paths and/or reference agents: 'random', 'idle', 'heuristic', "
                        "'rule_based' / 'rule_based_easy' / 'rule_based_hard' (B's baseline as agent)")
    p.add_argument("--opponent", default="rule_based", choices=OPPONENT_NAMES)
    p.add_argument("--matches", type=int, default=100, help="matches per agent (use multiples of 4)")
    p.add_argument("--seed-base", type=int, default=HELD_OUT_SEED_BASE)
    p.add_argument("--policy-mode", choices=("stochastic", "deterministic"), default="stochastic",
                   help="stochastic samples actions (matches differ); deterministic uses the mode")
    p.add_argument("--start-jitter", type=float, default=0.0,
                   help="+-units of seeded start-position jitter (0 = engine's fixed start)")
    p.add_argument("--reward", nargs="*", default=[], metavar="NAME=VALUE",
                   help="reward overrides (only changes the logged ep_return/r_* columns)")
    p.add_argument("--out", default=None, help="CSV path (default results/eval_<time>.csv)")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.seed_base < 100_000:
        print("WARNING: seed-base overlaps the training seed range; not held-out.", file=sys.stderr)
    spec = EnvSpec(opponent=args.opponent, start_jitter=args.start_jitter, randomize_sides=False,
                   reward_overrides=parse_reward_overrides(args.reward))
    deterministic = args.policy_mode == "deterministic"
    if deterministic and args.start_jitter == 0:
        print("WARNING: deterministic policy + deterministic engine/opponent + no start jitter "
              "=> only 4 distinct matches exist; extra matches repeat them. Use "
              "--policy-mode stochastic or --start-jitter for statistically useful samples.",
              file=sys.stderr)
    out = Path(args.out or f"results/eval_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec_name in args.agents:
        policy = load_policy(spec_name, deterministic=deterministic, device=args.device)
        name = getattr(policy, "name", spec_name)
        t0 = time.time()

        def progress(done, total, name=name):
            if done % 10 == 0 or done == total:
                print(f"  {name}: {done}/{total} matches", flush=True)

        rows += evaluate(policy, spec, n_matches=args.matches, seed_base=args.seed_base,
                         agent_name=name, policy_mode=args.policy_mode, progress=progress)
        print(f"{name}: {args.matches} matches in {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    df.to_csv(out, index=False)
    summary = summarize(df)
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\n" + summary[["agent", "matches", "wins", "draws", "losses", "win_rate",
                              "win_ci_low", "win_ci_high", "goal_diff", "goal_diff_ci_low",
                              "goal_diff_ci_high", "poss_self_frac", "shots"]]
              .round(3).to_string(index=False))
    print(f"\nPer-match CSV : {out}\nSummary CSV   : {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
