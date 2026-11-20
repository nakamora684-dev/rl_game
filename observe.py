"""Watch (or just run) full matches between two agents.

    uv run python observe.py                       # rule vs rule, graphical
    uv run python observe.py --headless --episodes 20 --swap
    uv run python observe.py --p0 ppo --model models/best.zip

Both modes call the SAME engine through agents.match.run_match. The graphical mode
only adds a render hook. PPO checkpoints (Member D) are run by agents.policy_agent, which
reproduces exactly what train.py / test.py evaluate (10 Hz decisions, C's observation).
"""

from __future__ import annotations

import argparse
import sys

from agents.difficulty import PRESETS, make_baseline
from agents.interface import PHYS_HZ
from agents.match import run_match
from play import load_engine


def load_agent(
    kind: str,
    index: int,
    model: str | None,
    difficulty: str = "normal",
    engine=None,
    deterministic: bool = True,
):
    if kind == "rule":
        return make_baseline(index, difficulty)
    if kind == "ppo":
        if not model:
            raise SystemExit("--model <checkpoint.pt> is required for a ppo agent")
        from agents.policy_agent import PolicyAgent
        from evaluation.policies import load_policy

        return PolicyAgent(load_policy(model, deterministic=deterministic), index, engine)
    raise ValueError(f"unknown agent kind: {kind}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Observe agents playing")
    ap.add_argument("--p0", default="rule", choices=["rule", "ppo"])
    ap.add_argument("--p1", default="rule", choices=["rule", "ppo"])
    ap.add_argument("--diff0", default="normal", choices=sorted(PRESETS))
    ap.add_argument("--diff1", default="normal", choices=sorted(PRESETS))
    ap.add_argument("--model", default=None, help="checkpoint (.pt) for ppo agents")
    ap.add_argument("--stochastic", action="store_true", help="ppo agents sample actions")
    ap.add_argument(
        "--ball-to",
        default="none",
        choices=["none", "p0", "p1"],
        help="viewing only: give the ball to p0 (blue) or p1 (red) at kickoff "
        "(setup helper; test.py/train.py always start with a free ball)",
    )
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swap", action="store_true", help="alternate attack direction each episode")
    ap.add_argument("--headless", action="store_true", help="no window, print results")
    ap.add_argument("--replay", default=None, help="replay file saved by play.py (JSON)")
    args = ap.parse_args(argv)

    if "ppo" in (args.p0, args.p1) and not args.replay:
        from agents.policy_agent import PolicyEngineAdapter

        engine, label = PolicyEngineAdapter(), "engine game/ via PolicyEngineAdapter (ppo agent)"
    else:
        engine, label = load_engine()
    print(f"Using {label}")
    if args.ball_to != "none":
        from agents.engine_adapter import to_view
        from game.player import PlayerId
        from game.simulation import assign_possession

        owner = PlayerId.A if args.ball_to == "p0" else PlayerId.B
        plain_reset = engine.reset

        def reset_with_ball(seed=None, swap_sides=False):
            plain_reset(seed=seed, swap_sides=swap_sides)
            engine.raw = assign_possession(engine.raw, owner, config=engine.config).state
            return to_view(engine.raw)

        engine.reset = reset_with_ball
    seed, swap_all = args.seed, False
    if args.replay:  # same seed + side + actions = same match
        from rendering.replay import ReplayAgent, load_replay

        data = load_replay(args.replay)
        agents = [ReplayAgent(data, 0), ReplayAgent(data, 1)]
        seed, swap_all, args.episodes, args.swap = data["seed"], data["swap_sides"], 1, False
    else:
        agents = [
            load_agent(args.p0, 0, args.model, args.diff0, engine, not args.stochastic),
            load_agent(args.p1, 1, args.model, args.diff1, engine, not args.stochastic),
        ]

    on_step, close = None, (lambda: None)
    if not args.headless:
        from rendering.renderer import Renderer

        r = Renderer()
        on_step = lambda st: r.render(st, fps=PHYS_HZ)
        close = r.close

    for ep in range(args.episodes):
        swap = swap_all or (args.swap and ep % 2 == 1)
        final = run_match(engine, agents, seed=seed + ep, swap_sides=swap, on_step=on_step)
        print(
            f"episode {ep}: score {final.score[0]}-{final.score[1]} "
            f"ticks {final.tick} swapped={swap}"
        )
    close()
    return 0


if __name__ == "__main__":
    sys.exit(main())