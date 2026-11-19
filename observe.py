"""Watch (or just run) full matches between two agents.

    uv run python observe.py                       # rule vs rule, graphical
    uv run python observe.py --headless --episodes 20 --swap
    uv run python observe.py --p0 ppo --model models/best.zip

Both modes call the SAME engine through agents.match.run_match. The graphical mode
only adds a render hook. Loading PPO checkpoints belongs to Member D (load_agent).
"""

from __future__ import annotations

import argparse
import sys

from agents.difficulty import PRESETS, make_baseline
from agents.interface import PHYS_HZ
from agents.match import run_match
from play import load_engine


def load_agent(kind: str, index: int, model: str | None, difficulty: str = "normal"):
    if kind == "rule":
        return make_baseline(index, difficulty)
    if kind == "ppo":
        # TODO(D): load the checkpoint and return an object with .act(GameState) -> Action
        # built on the shared observation (agents/observation_view.py + C's wrapper).
        raise NotImplementedError("PPO loading is provided by Member D (train.py / model config)")
    raise ValueError(f"unknown agent kind: {kind}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Observe agents playing")
    ap.add_argument("--p0", default="rule", choices=["rule", "ppo"])
    ap.add_argument("--p1", default="rule", choices=["rule", "ppo"])
    ap.add_argument("--diff0", default="normal", choices=sorted(PRESETS))
    ap.add_argument("--diff1", default="normal", choices=sorted(PRESETS))
    ap.add_argument("--model", default=None, help="checkpoint for ppo agents")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swap", action="store_true", help="alternate attack direction each episode")
    ap.add_argument("--headless", action="store_true", help="no window, print results")
    ap.add_argument("--replay", default=None, help="replay file saved by play.py (JSON)")
    args = ap.parse_args(argv)

    engine, label = load_engine()
    print(f"Using {label}")
    seed, swap_all = args.seed, False
    if args.replay:  # same seed + side + actions = same match
        from rendering.replay import ReplayAgent, load_replay

        data = load_replay(args.replay)
        agents = [ReplayAgent(data, 0), ReplayAgent(data, 1)]
        seed, swap_all, args.episodes, args.swap = data["seed"], data["swap_sides"], 1, False
    else:
        agents = [
            load_agent(args.p0, 0, args.model, args.diff0),
            load_agent(args.p1, 1, args.model, args.diff1),
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
