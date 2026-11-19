"""Human-playable graphical mode.   Run:  uv run python play.py [options]

    --difficulty easy|normal|hard   bot strength (default normal)
    --side left|right               your starting side (right = you attack toward -x)
    --swap                          flip your side at every rematch (play both directions)
    --seed N   --scale S   --no-menu   --controls keys.json

You are BLUE (player 0). Keys: arrows / ZQSD / WASD move | Shift sprint |
Space hold+release = kick | P or Esc pause | F1 debug | F12 screenshot.
The renderer only draws; the engine owns every gameplay rule (see session.py).
"""

from __future__ import annotations

import argparse
import sys
import time

from agents.interface import PHYS_HZ
from rendering.flow import DIFFICULTIES, FULLTIME, MENU, PAUSED, PLAYING, Flow, MatchOptions

HINT = "Move: arrows / WASD / ZQSD   Shift: sprint   Space: hold + release to kick   P: pause   F1: debug"


def load_engine():
    """Member A's engine (game/) behind agents.engine_adapter (one tick per step)."""
    from agents.engine_adapter import EngineAdapter

    return EngineAdapter(), "engine game/ (Member A) via agents.engine_adapter"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Human vs rule-based football")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.2)
    ap.add_argument("--difficulty", default="normal", choices=DIFFICULTIES)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--swap", action="store_true", help="flip side at each rematch")
    ap.add_argument("--no-menu", action="store_true", help="start the match immediately")
    ap.add_argument("--controls", default=None, help="JSON file with key bindings")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import pygame  # lazy: keep imports testable

    from rendering.controls import make_controls
    from rendering.renderer import Renderer
    from rendering.screens import Screens, keymap
    from rendering.session import Session

    engine, label = load_engine()
    print(f"Using {label}")
    options = MatchOptions(args.difficulty, args.side == "right", args.seed, args.swap)
    flow = Flow(options, start_in_menu=not args.no_menu)
    renderer = Renderer(scale=args.scale)
    screens = Screens(renderer)
    controls = make_controls(args.controls)
    keys = keymap(pygame)
    dt = 1.0 / PHYS_HZ
    session = None
    replay_path = None

    def new_session():
        nonlocal session, replay_path
        renderer.reset_effects()
        session, replay_path = Session(engine, flow.options), None

    if flow.state == PLAYING:
        new_session()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_F1:
                    renderer.toggle_debug()
                elif e.key == pygame.K_F12:
                    print(
                        "saved",
                        renderer.screenshot(f"results/screenshots/shot_{int(time.time())}.png"),
                    )
                else:
                    cmd = flow.handle(keys.get(e.key))
                    if cmd in ("start", "restart", "rematch"):
                        new_session()
                    elif cmd == "menu":
                        session = None
                    elif cmd == "quit":
                        running = False
        if flow.state == MENU:
            screens.menu(flow)
        elif flow.state == PLAYING and session is not None:
            session.tick(controls.read(dt))
            renderer.render(
                session.state, charge=controls.charge, fps=PHYS_HZ, show_final=False, hint=HINT
            )
            if session.finished:
                flow.finish()
                replay_path = session.save_replay()
        elif flow.state == PAUSED and session is not None:
            renderer.render(
                session.state,
                fps=30,
                show_final=False,
                freeze=True,
                overlay=lambda r: screens.pause(flow),
            )
        elif flow.state == FULLTIME and session is not None:
            renderer.render(
                session.state,
                fps=30,
                show_final=False,
                freeze=True,
                overlay=lambda r: screens.fulltime(
                    session.state, session.summary_row(), replay_path
                ),
            )
    renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
