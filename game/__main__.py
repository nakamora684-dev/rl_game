"""Headless movement/clock inspection: ``python -m game`` (not a football match)."""

import argparse
import csv
import json
import sys
from math import hypot

from game.actions import Direction, MovementAction
from game.ball import BallState, step_free_ball
from game.config import PHYSICS_HZ, BallConfig, MovementConfig
from game.player import PlayerState, step_player
from game.simulation import MatchPhase, reset, step_decision
from game.stamina import performance_factors


def inspect_stamina() -> None:
    """Log 20 s sprint + 15 s rest for one player in open space, at one-second intervals."""
    player = PlayerState(position=(0.0, 0.0))
    movement = MovementConfig()
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        ("seconds", "phase", "effort", "stamina", "speed", "speed_limit", "acceleration_limit")
    )
    writer.writerow((0, "initial", 0.0, 100.0, 0.0, movement.max_speed, movement.acceleration))
    for second in range(1, 36):
        sprinting = second <= 20
        action = MovementAction(Direction.EAST, 1.0) if sprinting else MovementAction()
        for _ in range(PHYSICS_HZ):
            player = step_player(player, action, movement=movement)
        speed_factor, acceleration_factor = performance_factors(player.stamina)
        writer.writerow(
            (
                second,
                "sprint" if sprinting else "rest",
                action.effort,
                round(player.stamina, 6),
                round(hypot(*player.velocity), 6),
                round(movement.max_speed * speed_factor, 6),
                round(movement.acceleration * acceleration_factor, 6),
            )
        )


def inspect_ball() -> None:
    """Log the approved reference roll in open space, including one second at rest."""
    config = BallConfig()
    ball = BallState(position=(0.0, 0.0), velocity=(config.reference_speed, 0.0))
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("seconds", "x", "y", "speed"))
    writer.writerow((0, 0.0, 0.0, config.reference_speed))
    for second in range(1, 5):
        for _ in range(PHYSICS_HZ):
            ball = step_free_ball(ball, config=config)
        writer.writerow(
            (second, *[round(value, 6) for value in ball.position], round(hypot(*ball.velocity), 6))
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stamina", action="store_true", help="print a sprint/rest inspection CSV")
    mode.add_argument("--ball", action="store_true", help="print the reference free-ball roll CSV")
    arguments = parser.parse_args()
    if arguments.stamina:
        inspect_stamina()
        return
    if arguments.ball:
        inspect_ball()
        return
    state = reset()
    checkpoints = []
    event_count = 0
    for decision in range(600):
        action = MovementAction(Direction.EAST, 1.0) if decision < 5 else MovementAction()
        result = step_decision(state, action)
        state = result.state
        event_count += len(result.events)
        if decision in (4, 44):
            checkpoints.append(
                {
                    "tick": state.simulation_tick,
                    "player_a_position": state.player_a.position,
                    "player_a_speed": hypot(*state.player_a.velocity),
                    "player_a_stamina": state.player_a.stamina,
                }
            )

    print(
        json.dumps(
            {
                "checkpoints": checkpoints,
                "final_tick": state.simulation_tick,
                "finished": state.phase == MatchPhase.FINISHED,
                "terminal_events": event_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
