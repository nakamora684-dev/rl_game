"""Headless engine inspection; add --match for a full idle/scripted match (no graphics)."""

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import replace
from math import hypot

from game.actions import Direction, MovementAction, PlayerAction
from game.ball import BallState, step_free_ball
from game.config import PHYSICS_HZ, TICKS_PER_DECISION, BallConfig, MovementConfig, SimulationConfig
from game.player import PlayerId, PlayerState, step_player
from game.runner import idle_provider, run_match
from game.simulation import (
    MatchPhase,
    SimulationState,
    assign_possession,
    release_kick,
    reset,
    step_decision,
    step_tick,
)
from game.stamina import performance_factors
from game.timeline import format_timeline


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


def inspect_dribbling() -> None:
    """Show attachment during forward movement, north-facing sideways coast, and rest."""
    state = assign_possession(reset(), PlayerId.A).state
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        (
            "tick",
            "owner",
            "last_toucher",
            "player_x",
            "player_y",
            "facing",
            "player_vx",
            "player_vy",
            "ball_x",
            "ball_y",
            "ball_vx",
            "ball_vy",
        )
    )

    def write_snapshot() -> None:
        player, ball = state.player_a, state.ball
        writer.writerow(
            (
                state.simulation_tick,
                ball.possession_owner.value,
                ball.last_touch_player.value,
                *[
                    round(value, 6)
                    for value in (
                        *player.position,
                        player.facing,
                        *player.velocity,
                        *ball.position,
                        *ball.velocity,
                    )
                ],
            )
        )

    write_snapshot()
    for action, decisions in (
        (MovementAction(Direction.EAST, 1.0), 1),
        (MovementAction(Direction.NORTH, 0.0), 3),
        (MovementAction(), 10),
    ):
        for _ in range(decisions):
            state = step_decision(state, action).state
        write_snapshot()


def inspect_kick() -> None:
    """Release at reference roll speed, then inspect free motion (not power tuning)."""
    state = reset()
    # Isolate the reference roll: the default opponent would now intercept it.
    state = replace(state, player_b=replace(state.player_b, position=(600.0, 400.0)))
    state = assign_possession(state, PlayerId.A).state
    result = release_kick(state, PlayerId.A, launch_speed=BallConfig().reference_speed)
    state = result.state
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("tick", "owner", "last_toucher", "ball_x", "ball_y", "speed", "events"))

    def write_snapshot(events: str) -> None:
        ball = state.ball
        writer.writerow(
            (
                state.simulation_tick,
                ball.possession_owner.value if ball.possession_owner is not None else "none",
                ball.last_touch_player.value,
                *[round(value, 6) for value in ball.position],
                round(hypot(*ball.velocity), 6),
                events,
            )
        )

    write_snapshot("|".join(event.kind for event in result.events))
    for _ in range(4):
        events = []
        for _ in range(PHYSICS_HZ // TICKS_PER_DECISION):
            result = step_decision(state)
            state = result.state
            events.extend(event.kind for event in result.events)
        write_snapshot("|".join(events))


def inspect_kick_actions() -> None:
    """Compare first-decision results from three independently reset kick powers."""
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("power", "tick", "ball_x", "speed", "cooldown_ticks", "kick_events"))
    for power in (0.0, 0.5, 1.0):
        initial = assign_possession(reset(), PlayerId.A).state
        result = step_decision(initial, PlayerAction(kick_trigger=1, kick_power=power))
        state = result.state
        writer.writerow(
            (
                power,
                state.simulation_tick,
                round(state.ball.position[0], 6),
                round(hypot(*state.ball.velocity), 6),
                state.player_a.kick_cooldown_ticks,
                sum(event.kind == "ball_kicked" for event in result.events),
            )
        )


def inspect_pickup() -> None:
    """Show a legal free-ball interception, distinct from setup and stealing."""
    state = assign_possession(reset(), PlayerId.A).state
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("tick", "owner", "last_toucher", "ball_x", "a_pickup_lockout", "events"))

    def write_snapshot(events: str) -> None:
        ball = state.ball
        writer.writerow(
            (
                state.simulation_tick,
                ball.possession_owner.value if ball.possession_owner is not None else "none",
                ball.last_touch_player.value,
                round(ball.position[0], 6),
                state.player_a.pickup_lockout_ticks,
                events,
            )
        )

    write_snapshot("setup")
    result = step_tick(state, PlayerAction(kick_trigger=1, kick_power=1.0))
    state = result.state
    write_snapshot("|".join(event.cause for event in result.events))
    while state.simulation_tick < 120:
        result = step_tick(state)
        state = result.state
        if state.simulation_tick == 12 or result.events:
            write_snapshot("|".join(event.cause for event in result.events))
        if state.ball.possession_owner is not None:
            break
    result = step_decision(state)
    state = result.state
    write_snapshot("|".join(event.cause for event in result.events))


def inspect_stealing() -> None:
    """Show uninterrupted contest, one steal, protection, and eventual steal-back."""
    state = reset()
    state = replace(state, player_b=replace(state.player_b, position=(235.0, 250.0)))
    state = assign_possession(state, PlayerId.A).state
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("tick", "owner", "contest_ticks", "protection_ticks", "events"))
    writer.writerow((0, state.ball.possession_owner.value, 0, 0, "setup"))
    while state.simulation_tick < 66:
        result = step_tick(state)
        state = result.state
        if state.simulation_tick in (17, 18, 48, 65, 66):
            writer.writerow(
                (
                    state.simulation_tick,
                    state.ball.possession_owner.value,
                    state.steal_progress_ticks,
                    state.possession_protection_ticks,
                    "|".join(event.cause for event in result.events),
                )
            )


def inspect_rules() -> None:
    """Compare an on-target goal and an off-target boundary exit, then their restarts."""
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        (
            "scenario",
            "tick",
            "phase",
            "score_a",
            "score_b",
            "owner",
            "ball_x",
            "restart_ticks",
            "events",
        )
    )
    for scenario, y in (("goal", 250.0), ("off_target", 100.0)):
        state = replace(
            reset(), ball=BallState((804.0, y), (300.0, 0.0), last_touch_player=PlayerId.A)
        )
        for _ in range(67):
            result = step_tick(state)
            state = result.state
            if state.simulation_tick in (1, 31, 61, 67):
                writer.writerow(
                    (
                        scenario,
                        state.simulation_tick,
                        state.phase.value,
                        state.score_a,
                        state.score_b,
                        state.ball.possession_owner.value
                        if state.ball.possession_owner is not None
                        else "none",
                        round(state.ball.position[0], 6),
                        state.restart_ticks_remaining,
                        "|".join(event.kind + ":" + event.cause for event in result.events),
                    )
                )


def inspect_contacts() -> None:
    """Isolated one-tick contact examples, not a full-match runner."""
    config = SimulationConfig(ball=BallConfig(deceleration=0.0))
    locked = reset()
    locked = replace(
        locked,
        player_a=replace(locked.player_a, pickup_lockout_ticks=12),
        ball=BallState((177.0, 250.0), (300.0, 0.0), last_touch_player=PlayerId.B),
    )
    post = replace(
        reset(), ball=BallState((780.0, 186.0), (1200.0, 0.0), last_touch_player=PlayerId.B)
    )
    controlled = reset()
    controlled = replace(
        controlled,
        player_a=replace(controlled.player_a, position=(765.0, 186.0), velocity=(180.0, 0.0)),
    )
    controlled = assign_possession(controlled, PlayerId.A).state
    goal = replace(
        reset(), ball=BallState((780.0, 304.0), (6000.0, 0.0), last_touch_player=PlayerId.A)
    )
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        ("scenario", "tick", "phase", "owner", "last_toucher", "x", "y", "vx", "vy", "events")
    )
    for scenario, initial in (
        ("locked_player", locked),
        ("post", post),
        ("controlled_post", controlled),
        ("post_then_goal", goal),
    ):
        result = step_tick(initial, config=config)
        state, ball = result.state, result.state.ball
        writer.writerow(
            (
                scenario,
                state.simulation_tick,
                state.phase.value,
                ball.possession_owner.value if ball.possession_owner is not None else "none",
                ball.last_touch_player.value if ball.last_touch_player is not None else "none",
                *[round(value, 6) for value in (*ball.position, *ball.velocity)],
                "|".join(event.kind + ":" + event.cause for event in result.events),
            )
        )


def inspect_match(script: str, *, swap_sides: bool = False, timeline: bool = False) -> None:
    """Run replaceable test controllers, not the team's rule-based opponent."""

    def scripted_provider(state: SimulationState, player: PlayerId) -> PlayerAction:
        decision = state.simulation_tick // TICKS_PER_DECISION
        if player == PlayerId.B:
            return PlayerAction(Direction.NORTH, 0.5) if decision < 15 else PlayerAction()
        if decision < 45:
            direction = Direction.EAST if state.attack_direction_a == 1 else Direction.WEST
            return PlayerAction(direction, 0.75)
        return PlayerAction(kick_trigger=int(decision == 45), kick_power=1.0)

    provider = idle_provider if script == "idle" else scripted_provider
    result = run_match(provider, provider, initial_state=reset(swap_sides=swap_sides))
    if timeline:
        print(format_timeline(result.events), file=sys.stderr)
    state = result.final_state
    event_counts = Counter(event.kind for event in result.events)
    diagnostics = {
        kind: {
            "count": event_counts[kind],
            "causes": dict(Counter(event.cause for event in result.events if event.kind == kind)),
        }
        for kind in ("ball_collision", "ball_constraint", "physics_limit")
    }
    print(
        json.dumps(
            {
                "controllers": script,
                "swap_sides": swap_sides,
                "score_a": state.score_a,
                "score_b": state.score_b,
                "final_tick": state.simulation_tick,
                "finished": state.phase == MatchPhase.FINISHED,
                "physics_ticks": result.physics_ticks,
                "decision_count": result.decision_count,
                "event_counts": dict(event_counts),
                "diagnostics": diagnostics,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stamina", action="store_true", help="print a sprint/rest inspection CSV")
    mode.add_argument("--ball", action="store_true", help="print the reference free-ball roll CSV")
    mode.add_argument("--dribble", action="store_true", help="print controlled-ball attachment CSV")
    mode.add_argument("--kick", action="store_true", help="print explicit-speed kick release CSV")
    mode.add_argument(
        "--kick-actions", action="store_true", help="compare decision kick powers CSV"
    )
    mode.add_argument("--pickup", action="store_true", help="print automatic interception CSV")
    mode.add_argument("--steal", action="store_true", help="print timed steal and protection CSV")
    mode.add_argument(
        "--rules", action="store_true", help="print goals, boundaries, and restarts CSV"
    )
    mode.add_argument("--contacts", action="store_true", help="print rebound and post-contact CSV")
    mode.add_argument(
        "--match",
        nargs="?",
        const="idle",
        choices=("idle", "scripted"),
        help="run a full headless match with idle (default) or fixed scripted controllers",
    )
    parser.add_argument(
        "--swap-sides", action="store_true", help="swap attacking sides for --match"
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="print --match event timeline to stderr (JSON stays on stdout)",
    )
    arguments = parser.parse_args()
    if arguments.swap_sides and arguments.match is None:
        parser.error("--swap-sides requires --match")
    if arguments.timeline and arguments.match is None:
        parser.error("--timeline requires --match")
    if arguments.match is not None:
        inspect_match(arguments.match, swap_sides=arguments.swap_sides, timeline=arguments.timeline)
        return
    if arguments.stamina:
        inspect_stamina()
        return
    if arguments.ball:
        inspect_ball()
        return
    if arguments.dribble:
        inspect_dribbling()
        return
    if arguments.kick:
        inspect_kick()
        return
    if arguments.kick_actions:
        inspect_kick_actions()
        return
    if arguments.pickup:
        inspect_pickup()
        return
    if arguments.steal:
        inspect_stealing()
        return
    if arguments.rules:
        inspect_rules()
        return
    if arguments.contacts:
        inspect_contacts()
        return
    state = reset()
    checkpoints = []
    event_count = 0
    for decision in range(600):
        action = MovementAction(Direction.EAST, 1.0) if decision < 5 else MovementAction()
        result = step_decision(state, action)
        state = result.state
        event_count += sum(event.kind == "match_finished" for event in result.events)
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
