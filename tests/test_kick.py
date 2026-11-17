import math
from dataclasses import replace

import pytest

from game.actions import Direction, MovementAction
from game.config import MATCH_TICKS, PHYSICS_DT, BallConfig, SimulationConfig
from game.player import PlayerId
from game.simulation import (
    MatchPhase,
    assign_possession,
    release_kick,
    reset,
    step_decision,
    step_tick,
)


@pytest.mark.parametrize("owner", list(PlayerId))
@pytest.mark.parametrize("angle", [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2])
def test_release_is_atomic_along_facing_and_emits_one_kick(owner, angle):
    initial = reset()
    field = "player_a" if owner == PlayerId.A else "player_b"
    carrier = replace(getattr(initial, field), facing=angle, velocity=(60.0, 20.0), stamina=37.0)
    initial = assign_possession(replace(initial, **{field: carrier}), owner).state
    result = release_kick(initial, owner, launch_speed=300.0)
    state = result.state
    assert state.ball.position == initial.ball.position
    assert state.ball.velocity == pytest.approx((300 * math.cos(angle), 300 * math.sin(angle)))
    assert math.hypot(*state.ball.velocity) == pytest.approx(300.0)
    assert state.ball.possession_owner is None
    assert state.ball.last_touch_player == owner
    assert not state.has_possession(PlayerId.A)
    assert not state.has_possession(PlayerId.B)
    released_carrier = getattr(state, field)
    assert released_carrier.pickup_lockout_ticks == 12
    assert replace(released_carrier, pickup_lockout_ticks=0) == carrier
    assert replace(state, ball=initial.ball, **{field: carrier}) == initial
    assert result.physics_ticks == 0
    assert initial.ball.possession_owner == owner  # Immutable input was not changed.
    assert len(result.events) == 1
    event = result.events[0]
    assert (event.kind, event.tick, event.previous_owner, event.new_owner, event.cause) == (
        "ball_kicked",
        initial.simulation_tick,
        owner,
        None,
        "kick",
    )


def test_release_synchronizes_attachment_from_current_carrier_and_config():
    initial = assign_possession(reset(), PlayerId.A).state
    initial = replace(initial, player_a=replace(initial.player_a, position=(785.0, 250.0)))
    config = SimulationConfig(ball=BallConfig(dribble_offset=30.0))
    result = release_kick(initial, PlayerId.A, launch_speed=150.0, config=config)
    assert result.state.ball.position == (815.0, 250.0)  # No pitch clamp.
    assert result.state.ball.velocity == (150.0, 0.0)


def test_free_motion_starts_on_next_tick_and_does_not_repeat_kick():
    initial = assign_possession(reset(), PlayerId.A).state
    released = release_kick(initial, PlayerId.A, launch_speed=300.0)
    assert released.state.simulation_tick == 0
    assert released.state.ball.position == (222.0, 250.0)
    tick = step_tick(released.state, MovementAction(Direction.NORTH, 1.0))
    distance = 300 * PHYSICS_DT - 0.5 * 100 * PHYSICS_DT**2
    assert tick.state.ball.position == pytest.approx((222.0 + distance, 250.0))
    assert tick.state.ball.velocity == pytest.approx((300 - 100 * PHYSICS_DT, 0.0))
    assert tick.state.ball.last_touch_player == PlayerId.A
    assert tick.events == ()
    assert step_decision(tick.state).events == ()
    with pytest.raises(ValueError, match="possession"):
        release_kick(tick.state, PlayerId.A, launch_speed=300.0)


def test_reference_kick_rolls_450_units_then_stays_stopped():
    initial = reset()
    initial = replace(initial, player_b=replace(initial.player_b, position=(600.0, 400.0)))
    state = release_kick(
        assign_possession(initial, PlayerId.A).state, PlayerId.A, launch_speed=300.0
    ).state
    for _ in range(180):
        result = step_tick(state)
        state = result.state
        assert result.events == ()
    assert state.ball.position == pytest.approx((672.0, 250.0))
    assert state.ball.velocity == (0.0, 0.0)
    assert step_tick(state).state.ball == state.ball


@pytest.mark.parametrize("speed", [1.0, 150.0, 300.0, 600.0])
def test_explicit_speed_is_not_capped_by_reference_speed(speed):
    initial = assign_possession(reset(), PlayerId.A).state
    assert release_kick(initial, PlayerId.A, launch_speed=speed).state.ball.velocity == (speed, 0.0)


@pytest.mark.parametrize("speed", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_nonpositive_or_nonfinite_speed_is_rejected(speed):
    initial = assign_possession(reset(), PlayerId.A).state
    with pytest.raises(ValueError, match="speed"):
        release_kick(initial, PlayerId.A, launch_speed=speed)


@pytest.mark.parametrize("player", ["A", None, 0])
def test_kicker_must_be_typed_player_id(player):
    initial = assign_possession(reset(), PlayerId.A).state
    with pytest.raises(TypeError):
        release_kick(initial, player, launch_speed=300.0)


def test_free_ball_and_nonowner_cannot_kick():
    initial = reset()
    with pytest.raises(ValueError, match="possession"):
        release_kick(initial, PlayerId.A, launch_speed=300.0)
    initial = assign_possession(initial, PlayerId.B).state
    with pytest.raises(ValueError, match="possession"):
        release_kick(initial, PlayerId.A, launch_speed=300.0)


@pytest.mark.parametrize(
    "phase", [MatchPhase.GOAL_RESTART, MatchPhase.OUT_OF_BOUNDS_RESTART, MatchPhase.FINISHED]
)
def test_release_is_forbidden_outside_playing(phase):
    initial = assign_possession(reset(), PlayerId.A).state
    initial = replace(
        initial,
        phase=phase,
        restart_ticks_remaining=0 if phase == MatchPhase.FINISHED else 6,
        simulation_tick=MATCH_TICKS if phase == MatchPhase.FINISHED else 0,
    )
    with pytest.raises(ValueError, match="playing"):
        release_kick(initial, PlayerId.A, launch_speed=300.0)


def test_swapped_side_release_uses_facing_and_identity():
    initial = assign_possession(reset(swap_sides=True), PlayerId.A).state
    result = release_kick(initial, PlayerId.A, launch_speed=300.0)
    assert result.state.ball.position == pytest.approx((578.0, 250.0))
    assert result.state.ball.velocity == pytest.approx((-300.0, 0.0))
    assert result.state.ball.last_touch_player == PlayerId.A


def test_release_replay_and_terminal_event_are_deterministic():
    initial = assign_possession(reset(), PlayerId.A).state
    initial = replace(initial, simulation_tick=MATCH_TICKS - 6)

    def trajectory():
        release = release_kick(initial, PlayerId.A, launch_speed=300.0)
        tick = step_decision(release.state)
        assert [event.kind for event in release.events + tick.events] == [
            "ball_kicked",
            "match_finished",
        ]
        assert tick.state.phase == MatchPhase.FINISHED
        assert step_tick(tick.state).events == ()
        return release, tick

    assert trajectory() == trajectory()
