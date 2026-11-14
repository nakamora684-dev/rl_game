"""Deterministic movement tests without graphics or RL libraries."""

import math
from dataclasses import replace

import pytest

from game.actions import Direction, MovementAction
from game.config import PHYSICS_DT, MovementConfig, PitchConfig, SimulationConfig, StaminaConfig
from game.player import PlayerState, constrain_to_pitch, step_idle_player
from game.player import step_player as step_player_with_stamina

# Isolate the existing movement equations; fatigue is covered in test_stamina.py.
_NO_FATIGUE = StaminaConfig(
    drain_rate=0.0, recovery_rate=0.0, min_speed_factor=1.0, min_acceleration_factor=1.0
)


def step_player(player, action, **kwargs):
    return step_player_with_stamina(player, action, stamina=_NO_FATIGUE, **kwargs)


@pytest.mark.parametrize("direction", list(Direction)[1:])
def test_acceleration_reaches_same_speed_in_every_direction(direction):
    player = PlayerState(position=(0.0, 0.0))
    action = MovementAction(direction, effort=1.0)
    for tick in range(1, 61):
        player = step_player(player, action)
        assert math.hypot(*player.velocity) == pytest.approx(min(160.0, tick * 320.0 / 60))
        assert math.hypot(*player.velocity) <= 160.0 + 1e-12
    assert player.velocity == pytest.approx(
        tuple(component * 160 for component in direction.vector)
    )


def test_first_tick_accelerates_before_position_update():
    player = PlayerState(position=(10.0, 20.0))
    result = step_player(player, MovementAction(Direction.EAST, 1.0))
    assert result.velocity == pytest.approx((320.0 / 60, 0.0))
    assert result.position == pytest.approx((10.0 + 320.0 / 3600, 20.0))
    assert player.position == (10.0, 20.0)
    assert player.velocity == (0.0, 0.0)


@pytest.mark.parametrize("effort", [0.0, 0.1, 0.5, 1.0])
def test_effort_sets_cruising_speed(effort):
    player = PlayerState(position=(0.0, 0.0), stamina=37.0)
    for _ in range(60):
        player = step_player(player, MovementAction(Direction.EAST, effort))
    assert player.velocity == pytest.approx((160.0 * effort, 0.0))
    assert player.stamina == 37.0  # Fatigue is explicitly disabled for this isolated test.


def test_lower_effort_decelerates_without_instant_speed_clamp():
    player = PlayerState(position=(0.0, 0.0), velocity=(160.0, 0.0))
    action = MovementAction(Direction.EAST, 0.5)
    result = step_player(player, action)
    assert result.velocity[0] == pytest.approx(160.0 - 320.0 / 60)
    for _ in range(14):
        result = step_player(result, action)
    assert result.velocity == pytest.approx((80.0, 0.0))
    assert step_player(result, action).velocity == pytest.approx((80.0, 0.0))


def test_small_positive_effort_and_zero_effort_use_documented_different_rates():
    player = PlayerState(position=(0.0, 0.0), velocity=(160.0, 0.0))
    coasting = step_player(player, MovementAction(Direction.EAST, 0.0))
    steering = step_player(player, MovementAction(Direction.EAST, 1e-6))
    assert coasting.velocity[0] == pytest.approx(160.0 - 40.0 / 60)
    assert steering.velocity[0] == pytest.approx(160.0 - 320.0 / 60)


def test_reversal_is_acceleration_limited():
    player = PlayerState(position=(0.0, 0.0), velocity=(160.0, 0.0))
    action = MovementAction(Direction.WEST, 1.0)
    for _ in range(60):
        result = step_player(player, action)
        change = math.dist(result.velocity, player.velocity)
        assert change <= 320.0 / 60 + 1e-12
        player = result
    assert player.velocity == pytest.approx((-160.0, 0.0))


@pytest.mark.parametrize("effort", [0.0, 0.5, 1.0])
def test_idle_ignores_effort_for_motion_and_preserves_facing(effort):
    player = PlayerState(position=(0.0, 0.0), velocity=(6.0, 0.0), facing=0.7)
    assert step_player(player, MovementAction(Direction.IDLE, effort)) == step_idle_player(player)


def test_zero_effort_brakes_but_nonidle_direction_can_turn_facing():
    player = PlayerState(position=(0.0, 0.0), velocity=(6.0, 0.0))
    result = step_player(player, MovementAction(Direction.NORTH, 0.0))
    assert result.position == step_idle_player(player).position
    assert result.velocity == step_idle_player(player).velocity
    assert result.facing == pytest.approx(-math.tau / 60)


def test_facing_turns_smoothly_independently_of_velocity():
    player = PlayerState(position=(0.0, 0.0), velocity=(160.0, 0.0))
    action = MovementAction(Direction.NORTH, 1.0)
    result = step_player(player, action)
    assert result.facing == pytest.approx(-math.tau / 60)
    assert result.velocity[0] > 0.0  # Still sliding east while turning north.
    for _ in range(14):
        result = step_player(result, action)
    assert result.facing == pytest.approx(-math.pi / 2)


def test_facing_wrap_uses_shortest_arc():
    player = PlayerState(position=(0.0, 0.0), facing=math.pi - 0.01)
    result = step_player(player, MovementAction(Direction.NORTHWEST, 0.0))
    delta = (result.facing - player.facing + math.pi) % math.tau - math.pi
    assert delta > 0.0
    assert delta == pytest.approx(math.tau * PHYSICS_DT)


def test_exact_opposite_facing_has_deterministic_counterclockwise_tie():
    player = PlayerState(position=(0.0, 0.0))
    result = step_player(player, MovementAction(Direction.WEST, 0.0))
    assert result.facing == pytest.approx(-math.tau / 60)


@pytest.mark.parametrize("direction", list(Direction)[1:])
def test_facing_forward_vector_matches_y_down_direction(direction):
    player = PlayerState(position=(0.0, 0.0))
    for _ in range(60):
        player = step_player(player, MovementAction(direction, 0.0))
    assert (math.cos(player.facing), math.sin(player.facing)) == pytest.approx(direction.vector)


def test_facing_can_turn_while_player_is_pinned_to_wall():
    player = PlayerState(position=(785.0, 250.0), facing=-math.pi / 2)
    result = step_player(player, MovementAction(Direction.EAST, 1.0), pitch=PitchConfig())
    assert result.position == player.position
    assert result.velocity == (0.0, 0.0)
    assert result.facing == pytest.approx(-math.pi / 2 + math.tau / 60)


def test_zero_turn_rate_keeps_orientation():
    player = PlayerState(position=(0.0, 0.0), facing=0.7)
    result = step_player(
        player, MovementAction(Direction.NORTH, 1.0), movement=MovementConfig(turn_rate=0.0)
    )
    assert result.facing == 0.7


def test_zero_acceleration_cannot_start_driven_motion():
    player = PlayerState(position=(0.0, 0.0))
    result = step_player(
        player, MovementAction(Direction.EAST, 1.0), movement=MovementConfig(acceleration=0.0)
    )
    assert result.position == player.position
    assert result.velocity == player.velocity


@pytest.mark.parametrize(
    ("position", "velocity", "expected_position", "expected_velocity"),
    [
        ((14.0, 100.0), (-10.0, 4.0), (15.0, 100.0), (0.0, 4.0)),
        ((786.0, 100.0), (10.0, 4.0), (785.0, 100.0), (0.0, 4.0)),
        ((100.0, 14.0), (4.0, -10.0), (100.0, 15.0), (4.0, 0.0)),
        ((100.0, 486.0), (4.0, 10.0), (100.0, 485.0), (4.0, 0.0)),
        ((786.0, 486.0), (10.0, 10.0), (785.0, 485.0), (0.0, 0.0)),
        ((15.0, 100.0), (10.0, 4.0), (15.0, 100.0), (10.0, 4.0)),
    ],
)
def test_pitch_bounds_cancel_only_outward_velocity(
    position, velocity, expected_position, expected_velocity
):
    result = constrain_to_pitch(PlayerState(position, velocity), PitchConfig())
    assert result.position == expected_position
    assert result.velocity == expected_velocity


def test_driven_motion_cannot_escape_pitch():
    player = PlayerState(position=(784.9, 250.0), velocity=(160.0, 0.0))
    for _ in range(60):
        player = step_player(player, MovementAction(Direction.EAST, 1.0), pitch=PitchConfig())
        assert player.position == (785.0, 250.0)
        assert player.velocity == (0.0, 0.0)


def test_player_can_move_away_from_wall():
    player = PlayerState(position=(785.0, 250.0))
    result = step_player(player, MovementAction(Direction.WEST, 1.0), pitch=PitchConfig())
    assert result.position[0] < player.position[0]
    assert result.velocity[0] < 0.0


@pytest.mark.parametrize("effort", [-0.1, 1.1, math.inf, math.nan])
def test_invalid_effort_is_rejected(effort):
    with pytest.raises(ValueError):
        MovementAction(Direction.EAST, effort)


def test_raw_direction_id_requires_explicit_adapter_conversion():
    with pytest.raises(TypeError):
        MovementAction(3, 1.0)
    assert MovementAction(Direction(3), 1.0).direction == Direction.EAST
    with pytest.raises(ValueError):
        Direction(9)


@pytest.mark.parametrize("field", ["max_speed", "acceleration", "braking", "turn_rate"])
@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_invalid_movement_parameters(field, value):
    with pytest.raises(ValueError):
        replace(MovementConfig(), **{field: value})


@pytest.mark.parametrize(
    "fields", [{"width": 0.0}, {"height": math.inf}, {"player_radius": -1.0}, {"width": 20.0}]
)
def test_invalid_pitch_parameters(fields):
    with pytest.raises(ValueError):
        PitchConfig(**fields)


def test_player_copies_mutable_vector_inputs():
    position, velocity = [10.0, 20.0], [3.0, 4.0]
    player = PlayerState(position, velocity)
    position[0], velocity[0] = 999.0, 999.0
    assert player.position == (10.0, 20.0)
    assert player.velocity == (3.0, 4.0)


def test_overflowing_velocity_magnitude_is_rejected_at_construction():
    with pytest.raises(ValueError, match="Velocity magnitude"):
        PlayerState(position=(0.0, 0.0), velocity=(1.7e308, 1.7e308))


def test_configs_do_not_share_mutable_objects():
    first, second = SimulationConfig(), SimulationConfig()
    assert first == second
    assert first.movement is not second.movement
