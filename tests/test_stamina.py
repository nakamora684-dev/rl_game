"""Energy bookkeeping and its integration with gradual player motion."""

import math
from dataclasses import replace

import pytest

from game.actions import Direction, MovementAction
from game.config import PHYSICS_DT, MovementConfig, PitchConfig, SimulationConfig, StaminaConfig
from game.player import PlayerState, step_player
from game.simulation import MatchPhase, reset, step_decision, step_tick
from game.stamina import performance_factors, update_stamina

SPRINT = MovementAction(Direction.EAST, 1.0)
REST = MovementAction()
FROZEN_ENERGY = StaminaConfig(drain_rate=0.0, recovery_rate=0.0)


def advance_energy(value, action, ticks):
    for _ in range(ticks):
        value = update_stamina(value, action)
    return value


def test_full_effort_consumes_five_points_per_second_and_twenty_seconds_to_empty():
    assert advance_energy(100.0, SPRINT, 60) == pytest.approx(95.0)
    assert advance_energy(100.0, SPRINT, 1200) == pytest.approx(0.0, abs=1e-9)
    assert advance_energy(100.0, SPRINT, 3600) == 0.0


def test_rest_recovers_empty_stamina_in_fifteen_seconds():
    assert advance_energy(0.0, REST, 60) == pytest.approx(100.0 / 15)
    assert advance_energy(0.0, REST, 900) == pytest.approx(100.0)
    assert advance_energy(0.0, REST, 3600) == 100.0


@pytest.mark.parametrize("value", [0.0, 37.0, 100.0])
def test_neutral_effort_neither_drains_nor_recovers(value):
    assert advance_energy(value, MovementAction(Direction.NORTH, 0.5), 600) == value


@pytest.mark.parametrize("effort", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_idle_always_counts_as_rest(effort):
    assert update_stamina(50.0, MovementAction(Direction.IDLE, effort)) == update_stamina(
        50.0, REST
    )


def test_rates_vary_linearly_with_effort():
    assert advance_energy(50.0, MovementAction(Direction.EAST, 0.75), 60) == pytest.approx(47.5)
    assert advance_energy(50.0, MovementAction(Direction.EAST, 0.25), 60) == pytest.approx(
        50 + 100.0 / 30
    )


def test_stamina_rate_is_continuous_at_neutral_effort():
    below = update_stamina(50.0, MovementAction(Direction.EAST, 0.5 - 1e-6))
    above = update_stamina(50.0, MovementAction(Direction.EAST, 0.5 + 1e-6))
    assert 0 < below - 50.0 < 1e-6
    assert 0 < 50.0 - above < 1e-6


def test_clamps_without_overshoot_at_both_limits():
    assert update_stamina(0.01, SPRINT) == 0.0
    assert update_stamina(99.99, REST) == 100.0
    assert update_stamina(0.0, SPRINT) == 0.0
    assert update_stamina(100.0, REST) == 100.0


@pytest.mark.parametrize(("value", "factor"), [(0.0, 0.5), (50.0, 0.75), (100.0, 1.0)])
def test_linear_speed_and_acceleration_factors(value, factor):
    assert performance_factors(value) == (factor, factor)


def test_speed_and_acceleration_minima_are_independently_configurable():
    config = StaminaConfig(min_speed_factor=0.6, min_acceleration_factor=0.4)
    assert performance_factors(0.0, config=config) == (0.6, 0.4)
    assert performance_factors(50.0, config=config) == pytest.approx((0.8, 0.7))


def test_stamina_is_updated_before_movement_limits_are_computed():
    result = step_player(PlayerState((0.0, 0.0)), SPRINT)
    expected_stamina = 100.0 - 5.0 * PHYSICS_DT
    expected_factor = 0.5 + 0.5 * expected_stamina / 100
    expected_speed = 320.0 * expected_factor * PHYSICS_DT
    assert result.stamina == pytest.approx(expected_stamina)
    assert result.velocity == pytest.approx((expected_speed, 0.0))
    assert result.position == pytest.approx((expected_speed * PHYSICS_DT, 0.0))


def test_fatigue_lowers_target_without_instantly_scaling_existing_velocity():
    player = PlayerState(position=(0.0, 0.0), velocity=(160.0, 0.0), stamina=50.0)
    result = step_player(player, SPRINT, stamina=FROZEN_ENERGY)
    assert result.velocity == pytest.approx((156.0, 0.0))  # 160 - (320 * 0.75)/60.
    assert result.velocity[0] > 120.0  # Fatigued target, not an instantaneous velocity clamp.
    assert player.velocity == (160.0, 0.0)


def test_factors_are_recomputed_from_base_configuration_without_compounding():
    movement = MovementConfig()
    player = PlayerState(position=(0.0, 0.0), stamina=50.0)
    first = step_player(player, SPRINT, movement=movement, stamina=FROZEN_ENERGY)
    second = step_player(first, SPRINT, movement=movement, stamina=FROZEN_ENERGY)
    assert first.velocity == pytest.approx((4.0, 0.0))
    assert second.velocity == pytest.approx((8.0, 0.0))
    assert movement.max_speed == 160.0
    assert movement.acceleration == 320.0


@pytest.mark.parametrize(("value", "speed"), [(0.0, 80.0), (50.0, 120.0), (100.0, 160.0)])
def test_fatigued_player_can_still_move_at_reduced_cruising_speed(value, speed):
    player = PlayerState(position=(0.0, 0.0), stamina=value)
    for _ in range(60):
        player = step_player(player, SPRINT, stamina=FROZEN_ENERGY)
    assert player.velocity == pytest.approx((speed, 0.0))


def test_idle_recovery_never_multiplies_existing_velocity():
    player = PlayerState(position=(0.0, 0.0), velocity=(80.0, 0.0), stamina=0.0)
    movement = MovementConfig(braking=0.0)
    for _ in range(60):
        player = step_player(player, REST, movement=movement)
    assert player.velocity == (80.0, 0.0)
    assert player.stamina == pytest.approx(100.0 / 15)


def test_braking_and_facing_do_not_depend_on_stamina():
    fresh = PlayerState(position=(0.0, 0.0), velocity=(6.0, 0.0), facing=0.0)
    tired = replace(fresh, stamina=0.0)
    action = MovementAction(Direction.NORTH, 0.0)
    first = step_player(fresh, action)
    second = step_player(tired, action)
    assert first.velocity == second.velocity
    assert first.position == second.position
    assert first.facing == second.facing


def test_nonidle_zero_effort_recovers_while_turning_and_coasting():
    player = PlayerState(position=(0.0, 0.0), velocity=(6.0, 0.0), stamina=50.0)
    result = step_player(player, MovementAction(Direction.NORTH, 0.0))
    assert result.stamina == pytest.approx(50.0 + (100.0 / 15) * PHYSICS_DT)
    assert result.facing == pytest.approx(-math.tau * PHYSICS_DT)
    assert result.velocity == pytest.approx((6.0 - 40.0 * PHYSICS_DT, 0.0))


def test_driving_against_wall_consumes_requested_effort():
    player = PlayerState(position=(785.0, 250.0), stamina=50.0)
    for _ in range(60):
        player = step_player(player, SPRINT, pitch=PitchConfig())
    assert player.position == (785.0, 250.0)
    assert player.velocity == (0.0, 0.0)
    assert player.stamina == pytest.approx(45.0)


def test_six_tick_decision_updates_energy_once_per_physics_tick_for_both_players():
    state = reset()
    state = replace(state, player_b=replace(state.player_b, stamina=50.0))
    result = step_decision(state, SPRINT, REST)
    assert result.state.player_a.stamina == pytest.approx(99.5)
    assert result.state.player_b.stamina == pytest.approx(50.0 + 100.0 / 150)
    assert result.state.player_a.velocity == pytest.approx((31.953333333333333, 0.0))


def test_both_players_use_the_same_stamina_rules():
    result = step_decision(reset(), SPRINT, MovementAction(Direction.WEST, 1.0))
    assert result.state.player_a.stamina == result.state.player_b.stamina
    assert math.hypot(*result.state.player_a.velocity) == math.hypot(
        *result.state.player_b.velocity
    )


def test_custom_stamina_config_reaches_simulation_step():
    config = SimulationConfig(stamina=StaminaConfig(drain_rate=10.0))
    result = step_decision(reset(), SPRINT, config=config)
    assert result.state.player_a.stamina == pytest.approx(99.0)


def test_truncated_final_decision_updates_stamina_only_until_whistle():
    state = reset()
    state = replace(
        state,
        simulation_tick=3599,
        player_a=replace(state.player_a, stamina=60.0),
        player_b=replace(state.player_b, stamina=40.0),
    )
    result = step_decision(state, SPRINT, REST)
    assert result.physics_ticks == 1
    assert result.state.player_a.stamina == pytest.approx(60.0 - 5.0 * PHYSICS_DT)
    assert result.state.player_b.stamina == pytest.approx(40.0 + (100.0 / 15) * PHYSICS_DT)
    assert step_decision(result.state, SPRINT, REST).state is result.state


def test_complete_match_sprint_rest_checkpoints_and_different_player_stamina():
    state = reset()
    state = replace(state, player_b=replace(state.player_b, stamina=40.0))
    neutral = MovementAction(Direction.WEST, 0.5)
    event_count = 0
    for decision in range(600):
        action = SPRINT if decision < 200 else REST if decision < 350 else neutral
        result = step_decision(state, action, neutral)
        state = result.state
        event_count += len(result.events)
        assert 0.0 <= state.player_a.stamina <= 100.0
        assert state.player_b.stamina == 40.0
        if decision == 199:
            assert state.player_a.stamina == pytest.approx(0.0, abs=1e-9)
        if decision == 349:
            assert state.player_a.stamina == pytest.approx(100.0)
    assert state.phase == MatchPhase.FINISHED
    assert event_count == 1


def test_reset_restores_full_stamina_without_mutating_old_snapshot():
    original = reset()
    tired = step_decision(original, SPRINT).state
    assert tired.player_a.stamina < 100.0
    assert original.player_a.stamina == 100.0
    assert reset().player_a.stamina == 100.0


def test_restart_scaffold_and_finished_states_do_not_update_stamina():
    state = reset()
    state = replace(
        state,
        player_a=replace(state.player_a, stamina=50.0),
        phase=MatchPhase.GOAL_RESTART,
        restart_ticks_remaining=6,
    )
    result = step_decision(state, SPRINT)
    assert result.state.player_a.stamina == 50.0
    assert step_tick(result.state).state.player_a.stamina > 50.0
    finished = replace(reset(), simulation_tick=3600, phase=MatchPhase.FINISHED)
    assert step_decision(finished, SPRINT).state is finished


@pytest.mark.parametrize("value", [-1.0, 101.0, math.inf, math.nan])
def test_invalid_stamina_is_rejected_by_pure_helpers(value):
    with pytest.raises(ValueError):
        update_stamina(value, REST)
    with pytest.raises(ValueError):
        performance_factors(value)


@pytest.mark.parametrize("field", ["drain_rate", "recovery_rate"])
@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_invalid_stamina_rates(field, value):
    with pytest.raises(ValueError):
        replace(StaminaConfig(), **{field: value})


@pytest.mark.parametrize("value", [0.0, 1.0, math.inf, math.nan])
def test_neutral_effort_must_be_strictly_inside_unit_interval(value):
    with pytest.raises(ValueError):
        StaminaConfig(neutral_effort=value)


@pytest.mark.parametrize("field", ["min_speed_factor", "min_acceleration_factor"])
@pytest.mark.parametrize("value", [-0.1, 1.1, math.inf, math.nan])
def test_invalid_performance_floors(field, value):
    with pytest.raises(ValueError):
        replace(StaminaConfig(), **{field: value})
