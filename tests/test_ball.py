import math
from dataclasses import replace

import pytest

from game.ball import BallState, step_free_ball
from game.config import PHYSICS_DT, BallConfig, SimulationConfig
from game.simulation import MatchPhase, reset, step_decision, step_tick


def test_reference_roll_stops_after_three_seconds_and_450_units():
    ball = BallState(position=(0.0, 0.0), velocity=(300.0, 0.0))
    for tick in range(1, 181):
        ball = step_free_ball(ball)
        assert ball.velocity[0] == pytest.approx(max(0.0, 300.0 - tick * 100.0 / 60))
        if tick < 180:
            assert ball.velocity[0] > 0.0
    assert ball.velocity == (0.0, 0.0)
    assert ball.position == pytest.approx((450.0, 0.0))
    assert step_free_ball(ball) == ball


def test_weaker_roll_stops_sooner_with_same_deceleration():
    ball = BallState(position=(0.0, 0.0), velocity=(150.0, 0.0))
    for _ in range(90):
        ball = step_free_ball(ball)
    assert ball.velocity == (0.0, 0.0)
    assert ball.position == pytest.approx((112.5, 0.0))


@pytest.mark.parametrize("velocity", [(180.0, 240.0), (-180.0, 240.0), (-180.0, -240.0)])
def test_deceleration_preserves_direction_and_distance(velocity):
    ball = BallState(position=(10.0, 20.0), velocity=velocity)
    initial = ball
    for _ in range(180):
        ball = step_free_ball(ball)
    assert ball.velocity == (0.0, 0.0)
    assert ball.position == pytest.approx(
        tuple(
            start + component / 300.0 * 450.0
            for start, component in zip(initial.position, velocity)
        )
    )
    assert initial.velocity == velocity


def test_first_tick_integrates_average_speed():
    initial = BallState(position=(1.0, 2.0), velocity=(300.0, 0.0))
    result = step_free_ball(initial)
    expected_speed = 300.0 - 100.0 * PHYSICS_DT
    assert result.velocity == pytest.approx((expected_speed, 0.0))
    assert result.position == pytest.approx((1.0 + (300.0 + expected_speed) / 2 * PHYSICS_DT, 2.0))
    assert initial.position == (1.0, 2.0)


def test_stops_partway_through_tick_without_reversing_or_losing_final_travel():
    initial = BallState(position=(0.0, 0.0), velocity=(0.1, 0.0))
    result = step_free_ball(initial)
    assert result.velocity == (0.0, 0.0)
    assert result.position == pytest.approx((0.1**2 / (2 * 100.0), 0.0))


def test_stationary_ball_stays_stationary():
    ball = BallState(position=(12.0, 24.0))
    assert step_free_ball(ball) == ball


def test_zero_deceleration_coasts_even_at_tiny_speed():
    config = BallConfig(deceleration=0.0)
    ball = BallState(position=(0.0, 0.0), velocity=(1e-15, -1e-15))
    result = step_free_ball(ball, config=config)
    assert result.velocity == ball.velocity
    assert result.position[0] > 0.0
    assert result.position[1] < 0.0


def test_reference_speed_is_not_a_speed_cap():
    ball = BallState(position=(0.0, 0.0), velocity=(600.0, 0.0))
    result = step_free_ball(ball)
    assert result.velocity[0] == pytest.approx(600.0 - 100.0 / 60)
    assert result.velocity[0] > BallConfig().reference_speed


def test_reset_places_stationary_ball_in_pitch_center():
    state = reset()
    assert state.ball == BallState(position=(400.0, 250.0))
    assert reset(swap_sides=True).ball == state.ball


def test_ball_can_cross_pitch_boundary_without_clamping_or_automatic_scoring():
    state = reset()
    state = replace(state, ball=BallState(position=(799.0, 250.0), velocity=(300.0, 0.0)))
    result = step_tick(state)
    assert result.state.ball.position[0] > 800.0
    assert result.state.phase == MatchPhase.PLAYING
    assert result.events == ()


def test_simulation_holds_six_separate_ball_updates_per_decision():
    state = reset()
    state = replace(state, ball=replace(state.ball, velocity=(300.0, 0.0)))
    result = step_decision(state)
    assert result.state.ball.velocity == pytest.approx((290.0, 0.0))
    assert result.state.ball.position == pytest.approx((429.5, 250.0))


def test_ball_config_is_used_by_simulation():
    state = reset()
    state = replace(state, ball=replace(state.ball, velocity=(300.0, 0.0)))
    result = step_decision(state, config=SimulationConfig(ball=BallConfig(deceleration=200.0)))
    assert result.state.ball.velocity == pytest.approx((280.0, 0.0))


def test_restart_and_finished_scaffolds_freeze_ball():
    state = reset()
    state = replace(
        state,
        ball=replace(state.ball, velocity=(300.0, 0.0)),
        phase=MatchPhase.GOAL_RESTART,
        restart_ticks_remaining=6,
    )
    result = step_decision(state)
    assert result.state.ball == state.ball
    assert step_tick(result.state).state.ball.position[0] > state.ball.position[0]
    finished = replace(
        state, simulation_tick=3600, phase=MatchPhase.FINISHED, restart_ticks_remaining=0
    )
    assert step_tick(finished).state.ball == state.ball


def test_reset_does_not_reuse_moving_ball_history():
    initial = reset()
    moving = replace(initial, ball=replace(initial.ball, velocity=(300.0, 0.0)))
    assert step_tick(moving).state.ball != initial.ball
    assert reset().ball == initial.ball


def test_same_initial_ball_state_replays_identically():
    initial = BallState(position=(0.0, 0.0), velocity=(180.0, 240.0))

    def trajectory():
        ball = initial
        states = []
        for _ in range(240):
            ball = step_free_ball(ball)
            states.append(ball)
        return states

    assert trajectory() == trajectory()


def test_ball_owns_immutable_vectors():
    position, velocity = [1.0, 2.0], [3.0, 4.0]
    ball = BallState(position=position, velocity=velocity)
    position[0], velocity[0] = 999.0, 999.0
    assert ball.position == (1.0, 2.0)
    assert ball.velocity == (3.0, 4.0)


@pytest.mark.parametrize(
    "fields",
    [
        {"position": (math.inf, 0.0)},
        {"position": (0.0,)},
        {"velocity": (math.nan, 0.0)},
        {"velocity": (0.0, 0.0, 0.0)},
        {"velocity": (1.7e308, 1.7e308)},
    ],
)
def test_invalid_ball_state_is_rejected(fields):
    with pytest.raises(ValueError):
        BallState(**{"position": (0.0, 0.0), **fields})


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_invalid_ball_deceleration_is_rejected(value):
    with pytest.raises(ValueError):
        BallConfig(deceleration=value)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_invalid_reference_speed_is_rejected(value):
    with pytest.raises(ValueError):
        BallConfig(reference_speed=value)
