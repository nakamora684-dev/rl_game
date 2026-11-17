"""Regression checks for the 17 Nov runner audit, not relaxed audit oracles."""

import math
from dataclasses import replace

import pytest

from game.ball import BallState
from game.config import PitchConfig, RulesConfig, SimulationConfig
from game.player import PlayerId, PlayerState
from game.scenarios import build_scenarios, run_scenario
from game.simulation import assign_possession, reset, step_decision, step_tick


@pytest.mark.parametrize(
    "name",
    [
        "14_coincident_bodies",
        "23_restart_expiry_no_retry",
        "39_known_touch_boundary_restart",
        "41_full_restart_freeze",
        "48_random_near_posts",
        "50_unsupported_narrow_reset",
    ],
)
def test_original_audit_findings_are_resolved_without_suppressing_checks(name):
    case = next(case for case in build_scenarios() if case.name == name)
    result = run_scenario(case)
    assert result["status"] == "PASS", (result["errors"], result["model_warnings"])
    assert result["replay_identical"]
    assert result["ticks_advanced"] == case.ticks


@pytest.mark.parametrize("lockout", [0, 12])
def test_stationary_body_separation_does_not_launch_a_stationary_free_ball(lockout):
    player = PlayerState((400.0, 250.0), pickup_lockout_ticks=lockout)
    state = replace(reset(), player_a=player, player_b=player, ball=BallState(player.position))
    result = step_tick(state)
    ball = result.state.ball
    assert ball.possession_owner is None
    assert ball.velocity == (0.0, 0.0)
    for body in (result.state.player_a, result.state.player_b):
        assert math.dist(ball.position, body.position) >= 22 - 1e-8
    assert all(event.velocity_after == (0.0, 0.0) for event in result.events)
    assert result.state.player_a.velocity == result.state.player_b.velocity == (0.0, 0.0)


def test_stationary_carrier_correction_into_post_does_not_invent_attachment_velocity():
    player = PlayerState((765.0, 186.0))
    state = replace(reset(), player_a=player, player_b=player)
    state = assign_possession(state, PlayerId.B).state
    result = step_tick(state)
    assert any(event.cause == "post_collision" for event in result.events)
    assert result.state.ball.velocity == (0.0, 0.0)
    assert all(
        event.velocity_before == event.velocity_after == (0.0, 0.0)
        for event in result.events
        if event.kind == "ball_collision"
    )


def test_real_moving_locked_player_still_transfers_normal_velocity_without_recoil():
    state = reset()
    state = replace(
        state,
        player_a=replace(state.player_a, velocity=(180.0, 0.0), pickup_lockout_ticks=12),
        ball=BallState((222.0, 250.0)),
    )
    result = step_tick(state)
    player = result.state.player_a
    assert player.velocity == pytest.approx((180.0 - 40.0 / 60, 0.0))
    collision = next(event for event in result.events if event.kind == "ball_collision")
    assert collision.velocity_before == (0.0, 0.0)
    assert collision.velocity_after == pytest.approx((1.5 * player.velocity[0], 0.0))
    assert result.state.ball.velocity[0] > 0.0
    assert result.state.ball.last_touch_player == PlayerId.A


@pytest.mark.parametrize("swap", [False, True])
def test_narrow_reset_is_separated_before_first_tick_and_swaps_identities(swap):
    config = SimulationConfig(
        pitch=PitchConfig(width=50.0, height=100.0), rules=RulesConfig(goal_width=40.0)
    )
    state = reset(config=config, swap_sides=swap)
    assert math.dist(state.player_a.position, state.player_b.position) >= 30
    assert state.ball.position == (25.0, 50.0)
    assert state.attack_direction_a == (-1 if swap else 1)
    other = reset(config=config, swap_sides=not swap)
    assert state.player_a == other.player_b
    assert state.player_b == other.player_a
    result = step_decision(state, config=config)
    assert result.physics_ticks == 6
    assert result.events == ()


def test_reset_rejects_geometry_that_cannot_fit_two_players_before_publishing_state():
    config = SimulationConfig(
        pitch=PitchConfig(width=32.0, height=32.0), rules=RulesConfig(goal_width=20.0)
    )
    with pytest.raises(ValueError, match="reset|Reset"):
        reset(config=config)


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("touch", [PlayerId.A, PlayerId.B, None])
@pytest.mark.parametrize("goal", [False, True])
def test_restart_event_describes_resulting_ball_not_previous_exit(swap, touch, goal):
    state = reset(swap_sides=swap)
    state = replace(
        state,
        ball=BallState((804.0, 250.0 if goal else 100.0), (300.0, 0.0), last_touch_player=touch),
    )
    crossed = step_tick(state)
    assert crossed.events[-1].last_touch_player == touch
    state = crossed.state
    for _ in range(60):
        restarted = step_tick(state)
        state = restarted.state
    (event,) = restarted.events
    assert event.last_touch_player == state.ball.last_touch_player
    assert event.new_owner == state.ball.possession_owner
    assert event.ball_position == state.ball.position
    assert event.tick == state.simulation_tick
