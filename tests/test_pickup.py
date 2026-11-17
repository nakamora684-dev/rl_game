import math
from dataclasses import replace

import pytest

from game.actions import Direction, MovementAction, PlayerAction
from game.ball import BallState
from game.config import MATCH_TICKS, BallConfig, PossessionConfig, SimulationConfig
from game.player import PlayerId
from game.possession import select_pickup_owner
from game.simulation import (
    MatchPhase,
    assign_possession,
    release_kick,
    reset,
    step_decision,
    step_tick,
)


def scene(ball_position=(200.0, 250.0), ball_velocity=(0.0, 0.0)):
    state = reset()
    return replace(state, ball=BallState(ball_position, ball_velocity))


@pytest.mark.parametrize("owner", list(PlayerId))
def test_contact_recovers_free_ball_once_without_changing_player_motion_or_stamina(owner):
    initial = reset()
    player = initial.player_a if owner == PlayerId.A else initial.player_b
    initial = replace(initial, ball=BallState(player.position, last_touch_player=PlayerId.B))
    result = step_tick(initial)
    state = result.state
    assert state.ball.possession_owner == state.ball.last_touch_player == owner
    assert state.ball.velocity == (0.0, 0.0)
    assert state.has_possession(PlayerId.A) != state.has_possession(PlayerId.B)
    assert state.player_a == initial.player_a
    assert state.player_b == initial.player_b
    assert len(result.events) == 1
    event = result.events[0]
    assert (event.kind, event.tick, event.previous_owner, event.new_owner, event.cause) == (
        "possession_changed",
        1,
        None,
        owner,
        "recovery",
    )
    assert initial.ball.possession_owner is None
    assert step_decision(state).events == ()


@pytest.mark.parametrize("distance,recovers", [(21.9, True), (22.0, True), (22.0001, False)])
def test_pickup_uses_sum_of_player_and_ball_radii(distance, recovers):
    result = step_tick(scene((200.0 + distance, 250.0)))
    assert (result.state.ball.possession_owner == PlayerId.A) is recovers


def test_custom_ball_radius_changes_contact_distance():
    config = SimulationConfig(ball=BallConfig(radius=10.0))
    assert step_tick(scene((225.0, 250.0)), config=config).state.ball.possession_owner == PlayerId.A
    assert step_tick(scene((225.001, 250.0)), config=config).state.ball.possession_owner is None


@pytest.mark.parametrize("speed", [300.0, 12000.0])
def test_fast_ball_crossing_player_recovers_even_when_endpoint_is_outside(speed):
    # At 12000 units/s, both endpoints are outside contact range: sweep is essential.
    initial = scene((177.0, 250.0), (speed, 0.0))
    result = step_tick(initial)
    assert result.state.ball.possession_owner == PlayerId.A
    assert result.state.ball.velocity == (0.0, 0.0)


def test_moving_player_can_sweep_into_stationary_ball():
    initial = scene((225.0, 250.0))
    initial = replace(initial, player_a=replace(initial.player_a, velocity=(6000.0, 0.0)))
    result = step_tick(initial)
    assert result.state.player_a.position[0] > 280
    assert result.state.ball.possession_owner == PlayerId.A
    assert result.state.ball.position[0] == pytest.approx(result.state.player_a.position[0] + 22)


def test_relative_sweep_does_not_pick_up_parallel_equal_speed_near_miss():
    initial = scene((225.0, 250.0), (160.0, 0.0))
    initial = replace(initial, player_a=replace(initial.player_a, velocity=(160.0, 0.0)))
    config = SimulationConfig(ball=BallConfig(deceleration=0.0))
    result = step_tick(initial, MovementAction(Direction.EAST, 1.0), config=config)
    assert result.state.ball.possession_owner is None


def test_earliest_contact_wins_not_nearest_endpoint_or_player_a_order():
    initial = scene((700.0, 250.0), (-36000.0, 0.0))
    result = step_tick(initial)
    assert result.state.ball.possession_owner == PlayerId.B
    # Reverse geometry as well; the rule is spatial, not identity-based.
    initial = scene((100.0, 250.0), (36000.0, 0.0))
    assert step_tick(initial).state.ball.possession_owner == PlayerId.A


def test_simultaneous_initial_contact_chooses_nearer_player():
    initial = scene((400.0, 250.0))
    initial = replace(
        initial,
        player_a=replace(initial.player_a, position=(420.0, 250.0)),
        player_b=replace(initial.player_b, position=(390.0, 250.0)),
    )
    result = step_tick(initial)
    assert result.state.ball.possession_owner == PlayerId.B


def test_exact_simultaneous_tie_stays_free_without_recovery_or_identity_bias():
    initial = scene((400.0, 250.0))
    initial = replace(
        initial,
        player_a=replace(initial.player_a, position=(380.0, 250.0)),
        player_b=replace(initial.player_b, position=(420.0, 250.0)),
    )
    result = step_tick(initial)
    assert result.state.ball.possession_owner is None
    assert all(event.kind == "ball_collision" for event in result.events)
    swapped = replace(initial, player_a=initial.player_b, player_b=initial.player_a)
    assert step_tick(swapped).state.ball.possession_owner is None


def test_simultaneous_tangent_contacts_leave_ball_free():
    initial = scene((300.0, 250.0), (12000.0, 0.0))
    initial = replace(
        initial,
        player_a=replace(initial.player_a, position=(400.0, 228.0)),
        player_b=replace(initial.player_b, position=(400.0, 272.0)),
    )
    result = step_tick(initial)
    assert result.state.ball.possession_owner is None
    assert result.events == ()


def test_tangent_contact_recovers_but_slight_miss_does_not():
    for y, expected in ((228.0, PlayerId.A), (227.99, None)):
        initial = scene((100.0, 250.0), (12000.0, 0.0))
        initial = replace(initial, player_a=replace(initial.player_a, position=(200.0, y)))
        assert step_tick(initial).state.ball.possession_owner == expected


def test_controlled_ball_near_defender_is_not_a_loose_recovery_or_steal():
    initial = reset()
    initial = replace(initial, player_b=replace(initial.player_b, position=(222.0, 250.0)))
    initial = assign_possession(initial, PlayerId.A).state
    result = step_tick(initial)
    assert result.state.ball.possession_owner == PlayerId.A
    assert result.events == ()


def test_failed_first_tick_kick_does_not_retry_after_same_tick_pickup():
    initial = scene()
    kick = PlayerAction(kick_trigger=1, kick_power=1.0)
    result = step_decision(initial, kick)
    assert [e.cause for e in result.events] == ["recovery"]
    assert result.state.ball.possession_owner == PlayerId.A
    next_decision = step_decision(result.state, kick)
    assert [e.kind for e in next_decision.events] == ["ball_kicked"]


def test_kick_starts_separate_pickup_lockout_and_opponent_can_intercept_same_tick():
    initial = reset()
    initial = replace(initial, player_b=replace(initial.player_b, position=(244.0, 250.0)))
    initial = assign_possession(initial, PlayerId.A).state
    result = step_tick(initial, PlayerAction(kick_trigger=1, kick_power=1.0))
    assert result.state.player_a.kick_cooldown_ticks == 11
    assert result.state.player_a.pickup_lockout_ticks == 11
    assert result.state.player_b.pickup_lockout_ticks == 0
    assert result.state.ball.possession_owner == result.state.ball.last_touch_player == PlayerId.B
    assert [(e.kind, e.cause, e.tick) for e in result.events] == [
        ("ball_kicked", "kick", 0),
        ("possession_changed", "recovery", 1),
    ]


def test_pickup_lockout_boundary_is_independent_of_kick_cooldown():
    initial = scene()
    initial = replace(
        initial, player_a=replace(initial.player_a, pickup_lockout_ticks=1, kick_cooldown_ticks=30)
    )
    first = step_tick(initial)
    assert first.state.ball.possession_owner is None
    assert first.state.player_a.pickup_lockout_ticks == 0
    second = step_tick(first.state)
    assert second.state.ball.possession_owner == PlayerId.A
    assert second.state.player_a.kick_cooldown_ticks == 28


def test_direct_release_initializes_pickup_lockout_without_advancing_clock():
    initial = assign_possession(reset(), PlayerId.A).state
    result = release_kick(initial, PlayerId.A, launch_speed=50.0)
    assert result.state.player_a.pickup_lockout_ticks == 12
    assert result.state.player_a.kick_cooldown_ticks == 0
    assert result.physics_ticks == 0
    assert result.state.simulation_tick == 0
    assert step_tick(result.state).state.ball.possession_owner is None


def test_custom_lockout_duration_does_not_change_kick_cooldown():
    config = SimulationConfig(possession=PossessionConfig(kicker_pickup_lockout_ticks=4))
    initial = assign_possession(reset(), PlayerId.A).state
    result = step_tick(initial, PlayerAction(kick_trigger=1), config=config)
    assert result.state.player_a.pickup_lockout_ticks == 3
    assert result.state.player_a.kick_cooldown_ticks == 11


@pytest.mark.parametrize("phase", [MatchPhase.GOAL_RESTART, MatchPhase.OUT_OF_BOUNDS_RESTART])
def test_restart_advances_lockout_but_does_not_pick_up(phase):
    initial = scene()
    initial = replace(
        initial,
        phase=phase,
        restart_ticks_remaining=2,
        player_a=replace(initial.player_a, pickup_lockout_ticks=2),
    )
    first = step_tick(initial)
    second = step_tick(first.state)
    assert first.events == second.events == ()
    assert second.state.player_a.pickup_lockout_ticks == 0
    assert second.state.ball.possession_owner is None
    assert step_tick(second.state).state.ball.possession_owner == PlayerId.A


def test_recovery_and_terminal_events_are_both_retained_and_finished_freezes_timers():
    initial = replace(scene(), simulation_tick=MATCH_TICKS - 1)
    result = step_tick(initial)
    assert [(e.cause, e.kind) for e in result.events] == [
        ("recovery", "possession_changed"),
        (None, "match_finished"),
    ]
    assert result.state.phase == MatchPhase.FINISHED
    assert step_tick(result.state).state is result.state
    assert step_tick(result.state).events == ()


def test_select_pickup_rejects_controlled_ball_and_does_not_mutate_inputs():
    initial = assign_possession(reset(), PlayerId.A).state
    with pytest.raises(ValueError, match="free"):
        select_pickup_owner(
            initial.ball,
            initial.ball,
            initial.player_a,
            initial.player_b,
            initial.player_a,
            initial.player_b,
        )


@pytest.mark.parametrize("radius", [0.0, -1.0, math.inf, math.nan])
def test_invalid_ball_radius(radius):
    with pytest.raises(ValueError):
        BallConfig(radius=radius)


@pytest.mark.parametrize("timer", [-1, 0.5, True])
def test_invalid_lockout_timers(timer):
    with pytest.raises(ValueError):
        PossessionConfig(kicker_pickup_lockout_ticks=timer)
    with pytest.raises(ValueError):
        replace(reset().player_a, pickup_lockout_ticks=timer)


def test_reset_clears_both_players_lockouts_and_replay_is_deterministic():
    def trajectory():
        state = scene()
        states, events = [], []
        for _ in range(600):
            result = step_decision(state, PlayerAction(kick_trigger=1, kick_power=0.0))
            state = result.state
            states.append(state)
            events.extend(result.events)
        assert events[0].cause == "recovery"
        assert events[1].kind == "ball_kicked"
        assert events[-1].kind == "match_finished"
        return states, events

    assert trajectory() == trajectory()
    assert reset().player_a.pickup_lockout_ticks == reset().player_b.pickup_lockout_ticks == 0


def test_default_opponent_really_intercepts_reference_kick():
    state = assign_possession(reset(), PlayerId.A).state
    result = step_tick(state, PlayerAction(kick_trigger=1, kick_power=1.0))
    state = result.state
    events = list(result.events)
    for _ in range(179):
        result = step_tick(state)
        state = result.state
        events.extend(result.events)
    assert state.ball.possession_owner == PlayerId.B
    assert state.ball.position == pytest.approx((578.0, 250.0))
    assert [(e.kind, e.cause, e.tick) for e in events] == [
        ("ball_kicked", "kick", 0),
        ("possession_changed", "recovery", 98),
    ]


def test_locked_player_does_not_win_even_when_already_touching():
    initial = scene((400.0, 250.0))
    initial = replace(
        initial,
        player_a=replace(initial.player_a, position=(400.0, 250.0), pickup_lockout_ticks=12),
        player_b=replace(initial.player_b, position=(420.0, 250.0)),
    )
    assert step_tick(initial).state.ball.possession_owner == PlayerId.B


def test_kicker_lockout_stays_active_until_exact_twelve_tick_boundary():
    initial = assign_possession(reset(), PlayerId.A).state
    result = step_tick(initial, PlayerAction(kick_trigger=1))
    state = result.state
    for tick in range(1, 12):
        assert state.simulation_tick == tick
        # Keep a stationary loose ball at the kicker to isolate lockout eligibility.
        state = replace(
            state, ball=BallState(state.player_a.position, last_touch_player=PlayerId.A)
        )
        result = step_tick(state)
        assert all(event.kind == "ball_collision" for event in result.events)
        assert result.state.ball.possession_owner is None
        state = result.state
    assert state.simulation_tick == 12
    assert state.player_a.pickup_lockout_ticks == 0
    result = step_tick(state)
    assert result.state.ball.possession_owner == PlayerId.A
    assert result.events[0].cause == "recovery"


def test_no_contact_preserves_free_ball_last_touch():
    initial = reset()
    initial = replace(initial, ball=replace(initial.ball, last_touch_player=PlayerId.B))
    result = step_tick(initial)
    assert result.state.ball == initial.ball
    assert result.events == ()


def test_new_kick_does_not_sweep_through_opponents_pre_release_motion():
    initial = reset()
    initial = replace(
        initial,
        player_b=replace(initial.player_b, position=(230.0, 350.0), velocity=(0.0, -12000.0)),
    )
    initial = assign_possession(initial, PlayerId.A).state
    result = step_tick(initial, PlayerAction(kick_trigger=1, kick_power=1.0))
    assert result.state.player_b.position[1] < 151.0
    assert result.state.ball.possession_owner is None
    assert [event.kind for event in result.events] == ["ball_kicked"]
