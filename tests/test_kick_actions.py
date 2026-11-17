import math
from dataclasses import FrozenInstanceError, replace

import pytest

from game.actions import Direction, MovementAction, PlayerAction
from game.config import MATCH_TICKS, PHYSICS_DT, KickConfig, SimulationConfig
from game.player import PlayerId
from game.simulation import MatchPhase, assign_possession, reset, step_decision, step_tick


@pytest.mark.parametrize("power,speed", [(0.0, 50.0), (0.25, 112.5), (0.5, 175.0), (1.0, 300.0)])
def test_linear_power_mapping(power, speed):
    assert KickConfig().speed_for_power(power) == speed


@pytest.mark.parametrize("power", [-0.1, 1.1, math.inf, math.nan])
def test_invalid_power_is_rejected_even_when_trigger_is_off(power):
    with pytest.raises(ValueError):
        PlayerAction(kick_power=power)
    with pytest.raises(ValueError):
        KickConfig().speed_for_power(power)


@pytest.mark.parametrize("trigger", [-1, 2, 0.5, 1.0, "1", None])
def test_trigger_must_be_binary_integer_or_bool(trigger):
    with pytest.raises(ValueError):
        PlayerAction(kick_trigger=trigger)


def test_action_validates_movement_and_is_immutable():
    with pytest.raises(TypeError):
        PlayerAction(direction=3)
    with pytest.raises(ValueError):
        PlayerAction(effort=1.1)
    with pytest.raises(FrozenInstanceError):
        PlayerAction().kick_trigger = 1
    assert PlayerAction(kick_trigger=True).kick_trigger == 1


@pytest.mark.parametrize("owner", list(PlayerId))
@pytest.mark.parametrize("power,speed", [(0.0, 50.0), (0.5, 175.0), (1.0, 300.0)])
def test_decision_consumes_one_kick_and_integrates_six_free_ticks(owner, power, speed):
    initial = assign_possession(reset(), owner).state
    kick = PlayerAction(kick_trigger=1, kick_power=power)
    args = (kick, MovementAction()) if owner == PlayerId.A else (MovementAction(), kick)
    result = step_decision(initial, *args)
    state = result.state
    sign = 1 if owner == PlayerId.A else -1
    assert result.physics_ticks == 6
    assert state.ball.possession_owner is None
    assert state.ball.last_touch_player == owner
    assert state.ball.velocity == pytest.approx((sign * (speed - 10), 0.0))
    assert state.ball.position == pytest.approx(
        (initial.ball.position[0] + sign * (speed * 0.1 - 0.5), 250.0)
    )
    assert [(e.kind, e.tick, e.previous_owner, e.new_owner) for e in result.events] == [
        ("ball_kicked", 0, owner, None)
    ]
    carrier = state.player_a if owner == PlayerId.A else state.player_b
    opponent = state.player_b if owner == PlayerId.A else state.player_a
    assert carrier.kick_cooldown_ticks == 6
    assert opponent.kick_cooldown_ticks == 0
    assert initial.ball.possession_owner == owner
    assert initial.player_a.kick_cooldown_ticks == initial.player_b.kick_cooldown_ticks == 0


def test_trigger_off_ignores_power_and_preserves_dribble():
    initial = assign_possession(reset(), PlayerId.A).state
    assert step_decision(initial, PlayerAction(kick_power=1.0)) == step_decision(initial)


def test_release_uses_updated_movement_and_facing_before_free_motion():
    initial = assign_possession(reset(), PlayerId.A).state
    action = PlayerAction(Direction.NORTH, 1.0, kick_trigger=1, kick_power=1.0)
    ordinary = step_tick(initial, MovementAction(Direction.NORTH, 1.0)).state
    result = step_tick(initial, action)
    carrier = result.state.player_a
    assert carrier.facing == ordinary.player_a.facing == pytest.approx(-2 * math.pi / 60)
    assert replace(carrier, kick_cooldown_ticks=0, pickup_lockout_ticks=0) == ordinary.player_a
    distance = 22 + 300 * PHYSICS_DT - 50 * PHYSICS_DT**2
    assert result.state.ball.position == pytest.approx(
        (
            carrier.position[0] + distance * math.cos(carrier.facing),
            carrier.position[1] + distance * math.sin(carrier.facing),
        )
    )
    assert result.state.player_b == ordinary.player_b
    assert carrier.kick_cooldown_ticks == 11


def test_simultaneous_requests_only_allow_current_owner():
    initial = assign_possession(reset(), PlayerId.B).state
    kick = PlayerAction(kick_trigger=1, kick_power=1.0)
    result = step_decision(initial, kick, kick)
    assert len(result.events) == 1
    assert result.events[0].previous_owner == PlayerId.B
    assert result.state.player_a.kick_cooldown_ticks == 0
    assert result.state.player_b.kick_cooldown_ticks == 6


def test_nonowner_and_free_ball_requests_fail_without_cooldown():
    kick = PlayerAction(kick_trigger=1)
    assert step_decision(reset(), kick) == step_decision(reset())
    initial = assign_possession(reset(), PlayerId.B).state
    assert step_decision(initial, kick) == step_decision(initial)


def test_cooldown_expiry_does_not_retry_failed_request_inside_decision():
    initial = assign_possession(reset(), PlayerId.A).state
    initial = replace(initial, player_a=replace(initial.player_a, kick_cooldown_ticks=1))
    kick = PlayerAction(kick_trigger=1, kick_power=1.0)
    failed = step_decision(initial, kick)
    assert failed.events == ()
    assert failed.state.ball.possession_owner == PlayerId.A
    assert failed.state.player_a.kick_cooldown_ticks == 0
    # Same object represents a NEW decision, not a rising-edge-only input.
    success = step_decision(failed.state, kick)
    assert [e.kind for e in success.events] == ["ball_kicked"]


@pytest.mark.parametrize("phase", [MatchPhase.GOAL_RESTART, MatchPhase.OUT_OF_BOUNDS_RESTART])
@pytest.mark.parametrize("countdown", [1, 2, 6])
def test_restart_discards_request_even_if_play_resumes_in_same_decision(phase, countdown):
    initial = assign_possession(reset(), PlayerId.A).state
    initial = replace(initial, phase=phase, restart_ticks_remaining=countdown)
    result = step_decision(initial, PlayerAction(kick_trigger=1))
    assert result.events == ()
    assert result.state.ball.possession_owner == PlayerId.A
    assert result.state.player_a.kick_cooldown_ticks == 0
    assert result.state.phase == MatchPhase.PLAYING


def test_restart_advances_cooldowns_but_freezes_motion_and_stamina():
    initial = reset()
    initial = replace(
        initial,
        phase=MatchPhase.GOAL_RESTART,
        restart_ticks_remaining=20,
        player_a=replace(initial.player_a, kick_cooldown_ticks=12, stamina=37.0),
        player_b=replace(initial.player_b, kick_cooldown_ticks=3),
    )
    result = step_decision(initial)
    assert result.state.player_a == replace(initial.player_a, kick_cooldown_ticks=6)
    assert result.state.player_b == replace(initial.player_b, kick_cooldown_ticks=0)
    assert result.state.ball == initial.ball


def test_exact_cooldown_boundary_after_successful_kick():
    kick = PlayerAction(kick_trigger=1)
    result = step_tick(assign_possession(reset(), PlayerId.A).state, kick)
    assert result.state.player_a.kick_cooldown_ticks == 11
    # Setup reacquisition isolates the kick guard independently of automatic pickup.
    state = assign_possession(result.state, PlayerId.A).state
    for tick in range(1, 12):
        assert state.simulation_tick == tick
        result = step_tick(state, kick)
        assert result.events == ()
        state = result.state
    assert state.simulation_tick == 12
    assert state.player_a.kick_cooldown_ticks == 0
    result = step_tick(state, kick)
    assert [(e.kind, e.tick) for e in result.events] == [("ball_kicked", 12)]


def test_decision_equals_first_full_action_then_five_movement_ticks():
    initial = assign_possession(reset(), PlayerId.A).state
    action = PlayerAction(Direction.NORTH, 0.5, 1, 0.5)
    expected = step_tick(initial, action)
    events = list(expected.events)
    state = expected.state
    for _ in range(5):
        expected = step_tick(state, MovementAction(action.direction, action.effort))
        state = expected.state
        events.extend(expected.events)
    result = step_decision(initial, action)
    assert result.state == state
    assert result.events == tuple(events)


def test_custom_kick_config_and_no_cooldown_mode():
    config = SimulationConfig(kick=KickConfig(min_speed=100, max_speed=200, cooldown_ticks=0))
    initial = assign_possession(reset(), PlayerId.A).state
    result = step_tick(initial, PlayerAction(kick_trigger=1, kick_power=0.5), config=config)
    assert result.state.ball.velocity == pytest.approx((150 - 100 * PHYSICS_DT, 0.0))
    assert result.state.player_a.kick_cooldown_ticks == 0


@pytest.mark.parametrize(
    "fields",
    [
        {"min_speed": 0},
        {"max_speed": -1},
        {"min_speed": math.nan},
        {"max_speed": math.inf},
        {"min_speed": 301},
        {"cooldown_ticks": -1},
        {"cooldown_ticks": 0.2},
        {"cooldown_ticks": True},
    ],
)
def test_invalid_kick_config(fields):
    with pytest.raises(ValueError):
        KickConfig(**fields)


@pytest.mark.parametrize("timer", [-1, 0.5, True])
def test_invalid_player_cooldown(timer):
    with pytest.raises(ValueError):
        replace(reset().player_a, kick_cooldown_ticks=timer)


def test_kick_and_whistle_events_are_both_kept_and_finished_state_is_idempotent():
    initial = assign_possession(reset(), PlayerId.A).state
    initial = replace(initial, simulation_tick=MATCH_TICKS - 1)
    kick = PlayerAction(kick_trigger=1)
    result = step_decision(initial, kick)
    assert result.physics_ticks == 1
    assert [(e.kind, e.tick) for e in result.events] == [
        ("ball_kicked", MATCH_TICKS - 1),
        ("match_finished", MATCH_TICKS),
    ]
    assert result.state.player_a.kick_cooldown_ticks == 11
    frozen = step_decision(result.state, kick)
    assert frozen.state is result.state
    assert frozen.events == ()
    assert frozen.physics_ticks == 0


def test_reset_clears_cooldown_and_full_replay_is_deterministic():
    def trajectory():
        state = assign_possession(reset(), PlayerId.A).state
        snapshots, events = [], []
        for _ in range(600):
            result = step_decision(state, PlayerAction(Direction.NORTH, 0.5, 1, 1.0))
            state = result.state
            assert state.player_a.kick_cooldown_ticks >= 0
            events.extend(result.events)
            snapshots.append(state)
        assert [e.kind for e in events] == ["ball_kicked", "match_finished"]
        return snapshots

    assert trajectory() == trajectory()
    assert reset().player_a.kick_cooldown_ticks == reset().player_b.kick_cooldown_ticks == 0


@pytest.mark.parametrize(
    "power,ticks,distance", [(0.0, 30, 12.5), (0.5, 105, 153.125), (1.0, 180, 450.0)]
)
def test_action_roll_stopping_time_and_distance(power, ticks, distance):
    initial = reset()
    # Keep the opponent away so this checks rolling, not interception.
    initial = replace(initial, player_b=replace(initial.player_b, position=(600.0, 400.0)))
    initial = assign_possession(initial, PlayerId.A).state
    result = step_tick(initial, PlayerAction(kick_trigger=1, kick_power=power))
    state = result.state
    for _ in range(ticks - 1):
        state = step_tick(state).state
    assert state.ball.velocity == (0.0, 0.0)
    assert state.ball.position == pytest.approx((222 + distance, 250.0))
