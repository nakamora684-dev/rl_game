import math
from dataclasses import replace

import pytest

from game.actions import Direction, MovementAction
from game.ball import BallState, step_free_ball
from game.config import BallConfig, SimulationConfig
from game.player import PlayerId
from game.possession import sync_controlled_ball
from game.simulation import MatchPhase, assign_possession, reset, step_decision, step_tick


@pytest.mark.parametrize("owner", list(PlayerId))
def test_setup_assignment_attaches_ball_and_emits_one_explicit_setup_event(owner):
    initial = reset()
    initial = replace(initial, ball=replace(initial.ball, velocity=(300.0, 0.0)))
    result = assign_possession(initial, owner)
    state = result.state
    carrier = state.player_a if owner == PlayerId.A else state.player_b
    expected = (
        carrier.position[0] + 22 * math.cos(carrier.facing),
        carrier.position[1] + 22 * math.sin(carrier.facing),
    )
    assert state.ball.position == pytest.approx(expected)
    assert state.ball.velocity == (0.0, 0.0)
    assert state.ball.possession_owner == owner
    assert state.ball.last_touch_player == owner
    assert state.has_possession(PlayerId.A) != state.has_possession(PlayerId.B)
    assert result.physics_ticks == 0
    assert state.simulation_tick == initial.simulation_tick
    assert state.player_a == initial.player_a
    assert state.player_b == initial.player_b
    assert initial.ball.possession_owner is None
    assert len(result.events) == 1
    event = result.events[0]
    assert event.kind == "possession_changed"
    assert event.previous_owner is None
    assert event.new_owner == owner
    assert event.cause == "setup"


def test_repeating_setup_assignment_does_not_repeat_transition_event():
    state = assign_possession(reset(), PlayerId.A).state
    result = assign_possession(state, PlayerId.A)
    assert result.state is state
    assert result.events == ()


def test_setup_transfer_changes_single_owner_without_touching_stamina():
    state = reset()
    state = replace(state, player_a=replace(state.player_a, stamina=37.0))
    state = assign_possession(state, PlayerId.A).state
    result = assign_possession(state, PlayerId.B)
    assert not result.state.has_possession(PlayerId.A)
    assert result.state.has_possession(PlayerId.B)
    assert result.state.ball.last_touch_player == PlayerId.B
    assert result.state.player_a.stamina == 37.0
    assert result.events[0].previous_owner == PlayerId.A
    assert result.events[0].new_owner == PlayerId.B
    assert result.events[0].cause == "setup"  # Not a steal or recovery reward signal.


@pytest.mark.parametrize("angle", [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2])
def test_attachment_uses_facing_and_fixed_distance(angle):
    state = reset()
    state = replace(state, player_a=replace(state.player_a, facing=angle))
    ball = assign_possession(state, PlayerId.A).state.ball
    displacement = tuple(b - p for b, p in zip(ball.position, state.player_a.position))
    assert displacement == pytest.approx((22 * math.cos(angle), 22 * math.sin(angle)))
    assert math.hypot(*displacement) == pytest.approx(22.0)


def test_controlled_ball_follows_updated_player_position_after_movement():
    state = assign_possession(reset(), PlayerId.A).state
    result = step_decision(state, MovementAction(Direction.EAST, 1.0))
    carrier = result.state.player_a
    assert carrier.position[0] > state.player_a.position[0]
    assert result.state.ball.position == pytest.approx(
        (carrier.position[0] + 22, carrier.position[1])
    )
    assert result.state.ball.velocity == (0.0, 0.0)
    assert result.events == ()


def test_sideways_coasting_ball_follows_facing_not_velocity():
    state = reset()
    carrier = replace(state.player_a, velocity=(60.0, 0.0), facing=-math.pi / 2)
    state = assign_possession(replace(state, player_a=carrier), PlayerId.A).state
    result = step_tick(state)
    carrier = result.state.player_a
    assert carrier.velocity[0] > 0.0
    assert carrier.velocity[1] == 0.0
    assert result.state.ball.position == pytest.approx(
        (carrier.position[0], carrier.position[1] - 22)
    )
    assert result.state.ball.velocity == (0.0, 0.0)


def test_attachment_follows_smooth_turn_while_stationary():
    state = assign_possession(reset(), PlayerId.A).state
    for _ in range(15):
        result = step_tick(state, MovementAction(Direction.NORTH, 0.0))
        state = result.state
        assert result.events == ()
    assert state.player_a.position == (200.0, 250.0)
    assert state.player_a.facing == pytest.approx(-math.pi / 2)
    assert state.ball.position == pytest.approx((200.0, 228.0))


def test_idle_keeps_ball_ahead_when_carrier_stops():
    state = assign_possession(reset(), PlayerId.A).state
    for _ in range(60):
        state = step_tick(state).state
    assert state.ball.position == (222.0, 250.0)
    assert state.ball.possession_owner == PlayerId.A


def test_custom_offset_is_used_by_setup_and_tick():
    config = SimulationConfig(ball=BallConfig(dribble_offset=30.0))
    state = assign_possession(reset(config=config), PlayerId.A, config=config).state
    assert state.ball.position == (230.0, 250.0)
    result = step_tick(state, MovementAction(Direction.EAST, 1.0), config=config)
    assert result.state.ball.position[0] - result.state.player_a.position[0] == pytest.approx(30.0)


def test_same_owner_can_resynchronize_new_offset_without_another_owner_event():
    state = assign_possession(reset(), PlayerId.A).state
    result = assign_possession(
        state, PlayerId.A, config=SimulationConfig(ball=BallConfig(dribble_offset=30.0))
    )
    assert result.state.ball.position == (230.0, 250.0)
    assert result.events == ()


def test_swapped_side_assignment_uses_player_identity_not_pitch_side():
    state = assign_possession(reset(swap_sides=True), PlayerId.A).state
    assert state.player_a.position == (600.0, 250.0)
    assert state.ball.position == pytest.approx((578.0, 250.0))


def test_attachment_after_player_pitch_clamp_does_not_clamp_ball():
    state = reset()
    state = replace(state, player_a=replace(state.player_a, position=(785.0, 250.0)))
    state = assign_possession(state, PlayerId.A).state
    result = step_tick(state, MovementAction(Direction.EAST, 1.0))
    assert result.state.player_a.position == (785.0, 250.0)
    assert result.state.ball.position == (807.0, 250.0)
    assert result.state.ball.possession_owner == PlayerId.A
    assert result.events == ()  # No boundary/goal rules yet.


def test_free_ball_update_rejects_controlled_ball():
    ball = assign_possession(reset(), PlayerId.A).state.ball
    with pytest.raises(ValueError, match="controlled"):
        step_free_ball(ball)


def test_attachment_update_rejects_free_ball():
    state = reset()
    with pytest.raises(ValueError, match="owner"):
        sync_controlled_ball(state.ball, state.player_a, state.player_b)


def test_last_toucher_can_persist_when_ball_is_free():
    # Directly construct the future release convention; this is not a kick API.
    owned = assign_possession(reset(), PlayerId.A).state.ball
    free = replace(owned, possession_owner=None, velocity=(300.0, 0.0))
    result = step_free_ball(free)
    assert result.possession_owner is None
    assert result.last_touch_player == PlayerId.A


def test_reset_clears_possession_and_last_touch():
    state = assign_possession(reset(), PlayerId.B).state
    assert state.ball.possession_owner == PlayerId.B
    fresh = reset()
    assert fresh.ball.possession_owner is None
    assert fresh.ball.last_touch_player is None
    assert not fresh.has_possession(PlayerId.A)
    assert not fresh.has_possession(PlayerId.B)


def test_restart_and_finish_scaffolds_freeze_controlled_ball():
    state = assign_possession(reset(), PlayerId.A).state
    restarted = replace(state, phase=MatchPhase.GOAL_RESTART, restart_ticks_remaining=6)
    assert step_decision(restarted, MovementAction(Direction.NORTH, 1.0)).state.ball == state.ball
    finished = replace(state, phase=MatchPhase.FINISHED, simulation_tick=3600)
    assert step_tick(finished).state is finished
    for frozen in (restarted, finished):
        with pytest.raises(ValueError):
            assign_possession(frozen, PlayerId.B)


def test_owned_ball_cannot_have_free_velocity_or_a_different_last_toucher():
    with pytest.raises(ValueError):
        BallState(
            (0.0, 0.0),
            velocity=(1.0, 0.0),
            possession_owner=PlayerId.A,
            last_touch_player=PlayerId.A,
        )
    with pytest.raises(ValueError):
        BallState((0.0, 0.0), possession_owner=PlayerId.A, last_touch_player=PlayerId.B)
    with pytest.raises(ValueError):
        BallState((0.0, 0.0), possession_owner=PlayerId.A)


@pytest.mark.parametrize("field", ["possession_owner", "last_touch_player"])
def test_raw_or_multiple_player_ids_are_rejected(field):
    for value in ("A", (PlayerId.A, PlayerId.B), 0):
        with pytest.raises(TypeError):
            BallState((0.0, 0.0), **{field: value})


def test_setup_assignment_and_possession_query_require_typed_player_id():
    for value in ("A", None, 0):
        with pytest.raises(TypeError):
            assign_possession(reset(), value)
        with pytest.raises(TypeError):
            reset().has_possession(value)


@pytest.mark.parametrize("offset", [0.0, -1.0, math.inf, math.nan])
def test_invalid_dribble_offset_is_rejected(offset):
    with pytest.raises(ValueError):
        BallConfig(dribble_offset=offset)


@pytest.mark.parametrize("owner", list(PlayerId))
def test_owned_full_match_replays_identically_without_repeated_possession_events(owner):
    initial = assign_possession(reset(), owner).state
    actions = [MovementAction(direction, 0.5) for direction in Direction]

    def trajectory():
        state = initial
        snapshots = []
        events = []
        for decision in range(600):
            action = actions[decision % len(actions)]
            result = step_decision(state, action, action)
            state = result.state
            carrier = state.player_a if owner == PlayerId.A else state.player_b
            assert state.ball.possession_owner == owner
            assert state.ball.last_touch_player == owner
            assert state.ball.velocity == (0.0, 0.0)
            assert math.dist(state.ball.position, carrier.position) == pytest.approx(22.0)
            events.extend(result.events)
            snapshots.append(state)
        assert [event.kind for event in events] == ["match_finished"]
        return snapshots

    assert trajectory() == trajectory()
