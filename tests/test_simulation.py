from dataclasses import replace
from math import isfinite, pi

import pytest

from game.actions import Direction, MovementAction
from game.config import (
    MATCH_TICKS,
    PHYSICS_DT,
    TICKS_PER_DECISION,
    MovementConfig,
    PitchConfig,
    SimulationConfig,
    StaminaConfig,
)
from game.simulation import MatchPhase, reset, step_decision, step_tick

_NO_FATIGUE = StaminaConfig(
    drain_rate=0.0, recovery_rate=0.0, min_speed_factor=1.0, min_acceleration_factor=1.0
)
_MOVEMENT_ONLY_CONFIG = SimulationConfig(stamina=_NO_FATIGUE)


def test_reset_positions_orientation_and_fresh_state():
    initial = reset()
    assert initial.player_a.position == (200.0, 250.0)
    assert initial.player_b.position == (600.0, 250.0)
    assert initial.player_a.facing == 0.0
    assert initial.player_b.facing == pi
    assert initial.player_a.stamina == initial.player_b.stamina == 100.0
    assert initial.remaining_seconds == 60.0
    assert initial.attack_direction_a == 1
    assert initial.attack_direction_b == -1
    changed = step_decision(initial, MovementAction(Direction.EAST, 1.0)).state
    assert changed != initial
    assert reset() == initial
    assert reset() is not initial


def test_reset_swaps_players_and_attack_directions():
    normal, swapped = reset(), reset(swap_sides=True)
    assert swapped.player_a == normal.player_b
    assert swapped.player_b == normal.player_a
    assert swapped.attack_direction_a == -1
    assert swapped.attack_direction_b == 1


def test_custom_pitch_and_movement_config_reach_both_layers():
    config = SimulationConfig(
        pitch=PitchConfig(width=1000.0, height=600.0, player_radius=20.0),
        movement=MovementConfig(max_speed=120.0, acceleration=240.0),
        stamina=_NO_FATIGUE,
    )
    state = reset(config=config)
    assert state.player_a.position == (250.0, 300.0)
    assert state.player_b.position == (750.0, 300.0)
    for _ in range(5):
        state = step_decision(state, MovementAction(Direction.EAST, 1.0), config=config).state
    assert state.player_a.velocity == pytest.approx((120.0, 0.0))
    assert state.player_a.position == pytest.approx((281.0, 300.0))


def test_both_players_use_identical_movement_rules():
    state = reset()
    action = MovementAction(Direction.SOUTH, 0.5)
    result = step_tick(state, action, action)
    assert result.state.player_a.velocity == result.state.player_b.velocity
    assert result.state.player_a.position[1] == result.state.player_b.position[1]
    assert result.physics_ticks == 1
    assert result.state.simulation_tick == 1
    assert result.state.remaining_seconds == pytest.approx(60.0 - PHYSICS_DT)
    assert result.events == ()


def test_one_decision_is_six_individual_ticks():
    initial = reset()
    action_a, action_b = MovementAction(Direction.EAST, 1.0), MovementAction(Direction.NORTH, 0.5)
    expected = initial
    for _ in range(TICKS_PER_DECISION):
        expected = step_tick(expected, action_a, action_b, config=_MOVEMENT_ONLY_CONFIG).state
    result = step_decision(initial, action_a, action_b, config=_MOVEMENT_ONLY_CONFIG)
    assert result.state == expected
    assert result.physics_ticks == 6
    assert result.state.simulation_tick == 6
    assert result.state.player_a.velocity == pytest.approx((32.0, 0.0))
    assert result.state.player_a.position[0] == pytest.approx(201.86666666666667)


@pytest.mark.parametrize("phase", [MatchPhase.GOAL_RESTART, MatchPhase.OUT_OF_BOUNDS_RESTART])
def test_restart_freezes_motion_but_not_clock(phase):
    initial = reset()
    moving = replace(initial.player_a, velocity=(80.0, 0.0))
    state = replace(initial, player_a=moving, phase=phase, restart_ticks_remaining=2)
    action = MovementAction(Direction.EAST, 1.0)
    first = step_tick(state, action).state
    assert first.simulation_tick == 1
    assert first.remaining_seconds < state.remaining_seconds
    assert first.player_a == state.player_a
    assert first.restart_ticks_remaining == 1
    second = step_tick(first, action).state
    assert second.phase == MatchPhase.PLAYING
    assert second.player_a == state.player_a
    assert second.restart_ticks_remaining == 0
    third = step_tick(second, action).state
    assert third.player_a.position[0] > second.player_a.position[0]


def test_decision_can_span_restart_expiry_with_held_movement():
    state = replace(reset(), phase=MatchPhase.GOAL_RESTART, restart_ticks_remaining=2)
    result = step_decision(state, MovementAction(Direction.EAST, 1.0), config=_MOVEMENT_ONLY_CONFIG)
    assert result.physics_ticks == 6
    assert result.state.phase == MatchPhase.PLAYING
    assert result.state.player_a.velocity == pytest.approx((4 * 320.0 / 60, 0.0))


def test_restart_of_exactly_six_ticks_consumes_one_decision_without_moving():
    state = replace(reset(), phase=MatchPhase.GOAL_RESTART, restart_ticks_remaining=6)
    result = step_decision(state, MovementAction(Direction.EAST, 1.0))
    assert result.physics_ticks == 6  # Physics ticks include frozen simulation time.
    assert result.state.simulation_tick == 6
    assert result.state.phase == MatchPhase.PLAYING
    assert result.state.player_a == state.player_a
    assert result.state.player_b == state.player_b


def test_clock_ends_after_exactly_600_decision_intervals():
    state = reset()
    events = []
    for decision in range(600):
        result = step_decision(state)
        state = result.state
        events.extend(result.events)
        assert state.simulation_tick == (decision + 1) * 6
    assert state.phase == MatchPhase.FINISHED
    assert state.simulation_tick == MATCH_TICKS
    assert state.remaining_seconds == 0.0
    assert [(event.kind, event.tick) for event in events] == [("match_finished", MATCH_TICKS)]
    result = step_decision(state, MovementAction(Direction.EAST, 1.0))
    assert result.state is state
    assert result.physics_ticks == 0
    assert result.events == ()


@pytest.mark.parametrize("ticks_left", [1, 2, 5])
def test_decision_stops_at_whistle_and_preserves_early_subtick_event(ticks_left):
    state = replace(reset(), simulation_tick=MATCH_TICKS - ticks_left)
    result = step_decision(state, MovementAction(Direction.EAST, 1.0))
    assert result.physics_ticks == ticks_left
    assert result.state.simulation_tick == MATCH_TICKS
    assert len(result.events) == 1
    assert result.events[0].kind == "match_finished"


def test_match_can_finish_during_restart():
    state = replace(
        reset(),
        simulation_tick=MATCH_TICKS - 1,
        phase=MatchPhase.OUT_OF_BOUNDS_RESTART,
        restart_ticks_remaining=120,
    )
    result = step_tick(state, MovementAction(Direction.EAST, 1.0))
    assert result.state.phase == MatchPhase.FINISHED
    assert result.state.restart_ticks_remaining == 0
    assert result.state.player_a == state.player_a
    assert len(result.events) == 1
    assert step_tick(result.state).events == ()


def test_full_match_replay_is_identical_and_states_stay_finite():
    actions = [
        MovementAction(direction, effort) for direction in Direction for effort in (0, 0.5, 1)
    ]

    def trajectory():
        state = reset()
        states = []
        for index in range(600):
            state = step_decision(
                state, actions[index % len(actions)], actions[-index % len(actions)]
            ).state
            for player in (state.player_a, state.player_b):
                assert all(
                    isfinite(value) for value in (*player.position, *player.velocity, player.facing)
                )
                assert 15 <= player.position[0] <= 785
                assert 15 <= player.position[1] <= 485
                assert 0.0 <= player.stamina <= 100.0
            states.append(state)
        return states

    assert trajectory() == trajectory()


@pytest.mark.parametrize(
    "fields",
    [
        {"simulation_tick": -1},
        {"simulation_tick": 0.5},
        {"simulation_tick": MATCH_TICKS},
        {"phase": MatchPhase.FINISHED},
        {"phase": MatchPhase.GOAL_RESTART},
        {"restart_ticks_remaining": 3},
        {"restart_ticks_remaining": -1},
        {"attack_direction_a": 0},
    ],
)
def test_inconsistent_clock_or_phase_is_rejected(fields):
    with pytest.raises(ValueError):
        replace(reset(), **fields)
