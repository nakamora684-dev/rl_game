import random
from dataclasses import replace

import pytest

from game.actions import Direction, MovementAction, PlayerAction
from game.ball import BallState
from game.config import MATCH_TICKS, BallConfig, MovementConfig, PitchConfig, SimulationConfig
from game.player import PlayerId
from game.runner import idle_provider, run_match
from game.simulation import MatchPhase, assign_possession, reset, step_decision


@pytest.mark.parametrize("swap", [False, True])
def test_idle_match_finishes_once_after_600_decisions(swap):
    initial = reset(swap_sides=swap)
    result = run_match(initial_state=initial)
    assert result.final_state.phase == MatchPhase.FINISHED
    assert result.final_state.simulation_tick == MATCH_TICKS
    assert result.final_state.attack_direction_a == initial.attack_direction_a
    assert (result.final_state.score_a, result.final_state.score_b) == (0, 0)
    assert result.physics_ticks == MATCH_TICKS
    assert result.decision_count == 600
    assert [(event.kind, event.tick) for event in result.events] == [
        ("match_finished", MATCH_TICKS)
    ]
    assert initial == reset(swap_sides=swap)


def test_providers_read_the_same_snapshot_including_during_restarts():
    initial = replace(
        reset(), ball=BallState((804.0, 250.0), (300.0, 0.0), last_touch_player=PlayerId.A)
    )
    calls = []

    def provider(state, player):
        calls.append((state, player))
        return PlayerAction()

    result = run_match(provider, provider, initial_state=initial)
    assert len(calls) == 2 * result.decision_count == 1200
    for index in range(0, len(calls), 2):
        state_a, player_a = calls[index]
        state_b, player_b = calls[index + 1]
        assert state_a is state_b
        assert (player_a, player_b) == (PlayerId.A, PlayerId.B)
        assert state_a.simulation_tick == index // 2 * 6
    assert calls[2][0].phase == MatchPhase.GOAL_RESTART
    assert calls[20][0].restart_ticks_remaining == 1
    assert calls[22][0].phase == MatchPhase.PLAYING


@pytest.mark.parametrize("swap", [False, True])
def test_seeded_action_scripts_replay_identical_snapshots_and_events(swap):
    def replay():
        rng = random.Random(17)
        scripts = {
            player: [
                PlayerAction(
                    Direction(rng.randrange(9)), rng.random(), rng.randrange(2), rng.random()
                )
                for _ in range(600)
            ]
            for player in PlayerId
        }
        snapshots = []

        def provider(state, player):
            snapshots.append((state, player))
            return scripts[player][state.simulation_tick // 6]

        result = run_match(provider, provider, initial_state=reset(swap_sides=swap))
        return snapshots, result

    assert replay() == replay()


@pytest.mark.parametrize(
    "y,first_kind,restart_cause",
    [
        (250.0, "goal", "kickoff"),
        (100.0, "out_of_bounds", "boundary_restart"),
    ],
)
def test_goals_and_boundaries_preserve_the_complete_engine_trace(y, first_kind, restart_cause):
    initial = replace(
        reset(), ball=BallState((804.0, y), (300.0, 0.0), last_touch_player=PlayerId.A)
    )
    expected_state = initial
    expected_events = []
    while expected_state.phase != MatchPhase.FINISHED:
        step = step_decision(expected_state)
        expected_state = step.state
        expected_events.extend(step.events)

    result = run_match(initial_state=initial)
    assert result.final_state == expected_state
    assert result.events == tuple(expected_events)
    assert result.events[0].kind == first_kind
    assert sum(event.kind == first_kind for event in result.events) == 1
    assert any(event.cause == restart_cause for event in result.events)
    assert result.physics_ticks == MATCH_TICKS
    assert result.events[-1].kind == "match_finished"


@pytest.mark.parametrize("ticks_left", [1, 2, 5, 7])
def test_resumed_match_counts_only_actual_remaining_ticks_and_queries(ticks_left):
    initial = replace(reset(), simulation_tick=MATCH_TICKS - ticks_left)
    calls = []

    def provider(state, player):
        calls.append((state.simulation_tick, player))
        return PlayerAction()

    result = run_match(provider, provider, initial_state=initial)
    assert result.physics_ticks == ticks_left
    assert result.decision_count == (ticks_left + 5) // 6
    assert len(calls) == 2 * result.decision_count
    assert result.final_state.simulation_tick == MATCH_TICKS
    assert [(event.kind, event.tick) for event in result.events] == [
        ("match_finished", MATCH_TICKS)
    ]


def test_finished_input_returns_unchanged_without_querying_or_repeating_whistle():
    initial = replace(reset(), simulation_tick=MATCH_TICKS, phase=MatchPhase.FINISHED)

    def forbidden(state, player):
        pytest.fail("A finished match must not query providers")

    result = run_match(forbidden, forbidden, initial_state=initial)
    assert result.final_state is initial
    assert result.physics_ticks == result.decision_count == 0
    assert result.events == ()


@pytest.mark.parametrize("ticks_left", [1, 5])
def test_goal_and_whistle_are_both_kept_even_with_restart_pending(ticks_left):
    initial = replace(
        reset(),
        simulation_tick=MATCH_TICKS - ticks_left,
        ball=BallState((804.0, 250.0), (300.0, 0.0), last_touch_player=PlayerId.A),
    )
    result = run_match(initial_state=initial)
    assert [event.kind for event in result.events] == ["goal", "match_finished"]
    assert result.events[0].tick == MATCH_TICKS - ticks_left + 1
    assert result.final_state.score_a == 1
    assert result.final_state.restart_plan is None
    assert result.physics_ticks == ticks_left


def test_restart_expiry_resumes_held_movement_but_does_not_retry_kick():
    shot = replace(
        reset(),
        simulation_tick=MATCH_TICKS - 8,
        ball=BallState((804.0, 250.0), (300.0, 0.0), last_touch_player=PlayerId.A),
    )
    restart = step_decision(shot).state  # Two match ticks remain, still in a frozen restart.
    initial = replace(restart, restart_ticks_remaining=1)
    action = PlayerAction(Direction.NORTH, effort=0.5, kick_trigger=1, kick_power=1.0)
    calls = []

    def provider(state, player):
        calls.append((state, player))
        return action

    expected = step_decision(initial, action, action)
    result = run_match(provider, provider, initial_state=initial)
    assert result.final_state == expected.state
    assert result.events == expected.events
    assert len(calls) == 2
    assert result.final_state.player_b.velocity[1] < 0
    assert any(event.cause == "kickoff" for event in result.events)
    assert not any(event.kind == "ball_kicked" for event in result.events)


def test_custom_configuration_is_used_for_default_reset_and_all_steps():
    config = SimulationConfig(
        pitch=PitchConfig(width=1000.0, height=600.0),
        movement=MovementConfig(max_speed=0.0),
    )

    def provider(state, player):
        return PlayerAction(Direction.EAST, effort=1.0)

    result = run_match(provider, provider, config=config)
    assert result.final_state.player_a.position == (250.0, 300.0)
    assert result.final_state.player_b.position == (750.0, 300.0)
    assert result.final_state.ball.position == (500.0, 300.0)
    assert result.physics_ticks == MATCH_TICKS


def test_diagnostic_events_are_not_filtered_or_reordered():
    config = SimulationConfig(ball=BallConfig(max_impacts_per_tick=1))
    state = reset(config=config)
    state = replace(
        state,
        simulation_tick=MATCH_TICKS - 1,
        player_a=replace(state.player_a, position=(773.0, 198.0)),
    )
    initial = assign_possession(state, PlayerId.A, config=config).state

    def provider(state, player):
        return PlayerAction(Direction.NORTH)

    expected = step_decision(initial, provider(initial, PlayerId.A), config=config)
    result = run_match(provider, idle_provider, initial_state=initial, config=config)
    assert result.events == expected.events
    assert any(event.kind == "physics_limit" for event in result.events)
    assert result.events[-1].kind == "match_finished"
    assert all(event.cause != "setup" for event in result.events)


def test_idle_provider_returns_a_full_noop_action():
    assert idle_provider(reset(), PlayerId.A) == PlayerAction()


@pytest.mark.parametrize("invalid", [None, 3, MovementAction(), {"direction": 0}])
def test_invalid_provider_output_is_rejected_without_coercion(invalid):
    def provider(state, player):
        return invalid

    with pytest.raises(TypeError, match="PlayerAction"):
        run_match(provider)


def test_provider_errors_propagate_instead_of_returning_a_fake_finished_match():
    def provider(state, player):
        raise RuntimeError("controller failed")

    with pytest.raises(RuntimeError, match="controller failed"):
        run_match(provider)


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"initial_state": 0}, "SimulationState"),
        ({"config": None}, "SimulationConfig"),
        ({"provider_a": None}, "callable"),
        ({"provider_b": None}, "callable"),
    ],
)
def test_invalid_runner_inputs_raise_clear_errors(fields, match):
    with pytest.raises(TypeError, match=match):
        run_match(**fields)


def test_invalid_b_action_names_the_provider():
    def provider(state, player):
        return None

    with pytest.raises(TypeError, match="Provider B must return a PlayerAction"):
        run_match(idle_provider, provider)


def test_post_collision_and_constraint_survive_alongside_final_whistle():
    initial = reset()
    initial = replace(
        initial,
        simulation_tick=MATCH_TICKS - 1,
        player_a=replace(initial.player_a, position=(765.0, 186.0), velocity=(180.0, 0.0)),
    )
    initial = assign_possession(initial, PlayerId.A).state
    expected = step_decision(initial)
    result = run_match(initial_state=initial)
    assert result.final_state == expected.state
    assert result.events == expected.events
    assert any(event.kind == "ball_collision" for event in result.events)
    assert any(event.kind == "ball_constraint" for event in result.events)
    assert result.events[-1].kind == "match_finished"


def test_action_kick_and_recovery_from_different_subticks_are_preserved():
    initial = assign_possession(reset(), PlayerId.A).state
    initial = replace(initial, simulation_tick=MATCH_TICKS - 110)
    action = PlayerAction(kick_trigger=1, kick_power=1.0)

    def provider(state, player):
        return action

    expected_state = initial
    expected_events = []
    while expected_state.phase != MatchPhase.FINISHED:
        step = step_decision(expected_state, action)
        expected_state = step.state
        expected_events.extend(step.events)
    result = run_match(provider, initial_state=initial)
    assert result.final_state == expected_state
    assert result.events == tuple(expected_events)
    assert sum(event.kind == "ball_kicked" for event in result.events) == 1
    assert any(event.cause == "recovery" for event in result.events)
