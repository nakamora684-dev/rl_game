"""The adapter must produce the SAME physics as the shared simulation."""

from dataclasses import replace

import pytest

from environment.contract import (
    Event,
    EventType,
    GameEngine,
    Phase,
    snapshot,
    validate_state,
    validate_step_result,
)
from environment.engine import SimulationEngine
from game.actions import PlayerAction
from game.ball import BallState
from game.config import MATCH_TICKS, SimulationConfig
from game.player import PlayerId
from game.scenarios import ScenarioAgent, build_scenarios
from game.simulation import assign_possession, reset, step_decision

pytestmark = pytest.mark.environment
IDLE = (PlayerAction(), PlayerAction())


def test_reset_protocol_orientation_and_no_hidden_history():
    engine = SimulationEngine()
    first = engine.reset()
    assert isinstance(engine, GameEngine)
    validate_state(first)
    assert first == engine.reset()
    assert first.tick == 0 and first.score == (0, 0)
    swapped = engine.reset(swap_sides=True)
    assert swapped.attack_dirs == (-1, 1)
    assert swapped.players[0].x == 600 and swapped.players[1].x == 200
    assert engine.reset() == first


@pytest.mark.parametrize("scenario", build_scenarios(), ids=lambda s: s.name)
def test_all_fifty_scenario_initial_decisions_match_shared_engine(scenario):
    engine = SimulationEngine(scenario.config)
    previous = engine.reset(initial_state=scenario.initial)
    a = ScenarioAgent(scenario.plan_a, scenario.initial.simulation_tick)
    b = ScenarioAgent(scenario.plan_b, scenario.initial.simulation_tick)
    actions = (a(scenario.initial, PlayerId.A), b(scenario.initial, PlayerId.B))
    expected = step_decision(scenario.initial, *actions, config=scenario.config)
    actual = engine.step(actions)
    assert engine.native_state == expected.state
    assert actual.state == snapshot(expected.state, scenario.config)
    assert actual.raw_events == expected.events
    assert actual.ticks_advanced == expected.physics_ticks
    validate_step_result(previous, actual)


def test_full_idle_match_is_600_decisions_and_finished_input_is_idempotent():
    engine = SimulationEngine()
    state = engine.reset()
    steps = 0
    while not state.is_over:
        result = engine.step(IDLE)
        validate_step_result(state, result)
        state = result.state
        steps += 1
    assert steps == 600 and state.tick == MATCH_TICKS and state.phase is Phase.FINISHED
    result = engine.step(IDLE)
    validate_step_result(state, result)
    assert result.ticks_advanced == 0 and not result.raw_events and not result.events


def test_first_tick_kick_is_valid_and_does_not_retry_after_later_pickup():
    engine = SimulationEngine()
    native = assign_possession(reset(), PlayerId.A).state
    previous = engine.reset(initial_state=native)
    result = engine.step((PlayerAction(kick_trigger=1, kick_power=1), PlayerAction()))
    validate_step_result(previous, result)
    assert [(e.type, e.tick) for e in result.events] == [(EventType.KICK, 0)]
    assert result.state.players[0].kick_cooldown_ticks == 6
    assert result.state.players[0].pickup_lockout_ticks == 6
    # A request while free is discarded, even if A recovers on tick 1.
    native = replace(reset(), ball=BallState((200, 250)))
    previous = engine.reset(initial_state=native)
    result = engine.step((PlayerAction(kick_trigger=1, kick_power=1), PlayerAction()))
    validate_step_result(previous, result)
    assert [e.type for e in result.events] == [EventType.LOOSE_BALL_RECOVERY]
    assert result.state.possession == 0


@pytest.mark.parametrize("swap", [False, True])
def test_goal_uses_scorer_not_last_toucher_and_does_not_skip_restart(swap):
    engine = SimulationEngine()
    native = replace(
        reset(swap_sides=swap),
        ball=BallState((806.9, 250), (60, 0), last_touch_player=PlayerId.B),
    )
    previous = engine.reset(initial_state=native)
    result = engine.step(IDLE)
    validate_step_result(previous, result)
    assert result.state.tick == 6 and result.state.restart_ticks_remaining == 55
    assert result.state.phase is Phase.GOAL_RESTART
    assert result.events[0].player == (1 if swap else 0)
    assert result.raw_events[0].last_touch_player is PlayerId.B


def test_neutral_boundary_is_unblamed_and_neutral_restart_is_not_recovery():
    engine = SimulationEngine()
    previous = engine.reset(initial_state=replace(reset(), ball=BallState((400, -6.9), (0, -60))))
    result = engine.step(IDLE)
    validate_step_result(previous, result)
    assert result.events[0].type is EventType.OUT_OF_BOUNDS
    assert result.events[0].player is None
    events = list(result.events)
    raw = list(result.raw_events)
    for _ in range(10):
        result = engine.step(IDLE)
        events.extend(result.events)
        raw.extend(result.raw_events)
    assert [e.type for e in events] == [EventType.OUT_OF_BOUNDS]
    assert any(e.cause == "neutral_restart" for e in raw)


def test_partial_final_decision_keeps_goal_and_terminal_events():
    engine = SimulationEngine()
    native = replace(
        reset(),
        simulation_tick=MATCH_TICKS - 1,
        ball=BallState((806.9, 250), (60, 0), last_touch_player=PlayerId.A),
    )
    previous = engine.reset(initial_state=native)
    result = engine.step(IDLE)
    validate_step_result(previous, result)
    assert result.ticks_advanced == 1 and result.state.is_over
    assert [e.kind for e in result.raw_events] == ["goal", "match_finished"]


def test_event_ids_do_not_collide_at_adjacent_decision_boundary():
    engine = SimulationEngine()
    # Contact occurs on completed tick 6; the next decision can kick at stamp 6.
    native = replace(reset(), ball=BallState((251, 250), (-300, 0)))
    engine.reset(initial_state=native)
    one = engine.step((PlayerAction(kick_trigger=1), PlayerAction()))
    two = engine.step((PlayerAction(kick_trigger=1), PlayerAction()))
    assert [(e.type, e.tick) for e in one.events] == [(EventType.LOOSE_BALL_RECOVERY, 6)]
    assert [(e.type, e.tick) for e in two.events] == [(EventType.KICK, 6)]
    assert two.events[0].event_id > one.events[0].event_id
    validate_step_result(one.state, two)


def test_validator_rejects_bad_clock_score_window_and_duplicate_ids():
    engine = SimulationEngine()
    prev = engine.reset()
    result = engine.step(IDLE)
    goal = Event(EventType.GOAL, 3, 0, None, 0)
    for bad in (
        replace(result, ticks_advanced=5),
        replace(result, events=(goal,)),
        replace(result, events=(replace(goal, tick=0),)),
        replace(result, events=(Event(EventType.KICK, 0, 0, None, 0),) * 2),
    ):
        with pytest.raises(ValueError):
            validate_step_result(prev, bad)


def test_adapter_guards_reset_and_action_types():
    engine = SimulationEngine(SimulationConfig())
    with pytest.raises(RuntimeError):
        engine.step(IDLE)
    engine.reset()
    with pytest.raises(TypeError):
        engine.step((object(), PlayerAction()))
