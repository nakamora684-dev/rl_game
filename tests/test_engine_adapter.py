"""agents/engine_adapter.py: the only place that knows both A's engine and the B-side view."""

import math
from dataclasses import replace

import pytest

from agents.engine_adapter import (
    JOG_EFFORT,
    EngineAdapter,
    convert_events,
    direction_from_vector,
    to_player_action,
    to_view,
)
from agents.interface import (
    EV_GOAL,
    EV_KICK,
    EV_OUT,
    EV_RECOVER,
    EV_STEAL,
    MATCH_TICKS,
    PHASE_FINISHED,
    PHASE_PLAYING,
    PHASE_RESTART,
    Action,
)
from game.actions import Direction, PlayerAction
from game.config import SimulationConfig
from game.player import PlayerId
from game.simulation import SimulationEvent, reset, step_decision

IDLE = Action()


def test_eight_directions_and_deadzone():
    table = {
        (1, 0): Direction.EAST,
        (1, 1): Direction.SOUTHEAST,
        (0, 1): Direction.SOUTH,
        (-1, 1): Direction.SOUTHWEST,
        (-1, 0): Direction.WEST,
        (-1, -1): Direction.NORTHWEST,
        (0, -1): Direction.NORTH,
        (1, -1): Direction.NORTHEAST,
        (0.1, -0.1): Direction.IDLE,
        (0, 0): Direction.IDLE,
    }
    for vec, expected in table.items():
        assert direction_from_vector(*vec) == expected


def test_effort_maps_jog_to_sprint_and_idle_has_none():
    walk = to_player_action(Action(1, 0, 0.0))
    sprint = to_player_action(Action(1, 0, 1.0))
    idle = to_player_action(Action(0, 0, 1.0))
    assert walk.effort == pytest.approx(JOG_EFFORT) and sprint.effort == 1.0
    assert idle.direction == Direction.IDLE and idle.effort == 0.0
    assert isinstance(walk, PlayerAction)


def test_a_player_really_moves_with_default_effort():
    eng = EngineAdapter()
    s = eng.reset()
    x0 = s.players[0].x
    for _ in range(60):
        s = eng.step([Action(1, 0, 0.0), IDLE])
    assert s.players[0].x > x0 + 40  # jogging east for 1 s


def test_view_matches_raw_state_and_swap():
    for swap in (False, True):
        eng = EngineAdapter()
        s = eng.reset(swap_sides=swap)
        raw = eng.raw
        assert s.attack_dirs == ((-1, 1) if swap else (1, -1))
        assert (s.players[0].x, s.players[0].y) == raw.player_a.position
        assert (s.players[1].x, s.players[1].y) == raw.player_b.position
        assert (s.ball.x, s.ball.y) == raw.ball.position
        assert s.phase == PHASE_PLAYING and s.possession is None and s.score == (0, 0)


def test_one_tick_per_step_and_clock():
    eng = EngineAdapter()
    eng.reset()
    for t in range(1, 11):
        assert eng.step([IDLE, IDLE]).tick == t


def test_six_adapter_steps_equal_one_engine_decision():
    """The adapter must not change A's physics: 6 steps == step_decision on the same input."""
    cfg = SimulationConfig()
    eng = EngineAdapter(cfg)
    eng.reset()
    raw = reset(config=cfg)
    for k in range(40):
        a0 = Action(math.cos(k), math.sin(k), (k % 3) / 2, 0, 0)
        a1 = Action(-1, 0.3, 1.0, 0, 0)
        for _ in range(6):
            eng.step([a0, a1])
        raw = step_decision(raw, to_player_action(a0), to_player_action(a1), config=cfg).state
        assert eng.raw == raw


def _owner_state(eng):
    """A has the ball at the centre, facing east, tick 0 (decision boundary)."""
    from game.simulation import assign_possession

    eng.reset()
    eng.raw = assign_possession(eng.raw, PlayerId.A, config=eng.config).state
    return eng


def test_kick_is_consumed_once_even_if_the_action_is_held():
    eng = _owner_state(EngineAdapter())
    kicks = 0
    for _ in range(18):  # held trigger for 3 decisions
        s = eng.step([Action(0, 0, 0, 1.0, 1.0), IDLE])
        kicks += sum(e.kind == EV_KICK for e in s.events)
    assert kicks == 1  # rising edge only


def test_a_press_in_the_middle_of_a_decision_is_latched_to_the_next_boundary():
    eng = _owner_state(EngineAdapter())
    for _ in range(3):
        eng.step([IDLE, IDLE])  # tick 3: not a boundary
    s = eng.step([Action(0, 0, 0, 1.0, 1.0), IDLE])  # human presses here
    assert not s.events
    kick_tick = None
    for _ in range(6):
        s = eng.step([IDLE, IDLE])  # released: key no longer down
        if any(e.kind == EV_KICK for e in s.events):
            kick_tick = s.tick
    assert kick_tick is not None and kick_tick <= 6 + 1  # fires at the next boundary
    assert math.hypot(s.ball.vx, s.ball.vy) > 100  # full power


def test_events_are_converted_once_and_unknown_kinds_are_ignored():
    ev = [
        SimulationEvent("goal", 5, scoring_player=PlayerId.B, last_touch_player=PlayerId.B),
        SimulationEvent("out_of_bounds", 6, last_touch_player=None),
        SimulationEvent("ball_kicked", 7, previous_owner=PlayerId.A, cause="kick"),
        SimulationEvent("possession_changed", 8, new_owner=PlayerId.B, cause="steal"),
        SimulationEvent("possession_changed", 9, new_owner=PlayerId.A, cause="recovery"),
        SimulationEvent("possession_changed", 9, new_owner=PlayerId.A, cause="kickoff"),
        SimulationEvent("ball_collision", 9, cause="goalpost"),
        SimulationEvent("match_finished", 10),
    ]
    out = convert_events(ev)
    assert [(e.kind, e.player, e.tick) for e in out] == [
        (EV_GOAL, 1, 5),
        (EV_OUT, None, 6),
        (EV_KICK, 0, 7),
        (EV_STEAL, 1, 8),
        (EV_RECOVER, 0, 9),
    ]


def test_full_match_goal_events_equal_score_and_restart_fields():
    from agents.difficulty import make_baseline

    for swap in (False, True):
        eng = EngineAdapter()
        s = eng.reset(swap_sides=swap)
        bots = [make_baseline(0), make_baseline(1)]
        for b in bots:
            b.reset()
        goals = [0, 0]
        saw_restart = False
        while s.phase != PHASE_FINISHED:
            s = eng.step([b.act(s) for b in bots])
            for e in s.events:
                if e.kind == EV_GOAL:
                    goals[e.player] += 1
            if s.phase == PHASE_RESTART:
                saw_restart = True
                assert 0.0 < s.restart_timer <= 1.0 + 1e-9
        assert s.tick == MATCH_TICKS and tuple(goals) == s.score
        assert saw_restart


def test_same_inputs_same_trajectory():
    def play():
        from agents.difficulty import make_baseline

        eng = EngineAdapter()
        s = eng.reset()
        bots = [make_baseline(0), make_baseline(1)]
        out = []
        for _ in range(1200):
            s = eng.step([b.act(s) for b in bots])
            out.append((s.tick, s.score, round(s.ball.x, 6), round(s.ball.y, 6)))
        return out

    assert play() == play()


def test_to_view_does_not_mutate_the_engine_state():
    eng = EngineAdapter()
    eng.reset()
    before = eng.raw
    to_view(eng.raw)
    assert eng.raw is before and replace(before) == before
