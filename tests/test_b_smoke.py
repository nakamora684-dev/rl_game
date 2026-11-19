"""Smoke tests for Member B's parts. Pure Python: no display, no pygame."""

import math

from agents.engine_adapter import EngineAdapter
from agents.interface import (
    MATCH_TICKS,
    PHASE_FINISHED,
    PITCH_W,
    Action,
    BallState,
    GameState,
    PlayerState,
    goal_mouth,
)
from agents.rule_based import RuleBasedAgent
from rendering.controls import KickCharger


def test_action_clipping():
    a = Action(5, -5, 2, -1, 9).clipped()
    assert a.to_list() == [1.0, -1.0, 1.0, 0.0, 1.0]
    assert len(Action.from_list([0, 0, 0, 0, 0]).to_list()) == 5


def test_goal_mouth_geometry():
    x0, t0, b0 = goal_mouth(0)
    x1, t1, b1 = goal_mouth(1)
    assert (x0, x1) == (0.0, PITCH_W)
    assert math.isclose(b0 - t0, 120.0)


def test_kick_charger_hold_then_release():
    k = KickCharger(charge_seconds=1.0)
    assert k.update(True, 0.5) == (0.0, 0.0)  # holding: no kick yet
    kick, power = k.update(False, 0.016)  # release: kick
    assert kick == 1.0 and math.isclose(power, 0.5)
    assert k.update(False, 0.016) == (0.0, 0.0)  # only one kick per release


def test_rule_agent_returns_valid_actions_and_both_sides():
    eng = EngineAdapter()
    for idx in (0, 1):
        for attack in ((1, -1), (-1, 1)):
            s = eng.reset()
            s.attack_dirs = attack
            a = RuleBasedAgent(idx).act(s)
            assert a.to_list() == a.clipped().to_list()


def test_rule_agent_goes_for_free_ball_and_shoots_when_close():
    bot = RuleBasedAgent(0)
    s = GameState(
        players=[PlayerState(100, 250), PlayerState(700, 250)],
        ball=BallState(400, 250),
        possession=None,
    )
    assert bot.act(s).move_x > 0  # runs toward the ball
    s = GameState(
        players=[PlayerState(650, 250, facing=0.0), PlayerState(100, 100)],
        ball=BallState(668, 250),
        possession=0,
    )
    assert bot.act(s).kick == 1.0  # close, facing goal: kick


def test_rule_agent_symmetry():
    s = GameState(
        players=[PlayerState(100, 250), PlayerState(700, 250)],
        ball=BallState(400, 250),
        possession=None,
        attack_dirs=(1, -1),
    )
    a0 = RuleBasedAgent(0).act(s)
    a1 = RuleBasedAgent(1).act(s)
    assert a0.move_x > 0 > a1.move_x  # each runs toward the ball


def test_engine_runs_a_full_match():
    eng = EngineAdapter()
    s = eng.reset(seed=1)
    bots = [RuleBasedAgent(0), RuleBasedAgent(1)]
    for _ in range(MATCH_TICKS):
        s = eng.step([b.act(s) for b in bots])
    assert s.phase == PHASE_FINISHED and s.time_left == 0.0
