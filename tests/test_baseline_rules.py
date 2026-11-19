"""Unit tests of single baseline rules (P3). Pure Python, no engine."""

import math

from agents.difficulty import PRESETS
from agents.interface import PHASE_RESTART
from agents.rule_based import RuleBasedAgent
from tests._helpers import make_state


def _act(state, idx=0, preset="normal"):
    bot = RuleBasedAgent(idx, PRESETS[preset])
    a = bot.act(state)
    return a, bot.last_rule


def test_idle_rule_name_on_restart():
    a, rule = _act(make_state(phase=PHASE_RESTART))
    assert rule == "idle" and a.to_list() == [0.0] * 5


def test_chases_free_ball():
    _, rule = _act(make_state())
    assert rule == "chase"


def test_leads_a_moving_ball_but_easy_does_not():
    moving = make_state(p0=(200, 250), ball=(400, 250))
    moving.ball.vy = 180.0  # units per second, moving down
    assert _act(moving)[0].move_y > 0.05  # normal anticipates
    assert abs(_act(moving, preset="easy")[0].move_y) < 1e-9  # easy runs at the ball


def test_defends_goal_side_when_far_then_contests_when_close():
    far = make_state(p0=(300, 250), p1=(600, 250), ball=(618, 250), possession=1)
    a, rule = _act(far)
    assert rule == "block" and a.move_x > 0
    near = make_state(p0=(300, 250), p1=(360, 250), ball=(378, 250), possession=1)
    assert _act(near)[1] == "contest"
    assert _act(far, preset="easy")[1] == "contest"  # easy has no goal-side logic


def test_aims_at_the_corner_far_from_the_opponent():
    me, opp_up = (400, 250), (700, 100)
    up = make_state(p0=me, p1=opp_up, ball=(418, 250), possession=0)
    down = make_state(p0=me, p1=(700, 400), ball=(418, 250), possession=0)
    assert _act(up)[0].move_y > 0  # opponent top -> aim bottom corner
    assert _act(down)[0].move_y < 0  # opponent bottom -> aim top corner
    assert _act(up)[1] == "carry"


def test_shoots_when_aligned_and_close_but_not_when_far():
    close = make_state(p0=(650, 250), p1=(100, 100), ball=(668, 250), possession=0, f0=0.0)
    assert _act(close)[1] == "shoot"
    far = make_state(p0=(300, 250), p1=(100, 100), ball=(318, 250), possession=0, f0=0.0)
    assert _act(far)[1] == "carry"


def test_does_not_shoot_wide_of_the_goal():
    wide = make_state(
        p0=(650, 250), p1=(100, 100), ball=(668, 250), possession=0, f0=math.radians(-60)
    )
    assert _act(wide)[0].kick == 0.0


def test_hard_shoots_from_farther_than_easy():
    s = make_state(p0=(600, 250), p1=(100, 100), ball=(618, 250), possession=0, f0=0.0)
    assert _act(s, preset="hard")[0].kick == 1.0  # 200 away, hard range 260
    assert _act(s, preset="easy")[0].kick == 0.0  # easy range 150


def test_swerves_toward_centre_near_a_touchline():
    s = make_state(p0=(400, 30), p1=(450, 20), ball=(418, 30), possession=0)
    assert _act(s)[0].move_y > 0  # away from the wall, not along it


def test_does_not_swerve_for_a_chaser_behind():
    s = make_state(p0=(500, 250), p1=(450, 300), ball=(518, 250), possession=0)
    a, _ = _act(s)
    assert abs(a.move_y) < 0.3 and a.move_x > 0.9  # runs on toward goal
    assert a.effort == 1.0  # and sprints away


def test_endgame_spends_the_stamina_reserve_only_when_not_winning():
    late = dict(tick=3300, st0=20.0)  # 5 s left, 20% stamina
    assert _act(make_state(**late))[0].effort == 1.0  # draw: all in
    assert _act(make_state(score=(0, 1), **late))[0].effort == 1.0  # losing: all in
    assert _act(make_state(score=(1, 0), **late))[0].effort == 0.0  # winning: rest
    assert _act(make_state(tick=100, st0=20.0))[0].effort == 0.0  # early: rest


def test_no_sprint_for_a_short_run():
    s = make_state(p0=(380, 250), ball=(400, 250))
    assert _act(s)[0].effort == 0.0
