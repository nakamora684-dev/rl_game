"""Cosmetic effects are deterministic, bounded and react to one-shot events once."""

import copy
import dataclasses

from agents.interface import EV_GOAL, EV_KICK, EV_STEAL, Event
from rendering.effects import BallTrail, Effects, Particles, Shake
from tests._helpers import make_state


def test_trail_grows_while_moving_fades_when_stopped_and_clears_on_teleport():
    t = BallTrail(n=5)
    for x in range(10):
        t.update(x * 5.0, 0.0)
    assert len(t.pts) == 5  # bounded
    pts = t.points()
    assert pts[-1][2] == 1.0 and pts[0][2] < pts[-1][2]  # newest is strongest
    for _ in range(3):
        t.update(45.0, 0.0)  # not moving
    assert len(t.pts) == 2
    t.update(700.0, 400.0)  # kick-off teleport
    assert len(t.pts) == 1


def test_particles_deterministic_bounded_and_expire():
    a, b = Particles(), Particles()
    a.burst(10, 10, [(1, 2, 3), (4, 5, 6)], 30, 100.0, 1.0, seed=7)
    b.burst(10, 10, [(1, 2, 3), (4, 5, 6)], 30, 100.0, 1.0, seed=7)
    assert a.items == b.items  # same seed, same burst
    for _ in range(20):
        a.burst(0, 0, [(0, 0, 0)], 100, 50.0, 1.0, seed=1)
    assert len(a.items) <= Particles.MAX
    for _ in range(200):
        a.update(1 / 60)
    assert a.items == []  # everything expired


def test_shake_decays_to_zero():
    s = Shake()
    assert s.offset() == (0, 0)
    s.kick(8.0)
    assert max(abs(v) for v in s.offset()) <= 8
    for _ in range(120):
        s.update(1 / 60)
    assert s.offset() == (0, 0)


def test_goal_event_triggers_burst_and_shake_once():
    fx = Effects()
    st = make_state()
    st.events = [Event(EV_GOAL, 0, 5)]
    fx.update(st, 1 / 60)
    assert len(fx.particles.items) == 60 and fx.shake.amp > 0
    st.events = []
    fx.update(st, 1 / 60)
    assert len(fx.particles.items) == 60  # no second burst without an event


def test_steal_and_kick_make_small_bursts():
    fx = Effects()
    st = make_state()
    st.events = [Event(EV_STEAL, 1, 2), Event(EV_KICK, 0, 2)]
    fx.update(st, 0.0)
    assert len(fx.particles.items) == 15


def test_trail_only_while_ball_is_free_and_clear_resets_everything():
    fx = Effects()
    st = make_state(possession=None)
    for i in range(5):
        st.ball.x = 100 + 10 * i
        fx.update(st, 1 / 60)
    assert len(fx.trail.pts) > 1 and fx.spin > 0
    st.possession = 0
    fx.update(st, 1 / 60)
    assert len(fx.trail.pts) == 0  # carried ball: no trail
    fx.clear()
    assert fx.spin == 0 and fx.particles.items == [] and fx.shake_offset() == (0, 0)


def test_effects_never_mutate_the_game_state():
    fx = Effects()
    st = make_state()
    st.events = [Event(EV_GOAL, 1, 3)]
    before = dataclasses.asdict(copy.deepcopy(st))
    fx.update(st, 1 / 60)
    assert dataclasses.asdict(st) == before
