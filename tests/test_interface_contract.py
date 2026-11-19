"""Member A's engine, seen through agents.engine_adapter, must honour interface v2."""

import pytest

from agents.engine_adapter import EngineAdapter
from agents.interface import (
    MATCH_TICKS,
    PHASE_FINISHED,
    PITCH_H,
    PITCH_W,
    Action,
    Event,
    GameState,
)


def _engines():
    return [("real", EngineAdapter)]


@pytest.fixture(params=_engines(), ids=lambda p: p[0])
def engine(request):
    return request.param[1]()


def test_reset_returns_valid_state(engine):
    s = engine.reset(seed=1)
    assert isinstance(s, GameState) and len(s.players) == 2
    assert s.tick == 0 and s.score == (0, 0) and s.possession in (None, 0, 1)
    assert s.attack_dirs in ((1, -1), (-1, 1))


def test_swap_sides_flips_attack_direction(engine):
    a = engine.reset(seed=1)
    d0 = a.attack_dirs
    b = engine.reset(seed=1, swap_sides=True)
    assert b.attack_dirs == (-d0[0], -d0[1])
    # player 0 starts on the side it attacks FROM: +x attackers start on the left
    assert (b.players[0].x < PITCH_W / 2) == (b.attack_dirs[0] > 0)


def test_step_advances_one_tick_and_keeps_invariants(engine):
    s = engine.reset(seed=2)
    for t in range(1, 200):
        s = engine.step([Action(1, 0, 1, 0, 0), Action(-1, 0, 0, 0, 0)])
        assert s.tick == t
        assert s.possession in (None, 0, 1)
        assert all(isinstance(e, Event) for e in s.events)
        assert -1 <= s.ball.x <= PITCH_W + 1 and -1 <= s.ball.y <= PITCH_H + 1
        for p in s.players:
            assert 0.0 <= p.stamina <= 100.0


def test_match_ends_at_match_ticks(engine):
    s = engine.reset(seed=3)
    for _ in range(MATCH_TICKS):
        s = engine.step([Action(), Action()])
    assert s.tick == MATCH_TICKS and s.phase == PHASE_FINISHED


def test_velocity_unit_matches_displacement(engine):
    """GameState velocities must be in the unit agents/config.py assumes.

    Per physics update (ratio ~ 1) or per second (ratio ~ 1/60). A's engine uses units
    per second, so agents/config.py has VEL_PER_TICK = False.
    """
    from agents.config import VEL_PER_TICK

    s = engine.reset(seed=4)
    for _ in range(30):
        prev = (s.players[0].x, s.players[0].y)
        s = engine.step([Action(0, 1, 0, 0, 0), Action(0, -1, 0, 0, 0)])
    p = s.players[0]
    dy, vy = p.y - prev[1], p.vy
    if abs(vy) < 1e-6 or abs(dy) < 1e-6:
        pytest.skip("player blocked: cannot infer the unit")
    per_tick = abs(dy / vy) > 0.3  # ~1 per tick, ~0.017 per second
    assert per_tick == VEL_PER_TICK
