"""Shared helpers for the baseline tests (no pygame, no engine needed)."""

from __future__ import annotations

import math
import random

from agents.interface import (
    MATCH_TICKS,
    PHASE_PLAYING,
    PITCH_H,
    PITCH_W,
    BallState,
    GameState,
    PlayerState,
)


def mirror_state(s: GameState) -> GameState:
    """Reflect the world in x and flip attack directions. Player indices are kept."""
    pl = [
        PlayerState(PITCH_W - p.x, p.y, -p.vx, p.vy, math.pi - p.facing, p.stamina)
        for p in s.players
    ]
    b = s.ball
    return GameState(
        players=pl,
        ball=BallState(PITCH_W - b.x, b.y, -b.vx, b.vy),
        possession=s.possession,
        score=s.score,
        tick=s.tick,
        phase=s.phase,
        attack_dirs=(-s.attack_dirs[0], -s.attack_dirs[1]),
        steal_progress_ticks=s.steal_progress_ticks,
    )


def random_state(rng: random.Random) -> GameState:
    poss = rng.choice([None, 0, 1])
    players = [
        PlayerState(
            rng.uniform(0, PITCH_W),
            rng.uniform(0, PITCH_H),
            rng.uniform(-160, 160),
            rng.uniform(-160, 160),
            rng.uniform(-math.pi, math.pi),
            rng.uniform(0, 100),
        )
        for _ in range(2)
    ]
    ball = BallState(rng.uniform(0, PITCH_W), rng.uniform(0, PITCH_H))
    if poss is None:
        ball.vx, ball.vy = rng.uniform(-300, 300), rng.uniform(-300, 300)
    return GameState(
        players=players,
        ball=ball,
        possession=poss,
        score=(rng.randint(0, 3), rng.randint(0, 3)),
        tick=rng.randint(0, MATCH_TICKS - 1),
        phase=PHASE_PLAYING,
        attack_dirs=rng.choice([(1, -1), (-1, 1)]),
        steal_progress_ticks=5 if rng.random() < 0.2 else 0,
    )


def make_state(
    p0=(200, 250),
    p1=(600, 250),
    ball=(400, 250),
    possession=None,
    f0=0.0,
    f1=math.pi,
    st0=100.0,
    st1=100.0,
    **kw,
) -> GameState:
    base = dict(
        players=[
            PlayerState(*p0, facing=f0, stamina=st0),
            PlayerState(*p1, facing=f1, stamina=st1),
        ],
        ball=BallState(*ball),
        possession=possession,
        attack_dirs=(1, -1),
    )
    base.update(kw)
    return GameState(**base)
