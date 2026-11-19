"""Attack-normalized, non-privileged view of the game for ONE player.

Both the rule-based baseline (B) and the RL observation (C) should read the same
information. Convention: the view is mirrored so that the player ALWAYS attacks
toward +x. Only x is mirrored (y is kept), so left/right side bias disappears.

Nothing here is hidden state: no reward bookkeeping, no last-touch history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agents.interface import (
    PHASE_PLAYING,
    PITCH_H,
    PITCH_W,
    STAMINA_MAX,
    GameState,
)

SELF, OPP, FREE = "self", "opp", "free"


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Body:
    x: float
    y: float
    vx: float
    vy: float
    facing: float  # radians, in the attack-right frame
    stamina: float  # [0, 1]


@dataclass(frozen=True)
class View:
    me: Body
    opp: Body
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    possession: str  # SELF / OPP / FREE
    time_left: float  # seconds
    score_diff: int  # my goals - opponent goals
    playing: bool
    contest_active: bool
    mirrored: bool  # True if the real attack direction is -x
    goal_x: float = PITCH_W
    goal_y: float = PITCH_H / 2


def make_view(state: GameState, index: int) -> View:
    mirrored = state.attack_dirs[index] < 0
    sx = -1.0 if mirrored else 1.0

    def fx(x: float) -> float:
        return PITCH_W - x if mirrored else x

    def body(i: int) -> Body:
        p = state.players[i]
        facing = _wrap(math.pi - p.facing) if mirrored else _wrap(p.facing)
        return Body(fx(p.x), p.y, sx * p.vx, p.vy, facing, p.stamina / STAMINA_MAX)

    if state.possession is None:
        poss = FREE
    else:
        poss = SELF if state.possession == index else OPP
    b = state.ball
    return View(
        me=body(index),
        opp=body(1 - index),
        ball_x=fx(b.x),
        ball_y=b.y,
        ball_vx=sx * b.vx,
        ball_vy=b.vy,
        possession=poss,
        time_left=state.time_left,
        score_diff=state.score[index] - state.score[1 - index],
        playing=state.phase == PHASE_PLAYING,
        contest_active=state.contest_active,
        mirrored=mirrored,
    )
