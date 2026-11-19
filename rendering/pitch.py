"""Pitch geometry (pure) and the one-time background drawing.

The background (stripes, markings, boxes, nets) never changes, so it is drawn
ONCE into a Surface and just blitted every frame.
"""

from __future__ import annotations

from agents.interface import PITCH_H, PITCH_W, goal_mouth
from game.config import SimulationConfig
from game.rules import goalpost_centres
from rendering import theme

_CFG = SimulationConfig()
POST_R = _CFG.rules.post_radius  # drawn where A's engine has its posts

PEN_DEPTH, PEN_H = 110.0, 280.0  # penalty box
SIX_DEPTH, SIX_H = 40.0, 180.0  # goal area
CENTER_R = 60.0
SPOT_DIST = 80.0
STRIPES = 10


def box(side: int, depth: float, height: float) -> tuple[float, float, float, float]:
    """(x, y, w, h) in logical units of a box centred on the goal of `side` (0 = left)."""
    x = 0.0 if side == 0 else PITCH_W - depth
    return x, (PITCH_H - height) / 2, depth, height


def penalty_spot(side: int) -> tuple[float, float]:
    return (SPOT_DIST if side == 0 else PITCH_W - SPOT_DIST), PITCH_H / 2


def net_rect(side: int) -> tuple[float, float, float, float]:
    """Net volume behind the goal line (logical units, outside the pitch)."""
    _, y0, y1 = goal_mouth(side)
    x = -theme.NET_DEPTH if side == 0 else PITCH_W
    return x, y0, theme.NET_DEPTH, y1 - y0


def build_background(pg, w: int, h: int, to_px, scale: float):
    """Return a Surface (w x h) with the whole static scene. `to_px` has no shake offset."""
    surf = pg.Surface((w, h))
    surf.fill(theme.BG)

    def rect(x, y, rw, rh):
        px, py = to_px(x, y)
        return pg.Rect(px, py, int(rw * scale) + 1, int(rh * scale) + 1)

    # nets first (they sit outside the pitch)
    for side in (0, 1):
        r = rect(*net_rect(side))
        pg.draw.rect(surf, theme.NET_BG, r)
        nx, ny, nw, nh = net_rect(side)
        step = 6.0
        k = 0.0
        while k <= nw:
            a, b = to_px(nx + k, ny), to_px(nx + k, ny + nh)
            pg.draw.line(surf, theme.NET_LINE, a, b, 1)
            k += step
        k = 0.0
        while k <= nh:
            a, b = to_px(nx, ny + k), to_px(nx + nw, ny + k)
            pg.draw.line(surf, theme.NET_LINE, a, b, 1)
            k += step
    # grass stripes
    sw = PITCH_W / STRIPES
    for i in range(STRIPES):
        pg.draw.rect(
            surf, theme.GRASS_A if i % 2 == 0 else theme.GRASS_B, rect(i * sw, 0, sw, PITCH_H)
        )
    L = theme.LINE
    pg.draw.rect(surf, L, rect(0, 0, PITCH_W, PITCH_H), 3)
    pg.draw.line(surf, L, to_px(PITCH_W / 2, 0), to_px(PITCH_W / 2, PITCH_H), 2)
    centre = to_px(PITCH_W / 2, PITCH_H / 2)
    pg.draw.circle(surf, L, centre, int(CENTER_R * scale), 2)
    pg.draw.circle(surf, L, centre, 4)
    for side in (0, 1):
        pg.draw.rect(surf, L, rect(*box(side, PEN_DEPTH, PEN_H)), 2)
        pg.draw.rect(surf, L, rect(*box(side, SIX_DEPTH, SIX_H)), 2)
        pg.draw.circle(surf, L, to_px(*penalty_spot(side)), 3)
        gx, gy0, gy1 = goal_mouth(side)
        pg.draw.line(surf, theme.GOAL, to_px(gx, gy0), to_px(gx, gy1), 6)
    for px_, py_ in goalpost_centres(_CFG):  # the 4 posts (ball bounces on them)
        c = to_px(px_, py_)
        r = max(3, int(POST_R * scale))
        pg.draw.circle(surf, theme.LINE, c, r)
        pg.draw.circle(surf, theme.shade(theme.LINE, 0.5), c, r, 1)
    return surf
