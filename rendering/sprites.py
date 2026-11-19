"""Players and ball. READ-ONLY drawing: everything comes from the arguments."""

from __future__ import annotations

import math

from rendering import theme


def _ellipse_shadow(pg, screen, c, r) -> None:
    rect = pg.Rect(c[0] - r + 3, c[1] - r // 2 + 4, 2 * r, int(1.5 * r))
    pg.draw.ellipse(screen, theme.SHADOW, rect)


def draw_player(
    pg,
    screen,
    c,
    r: int,
    color,
    facing: float,
    stamina_frac: float,
    has_ball: bool = False,
    pulse: float = 0.0,
) -> None:
    """Body + facing 'nose' + stamina ring (green -> amber -> red) + possession ring."""
    frac = max(0.0, min(1.0, stamina_frac))
    _ellipse_shadow(pg, screen, c, r)
    body = theme.shade(color, 0.65 + 0.35 * min(1.0, frac / 0.25)) if frac < 0.25 else color
    pg.draw.circle(screen, body, c, r)
    pg.draw.circle(screen, theme.shade(color, 0.55), c, r, 2)
    nose = (c[0] + int(0.62 * r * math.cos(facing)), c[1] + int(0.62 * r * math.sin(facing)))
    pg.draw.circle(screen, (250, 250, 250), nose, max(2, r // 3))
    ring = pg.Rect(c[0] - r - 4, c[1] - r - 4, 2 * r + 8, 2 * r + 8)
    col = theme.stamina_color(frac)
    if frac >= 0.999:
        pg.draw.circle(screen, col, c, r + 4, 2)
    elif frac > 0.0:
        pg.draw.arc(screen, col, ring, math.pi / 2, math.pi / 2 + 2 * math.pi * frac, 3)
    if has_ball:
        pg.draw.circle(screen, theme.GOAL, c, r + 8, 1)


def draw_contest(pg, screen, c, r: int, pulse: float) -> None:
    """Orange pulsing ring on the defender while a steal contest is running."""
    pg.draw.circle(screen, theme.CONTEST, c, int(r + 6 + 3 * pulse), 2)


def draw_ball(pg, screen, c, r: int, spin: float = 0.0) -> None:
    pg.draw.ellipse(
        screen, theme.SHADOW, pg.Rect(c[0] - r + 2, c[1] - r // 2 + 3, 2 * r, int(1.4 * r))
    )
    pg.draw.circle(screen, (255, 255, 255), c, r)
    for k in range(3):  # rolling pattern (visual only)
        a = spin + k * 2 * math.pi / 3
        pg.draw.circle(
            screen,
            (40, 40, 40),
            (c[0] + int(0.55 * r * math.cos(a)), c[1] + int(0.55 * r * math.sin(a))),
            max(1, r // 4),
        )
    pg.draw.circle(screen, (0, 0, 0), c, r, 1)


def draw_trail(pg, screen, pts_px, r: int) -> None:
    """pts_px: [(x, y, strength 0..1)] oldest first."""
    for x, y, k in pts_px:
        rr = max(1, int(r * (0.25 + 0.5 * k)))
        pg.draw.circle(screen, theme.mix(theme.GRASS_B, (255, 255, 255), 0.35 * k), (x, y), rr)
