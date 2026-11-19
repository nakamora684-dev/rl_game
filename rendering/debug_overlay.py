"""F1 debug overlay: possession ring, velocity vectors, steal contest, tick/phase.

READ-ONLY. STEAL_RANGE is a DISPLAY value only; the engine (A) owns the real one.
"""

from __future__ import annotations

from agents.interface import GameState

STEAL_RANGE = 30.0  # logical units; confirm the real value with Member A
RING = (255, 214, 0)
CONTEST = (255, 143, 0)
VEL = (255, 255, 0)


def draw_debug(pg, screen, to_px, scale: float, st: GameState, font) -> None:
    for i, p in enumerate(st.players):
        c = to_px(p.x, p.y)
        r = int(14 * scale)
        if st.possession == i:
            pg.draw.circle(screen, RING, c, r + 5, 2)
        if st.contest_active:
            pg.draw.circle(screen, CONTEST, c, int(STEAL_RANGE * scale), 1)
        pg.draw.line(
            screen, VEL, c, (c[0] + int(p.vx * 8 * scale), c[1] + int(p.vy * 8 * scale)), 2
        )
    b = to_px(st.ball.x, st.ball.y)
    pg.draw.line(
        screen, VEL, b, (b[0] + int(st.ball.vx * 4 * scale), b[1] + int(st.ball.vy * 4 * scale)), 2
    )
    txt = f"tick {st.tick}  phase {st.phase}  poss {st.possession}  events {[e.kind for e in st.events]}"
    screen.blit(font.render(txt, True, (255, 255, 255)), (8, screen.get_height() - 20))
