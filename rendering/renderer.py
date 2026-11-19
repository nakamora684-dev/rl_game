"""Pygame renderer. READ-ONLY view of GameState: it owns no gameplay rules.

All coordinates come in logical units (800 x 500) and are scaled here only.
Cosmetic state (trail, particles, shake, flashes) lives here, never in the engine.
Call render() once per physics tick: the one-shot events of a tick are consumed
there. While paused call render(..., freeze=True) so events are not replayed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from agents.interface import PITCH_H, PITCH_W, GameState
from rendering import sprites, theme
from rendering.debug_overlay import draw_debug
from rendering.effects import Effects
from rendering.hud import EventLog, FlashTracker, Hud, PossessionMeter
from rendering.pitch import build_background
from rendering.theme import BALL_R, HUD_H, MARGIN, PLAYER_R


class Renderer:
    def __init__(self, scale: float = 1.2, title: str = "Football DRL") -> None:
        import pygame

        self.pg = pygame
        pygame.init()
        pygame.display.set_caption(title)
        self.scale = scale
        self.w = int(PITCH_W * scale) + 2 * MARGIN
        self.h = int(PITCH_H * scale) + 2 * MARGIN + HUD_H
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.font = pygame.font.SysFont("arial", 22, bold=True)
        self.small = pygame.font.SysFont("arial", 14)
        self.big = pygame.font.SysFont("arial", 40, bold=True)
        pitch_rect = (MARGIN, MARGIN + HUD_H, int(PITCH_W * scale), int(PITCH_H * scale))
        self.hud = Hud(pygame, self.font, self.small, self.big, self.w, MARGIN, pitch_rect)
        self.flash = FlashTracker()
        self.log = EventLog()
        self.meter = PossessionMeter()
        self.fx = Effects()
        self.off = (0, 0)  # screen-shake offset (pixels)
        self.debug = False
        self.clock = pygame.time.Clock()
        self.bg = build_background(pygame, self.w, self.h, self._base, scale)

    # --- helpers -----------------------------------------------------------
    def _base(self, x: float, y: float) -> tuple[int, int]:
        return int(MARGIN + x * self.scale), int(MARGIN + HUD_H + y * self.scale)

    def _p(self, x: float, y: float) -> tuple[int, int]:
        bx, by = self._base(x, y)
        return bx + self.off[0], by + self.off[1]

    def toggle_debug(self) -> None:
        self.debug = not self.debug

    def reset_effects(self) -> None:
        """Call after a match reset so old flashes/particles do not linger."""
        self.flash.clear()
        self.log.clear()
        self.meter.clear()
        self.fx.clear()
        self.off = (0, 0)

    def screenshot(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pg.image.save(self.screen, str(path))
        return path

    # --- drawing -----------------------------------------------------------
    def _scene(self, st: GameState) -> None:
        pg, s = self.pg, self.screen
        r = max(6, int(PLAYER_R * self.scale))
        br = max(3, int(BALL_R * self.scale))
        trail = [(*self._p(x, y), k) for x, y, k in self.fx.trail.points()]
        if st.possession is None:
            sprites.draw_trail(pg, s, trail, br)
        pulse = 0.5 + 0.5 * math.sin(st.tick * 0.3)
        for i in (0, 1):
            p = st.players[i]
            c = self._p(p.x, p.y)
            sprites.draw_player(
                pg,
                s,
                c,
                r,
                theme.TEAM[i],
                p.facing,
                p.stamina / 100.0,
                has_ball=(st.possession == i),
            )
            if st.contest_active and st.possession == 1 - i:  # defender of the contest
                sprites.draw_contest(pg, s, c, r, pulse)
        sprites.draw_ball(pg, s, self._p(st.ball.x, st.ball.y), br, self.fx.spin)
        for x, y, life, color, size in self.fx.particles.view():
            pg.draw.circle(
                s,
                theme.mix(theme.GRASS_B, color, 0.4 + 0.6 * life),
                self._p(x, y),
                max(1, int(size * (0.5 + life))),
            )

    def render(
        self,
        st: GameState,
        charge: float = 0.0,
        fps: int = 60,
        show_final: bool = True,
        freeze: bool = False,
        overlay: Callable[[Renderer], None] | None = None,
        hint: str | None = None,
    ) -> None:
        pg = self.pg
        events = [] if freeze else st.events
        if not freeze:
            self.fx.update(st, 1.0 / max(1, fps))
        self.off = self.fx.shake_offset()
        self.screen.fill(theme.BG)
        self.screen.blit(self.bg, self.off)
        self._scene(st)
        if self.debug:
            draw_debug(pg, self.screen, self._p, self.scale, st, self.small)
        dt = 0.0 if freeze else 1.0 / max(1, fps)
        flashes = self.flash.update(events, dt)
        lines = self.log.update(events, st.time_left, dt)
        if not freeze:
            self.meter.update(st)
        self.hud.draw(
            self.screen,
            st,
            charge,
            flashes,
            show_final=show_final,
            log=lines,
            possession=self.meter.share(),
            hint=hint,
        )
        if overlay is not None:
            overlay(self)
        pg.display.flip()
        self.clock.tick(fps)

    def render_blank(self, overlay: Callable[[Renderer], None], fps: int = 30) -> None:
        """Menu background: the empty pitch plus an overlay drawn by screens.py."""
        self.screen.fill(theme.BG)
        self.screen.blit(self.bg, (0, 0))
        overlay(self)
        self.pg.display.flip()
        self.clock.tick(fps)

    def close(self) -> None:
        self.pg.quit()
