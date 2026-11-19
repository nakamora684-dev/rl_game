"""HUD: scoreboard panel, clock, stamina bars, kick-power bar, event flashes.

READ-ONLY: drawing from a GameState only. `FlashTracker`, `result_text`,
`format_clock` and `blink` are pure Python (unit-tested without a display).
"""

from __future__ import annotations

from agents.interface import (
    EV_GOAL,
    EV_KICK,
    EV_OUT,
    EV_RECOVER,
    EV_STEAL,
    PHASE_FINISHED,
    PHASE_PLAYING,
    PHASE_RESTART,
    PHYS_HZ,
    STAMINA_MAX,
    Event,
    GameState,
)
from rendering import theme

# kind -> (label, duration in seconds)
FLASH_RULES = {
    EV_GOAL: ("GOAL!", 1.4),
    EV_OUT: ("OUT", 0.9),
    EV_STEAL: ("STEAL", 0.6),
}


class FlashTracker:
    """Turns one-shot engine events into short on-screen messages."""

    def __init__(self) -> None:
        self.items: list[list] = []  # [label, player, seconds_left]

    def update(self, events: list[Event], dt: float) -> list[tuple[str, int | None]]:
        self.items = [[t, p, r - dt] for t, p, r in self.items if r - dt > 0]
        for ev in events:
            rule = FLASH_RULES.get(ev.kind)
            if rule:
                self.items.append([rule[0], ev.player, rule[1]])
        return [(t, p) for t, p, _ in self.items]

    def clear(self) -> None:
        self.items = []


# kind -> text of the match log (bottom-left of the pitch)
LOG_TEXT = {
    EV_GOAL: "GOAL",
    EV_OUT: "Out of bounds",
    EV_STEAL: "Steal",
    EV_RECOVER: "Loose ball won",
    EV_KICK: "Kick",
}


class EventLog:
    """Last few events as text lines with a time stamp (pure Python, unit-tested)."""

    def __init__(self, keep: int = 4, ttl: float = 6.0) -> None:
        self.keep, self.ttl = keep, ttl
        self.items: list[list] = []  # [text, player, seconds_left]

    def update(
        self, events: list[Event], time_left: float, dt: float
    ) -> list[tuple[str, int | None]]:
        self.items = [[t, p, r - dt] for t, p, r in self.items if r - dt > 0]
        for ev in events:
            label = LOG_TEXT.get(ev.kind)
            if label:
                who = f" {theme.TEAM_NAME[ev.player]}" if ev.player in (0, 1) else ""
                self.items.append([f"{format_clock(time_left)}  {label}{who}", ev.player, self.ttl])
        self.items = self.items[-self.keep :]
        return [(t, p) for t, p, _ in self.items]

    def clear(self) -> None:
        self.items = []


class PossessionMeter:
    """Share of controlled-ball time of each player while the game is running."""

    def __init__(self) -> None:
        self.ticks = [0, 0]

    def update(self, st: GameState) -> None:
        if st.phase == PHASE_PLAYING and st.possession in (0, 1):
            self.ticks[st.possession] += 1

    def share(self) -> float:
        """Fraction of player 0 (0.5 before anybody touched the ball)."""
        total = sum(self.ticks)
        return 0.5 if total == 0 else self.ticks[0] / total

    def clear(self) -> None:
        self.ticks = [0, 0]


def result_text(st: GameState) -> str:
    a, b = st.score
    if a == b:
        return "DRAW"
    return "BLUE WINS" if a > b else "RED WINS"


def format_clock(seconds: float) -> str:
    s = max(0, int(seconds + 0.999))  # round up: 0:00 only at the final whistle
    return f"{s // 60}:{s % 60:02d}"


def blink(tick: int, hz: float = 2.0) -> bool:
    """True/False alternating `hz` times per second (drives warning flashes)."""
    return int(tick * hz * 2 / PHYS_HZ) % 2 == 0


class Hud:
    def __init__(self, pg, font, small, big, width: int, margin: int, pitch_rect) -> None:
        self.pg, self.font, self.small, self.big = pg, font, small, big
        self.w, self.margin = width, margin
        self.pitch_rect = pitch_rect  # (x, y, w, h) in pixels

    def _text(self, screen, font, s, color, x, y, center=False) -> None:
        img = font.render(s, True, color)
        screen.blit(img, (x - img.get_width() // 2 if center else x, y))

    def draw(
        self,
        screen,
        st: GameState,
        charge: float,
        flashes: list[tuple[str, int | None]],
        show_final: bool = True,
        log: list[tuple[str, int | None]] | None = None,
        possession: float | None = None,
        hint: str | None = None,
    ) -> None:
        pg = self.pg
        cx = self.w // 2
        pg.draw.rect(screen, theme.PANEL, pg.Rect(cx - 110, 6, 220, 52), border_radius=10)
        pg.draw.rect(
            screen,
            theme.TEAM[0],
            pg.Rect(cx - 110, 6, 8, 52),
            border_top_left_radius=10,
            border_bottom_left_radius=10,
        )
        pg.draw.rect(
            screen,
            theme.TEAM[1],
            pg.Rect(cx + 102, 6, 8, 52),
            border_top_right_radius=10,
            border_bottom_right_radius=10,
        )
        self._text(screen, self.font, f"{st.score[0]}  -  {st.score[1]}", theme.TEXT, cx, 8, True)
        if possession is not None:  # thin possession bar under the scoreboard
            bar = pg.Rect(cx - 110, 60, 220, 4)
            pg.draw.rect(screen, theme.TEAM[1], bar, border_radius=2)
            pg.draw.rect(
                screen,
                theme.TEAM[0],
                pg.Rect(bar.x, bar.y, int(bar.w * possession), bar.h),
                border_radius=2,
            )
        last10 = st.time_left <= 10 and st.phase != PHASE_FINISHED
        clock_col = theme.STAMINA_LOW if last10 and blink(st.tick) else theme.MUTED
        self._text(screen, self.small, format_clock(st.time_left), clock_col, cx, 36, True)
        for i in (0, 1):
            self._stamina_bar(screen, i, st.players[i].stamina, st.tick)
        if charge > 0:  # kick power bar (human)
            pg.draw.rect(screen, theme.PANEL, pg.Rect(self.margin, 44, 120, 8), border_radius=4)
            pg.draw.rect(
                screen, theme.POWER, pg.Rect(self.margin, 44, int(120 * charge), 8), border_radius=4
            )
            self._text(screen, self.small, "POWER", theme.MUTED, self.margin + 128, 42)
        px, py, pw, ph = self.pitch_rect
        mx, my = px + pw // 2, py + ph // 2
        for k, (text, player) in enumerate(reversed(log or [])):  # newest line at the bottom
            img = self.small.render(text, True, theme.TEXT)
            row = pg.Rect(px + 8, py + ph - 26 - 22 * k, img.get_width() + 22, 20)
            pg.draw.rect(screen, theme.PANEL, row, border_radius=8)
            pg.draw.rect(
                screen,
                theme.TEAM[player] if player in (0, 1) else theme.MUTED,
                pg.Rect(row.x, row.y, 5, row.h),
                border_radius=3,
            )
            screen.blit(img, (row.x + 12, row.y + 2))
        if hint:
            self._text(screen, self.small, hint, theme.MUTED, self.w // 2, py + ph + 12, True)
        for k, (label, player) in enumerate(flashes):
            col = theme.TEAM[player] if player is not None else theme.TEXT
            shadow = self.big.render(label, True, (0, 0, 0))
            img = self.big.render(label, True, col)
            x, y = mx - img.get_width() // 2, my - 100 + 46 * k
            screen.blit(shadow, (x + 2, y + 2))
            screen.blit(img, (x, y))
        if st.phase == PHASE_RESTART:
            owner = theme.TEAM_NAME[st.restart_owner] if st.restart_owner in (0, 1) else "-"
            txt = f"{owner} ball  {st.restart_timer:.1f}s"
            img = self.font.render(txt, True, theme.TEXT)
            pill = pg.Rect(mx - img.get_width() // 2 - 14, py + ph - 44, img.get_width() + 28, 32)
            pg.draw.rect(screen, theme.PANEL, pill, border_radius=16)
            screen.blit(img, (mx - img.get_width() // 2, py + ph - 40))
        if show_final and st.phase == PHASE_FINISHED:
            img = self.big.render(f"FULL TIME - {result_text(st)}", True, theme.TEXT)
            sub = self.small.render("R: new match   Esc: quit", True, theme.TEXT)
            screen.blit(img, (mx - img.get_width() // 2, my - 20))
            screen.blit(sub, (mx - sub.get_width() // 2, my + 28))

    def _stamina_bar(self, screen, i: int, stamina: float, tick: int) -> None:
        pg = self.pg
        bw = 160
        x = self.margin if i == 0 else self.w - self.margin - bw
        frac = max(0.0, min(1.0, stamina / STAMINA_MAX))
        self._text(screen, self.small, f"{theme.TEAM_NAME[i]}  STAMINA", theme.MUTED, x, 6)
        pg.draw.rect(screen, theme.PANEL, pg.Rect(x, 24, bw, 12), border_radius=6)
        if frac > 0:
            pg.draw.rect(
                screen,
                theme.stamina_color(frac),
                pg.Rect(x, 24, max(6, int(bw * frac)), 12),
                border_radius=6,
            )
        if frac < 0.25 and blink(tick, 3.0):  # low stamina warning
            pg.draw.rect(
                screen, theme.STAMINA_LOW, pg.Rect(x - 2, 22, bw + 4, 16), 2, border_radius=8
            )
