"""Menu, pause and full-time screens (drawing only; logic is in rendering/flow.py)."""

from __future__ import annotations

from rendering import theme
from rendering.flow import MENU_ROWS, PAUSE_ROWS, Flow
from rendering.hud import result_text


def keymap(pg) -> dict:
    """pygame key constant -> plain key name used by Flow."""
    return {
        pg.K_UP: "up",
        pg.K_DOWN: "down",
        pg.K_LEFT: "left",
        pg.K_RIGHT: "right",
        pg.K_RETURN: "enter",
        pg.K_KP_ENTER: "enter",
        pg.K_ESCAPE: "escape",
        pg.K_p: "p",
        pg.K_r: "r",
        pg.K_m: "m",
    }


def fulltime_rows(row: dict) -> list[tuple[str, str, str]]:
    """(label, human/left value, bot/right value) from a MatchRecorder summary row. Pure."""
    pct = lambda k: tuple(f"{row[f'p{i}_{k}'] * 100:.0f}%" for i in (0, 1))
    num = lambda k: tuple(str(row[f"p{i}_{k}"]) for i in (0, 1))
    return [
        ("Shots", *num("shots")),
        ("Steals", *num("steals")),
        ("Recoveries", *num("recoveries")),
        ("Possession", *pct("possession_share")),
        ("Time sprinting", *pct("sprint_frac")),
        ("Stamina left", *pct("stamina_end")),
    ]


class Screens:
    def __init__(self, renderer) -> None:
        self.r, self.pg = renderer, renderer.pg

    # ---- helpers
    def _dim(self, alpha: int = 170) -> None:
        ov = self.pg.Surface((self.r.w, self.r.h), self.pg.SRCALPHA)
        ov.fill((0, 0, 0, alpha))
        self.r.screen.blit(ov, (0, 0))

    def _text(self, font, s, color, y, x=None) -> None:
        img = font.render(s, True, color)
        self.r.screen.blit(img, ((self.r.w // 2 if x is None else x) - img.get_width() // 2, y))

    def _panel(self, w: int, h: int) -> int:
        """Draw a centred panel; return its top y (pixels)."""
        top = self.r.h // 2 - h // 2
        rect = self.pg.Rect(self.r.w // 2 - w // 2, top, w, h)
        self.pg.draw.rect(self.r.screen, theme.PANEL, rect, border_radius=16)
        self.pg.draw.rect(self.r.screen, theme.MUTED, rect, 2, border_radius=16)
        return top

    # ---- screens
    def menu(self, flow: Flow) -> None:
        def overlay(_r) -> None:
            self._dim(150)
            top = self._panel(420, 300)
            self._text(self.r.big, "FOOTBALL DRL", theme.TEXT, top + 18)
            self._text(self.r.small, "1v1  -  you (blue) vs rule-based bot", theme.MUTED, top + 70)
            o = flow.options
            labels = {
                "difficulty": f"Difficulty:   < {o.difficulty.upper()} >",
                "side": f"Your side:   < {'RIGHT' if o.swap_sides else 'LEFT'} >",
                "start": "START MATCH",
            }
            for k, row in enumerate(MENU_ROWS):
                sel = k == flow.menu_row
                self._text(
                    self.r.font, labels[row], theme.GOAL if sel else theme.TEXT, top + 110 + 44 * k
                )
            self._text(
                self.r.small,
                "Up/Down: select   Left/Right: change   Enter: play   Esc: quit",
                theme.MUTED,
                top + 260,
            )

        self.r.render_blank(overlay)

    def pause(self, flow: Flow) -> None:
        self._dim(150)
        top = self._panel(300, 250)
        self._text(self.r.big, "PAUSED", theme.TEXT, top + 14)
        for k, label in enumerate(PAUSE_ROWS):
            self._text(
                self.r.font,
                label,
                theme.GOAL if k == flow.pause_row else theme.TEXT,
                top + 80 + 36 * k,
            )

    def fulltime(self, st, row: dict, replay_path=None) -> None:
        self._dim(170)
        top = self._panel(460, 400)
        self._text(self.r.small, "FULL TIME", theme.MUTED, top + 16)
        self._text(self.r.big, f"{st.score[0]}  -  {st.score[1]}", theme.TEXT, top + 36)
        self._text(self.r.font, result_text(st), theme.GOAL, top + 92)
        cx = self.r.w // 2
        for k, (label, a, b) in enumerate(fulltime_rows(row)):
            y = top + 140 + 32 * k
            self._text(self.r.font, a, theme.TEAM[0], y, cx - 150)
            self._text(self.r.small, label, theme.MUTED, y + 4, cx)
            self._text(self.r.font, b, theme.TEAM[1], y, cx + 150)
        foot = "Enter: rematch   M: menu   Esc: quit"
        if replay_path:
            foot += "   (replay saved)"
        self._text(self.r.small, foot, theme.MUTED, top + 365)
