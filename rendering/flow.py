"""Menu / pause / full-time state machine. PURE Python (keys are plain strings).

play.py translates pygame keys to names ("up", "enter", "p"...) and calls
`Flow.handle(name)`; the returned command tells play.py what to do:
    "start" | "resume" | "restart" | "rematch" | "menu" | "quit" | None
"""

from __future__ import annotations

from dataclasses import dataclass, replace

MENU, PLAYING, PAUSED, FULLTIME = "menu", "playing", "paused", "fulltime"
DIFFICULTIES = ("easy", "normal", "hard")
MENU_ROWS = ("difficulty", "side", "start")
PAUSE_ROWS = ("Resume", "Restart match", "Main menu", "Quit")
_PAUSE_CMD = {"Resume": "resume", "Restart match": "restart", "Main menu": "menu", "Quit": "quit"}


@dataclass(frozen=True)
class MatchOptions:
    difficulty: str = "normal"
    swap_sides: bool = False  # True: the human starts on the right and attacks toward -x
    seed: int = 0
    alternate: bool = False  # flip the side at every rematch (play both directions)


class Flow:
    def __init__(self, options: MatchOptions | None = None, start_in_menu: bool = True) -> None:
        self.options = options or MatchOptions()
        self.state = MENU if start_in_menu else PLAYING
        self.menu_row = 0
        self.pause_row = 0

    def finish(self) -> None:
        """The engine reported the final whistle."""
        if self.state == PLAYING:
            self.state = FULLTIME

    def handle(self, key: str | None) -> str | None:
        if key is None:
            return None
        return {
            MENU: self._menu,
            PLAYING: self._playing,
            PAUSED: self._paused,
            FULLTIME: self._fulltime,
        }[self.state](key)

    # ------------------------------------------------------------------
    def _menu(self, key: str) -> str | None:
        if key in ("up", "down"):
            self.menu_row = (self.menu_row + (1 if key == "down" else -1)) % len(MENU_ROWS)
        elif key in ("left", "right"):
            row, step = MENU_ROWS[self.menu_row], (1 if key == "right" else -1)
            if row == "difficulty":
                i = (DIFFICULTIES.index(self.options.difficulty) + step) % len(DIFFICULTIES)
                self.options = replace(self.options, difficulty=DIFFICULTIES[i])
            elif row == "side":
                self.options = replace(self.options, swap_sides=not self.options.swap_sides)
        elif key == "enter":
            self.state = PLAYING
            return "start"
        elif key == "escape":
            return "quit"
        return None

    def _playing(self, key: str) -> str | None:
        if key in ("p", "escape"):
            self.state, self.pause_row = PAUSED, 0
        return None

    def _paused(self, key: str) -> str | None:
        if key in ("p", "escape"):
            self.state = PLAYING
            return "resume"
        if key in ("up", "down"):
            self.pause_row = (self.pause_row + (1 if key == "down" else -1)) % len(PAUSE_ROWS)
        elif key == "enter":
            cmd = _PAUSE_CMD[PAUSE_ROWS[self.pause_row]]
            if cmd in ("resume", "restart"):
                self.state = PLAYING
            elif cmd == "menu":
                self.state = MENU
            return cmd
        return None

    def _fulltime(self, key: str) -> str | None:
        if key in ("enter", "r"):
            o = self.options
            self.options = replace(
                o, seed=o.seed + 1, swap_sides=(not o.swap_sides) if o.alternate else o.swap_sides
            )
            self.state = PLAYING
            return "rematch"
        if key == "m":
            self.state = MENU
            return "menu"
        if key == "escape":
            return "quit"
        return None
