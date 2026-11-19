"""Human controls: keyboard (+ optional gamepad) -> Action (the same 5 controls the RL agent uses).

Default keys (AZERTY and QWERTY both work):
    move   : arrows, or Z/Q/S/D, or W/A/S/D          effort : Shift (hold = sprint)
    kick   : Space. HOLD to charge power, RELEASE to kick.
Remap with a JSON file (play.py --controls keys.json), e.g.
    {"sprint": ["left ctrl"], "kick": ["space", "return"]}
Names are pygame key names (pygame.key.key_code). Gamepad: left stick = move,
LB/RB = sprint, A (button 0) = hold to charge / release to kick.
Pure parts (KickCharger, Bindings, axes_to_move) are tested without pygame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from agents.interface import Action

CHARGE_SECONDS = 0.8  # hold time that gives full power
MIN_POWER = 0.2
DEADZONE = 0.2
PAD_SPRINT_BUTTONS = (4, 5)
PAD_KICK_BUTTON = 0

DEFAULT_BINDINGS: dict[str, list[str]] = {
    "left": ["left", "q", "a"],
    "right": ["right", "d"],
    "up": ["up", "z", "w"],
    "down": ["down", "s"],
    "sprint": ["left shift", "right shift"],
    "kick": ["space"],
}


def load_bindings(path: str | Path | None = None) -> dict[str, list[str]]:
    """Defaults, overridden per action by a JSON file. Unknown actions are rejected."""
    out = {k: list(v) for k, v in DEFAULT_BINDINGS.items()}
    if path is None:
        return out
    data = json.loads(Path(path).read_text())
    for action, names in data.items():
        if action not in out:
            raise ValueError(f"unknown action {action!r}; valid: {sorted(out)}")
        if not (isinstance(names, list) and names and all(isinstance(n, str) for n in names)):
            raise ValueError(f"{action!r} must be a non-empty list of key names")
        out[action] = names
    return out


def axes_to_move(ax: float, ay: float, deadzone: float = DEADZONE) -> tuple[float, float]:
    """Radial dead zone, then rescale so the edge of the zone maps to 0 and full tilt to 1."""
    m = math.hypot(ax, ay)
    if m < deadzone:
        return 0.0, 0.0
    k = min(1.0, (m - deadzone) / (1.0 - deadzone))
    return ax / m * k, ay / m * k


class KickCharger:
    """Hold to charge, release to kick. Returns (kick, power) each frame."""

    def __init__(self, charge_seconds: float = CHARGE_SECONDS) -> None:
        self.charge_seconds = charge_seconds
        self.held = 0.0
        self.was_down = False

    def update(self, down: bool, dt: float) -> tuple[float, float]:
        kick, power = 0.0, 0.0
        if down:
            self.held += dt
        elif self.was_down:  # released this frame -> kick
            kick = 1.0
            power = max(MIN_POWER, min(1.0, self.held / self.charge_seconds))
            self.held = 0.0
        self.was_down = down
        return kick, power

    @property
    def charge(self) -> float:
        """0..1, for drawing a power bar in the HUD."""
        return min(1.0, self.held / self.charge_seconds)


class KeyboardControls:
    def __init__(self, bindings: dict[str, list[str]] | None = None) -> None:
        import pygame  # imported lazily: tests need no display

        self.pg = pygame
        self.charger = KickCharger()
        b = bindings or load_bindings()
        self.codes = {a: [pygame.key.key_code(n) for n in names] for a, names in b.items()}

    def _down(self, keys, action: str) -> bool:
        return any(keys[c] for c in self.codes[action])

    def read(self, dt: float) -> Action:
        k = self.pg.key.get_pressed()
        mx = float(self._down(k, "right")) - float(self._down(k, "left"))
        my = float(self._down(k, "down")) - float(self._down(k, "up"))
        n = (mx * mx + my * my) ** 0.5
        if n > 1.0:
            mx, my = mx / n, my / n
        effort = 1.0 if self._down(k, "sprint") else 0.0
        kick, power = self.charger.update(self._down(k, "kick"), dt)
        return Action(mx, my, effort, kick, power)


class GamepadControls:
    def __init__(self, joystick) -> None:
        self.js = joystick
        self.charger = KickCharger()

    def read(self, dt: float) -> Action:
        mx, my = axes_to_move(self.js.get_axis(0), self.js.get_axis(1))
        effort = 1.0 if any(self.js.get_button(b) for b in PAD_SPRINT_BUTTONS) else 0.0
        kick, power = self.charger.update(bool(self.js.get_button(PAD_KICK_BUTTON)), dt)
        return Action(mx, my, effort, kick, power)


class CombinedControls:
    """Keyboard always works; a gamepad (if plugged in) is merged on top."""

    def __init__(self, keyboard: KeyboardControls, pad: GamepadControls | None = None) -> None:
        self.keyboard, self.pad = keyboard, pad

    @property
    def charge(self) -> float:
        return max(self.keyboard.charger.charge, self.pad.charger.charge if self.pad else 0.0)

    def read(self, dt: float) -> Action:
        a = self.keyboard.read(dt)
        if self.pad is None:
            return a
        b = self.pad.read(dt)
        big = b if math.hypot(b.move_x, b.move_y) > math.hypot(a.move_x, a.move_y) else a
        kicker = a if a.kick else b
        return Action(
            big.move_x, big.move_y, max(a.effort, b.effort), max(a.kick, b.kick), kicker.power
        )


def make_controls(bindings_path: str | Path | None = None) -> CombinedControls:
    import pygame

    kb = KeyboardControls(load_bindings(bindings_path))
    pad = None
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        pad = GamepadControls(js)
    return CombinedControls(kb, pad)
