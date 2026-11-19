"""Visual constants and tiny pure helpers (no pygame). One place to restyle the game."""

from __future__ import annotations

from game.config import SimulationConfig

_CFG = SimulationConfig()

MARGIN = 40  # pixels around the pitch (room for the nets)
HUD_H = 64  # pixels above the pitch
PLAYER_R = _CFG.pitch.player_radius  # logical radii come from game/config.py (A owns them)
BALL_R = _CFG.ball.radius
NET_DEPTH = 18.0  # logical units drawn behind each goal line

BG = (20, 24, 30)
PANEL = (32, 38, 48)
TEXT = (235, 235, 235)
MUTED = (150, 158, 170)
GRASS_A = (46, 125, 50)
GRASS_B = (56, 142, 60)
SHADOW = (30, 92, 38)
LINE = (240, 240, 240)
GOAL = (255, 214, 0)
NET_BG = (38, 44, 54)
NET_LINE = (110, 118, 130)
TEAM = [(30, 136, 229), (229, 57, 53)]
TEAM_NAME = ("BLUE", "RED")
STAMINA_OK = (102, 187, 106)
STAMINA_MID = (255, 193, 7)
STAMINA_LOW = (239, 83, 80)
POWER = (255, 214, 0)
CONTEST = (255, 143, 0)


def mix(a: tuple, b: tuple, t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def shade(c: tuple, k: float) -> tuple[int, int, int]:
    """Multiply brightness (k < 1 darkens). Used for tired players."""
    return tuple(int(max(0, min(255, v * k))) for v in c[:3])


def stamina_color(frac: float) -> tuple[int, int, int]:
    if frac > 0.5:
        return STAMINA_OK
    if frac > 0.25:
        return STAMINA_MID
    return STAMINA_LOW
