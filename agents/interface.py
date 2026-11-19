"""B-side VIEW of the game state (what rendering/ and agents/ read).

It is NOT the engine: Member A's physics lives in game/ and is untouched.
agents/engine_adapter.py converts A's SimulationState into these read-only types
(see the vocabulary table in that file). Constants come from game/config.py.

Units: logical units (pitch is 800 x 500). The renderer scales to pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

INTERFACE_VERSION: int = 2  # bump when a field changes (A, B, C must agree)

# ---- Constants: read from Member A's game/config.py (single source of truth) ----
from game.config import MATCH_TICKS, PHYSICS_HZ, TICKS_PER_DECISION, SimulationConfig

_CFG = SimulationConfig()
PITCH_W: float = _CFG.pitch.width
PITCH_H: float = _CFG.pitch.height
GOAL_W: float = _CFG.rules.goal_width  # goal opening, centered on each short edge
PHYS_HZ: int = PHYSICS_HZ  # physics updates per second
DECISION_EVERY: int = TICKS_PER_DECISION  # agent decides every 6 physics updates (10 Hz)
MATCH_SECONDS: int = MATCH_TICKS // PHYS_HZ
STAMINA_MAX: float = 100.0

# Match phases (one explicit phase, see "unclear restart semantics" risk).
PHASE_PLAYING = "playing"
PHASE_RESTART = "restart"
PHASE_FINISHED = "finished"


# One-shot event kinds. The engine emits each event ONCE, on the tick it happens.
EV_GOAL = "goal"  # player = scorer
EV_OUT = "out_of_bounds"  # player = last toucher (responsible)
EV_STEAL = "steal"  # player = new owner (opponent -> player)
EV_RECOVER = "loose_ball_recovery"  # player = new owner (free ball -> player)
EV_KICK = "kick"  # player = kicker


@dataclass(frozen=True)
class Event:
    """Same fields as environment.contract.Event (`kind` is its `type`, as a string)."""

    kind: str
    player: int | None
    tick: int
    other: int | None = None  # steal: the player who lost the ball
    event_id: int = -1  # episode-local id given by the adapter (-1 = unset)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class Action:
    """The 5 continuous controls (plan, section 2 'Action')."""

    move_x: float = 0.0  # [-1, 1]
    move_y: float = 0.0  # [-1, 1]
    effort: float = 0.0  # [0, 1]  sprint effort, consumes stamina
    kick: float = 0.0  # [0, 1]  kick trigger (>= 0.5 means kick)
    power: float = 0.0  # [0, 1]  kick power

    def clipped(self) -> Action:
        return Action(
            _clip(self.move_x, -1.0, 1.0),
            _clip(self.move_y, -1.0, 1.0),
            _clip(self.effort, 0.0, 1.0),
            _clip(self.kick, 0.0, 1.0),
            _clip(self.power, 0.0, 1.0),
        )

    def to_list(self) -> list[float]:
        return [self.move_x, self.move_y, self.effort, self.kick, self.power]

    @staticmethod
    def from_list(v) -> Action:
        return Action(*[float(x) for x in v]).clipped()


@dataclass
class PlayerState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    facing: float = 0.0  # radians, persists while stopped
    stamina: float = STAMINA_MAX  # [0, 100]


@dataclass
class BallState:
    x: float = PITCH_W / 2
    y: float = PITCH_H / 2
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class GameState:
    """Read-only snapshot. The engine keeps ONE possession owner."""

    players: list[PlayerState] = field(default_factory=list)  # index 0 and 1
    ball: BallState = field(default_factory=BallState)
    possession: int | None = None  # 0, 1 or None (free ball)
    score: tuple[int, int] = (0, 0)
    tick: int = 0
    phase: str = PHASE_PLAYING
    attack_dirs: tuple[int, int] = (1, -1)  # +1: player attacks toward +x
    # --- names shared with environment.contract.MatchState (timers in physics ticks) ---
    last_toucher: int | None = None
    steal_progress_ticks: int = 0
    possession_protection_ticks: int = 0
    restart_ticks_remaining: int = 0
    restart_owner: int | None = None  # player who receives the ball
    events: list[Event] = field(default_factory=list)  # events of THIS tick only (B-side)

    @property
    def contest_active(self) -> bool:
        """A steal contest is in progress (derived from steal_progress_ticks)."""
        return self.steal_progress_ticks > 0

    @property
    def restart_timer(self) -> float:
        """Seconds left in a restart (derived from restart_ticks_remaining)."""
        return self.restart_ticks_remaining / PHYS_HZ

    @property
    def time_left(self) -> float:
        return max(0.0, (MATCH_TICKS - self.tick) / PHYS_HZ)


class GameEngine(Protocol):
    """What Member A must provide. Same class for graphical and headless."""

    def reset(self, seed: int | None = None, swap_sides: bool = False) -> GameState:
        """swap_sides=True mirrors the start: player 0 attacks toward -x."""
        ...

    def step(self, actions: list[Action]) -> GameState:
        """Advance ONE physics update (1/60 s) with one Action per player."""
        ...


def goal_mouth(side: int) -> tuple[float, float, float]:
    """Return (x, y_top, y_bottom) of the goal line. side 0 = left, 1 = right."""
    x = 0.0 if side == 0 else PITCH_W
    return x, (PITCH_H - GOAL_W) / 2, (PITCH_H + GOAL_W) / 2
