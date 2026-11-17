"""Hybrid player intent in logical coordinates: x right, y down (not grid positions)."""

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite, sqrt


class Direction(IntEnum):
    """Engine direction IDs; the final PPO encoding is a separate interface."""

    IDLE = 0
    NORTH = 1
    NORTHEAST = 2
    EAST = 3
    SOUTHEAST = 4
    SOUTH = 5
    SOUTHWEST = 6
    WEST = 7
    NORTHWEST = 8

    @property
    def vector(self) -> tuple[float, float]:
        """A unit direction, or the zero vector for idle."""
        return _DIRECTION_VECTORS[self.value]


_DIAGONAL = 1.0 / sqrt(2.0)
_DIRECTION_VECTORS = (
    (0.0, 0.0),
    (0.0, -1.0),
    (_DIAGONAL, -_DIAGONAL),
    (1.0, 0.0),
    (_DIAGONAL, _DIAGONAL),
    (0.0, 1.0),
    (-_DIAGONAL, _DIAGONAL),
    (-1.0, 0.0),
    (-_DIAGONAL, -_DIAGONAL),
)


@dataclass(frozen=True)
class MovementAction:
    """Movement-only subset, also accepted by simulation for backward compatibility.

    Use Direction(raw_id) when adapting numerical controls to this typed interface.
    Invalid efforts are rejected rather than silently clipped inside physics.
    """

    direction: Direction = Direction.IDLE
    effort: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.direction, Direction):
            raise TypeError("Direction must be a Direction enum member.")
        if not isfinite(self.effort) or not 0.0 <= self.effort <= 1.0:
            raise ValueError("Effort must be finite and within [0, 1].")


@dataclass(frozen=True)
class PlayerAction(MovementAction):
    """Hybrid engine action, not an SB3 action-space encoding.

    Trigger 1 requests one kick per decision; power 0 still requests the minimum
    speed. A trigger in step_tick represents a fresh single-tick request.
    """

    kick_trigger: int = 0
    kick_power: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.kick_trigger) not in (int, bool) or self.kick_trigger not in (0, 1):
            raise ValueError("Kick trigger must be binary (0 or 1).")
        if not isfinite(self.kick_power) or not 0.0 <= self.kick_power <= 1.0:
            raise ValueError("Kick power must be finite and within [0, 1].")
