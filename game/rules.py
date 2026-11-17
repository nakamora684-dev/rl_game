"""Swept whole-ball goal/boundary classification, independent of rewards/rendering."""

from dataclasses import dataclass
from enum import StrEnum
from math import isclose, isfinite

from game.config import SimulationConfig
from game.player import PlayerId


class BoundarySide(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


def goalpost_centres(config: SimulationConfig) -> tuple[tuple[float, float], ...]:
    """Four posts whose INNER edges preserve the configured clear goal opening."""
    top = (config.pitch.height - config.rules.goal_width) / 2 - config.rules.post_radius
    bottom = (config.pitch.height + config.rules.goal_width) / 2 + config.rules.post_radius
    return ((0.0, top), (0.0, bottom), (config.pitch.width, top), (config.pitch.width, bottom))


@dataclass(frozen=True)
class BallExit:
    fraction: float
    position: tuple[float, float]
    side: BoundarySide
    is_goal: bool

    def __post_init__(self) -> None:
        if not isfinite(self.fraction) or not 0.0 <= self.fraction <= 1.0:
            raise ValueError("Exit fraction must be finite and within [0, 1].")
        if len(self.position) != 2 or not all(isfinite(value) for value in self.position):
            raise ValueError("Exit position must have two finite coordinates.")
        object.__setattr__(self, "position", tuple(self.position))
        if not isinstance(self.side, BoundarySide) or type(self.is_goal) is not bool:
            raise TypeError("Exit requires a typed boundary side and boolean goal classification.")


@dataclass(frozen=True)
class RestartPlan:
    """Owner is the kickoff/restart receiver; None means unknown-touch neutral restart."""

    owner: PlayerId | None
    exit: BallExit

    def __post_init__(self) -> None:
        if self.owner is not None and not isinstance(self.owner, PlayerId):
            raise TypeError("Restart owner must be PlayerId or None.")
        if not isinstance(self.exit, BallExit):
            raise TypeError("Restart plan must contain a BallExit.")


def detect_ball_exit(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    config: SimulationConfig,
) -> BallExit | None:
    """Earliest strict whole-ball exit on a linear snapshot sweep.

    Merely reaching centre x=width+radius (or the other three thresholds) is not
    an exit; moving beyond is. Already-outside setup snapshots are handled at
    fraction zero. A simultaneous corner exit is out of bounds, never a goal.
    Goal-post impact physics is not modelled; the whole ball must fit the opening.
    """
    pitch, radius = config.pitch, config.ball.radius
    low = (pitch.height - config.rules.goal_width) / 2 + radius
    high = (pitch.height + config.rules.goal_width) / 2 - radius
    candidates = []
    for side, axis, threshold, sign in (
        (BoundarySide.TOP, 1, -radius, -1),
        (BoundarySide.BOTTOM, 1, pitch.height + radius, 1),
        (BoundarySide.LEFT, 0, -radius, -1),
        (BoundarySide.RIGHT, 0, pitch.width + radius, 1),
    ):
        before, after = sign * (start[axis] - threshold), sign * (end[axis] - threshold)
        if before > 0.0:
            fraction = 0.0
        elif after > 0.0:
            fraction = -before / (after - before)
        else:
            continue
        position = tuple(a + fraction * (b - a) for a, b in zip(start, end))
        goal = side in (BoundarySide.LEFT, BoundarySide.RIGHT) and low <= position[1] <= high
        candidates.append(BallExit(fraction, position, side, goal))
    if not candidates:
        return None
    earliest = min(item.fraction for item in candidates)
    tied = [
        item for item in candidates if isclose(item.fraction, earliest, rel_tol=0.0, abs_tol=1e-12)
    ]
    return min(tied, key=lambda item: item.is_goal)
