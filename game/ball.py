"""Ball state and free motion; controlled attachment is handled by game.possession."""

from dataclasses import dataclass, replace
from math import hypot, isclose, isfinite

from game.config import PHYSICS_DT, BallConfig
from game.player import PlayerId

_DEFAULT_BALL = BallConfig()


@dataclass(frozen=True)
class BallState:
    """One authoritative owner; attached position is synchronized from the carrier.

    Velocity represents free-ball motion only and must be zero while controlled.
    Last toucher persists separately when possession ends.
    """

    position: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    possession_owner: PlayerId | None = None
    last_touch_player: PlayerId | None = None

    def __post_init__(self) -> None:
        for vector in (self.position, self.velocity):
            if len(vector) != 2 or not all(isfinite(value) for value in vector):
                raise ValueError("Ball position and velocity must contain two finite values.")
        object.__setattr__(self, "position", tuple(self.position))
        object.__setattr__(self, "velocity", tuple(self.velocity))
        if not isfinite(hypot(*self.velocity)):
            raise ValueError("Ball speed must be finite.")
        for player in (self.possession_owner, self.last_touch_player):
            if player is not None and not isinstance(player, PlayerId):
                raise TypeError("Owner and last toucher must be PlayerId members or None.")
        if self.possession_owner is not None:
            if self.velocity != (0.0, 0.0):
                raise ValueError("A controlled ball cannot also have free-ball velocity.")
            if self.last_touch_player != self.possession_owner:
                raise ValueError("A controlled ball's last toucher must be its owner.")


def step_free_ball(ball: BallState, *, config: BallConfig = _DEFAULT_BALL) -> BallState:
    """Integrate constant deceleration over one fixed tick, without clamping to a pitch.

    Keep velocity direction, reduce speed without reversal, and integrate average
    speed over the moving portion of the tick. Unlike the player's approximate
    semi-implicit integration, this preserves the analytic roll stopping distance.
    A weaker roll stops sooner: three seconds applies only to reference speed 300.
    """
    return integrate_free_ball(ball, PHYSICS_DT, config=config)


def integrate_free_ball(
    ball: BallState, duration: float, *, config: BallConfig = _DEFAULT_BALL
) -> BallState:
    """Analytic roll over a finite nonnegative duration, also used between impacts."""
    if ball.possession_owner is not None:
        raise ValueError("A controlled ball must use attachment, not free-ball integration.")
    if not isfinite(duration) or duration < 0.0:
        raise ValueError("Integration duration must be finite and nonnegative.")
    vx, vy = ball.velocity
    speed = hypot(vx, vy)
    if speed == 0.0 or duration == 0.0:
        return ball

    if config.deceleration == 0.0:
        displacement_x, displacement_y = vx * duration, vy * duration
    else:
        speed_loss = config.deceleration * duration
        moving_time = min(duration, speed / config.deceleration)
        if speed <= speed_loss or isclose(speed, speed_loss, rel_tol=1e-12, abs_tol=0.0):
            next_speed = 0.0
        else:
            next_speed = speed - speed_loss
        distance = (speed / 2.0 + next_speed / 2.0) * moving_time
        direction_x, direction_y = vx / speed, vy / speed
        displacement_x, displacement_y = direction_x * distance, direction_y * distance
        vx, vy = direction_x * next_speed, direction_y * next_speed

    x, y = ball.position
    return replace(ball, position=(x + displacement_x, y + displacement_y), velocity=(vx, vy))
