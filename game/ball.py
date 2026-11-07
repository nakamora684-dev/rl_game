"""Free-ball motion only: no possession, kicks, collision, goal, or boundary rules."""

from dataclasses import dataclass, replace
from math import hypot, isclose, isfinite

from game.config import PHYSICS_DT, BallConfig

_DEFAULT_BALL = BallConfig()


@dataclass(frozen=True)
class BallState:
    """Authoritative free-ball position/velocity; ownership is a later mechanic."""

    position: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        for vector in (self.position, self.velocity):
            if len(vector) != 2 or not all(isfinite(value) for value in vector):
                raise ValueError("Ball position and velocity must contain two finite values.")
        object.__setattr__(self, "position", tuple(self.position))
        object.__setattr__(self, "velocity", tuple(self.velocity))
        if not isfinite(hypot(*self.velocity)):
            raise ValueError("Ball speed must be finite.")


def step_free_ball(ball: BallState, *, config: BallConfig = _DEFAULT_BALL) -> BallState:
    """Integrate constant deceleration over one fixed tick, without clamping to a pitch.

    Keep velocity direction, reduce speed without reversal, and integrate average
    speed over the moving portion of the tick. Unlike the player's approximate
    semi-implicit integration, this preserves the analytic roll stopping distance.
    A weaker roll stops sooner: three seconds applies only to reference speed 300.
    """
    vx, vy = ball.velocity
    speed = hypot(vx, vy)
    if speed == 0.0:
        return ball

    if config.deceleration == 0.0:
        displacement_x, displacement_y = vx * PHYSICS_DT, vy * PHYSICS_DT
    else:
        speed_loss = config.deceleration * PHYSICS_DT
        moving_time = min(PHYSICS_DT, speed / config.deceleration)
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
