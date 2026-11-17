"""Fixed-step player movement, without rendering or wall time.

Positions use logical units, velocity units/second, and facing radians.
Facing zero points east; positive angles turn clockwise in the y-down world.
Shared simulation resolves player-player contact separately after movement updates.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from math import atan2, hypot, isclose, isfinite, pi, tau

from game.actions import Direction, MovementAction
from game.config import DEFAULT_BRAKING, PHYSICS_DT, MovementConfig, PitchConfig, StaminaConfig
from game.stamina import performance_factors, update_stamina

_DEFAULT_MOVEMENT = MovementConfig()
_DEFAULT_STAMINA = StaminaConfig()


class PlayerId(StrEnum):
    """Stable identity, independent of spawn side or attack orientation."""

    A = "A"
    B = "B"


@dataclass(frozen=True)
class PlayerState:
    """Player snapshot; interaction timers advance in shared simulation, not movement."""

    position: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    facing: float = 0.0
    stamina: float = 100.0
    kick_cooldown_ticks: int = 0
    pickup_lockout_ticks: int = 0

    def __post_init__(self) -> None:
        for vector in (self.position, self.velocity):
            if len(vector) != 2 or not all(isfinite(value) for value in vector):
                raise ValueError("Position and velocity must each contain two finite values.")
        # Own immutable vectors even if a caller supplies a list at an API boundary.
        object.__setattr__(self, "position", tuple(self.position))
        object.__setattr__(self, "velocity", tuple(self.velocity))
        if not isfinite(hypot(*self.velocity)):
            raise ValueError("Velocity magnitude must be finite.")
        if not isfinite(self.facing):
            raise ValueError("Facing must be finite.")
        if not isfinite(self.stamina) or not 0.0 <= self.stamina <= 100.0:
            raise ValueError("Stamina must be finite and within [0, 100].")
        if type(self.kick_cooldown_ticks) is not int or self.kick_cooldown_ticks < 0:
            raise ValueError("Kick cooldown must be a nonnegative integer number of ticks.")
        if type(self.pickup_lockout_ticks) is not int or self.pickup_lockout_ticks < 0:
            raise ValueError("Pickup lockout must be a nonnegative integer number of ticks.")


def step_idle_player(player: PlayerState, *, braking: float = DEFAULT_BRAKING) -> PlayerState:
    """Return a new snapshot after one 1/60-second idle tick in open space.

    Reduce speed by braking * dt, without changing velocity direction or reversing.
    Update velocity first, then position using that new velocity (semi-implicit Euler).
    This discretization underestimates continuous stopping distance, especially on
    the final tick. Facing and stamina are deliberately unchanged in this slice.
    """
    if not isfinite(braking) or braking < 0.0:
        raise ValueError("Braking must be finite and nonnegative.")

    vx, vy = player.velocity
    speed = hypot(vx, vy)
    speed_loss = braking * PHYSICS_DT
    # The relative tolerance handles roundoff at an exact stopping tick, not a
    # gameplay speed cutoff. Tiny velocities with zero braking still coast.
    if speed <= speed_loss or isclose(speed, speed_loss, rel_tol=1e-12, abs_tol=0.0):
        vx, vy = 0.0, 0.0
    elif braking > 0.0:
        scale = (speed - speed_loss) / speed
        vx, vy = vx * scale, vy * scale

    x, y = player.position
    return replace(
        player,
        position=(x + vx * PHYSICS_DT, y + vy * PHYSICS_DT),
        velocity=(vx, vy),
    )


def step_player(
    player: PlayerState,
    action: MovementAction,
    *,
    movement: MovementConfig = _DEFAULT_MOVEMENT,
    stamina: StaminaConfig = _DEFAULT_STAMINA,
    pitch: PitchConfig | None = None,
) -> PlayerState:
    """Advance one fixed tick using acceleration-limited target velocity.

    Update stamina first, then fatigue the target speed and acceleration limit.
    Never multiply current velocity by the fatigue factor. Effort scales target
    speed, and active steering changes velocity by at most acceleration * dt.
    Idle or zero effort uses constant coasting braking instead. A non-idle command
    turns facing even at zero effort; it never derives facing from sliding velocity.
    Velocity is updated before position. Braking and turning are unaffected by fatigue.
    """
    player = replace(player, stamina=update_stamina(player.stamina, action, config=stamina))
    speed_factor, acceleration_factor = performance_factors(player.stamina, config=stamina)
    movement = replace(
        movement,
        max_speed=movement.max_speed * speed_factor,
        acceleration=movement.acceleration * acceleration_factor,
    )
    dx, dy = action.direction.vector
    if action.direction == Direction.IDLE or action.effort == 0.0:
        result = step_idle_player(player, braking=movement.braking)
    else:
        target_speed = action.effort * movement.max_speed
        target_vx, target_vy = dx * target_speed, dy * target_speed
        change_x = target_vx - player.velocity[0]
        change_y = target_vy - player.velocity[1]
        change_length = hypot(change_x, change_y)
        limit = movement.acceleration * PHYSICS_DT
        if change_length <= limit or isclose(change_length, limit, rel_tol=1e-12):
            vx, vy = target_vx, target_vy
        else:
            scale = limit / change_length
            vx = player.velocity[0] + change_x * scale
            vy = player.velocity[1] + change_y * scale
        x, y = player.position
        result = replace(
            player,
            velocity=(vx, vy),
            position=(x + vx * PHYSICS_DT, y + vy * PHYSICS_DT),
        )

    if action.direction != Direction.IDLE and movement.turn_rate > 0.0:
        target_angle = atan2(dy, dx)
        difference = (target_angle - player.facing + pi) % tau - pi
        max_turn = movement.turn_rate * PHYSICS_DT
        turn = max(-max_turn, min(max_turn, difference))
        result = replace(result, facing=(player.facing + turn + pi) % tau - pi)

    if pitch is not None:
        result = constrain_to_pitch(result, pitch)
    return result


def constrain_to_pitch(player: PlayerState, pitch: PitchConfig) -> PlayerState:
    """Keep the whole player inside; cancel outward normal velocity, not sliding."""
    radius = pitch.player_radius
    x = max(radius, min(pitch.width - radius, player.position[0]))
    y = max(radius, min(pitch.height - radius, player.position[1]))
    vx, vy = player.velocity
    if (x == radius and vx < 0.0) or (x == pitch.width - radius and vx > 0.0):
        vx = 0.0
    if (y == radius and vy < 0.0) or (y == pitch.height - radius and vy > 0.0):
        vy = 0.0
    return replace(player, position=(x, y), velocity=(vx, vy))
