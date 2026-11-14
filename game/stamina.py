"""Fixed-step energy bookkeeping and fatigue factors, without modifying momentum."""

from math import isfinite

from game.actions import Direction, MovementAction
from game.config import PHYSICS_DT, StaminaConfig

_DEFAULT_STAMINA = StaminaConfig()


def _validate_stamina(value: float) -> None:
    if not isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError("Stamina must be finite and within [0, 100].")


def update_stamina(
    value: float, action: MovementAction, *, config: StaminaConfig = _DEFAULT_STAMINA
) -> float:
    """Advance stamina one tick from requested active effort, not measured speed.

    Idle counts as rest even when its unused effort is high. Driving into a wall
    still costs effort. Rates vary linearly on either side of neutral effort.
    """
    _validate_stamina(value)
    effort = 0.0 if action.direction == Direction.IDLE else action.effort
    if effort > config.neutral_effort:
        rate = -config.drain_rate * (effort - config.neutral_effort) / (1.0 - config.neutral_effort)
    else:
        rate = config.recovery_rate * (config.neutral_effort - effort) / config.neutral_effort
    return max(0.0, min(100.0, value + rate * PHYSICS_DT))


def performance_factors(
    value: float, *, config: StaminaConfig = _DEFAULT_STAMINA
) -> tuple[float, float]:
    """Return speed/acceleration multipliers; never apply these to current velocity."""
    _validate_stamina(value)
    fraction = value / 100.0
    return (
        config.min_speed_factor + (1.0 - config.min_speed_factor) * fraction,
        config.min_acceleration_factor + (1.0 - config.min_acceleration_factor) * fraction,
    )
