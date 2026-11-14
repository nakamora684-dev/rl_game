"""Logical units and configurable starting values, not finalized physics tuning."""

from dataclasses import dataclass, field
from math import isfinite, tau

PHYSICS_HZ = 60
PHYSICS_DT = 1.0 / PHYSICS_HZ
DECISION_HZ = 10
TICKS_PER_DECISION = PHYSICS_HZ // DECISION_HZ
MATCH_TICKS = 60 * PHYSICS_HZ
DEFAULT_BRAKING = 40.0


@dataclass(frozen=True)
class MovementConfig:
    """Speeds in units/s, acceleration/braking in units/s², turning in radians/s."""

    max_speed: float = 160.0
    acceleration: float = 320.0
    braking: float = DEFAULT_BRAKING
    turn_rate: float = tau

    def __post_init__(self) -> None:
        for value in (self.max_speed, self.acceleration, self.braking, self.turn_rate):
            if not isfinite(value) or value < 0.0:
                raise ValueError("Movement parameters must be finite and nonnegative.")


@dataclass(frozen=True)
class PitchConfig:
    width: float = 800.0
    height: float = 500.0
    player_radius: float = 15.0

    def __post_init__(self) -> None:
        for value in (self.width, self.height, self.player_radius):
            if not isfinite(value) or value <= 0.0:
                raise ValueError("Pitch dimensions and player radius must be finite and positive.")
        if 2.0 * self.player_radius > min(self.width, self.height):
            raise ValueError("The player diameter must fit inside the pitch.")


@dataclass(frozen=True)
class StaminaConfig:
    """Approved starting tuning: rates are stamina points per simulated second."""

    drain_rate: float = 5.0
    recovery_rate: float = 100.0 / 15.0
    neutral_effort: float = 0.5
    min_speed_factor: float = 0.5
    min_acceleration_factor: float = 0.5

    def __post_init__(self) -> None:
        for value in (self.drain_rate, self.recovery_rate):
            if not isfinite(value) or value < 0.0:
                raise ValueError("Stamina rates must be finite and nonnegative.")
        if not isfinite(self.neutral_effort) or not 0.0 < self.neutral_effort < 1.0:
            raise ValueError("Neutral effort must be finite and strictly between 0 and 1.")
        for value in (self.min_speed_factor, self.min_acceleration_factor):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Minimum performance factors must be finite and within [0, 1].")


@dataclass(frozen=True)
class BallConfig:
    """Reference roll tuning, not a kick-power mapping or ball speed cap."""

    reference_speed: float = 300.0
    deceleration: float = 100.0

    def __post_init__(self) -> None:
        if not isfinite(self.reference_speed) or self.reference_speed <= 0.0:
            raise ValueError("Reference ball speed must be finite and positive.")
        if not isfinite(self.deceleration) or self.deceleration < 0.0:
            raise ValueError("Ball deceleration must be finite and nonnegative.")


@dataclass(frozen=True)
class SimulationConfig:
    movement: MovementConfig = field(default_factory=MovementConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    stamina: StaminaConfig = field(default_factory=StaminaConfig)
    ball: BallConfig = field(default_factory=BallConfig)
