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
    """Roll/radius/attachment tuning, not a kick-power mapping or ball speed cap."""

    reference_speed: float = 300.0
    deceleration: float = 100.0
    dribble_offset: float = 22.0  # Center-to-center; equals 15+7 with approved starting radii.
    radius: float = 7.0
    player_restitution: float = 0.5
    post_restitution: float = 0.7
    max_impacts_per_tick: int = 32

    def __post_init__(self) -> None:
        if not isfinite(self.reference_speed) or self.reference_speed <= 0.0:
            raise ValueError("Reference ball speed must be finite and positive.")
        if not isfinite(self.deceleration) or self.deceleration < 0.0:
            raise ValueError("Ball deceleration must be finite and nonnegative.")
        if not isfinite(self.dribble_offset) or self.dribble_offset <= 0.0:
            raise ValueError("Dribble offset must be finite and positive.")
        if not isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("Ball radius must be finite and positive.")
        for coefficient in (self.player_restitution, self.post_restitution):
            if not isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
                raise ValueError("Restitution must be finite and within [0, 1].")
        if type(self.max_impacts_per_tick) is not int or self.max_impacts_per_tick <= 0:
            raise ValueError("Impact budget must be a positive integer.")


@dataclass(frozen=True)
class KickConfig:
    """Approved experiment-0 kick speeds (units/s) and cooldown (60 Hz ticks)."""

    min_speed: float = 50.0
    max_speed: float = 300.0
    cooldown_ticks: int = 12

    def __post_init__(self) -> None:
        if not all(isfinite(value) and value > 0.0 for value in (self.min_speed, self.max_speed)):
            raise ValueError("Kick speeds must be finite and positive.")
        if self.min_speed > self.max_speed:
            raise ValueError("Minimum kick speed cannot exceed maximum kick speed.")
        if type(self.cooldown_ticks) is not int or self.cooldown_ticks < 0:
            raise ValueError("Kick cooldown must be a nonnegative integer number of ticks.")

    def speed_for_power(self, power: float) -> float:
        """Linear mapping; trigger off is handled separately, never by power."""
        if not isfinite(power) or not 0.0 <= power <= 1.0:
            raise ValueError("Kick power must be finite and within [0, 1].")
        return self.min_speed + (self.max_speed - self.min_speed) * power


@dataclass(frozen=True)
class PossessionConfig:
    """Approved starting pickup/steal rules; distances in units, durations in ticks."""

    kicker_pickup_lockout_ticks: int = 12
    steal_distance: float = 35.0
    steal_contest_ticks: int = 18
    protection_ticks: int = 30

    def __post_init__(self) -> None:
        if (
            type(self.kicker_pickup_lockout_ticks) is not int
            or self.kicker_pickup_lockout_ticks < 0
        ):
            raise ValueError("Pickup lockout must be a nonnegative integer number of ticks.")
        if not isfinite(self.steal_distance) or self.steal_distance <= 0.0:
            raise ValueError("Steal distance must be finite and positive.")
        if type(self.steal_contest_ticks) is not int or self.steal_contest_ticks <= 0:
            raise ValueError("Steal contest duration must be a positive integer number of ticks.")
        if type(self.protection_ticks) is not int or self.protection_ticks < 0:
            raise ValueError("Possession protection must be a nonnegative integer number of ticks.")


@dataclass(frozen=True)
class RulesConfig:
    """Goal opening width in logical units and restart duration in 60 Hz ticks."""

    goal_width: float = 120.0
    restart_ticks: int = 60
    post_radius: float = 4.0

    def __post_init__(self) -> None:
        if not isfinite(self.goal_width) or self.goal_width <= 0.0:
            raise ValueError("Goal opening width must be finite and positive.")
        if type(self.restart_ticks) is not int or self.restart_ticks <= 0:
            raise ValueError("Restart duration must be a positive integer number of ticks.")
        if not isfinite(self.post_radius) or self.post_radius <= 0.0:
            raise ValueError("Goalpost radius must be finite and positive.")


@dataclass(frozen=True)
class SimulationConfig:
    movement: MovementConfig = field(default_factory=MovementConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    stamina: StaminaConfig = field(default_factory=StaminaConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    kick: KickConfig = field(default_factory=KickConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)

    def __post_init__(self) -> None:
        if not 2 * self.ball.radius < self.rules.goal_width <= self.pitch.height:
            raise ValueError("Goal opening must fit the pitch and be wider than the ball diameter.")
