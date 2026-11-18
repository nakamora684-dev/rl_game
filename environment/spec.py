"""Approved hybrid controls and versioned observation order for Members B/C/D.

This Dict action space preserves gameplay exactly. Standard SB3 PPO does NOT
accept it directly; its mixed distribution/encoding is a separate C/D task.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from game.actions import Direction, PlayerAction
from game.config import (
    DECISION_HZ,
    MATCH_TICKS,
    PHYSICS_HZ,
    TICKS_PER_DECISION,
    SimulationConfig,
)

DEFAULT_CONFIG = SimulationConfig()
PITCH_W = DEFAULT_CONFIG.pitch.width
PITCH_H = DEFAULT_CONFIG.pitch.height
GOAL_W = DEFAULT_CONFIG.rules.goal_width
FRAMES_PER_DECISION = PHYSICS_HZ // DECISION_HZ
assert FRAMES_PER_DECISION == TICKS_PER_DECISION
MAX_PHYSICS_TICKS = MATCH_TICKS
MATCH_SECONDS = MATCH_TICKS / PHYSICS_HZ
MAX_DECISIONS = (MATCH_TICKS + TICKS_PER_DECISION - 1) // TICKS_PER_DECISION
MAX_SCORE_DIFF = 3
# Observation scale only, NOT an engine speed cap. Rebounds can exceed kick speed.
BALL_SPEED_SCALE = 600.0

ACTION_NAMES = ("direction", "effort", "kick_trigger", "kick_power")
ACTION_DIM = len(ACTION_NAMES)  # Four heterogeneous components, NOT a flat Box dimension.
OBSERVATION_VERSION = 2


def _discrete(value, name: str, maximum: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a rounded continuous value.")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be within [0,{maximum}].")
    return int(value)


def _continuous(value, name: str) -> float:
    array = np.asarray(value)
    if array.shape != (1,) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric array of shape (1,).")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real.")
    number = float(array[0])
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be finite and within [0,1].")
    return number


def decode_action(raw: Mapping) -> PlayerAction:
    """Decode an exact hybrid policy-space action; reject, never clip/round.

    direction/kick_trigger are integer scalars; effort/kick_power are length-one
    arrays as declared by the Gymnasium Dict. No independent aiming or charging.
    """
    if not isinstance(raw, Mapping) or set(raw) != set(ACTION_NAMES):
        raise ValueError(f"Expected an action mapping with keys {ACTION_NAMES}.")
    return PlayerAction(
        direction=Direction(_discrete(raw["direction"], "direction", 8)),
        effort=_continuous(raw["effort"], "effort"),
        kick_trigger=_discrete(raw["kick_trigger"], "kick_trigger", 1),
        kick_power=_continuous(raw["kick_power"], "kick_power"),
    )


_MIRROR_DIRECTION = (
    Direction.IDLE,
    Direction.NORTH,
    Direction.NORTHWEST,
    Direction.WEST,
    Direction.SOUTHWEST,
    Direction.SOUTH,
    Direction.SOUTHEAST,
    Direction.EAST,
    Direction.NORTHEAST,
)


def policy_to_world(action: PlayerAction, attack_dir: int) -> PlayerAction:
    """Mirror discrete directions horizontally; effort/kick semantics are unchanged."""
    if attack_dir not in (1, -1):
        raise ValueError("attack_dir must be +1 or -1.")
    return (
        action
        if attack_dir == 1
        else replace(action, direction=_MIRROR_DIRECTION[action.direction])
    )


@dataclass(frozen=True)
class ObsFeature:
    name: str
    low: float
    high: float
    group: str


def _f(group: str, names: list[str], low: float, high: float) -> list[ObsFeature]:
    return [ObsFeature(name, low, high, group) for name in names]


# Version 2: retain C's first 35 features, then describe actual engine interactions.
OBS_FEATURES = tuple(
    _f("self", ["self_x", "self_y", "self_vx", "self_vy"], -1, 1)
    + _f("self", ["self_face_sin", "self_face_cos"], -1, 1)
    + _f("self", ["self_stamina"], 0, 1)
    + _f("opp", ["opp_rel_x", "opp_rel_y", "opp_vx", "opp_vy"], -1, 1)
    + _f("opp", ["opp_face_sin", "opp_face_cos"], -1, 1)
    + _f("opp", ["opp_stamina"], 0, 1)
    + _f("ball", ["ball_rel_x", "ball_rel_y", "ball_vx", "ball_vy"], -1, 1)
    + _f("ball", ["poss_self", "poss_opp", "poss_free"], 0, 1)
    + [
        feature
        for target in ("ball", "opp", "own_goal", "opp_goal")
        for feature in (
            *_f("geom", [f"to_{target}_dx", f"to_{target}_dy"], -1, 1),
            *_f("geom", [f"to_{target}_dist"], 0, 1),
        )
    ]
    + _f("context", ["time_remaining"], 0, 1)
    + _f("context", ["score_diff"], -1, 1)
    + _f(
        "interaction",
        [
            "contest_active",
            "contest_progress",
            "possession_protection",
            "self_kick_cooldown",
            "opp_kick_cooldown",
            "self_pickup_lockout",
            "opp_pickup_lockout",
        ],
        0,
        1,
    )
    + _f(
        "phase",
        [
            "phase_playing",
            "phase_goal_restart",
            "phase_out_of_bounds_restart",
            "phase_finished",
            "restart_remaining",
            "restart_self",
            "restart_opp",
            "restart_neutral",
        ],
        0,
        1,
    )
)
OBS_DIM = len(OBS_FEATURES)  # 50
OBS_NAMES = tuple(f.name for f in OBS_FEATURES)
OBS_INDEX = {name: i for i, name in enumerate(OBS_NAMES)}
assert len(OBS_INDEX) == OBS_DIM


def obs_bounds() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([f.low for f in OBS_FEATURES], dtype=np.float32),
        np.array([f.high for f in OBS_FEATURES], dtype=np.float32),
    )


def to_policy_x(x: float, attack_dir: int, pitch_width: float = PITCH_W) -> float:
    return x if attack_dir == 1 else pitch_width - x


def to_policy_angle(theta: float, attack_dir: int) -> float:
    return theta if attack_dir == 1 else float(np.pi - theta)


def make_action_space():
    from gymnasium import spaces

    return spaces.Dict(
        {
            "direction": spaces.Discrete(9),
            "effort": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "kick_trigger": spaces.Discrete(2),
            "kick_power": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }
    )


def make_observation_space():
    from gymnasium import spaces

    low, high = obs_bounds()
    return spaces.Box(low=low, high=high, dtype=np.float32)


if __name__ == "__main__":
    print(f"Hybrid action components: {ACTION_NAMES}")
    print(f"Observation version {OBSERVATION_VERSION}: {OBS_DIM} features")
    for i, feature in enumerate(OBS_FEATURES):
        print(f"{i:2d} {feature.group:<12s} {feature.name:<30s}")
