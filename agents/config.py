"""All tunable numbers of the rule-based baseline in ONE place (tune in P5).

`BaselineConfig()` with no arguments is the "normal" preset. Presets for
easy / hard live in agents/difficulty.py.

UNIT NOTE: Member A's engine reports velocities in logical units per SECOND, so
`VEL_PER_TICK = False` (agents/engine_adapter.py copies them unchanged).
tests/test_interface_contract.py (test_velocity_unit_matches_displacement) checks it.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.interface import PHYS_HZ

VEL_PER_TICK: bool = False


def to_per_second(v: float) -> float:
    """Convert a GameState velocity component to logical units per second."""
    return v * PHYS_HZ if VEL_PER_TICK else v


@dataclass(frozen=True)
class BaselineConfig:
    name: str = "normal"
    # --- shooting ---------------------------------------------------------
    shoot_dist: float = 220.0  # max distance to the aim point to shoot
    face_tol: float = 0.35  # rad: max angle between facing and aim line
    aim_corners: bool = True  # aim at the corner farther from the opponent
    post_margin: float = 14.0  # stay this far inside the posts
    snap_shot: bool = True  # shoot early (looser angle) when pressed
    close_power: float = 0.85  # kick power when closer than 150 units
    # --- dribbling --------------------------------------------------------
    opp_avoid_dist: float = 90.0  # swerve when the opponent is closer
    wall_margin: float = 45.0  # near a touchline: swerve toward the centre
    # --- defending --------------------------------------------------------
    goal_side: bool = True  # stand between opponent and own goal first
    block_dist: float = 60.0  # distance in front of the carrier to block
    contest_dist: float = 70.0  # closer than this: go for the steal
    # --- chasing a free ball ---------------------------------------------
    lead_time_max: float = 0.4  # s: max anticipation of the ball movement
    ref_speed: float = 180.0  # units/s used to estimate the time to reach
    # --- stamina ----------------------------------------------------------
    sprint_reserve: float = 0.30  # stop sprinting under this stamina share
    sprint_resume: float = 0.50  # resume above this (hysteresis)
    sprint_min_dist: float = 50.0  # do not sprint for short distances
    endgame_all_in: bool = True  # spend the reserve when losing/drawing late
    endgame_secs: float = 10.0
    endgame_reserve: float = 0.10
    endgame_resume: float = 0.30
