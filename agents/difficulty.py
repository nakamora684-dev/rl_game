"""Difficulty presets for the rule-based baseline.

easy   : short shots, no goal-side defence, no anticipation, tires quickly.
normal : the reference baseline (BaselineConfig defaults). Used for evaluation.
hard   : earlier and more accurate shots, wider swerve, better anticipation.

They are also usable as a simple curriculum for PPO (easy -> normal -> hard).
All presets use the SAME physics and stamina rules and the SAME View: they only
change decision thresholds.
"""

from __future__ import annotations

from dataclasses import replace

from agents.base import HeldAgent
from agents.config import BaselineConfig
from agents.interface import DECISION_EVERY
from agents.rule_based import RuleBasedAgent

PRESETS: dict[str, BaselineConfig] = {
    "easy": replace(
        BaselineConfig(),
        name="easy",
        shoot_dist=150.0,
        face_tol=0.60,
        aim_corners=False,
        snap_shot=False,
        opp_avoid_dist=60.0,
        goal_side=False,
        lead_time_max=0.0,
        sprint_reserve=0.50,
        sprint_resume=0.70,
        endgame_all_in=False,
    ),
    "normal": BaselineConfig(),
    "hard": replace(
        BaselineConfig(),
        name="hard",
        shoot_dist=260.0,
        face_tol=0.25,
        opp_avoid_dist=110.0,
        contest_dist=80.0,
        lead_time_max=0.6,
        sprint_reserve=0.25,
        sprint_resume=0.45,
    ),
}


def make_baseline(index: int, difficulty: str = "normal", held: bool = True):
    """Build the baseline for player `index`. `held=True` wraps it at 10 Hz."""
    if difficulty not in PRESETS:
        raise ValueError(f"unknown difficulty {difficulty!r}; choose from {sorted(PRESETS)}")
    agent = RuleBasedAgent(index, PRESETS[difficulty])
    return HeldAgent(agent, every=DECISION_EVERY) if held else agent
