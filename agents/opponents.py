"""Opponent registry for training / evaluation.

An opponent is an ``ActionProvider`` -- ``(SimulationState, PlayerId) -> PlayerAction``
returning WORLD-space controls (same contract as ``game.runner``).

* ``idle``       - does nothing (inspection only).
* ``chaser``     - TEMPORARY stand-in so Member D can test the pipeline before
                   Member B delivers the real baseline. NOT the evaluation baseline.
* ``rule_based`` - Member B's baseline (``agents.provider.rule_based_provider``), difficulty
                   "normal". ``rule_based_easy`` / ``rule_based_hard`` pick the other presets.
"""

from __future__ import annotations

import math

from game.actions import Direction, PlayerAction
from game.player import PlayerId
from game.runner import idle_provider

# angle sector (angle / 45 deg, y-down world) -> Direction id
_SECTOR_TO_DIRECTION = {
    0: Direction.EAST,
    -1: Direction.NORTHEAST,
    -2: Direction.NORTH,
    -3: Direction.NORTHWEST,
    4: Direction.WEST,
    -4: Direction.WEST,
    3: Direction.SOUTHWEST,
    2: Direction.SOUTH,
    1: Direction.SOUTHEAST,
}


def direction_towards(dx: float, dy: float) -> Direction:
    if math.hypot(dx, dy) < 1e-6:
        return Direction.IDLE
    return _SECTOR_TO_DIRECTION[round(math.atan2(dy, dx) / (math.pi / 4))]


class ChaserOpponent:
    """Chase the ball; with the ball, run at the goal and shoot from range."""

    def __init__(self, shoot_range: float = 230.0, effort: float = 0.7) -> None:
        self.shoot_range = shoot_range
        self.effort = effort

    def reset(self) -> None:  # stateless, kept for interface symmetry
        pass

    def __call__(self, state, player: PlayerId) -> PlayerAction:
        me = state.player_a if player == PlayerId.A else state.player_b
        attack = state.attack_direction_a if player == PlayerId.A else -state.attack_direction_a
        width, height = 800.0, 500.0  # default pitch; only used for the goal target
        if state.ball.possession_owner == player:
            goal = (width if attack == 1 else 0.0, height / 2)
            dx, dy = goal[0] - me.position[0], goal[1] - me.position[1]
            shoot = abs(dx) < self.shoot_range
            return PlayerAction(
                direction=direction_towards(dx, dy),
                effort=self.effort,
                kick_trigger=int(shoot),
                kick_power=1.0 if shoot else 0.0,
            )
        dx = state.ball.position[0] - me.position[0]
        dy = state.ball.position[1] - me.position[1]
        return PlayerAction(direction=direction_towards(dx, dy), effort=self.effort)


OPPONENT_NAMES = ("idle", "chaser", "rule_based", "rule_based_easy", "rule_based_hard")


def make_opponent(name: str):
    """Build a FRESH opponent instance (one per environment)."""
    if name == "idle":
        return idle_provider
    if name == "chaser":
        return ChaserOpponent()
    if name.startswith("rule_based"):
        from agents.provider import rule_based_provider

        difficulty = name.removeprefix("rule_based").removeprefix("_") or "normal"
        return rule_based_provider(difficulty)  # has reset(); EnvRunner calls it each episode
    raise ValueError(f"Unknown opponent {name!r}; choose from {OPPONENT_NAMES}.")
