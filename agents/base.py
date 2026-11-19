"""Common interface for any agent (rule-based now, PPO policy later)."""

from __future__ import annotations

from typing import Protocol

from agents.interface import Action, GameState


class Agent(Protocol):
    def act(self, state: GameState) -> Action:
        """Return the 5 controls for the player this agent controls."""
        ...


class HeldAgent:
    """Wraps an agent so it decides at 10 Hz: one decision, held for `every` ticks.

    The plan fixes 60 Hz physics and 6 physics updates per decision. Using this
    wrapper for the baseline keeps it on the same clock as the RL agent.
    """

    def __init__(self, agent, every: int = 6) -> None:
        self.agent, self.every = agent, every
        self._cache = None
        self._last = 0

    def reset(self) -> None:
        self._cache = None
        if hasattr(self.agent, "reset"):
            self.agent.reset()

    def act(self, state: GameState):
        if (
            self._cache is None or state.tick - self._last >= self.every or state.tick < self._last
        ):  # new match: tick went back
            self._cache, self._last = self.agent.act(state), state.tick
        return self._cache
