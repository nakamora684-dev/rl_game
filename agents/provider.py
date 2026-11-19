"""The rule-based baseline as a native `game.runner.ActionProvider`.

    provider(state: SimulationState, player: PlayerId) -> PlayerAction     (world space)

This is the exact signature of `game.runner.run_match(provider_a, provider_b)` and of
`FootballEnv(opponent=provider)` (docs/environment.md). It is queried once per six-tick
decision, restarts included, with A's immutable pre-decision state; the baseline then
returns idle during restarts by itself.

The provider keeps private state (sprint hysteresis): call `reset()` (or build a new one)
before replaying a match, as the wrapper never reseeds a teammate's controller.
"""

from __future__ import annotations

from agents.difficulty import PRESETS, make_baseline
from agents.engine_adapter import index_of, to_player_action, to_view
from game.actions import PlayerAction
from game.player import PlayerId
from game.simulation import SimulationState


class RuleBasedProvider:
    def __init__(self, difficulty: str = "normal") -> None:
        if difficulty not in PRESETS:
            raise ValueError(f"unknown difficulty {difficulty!r}; choose from {sorted(PRESETS)}")
        self.difficulty = difficulty
        self._agents: dict[int, object] = {}

    def reset(self) -> None:
        self._agents = {}

    def __call__(self, state: SimulationState, player: PlayerId) -> PlayerAction:
        i = index_of(player)
        agent = self._agents.get(i)
        if agent is None:  # one baseline per identity, so it works as A or B
            agent = self._agents[i] = make_baseline(i, self.difficulty, held=False)
        return to_player_action(agent.act(to_view(state)))


def rule_based_provider(difficulty: str = "normal") -> RuleBasedProvider:
    """Factory: a fresh provider (own private state) for FootballEnv / game.runner."""
    return RuleBasedProvider(difficulty)
