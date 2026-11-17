"""Headless full-match orchestration; controllers supply actions, not physics."""

from collections.abc import Callable
from dataclasses import dataclass

from game.actions import PlayerAction
from game.config import SimulationConfig
from game.player import PlayerId
from game.simulation import MatchPhase, SimulationEvent, SimulationState, reset, step_decision

ActionProvider = Callable[[SimulationState, PlayerId], PlayerAction]
_DEFAULT_CONFIG = SimulationConfig()
_IDLE_ACTION = PlayerAction()


def idle_provider(state: SimulationState, player: PlayerId) -> PlayerAction:
    """A no-op controller for inspection/tests, not a rule-based opponent."""
    return _IDLE_ACTION


@dataclass(frozen=True)
class MatchResult:
    """Only this invocation's events/ticks/decisions; score lives in final_state."""

    final_state: SimulationState
    events: tuple[SimulationEvent, ...]
    physics_ticks: int
    decision_count: int


def run_match(
    provider_a: ActionProvider = idle_provider,
    provider_b: ActionProvider = idle_provider,
    *,
    initial_state: SimulationState | None = None,
    config: SimulationConfig = _DEFAULT_CONFIG,
) -> MatchResult:
    """Run a fresh or resumed match to the simulation's final whistle.

    Query A then B once per decision, including during restarts, with the same
    immutable pre-step state. Each action must be a PlayerAction. Providers should
    keep any private script/RNG state independent; reset it before replaying.
    No provider is queried for an already-finished state.

    Decisions span up to six ticks starting at the supplied state's tick, even if
    it is not a multiple of six. Movement can resume within a restart-expiry block;
    a first-tick kick rejected during restart is not retried. The final block may
    be shorter. All event order/timestamps are preserved exactly from the engine.

    Callers supplying a state must use its matching config. There is no engine
    seed or random reset. Provider/physics errors propagate, not partial success.
    This runner does not pace wall time, render, compute rewards, or keep a replay
    of actions/states. Its full engine state is not a finalized RL observation.
    """
    if not isinstance(config, SimulationConfig):
        raise TypeError("Config must be a SimulationConfig.")
    if initial_state is not None and not isinstance(initial_state, SimulationState):
        raise TypeError("Initial state must be a SimulationState or None.")
    if not callable(provider_a) or not callable(provider_b):
        raise TypeError("Both action providers must be callable.")

    state = reset(config=config) if initial_state is None else initial_state
    events = []
    physics_ticks = 0
    decision_count = 0
    while state.phase != MatchPhase.FINISHED:
        action_a = provider_a(state, PlayerId.A)
        action_b = provider_b(state, PlayerId.B)
        for player, action in ((PlayerId.A, action_a), (PlayerId.B, action_b)):
            if not isinstance(action, PlayerAction):
                raise TypeError(f"Provider {player.value} must return a PlayerAction.")
        result = step_decision(state, action_a, action_b, config=config)
        state = result.state
        events.extend(result.events)
        physics_ticks += result.physics_ticks
        decision_count += 1

    return MatchResult(state, tuple(events), physics_ticks, decision_count)
