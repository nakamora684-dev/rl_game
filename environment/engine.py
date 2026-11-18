"""State-owning adapter around the unchanged, pure shared simulation."""

from environment.contract import Event, EventType, MatchState, StepResult, player_index, snapshot
from game.actions import PlayerAction
from game.config import SimulationConfig
from game.simulation import SimulationState, reset, step_decision


class SimulationEngine:
    """One instance per environment; no physics, reward, RNG, or restart fast-forward."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = SimulationConfig() if config is None else config
        self._state: SimulationState | None = None
        self._next_event_id = 0

    @property
    def native_state(self) -> SimulationState:
        if self._state is None:
            raise RuntimeError("Call reset() before reading or stepping the engine.")
        return self._state

    def reset(
        self, *, swap_sides: bool = False, initial_state: SimulationState | None = None
    ) -> MatchState:
        """An explicit resumed/test state must use this adapter's matching config.

        There is no engine seed: the current physics/reset contain no randomness.
        With initial_state, its existing orientation overrides swap_sides.
        """
        if initial_state is not None and not isinstance(initial_state, SimulationState):
            raise TypeError("initial_state must be SimulationState or None.")
        self._state = (
            reset(config=self.config, swap_sides=swap_sides)
            if initial_state is None
            else initial_state
        )
        self._next_event_id = 0
        return snapshot(self._state, self.config)

    def step(self, actions: tuple[PlayerAction, PlayerAction]) -> StepResult:
        if len(actions) != 2 or not all(isinstance(a, PlayerAction) for a in actions):
            raise TypeError("Supply two world-space PlayerAction objects, ordered A then B.")
        result = step_decision(self.native_state, *actions, config=self.config)
        self._state = result.state
        events = []
        for raw in result.events:
            event_id = self._next_event_id
            self._next_event_id += 1
            kind = None
            player = other = None
            if raw.kind == "goal":
                kind, player = EventType.GOAL, raw.scoring_player
            elif raw.kind == "out_of_bounds":
                kind, player = EventType.OUT_OF_BOUNDS, raw.last_touch_player
            elif raw.kind == "ball_kicked":
                kind, player = EventType.KICK, raw.previous_owner
            elif raw.kind == "possession_changed" and raw.cause == "recovery":
                kind, player = EventType.LOOSE_BALL_RECOVERY, raw.new_owner
            elif raw.kind == "possession_changed" and raw.cause == "steal":
                kind, player, other = EventType.STEAL, raw.new_owner, raw.previous_owner
            # Setup, restart acquisition, post loss and diagnostics are NOT recoveries.
            # All original events (including unmapped ones) remain in raw_events.
            if kind is not None:
                events.append(
                    Event(kind, raw.tick, player_index(player), player_index(other), event_id)
                )
        return StepResult(
            state=snapshot(result.state, self.config),
            events=tuple(events),
            ticks_advanced=result.physics_ticks,
            raw_events=result.events,
        )
