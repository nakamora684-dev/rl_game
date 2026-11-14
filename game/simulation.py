"""Deterministic two-player stepping foundation, not yet a complete football game.

No scoring, kick processing, possession, player-player collisions, rewards,
or rendering. Restart phases only provide a clock/freeze scaffold for future rules.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from math import pi

from game.actions import MovementAction
from game.ball import BallState, step_free_ball
from game.config import MATCH_TICKS, PHYSICS_DT, TICKS_PER_DECISION, SimulationConfig
from game.player import PlayerState, constrain_to_pitch, step_player

_DEFAULT_CONFIG = SimulationConfig()
_IDLE_ACTION = MovementAction()


class MatchPhase(StrEnum):
    PLAYING = "playing"
    GOAL_RESTART = "goal_restart"
    OUT_OF_BOUNDS_RESTART = "out_of_bounds_restart"
    FINISHED = "finished"


@dataclass(frozen=True)
class SimulationState:
    player_a: PlayerState
    player_b: PlayerState
    ball: BallState
    simulation_tick: int = 0
    phase: MatchPhase = MatchPhase.PLAYING
    restart_ticks_remaining: int = 0
    attack_direction_a: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.phase, MatchPhase):
            raise TypeError("Phase must be a MatchPhase enum member.")
        if type(self.simulation_tick) is not int or not 0 <= self.simulation_tick <= MATCH_TICKS:
            raise ValueError("Simulation tick must be an integer within the match duration.")
        if type(self.restart_ticks_remaining) is not int or self.restart_ticks_remaining < 0:
            raise ValueError("Restart countdown must be a nonnegative integer.")
        restarting = self.phase in (MatchPhase.GOAL_RESTART, MatchPhase.OUT_OF_BOUNDS_RESTART)
        if restarting != (self.restart_ticks_remaining > 0):
            raise ValueError("A positive restart countdown is required exactly in restart phases.")
        if (self.phase == MatchPhase.FINISHED) != (self.simulation_tick == MATCH_TICKS):
            raise ValueError("Finished phase must coincide with the final match tick.")
        if type(self.attack_direction_a) is not int or self.attack_direction_a not in (-1, 1):
            raise ValueError("Attack direction must be -1 (left) or 1 (right).")

    @property
    def remaining_seconds(self) -> float:
        return (MATCH_TICKS - self.simulation_tick) * PHYSICS_DT

    @property
    def attack_direction_b(self) -> int:
        return -self.attack_direction_a


@dataclass(frozen=True)
class SimulationEvent:
    """One-shot terminal event; football transition events will be added later."""

    kind: str
    tick: int


@dataclass(frozen=True)
class StepResult:
    state: SimulationState
    events: tuple[SimulationEvent, ...] = ()
    physics_ticks: int = 0


def reset(
    *, config: SimulationConfig = _DEFAULT_CONFIG, swap_sides: bool = False
) -> SimulationState:
    """Create a fresh state; there is no randomness or hidden action history yet."""
    pitch = config.pitch
    left = constrain_to_pitch(PlayerState(position=(pitch.width * 0.25, pitch.height * 0.5)), pitch)
    right = constrain_to_pitch(
        PlayerState(position=(pitch.width * 0.75, pitch.height * 0.5), facing=pi), pitch
    )
    return SimulationState(
        player_a=right if swap_sides else left,
        player_b=left if swap_sides else right,
        ball=BallState(position=(pitch.width * 0.5, pitch.height * 0.5)),
        attack_direction_a=-1 if swap_sides else 1,
    )


def step_tick(
    state: SimulationState,
    action_a: MovementAction = _IDLE_ACTION,
    action_b: MovementAction = _IDLE_ACTION,
    *,
    config: SimulationConfig = _DEFAULT_CONFIG,
) -> StepResult:
    """Advance one physics tick; the clock runs even while restart motion is frozen.

    Both players read the same input snapshot. On the last restart tick they stay
    frozen; movement resumes on the following tick. Finished states are idempotent.
    """
    if state.phase == MatchPhase.FINISHED:
        return StepResult(state)

    player_a, player_b, ball = state.player_a, state.player_b, state.ball
    phase, countdown = state.phase, state.restart_ticks_remaining
    if phase == MatchPhase.PLAYING:
        player_a = step_player(
            player_a, action_a, movement=config.movement, stamina=config.stamina, pitch=config.pitch
        )
        player_b = step_player(
            player_b, action_b, movement=config.movement, stamina=config.stamina, pitch=config.pitch
        )
        ball = step_free_ball(ball, config=config.ball)
    else:
        countdown -= 1
        if countdown == 0:
            phase = MatchPhase.PLAYING

    tick = state.simulation_tick + 1
    events = ()
    if tick == MATCH_TICKS:
        phase, countdown = MatchPhase.FINISHED, 0
        events = (SimulationEvent(kind="match_finished", tick=tick),)

    return StepResult(
        replace(
            state,
            player_a=player_a,
            player_b=player_b,
            ball=ball,
            simulation_tick=tick,
            phase=phase,
            restart_ticks_remaining=countdown,
        ),
        events=events,
        physics_ticks=1,
    )


def step_decision(
    state: SimulationState,
    action_a: MovementAction = _IDLE_ACTION,
    action_b: MovementAction = _IDLE_ACTION,
    *,
    config: SimulationConfig = _DEFAULT_CONFIG,
) -> StepResult:
    """Hold movement for up to six distinct physics updates; stop at the whistle.

    Events are aggregated across all sub-ticks. Restart handling currently uses the
    same six-tick cadence; if a restart expires, held movement resumes within this
    block. Policy scheduling/observations across restarts remain C's interface work.
    """
    events = []
    ticks = 0
    for _ in range(TICKS_PER_DECISION):
        result = step_tick(state, action_a, action_b, config=config)
        state = result.state
        events.extend(result.events)
        ticks += result.physics_ticks
        if state.phase == MatchPhase.FINISHED:
            break
    return StepResult(state, events=tuple(events), physics_ticks=ticks)
