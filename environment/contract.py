"""Read-only projections of the real engine, not a second physics specification.

Player indices 0/1 mean stable identities A/B, independently of attacking side.
The authoritative state and complete event stream remain in game.simulation.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol, runtime_checkable

from game.actions import PlayerAction
from game.config import MATCH_TICKS, TICKS_PER_DECISION, SimulationConfig
from game.player import PlayerId
from game.simulation import MatchPhase, SimulationEvent, SimulationState

Phase = MatchPhase
PLAYER_IDS = (PlayerId.A, PlayerId.B)


def player_index(player: PlayerId | None) -> int | None:
    return None if player is None else PLAYER_IDS.index(player)


class EventType(Enum):
    GOAL = "goal"
    STEAL = "steal"
    LOOSE_BALL_RECOVERY = "loose_ball_recovery"
    OUT_OF_BOUNDS = "out_of_bounds"
    KICK = "kick"


@dataclass(frozen=True)
class Event:
    """Reward-relevant classification; id is unique within one adapter episode.

    tick keeps the original engine stamp. A kick uses its pre-integration tick;
    goals/acquisitions use the completed tick. id, NOT (type, tick, player),
    distinguishes multiple real transitions within a bounded multi-impact tick.
    """

    type: EventType
    tick: int
    player: int | None
    other: int | None
    event_id: int


@dataclass(frozen=True)
class PlayerState:
    x: float
    y: float
    vx: float
    vy: float
    facing: float
    stamina: float
    kick_cooldown_ticks: int = 0
    pickup_lockout_ticks: int = 0


@dataclass(frozen=True)
class BallState:
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True)
class MatchState:
    tick: int
    players: tuple[PlayerState, PlayerState]
    ball: BallState
    possession: int | None
    last_toucher: int | None
    score: tuple[int, int]
    attack_dirs: tuple[int, int]
    phase: Phase
    steal_progress_ticks: int
    possession_protection_ticks: int
    restart_ticks_remaining: int
    restart_owner: int | None
    config: SimulationConfig

    @property
    def ticks_remaining(self) -> int:
        return max(0, MATCH_TICKS - self.tick)

    @property
    def time_remaining_frac(self) -> float:
        return self.ticks_remaining / MATCH_TICKS

    @property
    def is_over(self) -> bool:
        return self.phase is Phase.FINISHED


def snapshot(state: SimulationState, config: SimulationConfig) -> MatchState:
    """Project, never integrate or mutate, a shared immutable simulation state."""
    players = tuple(
        PlayerState(
            *p.position,
            *p.velocity,
            p.facing,
            p.stamina,
            p.kick_cooldown_ticks,
            p.pickup_lockout_ticks,
        )
        for p in (state.player_a, state.player_b)
    )
    return MatchState(
        tick=state.simulation_tick,
        players=players,
        ball=BallState(*state.ball.position, *state.ball.velocity),
        possession=player_index(state.ball.possession_owner),
        last_toucher=player_index(state.ball.last_touch_player),
        score=(state.score_a, state.score_b),
        attack_dirs=(state.attack_direction_a, state.attack_direction_b),
        phase=state.phase,
        steal_progress_ticks=state.steal_progress_ticks,
        possession_protection_ticks=state.possession_protection_ticks,
        restart_ticks_remaining=state.restart_ticks_remaining,
        restart_owner=player_index(state.restart_plan.owner) if state.restart_plan else None,
        config=config,
    )


@dataclass(frozen=True)
class StepResult:
    state: MatchState
    events: tuple[Event, ...]
    ticks_advanced: int
    raw_events: tuple[SimulationEvent, ...] = ()


@runtime_checkable
class GameEngine(Protocol):
    config: SimulationConfig

    @property
    def native_state(self) -> SimulationState: ...

    def reset(
        self, *, swap_sides: bool = False, initial_state: SimulationState | None = None
    ) -> MatchState: ...

    def step(self, actions: tuple[PlayerAction, PlayerAction]) -> StepResult: ...


def validate_state(s: MatchState) -> None:
    if type(s.tick) is not int or not 0 <= s.tick <= MATCH_TICKS:
        raise ValueError("Tick must be an integer within the match duration.")
    if len(s.players) != 2 or s.attack_dirs not in ((1, -1), (-1, 1)):
        raise ValueError("Expected two players with opposing attack directions.")
    if not isinstance(s.phase, Phase) or s.is_over != (s.tick == MATCH_TICKS):
        raise ValueError("Finished phase must coincide with the final match tick.")
    pitch = s.config.pitch
    timers = [s.steal_progress_ticks, s.possession_protection_ticks, s.restart_ticks_remaining]
    for p in s.players:
        if not all(isfinite(v) for v in (p.x, p.y, p.vx, p.vy, p.facing, p.stamina)):
            raise ValueError("Nonfinite player state.")
        if not (0 <= p.x <= pitch.width and 0 <= p.y <= pitch.height):
            raise ValueError("Player outside pitch.")
        if not 0 <= p.stamina <= 100:
            raise ValueError("Stamina outside [0,100].")
        timers.extend((p.kick_cooldown_ticks, p.pickup_lockout_ticks))
    if not all(isfinite(v) for v in (s.ball.x, s.ball.y, s.ball.vx, s.ball.vy)):
        raise ValueError("Nonfinite ball state.")
    for owner in (s.possession, s.last_toucher, s.restart_owner):
        if owner is not None and (type(owner) is not int or owner not in (0, 1)):
            raise ValueError("Invalid player index.")
    if len(s.score) != 2 or any(type(v) is not int or v < 0 for v in s.score):
        raise ValueError("Invalid score.")
    if any(type(t) is not int or t < 0 for t in timers):
        raise ValueError("Timers must be nonnegative integer ticks.")
    if s.possession is None and (s.steal_progress_ticks or s.possession_protection_ticks):
        raise ValueError("Free ball cannot have contest/protection timers.")
    if s.possession is not None and (
        s.last_toucher != s.possession or s.ball.vx != 0 or s.ball.vy != 0
    ):
        raise ValueError("Controlled ball must have matching last touch and zero free velocity.")
    restarting = s.phase in (Phase.GOAL_RESTART, Phase.OUT_OF_BOUNDS_RESTART)
    if restarting != (s.restart_ticks_remaining > 0):
        raise ValueError("Restart phase and countdown disagree.")
    if not restarting and s.restart_owner is not None:
        raise ValueError("Restart owner outside restart phase.")


def validate_step_result(prev: MatchState, result: StepResult) -> None:
    """Keep original event order; stamps are not all on the same side of a tick.

    A tick's completion event at t+1 can precede the next tick's kick at t+1.
    We allow that tie, but only kicks may use the window's initial tick.
    """
    new = result.state
    validate_state(new)
    expected = min(TICKS_PER_DECISION, prev.ticks_remaining)
    if type(result.ticks_advanced) is not int or result.ticks_advanced != expected:
        raise ValueError("A decision must advance six ticks, or the remaining terminal ticks.")
    if new.tick != prev.tick + result.ticks_advanced or new.config != prev.config:
        raise ValueError("Inconsistent clock or config.")
    last_tick, ids = prev.tick, set()
    for e in result.events:
        valid_tick = (
            prev.tick <= e.tick < new.tick
            if e.type is EventType.KICK
            else prev.tick < e.tick <= new.tick
        )
        if not valid_tick or e.tick < last_tick:
            raise ValueError("Event outside its engine timestamp window or out of order.")
        last_tick = e.tick
        if e.player not in (0, 1) and not (e.type is EventType.OUT_OF_BOUNDS and e.player is None):
            raise ValueError("Invalid event player.")
        if e.other not in (None, 0, 1):
            raise ValueError("Invalid secondary player.")
        if type(e.event_id) is not int or e.event_id < 0 or e.event_id in ids:
            raise ValueError("Invalid or duplicate event id.")
        ids.add(e.event_id)
    for i in (0, 1):
        goals = sum(e.type is EventType.GOAL and e.player == i for e in result.events)
        if goals != new.score[i] - prev.score[i]:
            raise ValueError("Goal events disagree with score change.")
