"""Bridge between Member A's engine (game/) and the Member B code (agents/, rendering/).

Member A owns the physics; this file only TRANSLATES, it never changes a rule:

    game.simulation.SimulationState  --to_view-->  agents.interface.GameState   (read only)
    agents.interface.Action          --to_player_action-->  game.actions.PlayerAction

Vocabulary map (A  ->  B view):
    player_a / player_b  (PlayerId.A / .B)      -> players[0] / players[1]
    position (x, y), velocity (vx, vy) [units/s] -> x, y, vx, vy
    ball.possession_owner                        -> possession (0 / 1 / None)
    score_a / score_b                            -> score (a, b)
    simulation_tick                              -> tick
    attack_direction_a / _b                      -> attack_dirs (a, b)
    MatchPhase.PLAYING / *_RESTART / FINISHED    -> "playing" / "restart" / "finished"
    steal_progress_ticks, possession_protection_ticks, restart_ticks_remaining -> same names
    restart_plan.owner                           -> restart_owner;  ball.last_touch_player -> last_toucher
    (contest_active and restart_timer stay available as derived properties)
    SimulationEvent kinds                        -> goal, out_of_bounds, kick, steal, recover
        (possession_changed + cause "steal" / "recovery"; ball_kicked; goal; out_of_bounds)

Actions: A's engine takes 9 discrete directions + effort + (kick_trigger, kick_power).
The B-side 5 continuous controls are quantised to the nearest of the 8 compass
directions (idle below DEADZONE); B's effort 0..1 maps to A's 0.5..1 (jog..sprint). Kicks follow A's contract: a request is consumed
on the FIRST physics tick of a 6-tick decision. A fresh key press that arrives in the
middle of a decision is latched until the next decision boundary (<= 0.08 s later);
a kick held across ticks is NOT repeated (only the rising edge counts).
"""

from __future__ import annotations

from math import atan2, hypot, pi

from agents import interface as view
from game.actions import Direction, PlayerAction
from game.config import TICKS_PER_DECISION, SimulationConfig, StaminaConfig
from game.player import PlayerId
from game.simulation import MatchPhase, SimulationEvent, SimulationState, reset, step_tick

DEADZONE: float = 0.25  # |(move_x, move_y)| below this = idle
# A's `effort` is the SPEED command (target speed = effort * max_speed; effort 0 = braking)
# and 0.5 is the stamina-neutral effort. B's Action.effort is "sprint on top of jogging":
#   0 -> jog at the neutral effort (no stamina drain), 1 -> full sprint (drains stamina).
JOG_EFFORT: float = StaminaConfig().neutral_effort

# sector k (0..7) = round(angle / 45 deg), angle = atan2(y, x) in the y-down world
_SECTOR_TO_DIRECTION = (
    Direction.EAST,
    Direction.SOUTHEAST,
    Direction.SOUTH,
    Direction.SOUTHWEST,
    Direction.WEST,
    Direction.NORTHWEST,
    Direction.NORTH,
    Direction.NORTHEAST,
)


def index_of(player: PlayerId | None) -> int | None:
    return None if player is None else (0 if player == PlayerId.A else 1)


def direction_from_vector(move_x: float, move_y: float) -> Direction:
    if hypot(move_x, move_y) < DEADZONE:
        return Direction.IDLE
    sector = round(atan2(move_y, move_x) / (pi / 4)) % 8
    return _SECTOR_TO_DIRECTION[sector]


def to_player_action(
    action: view.Action, kick_trigger: int | None = None, kick_power: float | None = None
) -> PlayerAction:
    """Decision-level conversion: `kick >= 0.5` is one kick request (EngineAdapter overrides it)."""
    a = action.clipped()
    if kick_trigger is None:
        kick_trigger = 1 if a.kick >= 0.5 else 0
    direction = direction_from_vector(a.move_x, a.move_y)
    effort = 0.0 if direction == Direction.IDLE else JOG_EFFORT + (1.0 - JOG_EFFORT) * a.effort
    return PlayerAction(
        direction=direction,
        effort=effort,
        kick_trigger=kick_trigger,
        kick_power=a.power if kick_power is None else kick_power,
    )


def convert_events(events: tuple[SimulationEvent, ...] | list[SimulationEvent]) -> list[view.Event]:
    """Keep only what the renderer / metrics need; every one is emitted once by A."""
    out: list[view.Event] = []
    for e in events:
        if e.kind == "goal":
            out.append(view.Event(view.EV_GOAL, index_of(e.scoring_player), e.tick))
        elif e.kind == "out_of_bounds":
            out.append(view.Event(view.EV_OUT, index_of(e.last_touch_player), e.tick))
        elif e.kind == "ball_kicked":
            out.append(view.Event(view.EV_KICK, index_of(e.previous_owner), e.tick))
        elif e.kind == "possession_changed" and e.cause == "steal":
            out.append(view.Event(view.EV_STEAL, index_of(e.new_owner), e.tick))
        elif e.kind == "possession_changed" and e.cause == "recovery":
            out.append(view.Event(view.EV_RECOVER, index_of(e.new_owner), e.tick))
    return out


def _player(p) -> view.PlayerState:
    return view.PlayerState(
        p.position[0], p.position[1], p.velocity[0], p.velocity[1], p.facing, p.stamina
    )


def to_view(state: SimulationState, events: list[view.Event] | None = None) -> view.GameState:
    if state.phase == MatchPhase.FINISHED:
        phase = view.PHASE_FINISHED
    elif state.phase == MatchPhase.PLAYING:
        phase = view.PHASE_PLAYING
    else:
        phase = view.PHASE_RESTART
    plan = state.restart_plan
    return view.GameState(
        players=[_player(state.player_a), _player(state.player_b)],
        ball=view.BallState(
            state.ball.position[0],
            state.ball.position[1],
            state.ball.velocity[0],
            state.ball.velocity[1],
        ),
        possession=index_of(state.ball.possession_owner),
        score=(state.score_a, state.score_b),
        tick=state.simulation_tick,
        phase=phase,
        attack_dirs=(state.attack_direction_a, state.attack_direction_b),
        last_toucher=index_of(state.ball.last_touch_player),
        steal_progress_ticks=state.steal_progress_ticks,
        possession_protection_ticks=state.possession_protection_ticks,
        restart_ticks_remaining=state.restart_ticks_remaining,
        restart_owner=index_of(plan.owner) if plan is not None else None,
        events=list(events or []),
    )


class EngineAdapter:
    """`agents.interface.GameEngine` implemented on top of game.simulation.

    One `step()` = ONE physics tick (1/60 s), exactly like the plan; the returned
    GameState carries the events of that tick. `raw` is A's immutable state.
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.raw: SimulationState = reset(config=self.config)
        self._was_kicking = [False, False]
        self._latched: list[float | None] = [None, None]

    def reset(self, seed: int | None = None, swap_sides: bool = False) -> view.GameState:
        # A's engine has no randomness: `seed` is accepted for the shared interface only.
        self.raw = reset(config=self.config, swap_sides=swap_sides)
        self._was_kicking = [False, False]
        self._latched = [None, None]
        return to_view(self.raw)

    def _kick_requests(self, actions: list[view.Action]) -> list[tuple[int, float | None]]:
        at_boundary = self.raw.simulation_tick % TICKS_PER_DECISION == 0
        out = []
        for i, act in enumerate(actions):
            a = act.clipped()
            wants = a.kick >= 0.5
            # rising edge only; a request on a decision boundary is a fresh decision (so two
            # consecutive 10 Hz decisions that both kick are both honoured, like step_decision)
            if wants and (not self._was_kicking[i] or at_boundary):
                self._latched[i] = a.power
            self._was_kicking[i] = wants
            if at_boundary and self._latched[i] is not None:
                out.append((1, self._latched[i]))
                self._latched[i] = None
            else:
                out.append((0, None))
        return out

    def step(self, actions: list[view.Action]) -> view.GameState:
        if len(actions) != 2:
            raise ValueError("Exactly one Action per player is required.")
        kicks = self._kick_requests(actions)
        pa = [to_player_action(a, k, p) for a, (k, p) in zip(actions, kicks)]
        result = step_tick(self.raw, pa[0], pa[1], config=self.config)
        self.raw = result.state
        return to_view(self.raw, convert_events(result.events))
