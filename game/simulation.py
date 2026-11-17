"""Deterministic shared football physics/rules; no rewards or rendering."""

from dataclasses import dataclass, replace
from enum import StrEnum
from math import cos, dist, isclose, isfinite, pi, sin

from game.actions import MovementAction, PlayerAction
from game.ball import BallState
from game.ball_contacts import BallFlight, advance_free_ball, first_post_contact
from game.collisions import resolve_players
from game.config import MATCH_TICKS, PHYSICS_DT, TICKS_PER_DECISION, SimulationConfig
from game.player import PlayerId, PlayerState, constrain_to_pitch, step_player
from game.possession import sync_controlled_ball
from game.rebounds import project_outside_circles
from game.rules import BallExit, BoundarySide, RestartPlan, detect_ball_exit, goalpost_centres

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
    steal_progress_ticks: int = 0
    possession_protection_ticks: int = 0
    score_a: int = 0
    score_b: int = 0
    restart_plan: RestartPlan | None = None

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
        for timer in (self.steal_progress_ticks, self.possession_protection_ticks):
            if type(timer) is not int or timer < 0:
                raise ValueError("Possession timers must be nonnegative integers.")
        if self.ball.possession_owner is None and (
            self.steal_progress_ticks or self.possession_protection_ticks
        ):
            raise ValueError("A free ball cannot have active possession contest/protection.")
        if any(type(score) is not int or score < 0 for score in (self.score_a, self.score_b)):
            raise ValueError("Scores must be nonnegative integers.")
        if self.restart_plan is not None:
            if not isinstance(self.restart_plan, RestartPlan):
                raise TypeError("Restart plan must be RestartPlan or None.")
            if not restarting:
                raise ValueError("A restart plan is only valid during a restart phase.")
            if self.restart_plan.exit.is_goal != (self.phase == MatchPhase.GOAL_RESTART):
                raise ValueError("Restart phase and exit classification must agree.")
            if self.phase == MatchPhase.GOAL_RESTART and self.restart_plan.owner is None:
                raise ValueError("A goal restart requires a conceding kickoff owner.")

    @property
    def remaining_seconds(self) -> float:
        return (MATCH_TICKS - self.simulation_tick) * PHYSICS_DT

    @property
    def attack_direction_b(self) -> int:
        return -self.attack_direction_a

    def has_possession(self, player: PlayerId) -> bool:
        """Derive possession from the ball; never store contradictory player flags."""
        if not isinstance(player, PlayerId):
            raise TypeError("Player must be a PlayerId enum member.")
        return self.ball.possession_owner == player


@dataclass(frozen=True)
class SimulationEvent:
    """One-shot transition data, not reward arithmetic.

    Possession changes carry previous/new owner and cause. A 'setup' assignment
    is not a gameplay recovery or steal and must not receive those rewards.
    A ball_kicked event includes its controlled-to-free transition; do not emit
    another possession_changed event for that same release.
    """

    kind: str
    tick: int
    previous_owner: PlayerId | None = None
    new_owner: PlayerId | None = None
    cause: str | None = None
    scoring_player: PlayerId | None = None
    last_touch_player: PlayerId | None = None
    ball_position: tuple[float, float] | None = None
    restart_owner: PlayerId | None = None
    tick_fraction: float | None = None
    collision_index: int | None = None
    contact_player: PlayerId | None = None
    post_index: int | None = None
    velocity_before: tuple[float, float] | None = None
    velocity_after: tuple[float, float] | None = None


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
    if dist(left.position, right.position) < 2 * pitch.player_radius:
        # Narrow pitches can fit two bodies vertically even when the usual
        # left/right quarter-pitch spawns overlap. Keep facing/attack conventions.
        left = constrain_to_pitch(
            replace(left, position=(pitch.width * 0.5, pitch.height * 0.25)), pitch
        )
        right = constrain_to_pitch(
            replace(right, position=(pitch.width * 0.5, pitch.height * 0.75)), pitch
        )
        if dist(left.position, right.position) < 2 * pitch.player_radius:
            left = replace(left, position=(pitch.player_radius, pitch.player_radius))
            right = replace(
                right,
                position=(pitch.width - pitch.player_radius, pitch.height - pitch.player_radius),
            )
            if dist(left.position, right.position) < 2 * pitch.player_radius:
                raise ValueError("Reset pitch cannot fit two non-overlapping player bodies.")
    try:
        left, right, _ = resolve_players(left, right, left, right, config=config)
    except ValueError as exc:
        raise ValueError("Reset layout cannot satisfy player/pitch/post constraints.") from exc
    return SimulationState(
        player_a=right if swap_sides else left,
        player_b=left if swap_sides else right,
        ball=BallState(position=(pitch.width * 0.5, pitch.height * 0.5)),
        attack_direction_a=-1 if swap_sides else 1,
    )


def assign_possession(
    state: SimulationState,
    owner: PlayerId,
    *,
    config: SimulationConfig = _DEFAULT_CONFIG,
) -> StepResult:
    """Explicit setup/test assignment, not distance-based pickup or stealing.

    Attach immediately, clear free velocity, record last touch, preserve both
    players/stamina and the clock. Repeating the owner emits no new transition.
    This helper is restricted to playing states; referee restart code is separate.
    """
    if not isinstance(owner, PlayerId):
        raise TypeError("Owner must be a PlayerId enum member.")
    if state.phase != MatchPhase.PLAYING:
        raise ValueError("Setup possession assignment requires a playing state.")
    previous = state.ball.possession_owner
    ball = replace(state.ball, possession_owner=owner, last_touch_player=owner, velocity=(0.0, 0.0))
    ball = sync_controlled_ball(ball, state.player_a, state.player_b, config=config.ball)
    if previous == owner:
        updated = state if ball == state.ball else replace(state, ball=ball)
    else:
        updated = replace(state, ball=ball, steal_progress_ticks=0, possession_protection_ticks=0)
    if previous == owner:
        return StepResult(updated)
    event = SimulationEvent(
        kind="possession_changed",
        tick=state.simulation_tick,
        previous_owner=previous,
        new_owner=owner,
        cause="setup",
        last_touch_player=owner,
        ball_position=ball.position,
    )
    return StepResult(updated, events=(event,))


def release_kick(
    state: SimulationState,
    player: PlayerId,
    *,
    launch_speed: float,
    config: SimulationConfig = _DEFAULT_CONFIG,
) -> StepResult:
    """Release an owned ball along current facing, without advancing time.

    This low-level transition is not the decision-action/request scheduler and
    does not enforce or start kick cooldown; gameplay callers use shared stepping.
    It does initialize the kicker-only pickup lockout to prevent self-recovery.
    Speed is explicit, finite and positive in world units/s; it replaces ball
    velocity without adding carrier momentum. This primitive takes no power or
    trigger; shared stepping handles the approved mapping and discards failed requests.
    Invalid direct calls raise programmer errors.
    The next ordinary physics tick integrates the newly free ball.
    """
    if not isinstance(player, PlayerId):
        raise TypeError("Kicker must be a PlayerId enum member.")
    if not isfinite(launch_speed) or launch_speed <= 0.0:
        raise ValueError("Launch speed must be finite and positive.")
    if state.phase != MatchPhase.PLAYING:
        raise ValueError("Kick release requires a playing state.")
    if state.ball.possession_owner != player:
        raise ValueError("Kicker must have possession of the ball.")
    carrier = state.player_a if player == PlayerId.A else state.player_b
    attached = sync_controlled_ball(state.ball, state.player_a, state.player_b, config=config.ball)
    ball = replace(
        attached,
        possession_owner=None,
        velocity=(launch_speed * cos(carrier.facing), launch_speed * sin(carrier.facing)),
        last_touch_player=player,
    )
    event = SimulationEvent(
        kind="ball_kicked",
        tick=state.simulation_tick,
        previous_owner=player,
        new_owner=None,
        cause="kick",
        last_touch_player=player,
        ball_position=ball.position,
    )
    carrier = replace(carrier, pickup_lockout_ticks=config.possession.kicker_pickup_lockout_ticks)
    updated = replace(
        state,
        ball=ball,
        steal_progress_ticks=0,
        possession_protection_ticks=0,
        **{"player_a" if player == PlayerId.A else "player_b": carrier},
    )
    return StepResult(updated, events=(event,))


def _begin_restart(
    state: SimulationState, crossing: BallExit, *, config: SimulationConfig
) -> StepResult:
    """Record one goal/exit, stop the ball, and freeze until a planned restart."""
    scorer = None
    score_a, score_b = state.score_a, state.score_b
    if crossing.is_goal:
        direction = 1 if crossing.side == BoundarySide.RIGHT else -1
        scorer = PlayerId.A if state.attack_direction_a == direction else PlayerId.B
        receiver = PlayerId.B if scorer == PlayerId.A else PlayerId.A
        score_a += scorer == PlayerId.A
        score_b += scorer == PlayerId.B
        phase, kind, cause = MatchPhase.GOAL_RESTART, "goal", "goal"
    else:
        toucher = state.ball.last_touch_player
        receiver = None if toucher is None else PlayerId.B if toucher == PlayerId.A else PlayerId.A
        phase, kind, cause = MatchPhase.OUT_OF_BOUNDS_RESTART, "out_of_bounds", "boundary"
    event = SimulationEvent(
        kind=kind,
        tick=state.simulation_tick + 1,
        previous_owner=state.ball.possession_owner,
        new_owner=None,
        cause=cause,
        scoring_player=scorer,
        last_touch_player=state.ball.last_touch_player,
        ball_position=crossing.position,
        restart_owner=receiver,
        tick_fraction=crossing.fraction,
    )
    return StepResult(
        replace(
            state,
            ball=BallState(crossing.position, last_touch_player=state.ball.last_touch_player),
            score_a=score_a,
            score_b=score_b,
            phase=phase,
            restart_ticks_remaining=config.rules.restart_ticks,
            restart_plan=RestartPlan(receiver, crossing),
            steal_progress_ticks=0,
            possession_protection_ticks=0,
        ),
        events=(event,),
    )


def _complete_restart(state: SimulationState, *, config: SimulationConfig) -> StepResult:
    """Reposition on the last frozen tick, preserving stamina, score, clock and sides."""
    plan = state.restart_plan
    if plan is None:
        raise ValueError("An actual football restart requires a restart plan.")
    fresh = reset(config=config, swap_sides=state.attack_direction_a == -1)
    player_a = replace(fresh.player_a, stamina=state.player_a.stamina)
    player_b = replace(fresh.player_b, stamina=state.player_b.stamina)
    ball, protection = fresh.ball, 0
    cause = "neutral_restart"
    if plan.owner is not None:
        carrier = player_a if plan.owner == PlayerId.A else player_b
        if state.phase == MatchPhase.GOAL_RESTART:
            carrier = constrain_to_pitch(
                replace(
                    carrier,
                    position=(
                        config.pitch.width / 2 - config.ball.dribble_offset * cos(carrier.facing),
                        config.pitch.height / 2 - config.ball.dribble_offset * sin(carrier.facing),
                    ),
                ),
                config.pitch,
            )
            cause = "kickoff"
        else:
            facing = {
                BoundarySide.LEFT: 0.0,
                BoundarySide.RIGHT: pi,
                BoundarySide.TOP: pi / 2,
                BoundarySide.BOTTOM: -pi / 2,
            }[plan.exit.side]
            carrier = constrain_to_pitch(
                replace(carrier, position=plan.exit.position, facing=facing), config.pitch
            )
            cause = "boundary_restart"
        if plan.owner == PlayerId.A:
            player_a = carrier
        else:
            player_b = carrier
        player_a, player_b, _ = resolve_players(
            player_a, player_b, player_a, player_b, config=config
        )
        ball = BallState(
            carrier.position, possession_owner=plan.owner, last_touch_player=plan.owner
        )
        ball = sync_controlled_ball(ball, player_a, player_b, config=config.ball)
        protection = config.possession.protection_ticks
    event = SimulationEvent(
        kind="possession_changed" if plan.owner is not None else "restart_completed",
        tick=state.simulation_tick + 1,
        previous_owner=None,
        new_owner=plan.owner,
        cause=cause,
        last_touch_player=ball.last_touch_player,
        ball_position=ball.position,
        restart_owner=plan.owner,
    )
    return StepResult(
        replace(
            state,
            player_a=player_a,
            player_b=player_b,
            ball=ball,
            phase=MatchPhase.PLAYING,
            restart_ticks_remaining=0,
            restart_plan=None,
            steal_progress_ticks=0,
            possession_protection_ticks=protection,
        ),
        events=(event,),
    )


def _collision_events(
    flight: BallFlight, tick: int, offset: int = 0
) -> tuple[SimulationEvent, ...]:
    events = [
        SimulationEvent(
            kind="ball_collision",
            tick=tick,
            cause="player_contact" if impact.player else "goalpost",
            last_touch_player=impact.last_touch_player,
            ball_position=impact.position,
            tick_fraction=impact.fraction,
            collision_index=offset + index,
            contact_player=impact.player,
            post_index=impact.post_index,
            velocity_before=impact.before,
            velocity_after=impact.after,
        )
        for index, impact in enumerate(flight.impacts)
    ]
    if flight.limited:
        events.append(
            SimulationEvent(
                kind="physics_limit",
                tick=tick,
                cause="impact_budget",
                tick_fraction=1.0,
                ball_position=flight.ball.position,
                last_touch_player=flight.ball.last_touch_player,
            )
        )
    if flight.constraint is not None:
        events.append(
            SimulationEvent(
                kind="ball_constraint",
                tick=tick,
                cause="opposed_contacts",
                tick_fraction=1.0,
                ball_position=flight.constraint.position,
                last_touch_player=flight.ball.last_touch_player,
                velocity_before=flight.constraint.before,
                velocity_after=flight.constraint.after,
            )
        )
    return tuple(events)


def _post_before_exit(
    start: tuple[float, float], end: tuple[float, float], config: SimulationConfig
) -> float | None:
    fraction = first_post_contact(start, end, config=config)
    crossing = detect_ball_exit(start, end, config=config)
    if fraction is None or (
        crossing is not None
        and crossing.fraction < fraction
        and not isclose(crossing.fraction, fraction, rel_tol=0.0, abs_tol=1e-12)
    ):
        return None
    return fraction


def _release_on_post(
    state: SimulationState,
    ball: BallState,
    player_a: PlayerState,
    player_b: PlayerState,
    point: tuple[float, float],
    velocity: tuple[float, float],
    fraction: float,
    *,
    config: SimulationConfig,
) -> StepResult:
    """Contact loss, not a kick: preserve kick cooldown and start pickup lockout."""
    owner = ball.possession_owner
    carrier = player_a if owner == PlayerId.A else player_b
    carrier = replace(carrier, pickup_lockout_ticks=config.possession.kicker_pickup_lockout_ticks)
    if owner == PlayerId.A:
        player_a = carrier
    else:
        player_b = carrier
    free = BallState(point, velocity, last_touch_player=owner)
    event = SimulationEvent(
        kind="possession_changed",
        tick=state.simulation_tick + 1,
        cause="post_collision",
        previous_owner=owner,
        new_owner=None,
        last_touch_player=owner,
        ball_position=point,
        tick_fraction=fraction,
    )
    return StepResult(
        replace(
            state,
            player_a=player_a,
            player_b=player_b,
            ball=free,
            steal_progress_ticks=0,
            possession_protection_ticks=0,
        ),
        events=(event,),
    )


def step_tick(
    state: SimulationState,
    action_a: MovementAction = _IDLE_ACTION,
    action_b: MovementAction = _IDLE_ACTION,
    *,
    config: SimulationConfig = _DEFAULT_CONFIG,
) -> StepResult:
    """Advance one physics tick; the clock runs even while restart motion is frozen.

    Both players read the same input snapshot. Restart placement occurs on the
    last frozen tick; normal motion resumes next tick. Finished states are idempotent.
    A PlayerAction trigger is one fresh request on this tick, after movement/facing
    and before ball integration. Unavailable requests are discarded. Use
    step_decision to consume a trigger once while holding movement for six ticks.
    """
    if state.phase == MatchPhase.FINISHED:
        return StepResult(state)

    player_a, player_b, ball = state.player_a, state.player_b, state.ball
    phase, countdown = state.phase, state.restart_ticks_remaining
    progress, protection = state.steal_progress_ticks, state.possession_protection_ticks
    plan = state.restart_plan
    pending_exit = None
    acquired = False
    events = []
    if phase == MatchPhase.PLAYING:
        # Repair injected initial overlaps before constructing movement sweeps.
        # A positional correction is not a moving bat that can launch the ball.
        motion_start_a, motion_start_b, initial_limited = resolve_players(
            player_a, player_b, player_a, player_b, config=config
        )
        pickup_start_a, pickup_start_b = motion_start_a, motion_start_b
        player_a = step_player(
            motion_start_a,
            action_a,
            movement=config.movement,
            stamina=config.stamina,
            pitch=config.pitch,
        )
        player_b = step_player(
            motion_start_b,
            action_b,
            movement=config.movement,
            stamina=config.stamina,
            pitch=config.pitch,
        )
        player_a, player_b, players_limited = resolve_players(
            motion_start_a, motion_start_b, player_a, player_b, config=config
        )
        # Snapshot sweeps include geometric correction; impulses must not turn
        # that correction into velocity/energy. Use resolved physical velocities.
        surface_velocities = (player_a.velocity, player_b.velocity)
        if initial_limited or players_limited:
            events.append(
                SimulationEvent(
                    kind="physics_limit", tick=state.simulation_tick + 1, cause="player_post_budget"
                )
            )
        owner = ball.possession_owner
        if owner is not None:
            action = action_a if owner == PlayerId.A else action_b
            carrier = player_a if owner == PlayerId.A else player_b
            if (
                isinstance(action, PlayerAction)
                and action.kick_trigger
                and carrier.kick_cooldown_ticks == 0
            ):
                release = release_kick(
                    replace(state, player_a=player_a, player_b=player_b),
                    owner,
                    launch_speed=config.kick.speed_for_power(action.kick_power),
                    config=config,
                )
                ball = release.state.ball
                progress, protection = 0, 0
                events.extend(release.events)
                player_a, player_b = release.state.player_a, release.state.player_b
                carrier = player_a if owner == PlayerId.A else player_b
                carrier = replace(carrier, kick_cooldown_ticks=config.kick.cooldown_ticks)
                if owner == PlayerId.A:
                    player_a = carrier
                else:
                    player_b = carrier
                # Release happens AFTER player motion, so do not sweep the shot
                # backwards through the players' pre-release movement paths.
                pickup_start_a, pickup_start_b = player_a, player_b
                surface_velocities = ((0.0, 0.0), (0.0, 0.0))

        def record_flight(flight: BallFlight):
            """Apply a contact trace and at most one acquisition; no time advancement."""
            nonlocal ball, progress, protection, acquired, pending_exit
            ball = flight.ball
            offset = sum(event.kind == "ball_collision" for event in events)
            events.extend(_collision_events(flight, state.simulation_tick + 1, offset))
            pending_exit = flight.crossing
            if flight.pickup is not None:
                recovered_by = flight.pickup.owner
                contact_position = ball.position
                incoming_velocity = ball.velocity
                ball = replace(
                    ball,
                    possession_owner=recovered_by,
                    last_touch_player=recovered_by,
                    velocity=(0.0, 0.0),
                )
                ball = sync_controlled_ball(ball, player_a, player_b, config=config.ball)
                progress, protection, acquired = 0, config.possession.protection_ticks, True
                events.append(
                    SimulationEvent(
                        kind="possession_changed",
                        tick=state.simulation_tick + 1,
                        previous_owner=None,
                        new_owner=recovered_by,
                        cause="recovery",
                        last_touch_player=recovered_by,
                        ball_position=ball.position,
                        tick_fraction=flight.pickup.fraction,
                    )
                )
                pending_exit = detect_ball_exit(contact_position, ball.position, config=config)
                if pending_exit is not None:
                    pending_exit = replace(pending_exit, fraction=flight.pickup.fraction)
                return (
                    contact_position,
                    ball.position,
                    incoming_velocity,
                    flight.pickup.fraction,
                    True,
                )
            return None

        if ball.possession_owner is None:
            control_path = record_flight(
                advance_free_ball(
                    ball,
                    pickup_start_a,
                    pickup_start_b,
                    player_a,
                    player_b,
                    config=config,
                    surface_velocities=surface_velocities,
                )
            )
        else:
            attached_start = ball.position
            ball = sync_controlled_ball(ball, player_a, player_b, config=config.ball)
            carrier = player_a if ball.possession_owner == PlayerId.A else player_b
            previous_carrier = (
                state.player_a if ball.possession_owner == PlayerId.A else state.player_b
            )
            # Physical translation plus facing-offset motion, excluding body/post
            # depenetration and other positional correction from launch energy.
            control_velocity = tuple(
                v
                + config.ball.dribble_offset
                * (trig(carrier.facing) - trig(previous_carrier.facing))
                / PHYSICS_DT
                for v, trig in zip(carrier.velocity, (cos, sin))
            )
            control_path = attached_start, ball.position, control_velocity, 0.0, False
            pending_exit = detect_ball_exit(attached_start, ball.position, config=config)
            if (
                pending_exit is None
                and _post_before_exit(attached_start, ball.position, config) is None
                and protection == 0
                and dist(player_a.position, player_b.position) <= config.possession.steal_distance
            ):
                progress += 1
                if progress >= config.possession.steal_contest_ticks:
                    previous = ball.possession_owner
                    defender = PlayerId.B if previous == PlayerId.A else PlayerId.A
                    before_transfer = ball.position
                    ball = replace(ball, possession_owner=defender, last_touch_player=defender)
                    ball = sync_controlled_ball(ball, player_a, player_b, config=config.ball)
                    progress, protection, acquired = 0, config.possession.protection_ticks, True
                    events.append(
                        SimulationEvent(
                            kind="possession_changed",
                            tick=state.simulation_tick + 1,
                            previous_owner=previous,
                            new_owner=defender,
                            cause="steal",
                            last_touch_player=defender,
                            ball_position=ball.position,
                        )
                    )
                    pending_exit = detect_ball_exit(before_transfer, ball.position, config=config)
                    if pending_exit is not None:
                        pending_exit = replace(pending_exit, fraction=1.0)
                    control_path = before_transfer, ball.position, control_velocity, 1.0, True
            else:
                progress = 0
        # At most the two available players can normally acquire/lose control in
        # one tick. A configurable bounded loop also covers zero-lockout fixtures.
        for _ in range(config.ball.max_impacts_per_tick):
            if control_path is None or ball.possession_owner is None:
                break
            start, end, incoming_velocity, base_fraction, instantaneous = control_path
            contact_fraction = _post_before_exit(start, end, config)
            if contact_fraction is None:
                break
            point = tuple(a + contact_fraction * (b - a) for a, b in zip(start, end))
            fraction = (
                base_fraction
                if instantaneous
                else base_fraction + (1 - base_fraction) * contact_fraction
            )
            released = _release_on_post(
                state, ball, player_a, player_b, point, incoming_velocity, fraction, config=config
            )
            ball, player_a, player_b = (
                released.state.ball,
                released.state.player_a,
                released.state.player_b,
            )
            progress, protection, acquired = 0, 0, False
            events.extend(released.events)
            used = sum(event.kind == "ball_collision" for event in events)
            control_path = record_flight(
                advance_free_ball(
                    ball,
                    pickup_start_a,
                    pickup_start_b,
                    player_a,
                    player_b,
                    config=config,
                    start_fraction=fraction,
                    impact_budget=max(0, config.ball.max_impacts_per_tick - used),
                    surface_velocities=surface_velocities,
                )
            )
        if (
            control_path is not None
            and ball.possession_owner is not None
            and _post_before_exit(control_path[0], control_path[1], config) is not None
        ):
            # Do not derive launch energy from an instantaneous attachment jump,
            # or allow a zero-lockout setup to loop without advancing the clock.
            circles = tuple(
                (c, config.ball.radius + config.rules.post_radius) for c in goalpost_centres(config)
            )
            circles += tuple(
                (p.position, config.ball.radius + config.pitch.player_radius)
                for p in (player_a, player_b)
            )
            point = project_outside_circles(ball.position, circles)
            pending_exit = detect_ball_exit(ball.position, point, config=config)
            if pending_exit is not None:
                pending_exit = replace(pending_exit, fraction=1.0)
            ball = BallState(point, last_touch_player=ball.last_touch_player)
            progress, protection, acquired = 0, 0, False
            events.append(
                SimulationEvent(
                    kind="physics_limit",
                    tick=state.simulation_tick + 1,
                    cause="attachment_contact_budget",
                    ball_position=point,
                )
            )
    else:
        progress = 0  # A restart interrupts the consecutive playing-tick contest.
        countdown -= 1
        if countdown == 0:
            phase = MatchPhase.PLAYING
            if plan is not None:
                restarted = _complete_restart(state, config=config)
                player_a, player_b, ball = (
                    restarted.state.player_a,
                    restarted.state.player_b,
                    restarted.state.ball,
                )
                protection = restarted.state.possession_protection_ticks
                acquired = ball.possession_owner is not None
                events.extend(restarted.events)
                plan = None

    # Count elapsed simulation time, including the release tick and restart ticks.
    # A kick at tick t is eligible again at the start of tick t + cooldown_ticks.
    if player_a.kick_cooldown_ticks:
        player_a = replace(player_a, kick_cooldown_ticks=player_a.kick_cooldown_ticks - 1)
    if player_b.kick_cooldown_ticks:
        player_b = replace(player_b, kick_cooldown_ticks=player_b.kick_cooldown_ticks - 1)
    if player_a.pickup_lockout_ticks:
        player_a = replace(player_a, pickup_lockout_ticks=player_a.pickup_lockout_ticks - 1)
    if player_b.pickup_lockout_ticks:
        player_b = replace(player_b, pickup_lockout_ticks=player_b.pickup_lockout_ticks - 1)
    if not acquired:
        protection = max(0, protection - 1)
    updated = replace(
        state,
        player_a=player_a,
        player_b=player_b,
        ball=ball,
        phase=phase,
        restart_ticks_remaining=countdown,
        restart_plan=plan,
        steal_progress_ticks=progress,
        possession_protection_ticks=protection,
    )
    if pending_exit is not None:
        restarted = _begin_restart(updated, pending_exit, config=config)
        updated = restarted.state
        events.extend(restarted.events)
    tick = state.simulation_tick + 1
    if tick == MATCH_TICKS:
        updated = replace(
            updated,
            simulation_tick=tick,
            phase=MatchPhase.FINISHED,
            restart_ticks_remaining=0,
            restart_plan=None,
        )
        events.append(SimulationEvent(kind="match_finished", tick=tick))
    else:
        updated = replace(updated, simulation_tick=tick)

    return StepResult(
        updated,
        events=tuple(events),
        physics_ticks=1,
    )


def step_decision(
    state: SimulationState,
    action_a: MovementAction = _IDLE_ACTION,
    action_b: MovementAction = _IDLE_ACTION,
    *,
    config: SimulationConfig = _DEFAULT_CONFIG,
) -> StepResult:
    """Hold movement for up to six ticks, attempting kicks only on the first.

    Events are aggregated across all sub-ticks. Restart handling currently uses the
    same six-tick cadence; if a restart expires, held movement resumes within this
    block but the kick request has already expired. Each new call is a new decision
    and may request a new kick, even if the same immutable action object is reused.
    """
    events = []
    ticks = 0
    held_a = MovementAction(action_a.direction, action_a.effort)
    held_b = MovementAction(action_b.direction, action_b.effort)
    for sub_tick in range(TICKS_PER_DECISION):
        result = step_tick(
            state,
            action_a if sub_tick == 0 else held_a,
            action_b if sub_tick == 0 else held_b,
            config=config,
        )
        state = result.state
        events.extend(result.events)
        ticks += result.physics_ticks
        if state.phase == MatchPhase.FINISHED:
            break
    return StepResult(state, events=tuple(events), physics_ticks=ticks)
