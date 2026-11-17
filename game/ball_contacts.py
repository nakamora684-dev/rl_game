"""Bounded, ordered free-ball contacts within one fixed physics tick.

Each remaining flight segment uses analytic roll endpoints and a linear snapshot
sweep. This is deterministic multi-contact processing, not exact accelerated TOI.
"""

from dataclasses import dataclass, replace
from math import dist, isclose, isfinite

from game.ball import BallState, integrate_free_ball
from game.config import PHYSICS_DT, SimulationConfig
from game.geometry import swept_circle_contact
from game.player import PlayerId, PlayerState
from game.possession import PickupContact, find_pickup_contact
from game.rebounds import circle_normal, project_outside_circles, reflect_velocity
from game.rules import BallExit, detect_ball_exit, goalpost_centres

Point = tuple[float, float]


def _lerp(start: Point, end: Point, fraction: float) -> Point:
    return tuple(a + fraction * (b - a) for a, b in zip(start, end))


@dataclass(frozen=True)
class BallImpact:
    fraction: float
    position: Point
    before: Point
    after: Point
    player: PlayerId | None = None
    post_index: int | None = None
    last_touch_player: PlayerId | None = None


@dataclass(frozen=True)
class BallConstraint:
    position: Point
    before: Point
    after: Point


@dataclass(frozen=True)
class BallFlight:
    ball: BallState
    impacts: tuple[BallImpact, ...] = ()
    pickup: PickupContact | None = None
    crossing: BallExit | None = None
    limited: bool = False
    constraint: BallConstraint | None = None


@dataclass(frozen=True)
class _Solid:
    fraction: float
    centre: Point
    velocity: Point
    radius: float
    restitution: float
    player: PlayerId | None = None
    post_index: int | None = None
    eligible: bool = False


def first_post_contact(start: Point, end: Point, *, config: SimulationConfig) -> float | None:
    """First genuine attachment/post impact, excluding separating/tangent touch."""
    velocity = tuple((b - a) / PHYSICS_DT for a, b in zip(start, end))
    fractions = []
    radius = config.ball.radius + config.rules.post_radius
    for centre in goalpost_centres(config):
        contact = swept_circle_contact(start, end, centre, centre, radius)
        if contact is None:
            continue
        fraction, _ = contact
        normal = circle_normal(_lerp(start, end, fraction), centre)
        if (
            dist(start, centre) < radius - 1e-9
            or sum(v * n for v, n in zip(velocity, normal)) < -1e-10
        ):
            fractions.append(fraction)
    return min(fractions) if fractions else None


def advance_free_ball(
    ball: BallState,
    previous_a: PlayerState,
    previous_b: PlayerState,
    player_a: PlayerState,
    player_b: PlayerState,
    *,
    config: SimulationConfig,
    start_fraction: float = 0.0,
    impact_budget: int | None = None,
    surface_velocities: tuple[Point, Point] | None = None,
) -> BallFlight:
    """Advance until tick end, recovery, or whole-ball exit; rebound and continue.

    Posts and pickup-locked players are solid. Eligible player contact traps the
    ball first; an exact contested pickup remains free and resolves solid contact.
    Snapshots supply contact geometry, not impulse energy: surface velocity defaults
    to each updated player's physical velocity, excluding overlap correction. A
    freshly kicked ball uses stationary release snapshots/zero surface velocities.
    """
    if ball.possession_owner is not None:
        raise ValueError("Free-ball contact processing requires a free ball.")
    if not 0.0 <= start_fraction <= 1.0:
        raise ValueError("Start fraction must be within [0, 1].")
    posts = goalpost_centres(config)
    budget = config.ball.max_impacts_per_tick if impact_budget is None else impact_budget
    if type(budget) is not int or budget < 0:
        raise ValueError("Remaining impact budget must be a nonnegative integer.")
    if surface_velocities is None:
        surface_velocities = (player_a.velocity, player_b.velocity)
    if len(surface_velocities) != 2 or any(
        len(vector) != 2 or not all(isfinite(value) for value in vector)
        for vector in surface_velocities
    ):
        raise ValueError("Surface velocities require two finite 2D vectors.")
    elapsed = start_fraction
    impacts = []
    recent = []
    recent_fraction = -1.0
    while True:
        remaining = (1.0 - elapsed) * PHYSICS_DT
        end = integrate_free_ball(ball, remaining, config=config.ball)
        starts = tuple(
            replace(after, position=_lerp(before.position, after.position, elapsed))
            for before, after in ((previous_a, player_a), (previous_b, player_b))
        )
        pickup = find_pickup_contact(ball, end, *starts, player_a, player_b, config=config)
        crossing = detect_ball_exit(ball.position, end.position, config=config)
        solids = []
        circles = []
        for index, centre in enumerate(posts):
            solids.append(
                (
                    centre,
                    centre,
                    (0.0, 0.0),
                    config.ball.radius + config.rules.post_radius,
                    config.ball.post_restitution,
                    None,
                    index,
                    False,
                )
            )
        for identity, current_start, after, velocity in (
            (PlayerId.A, starts[0], player_a, surface_velocities[0]),
            (PlayerId.B, starts[1], player_b, surface_velocities[1]),
        ):
            solids.append(
                (
                    current_start.position,
                    after.position,
                    velocity,
                    config.pitch.player_radius + config.ball.radius,
                    config.ball.player_restitution,
                    identity,
                    None,
                    after.pickup_lockout_ticks == 0,
                )
            )
        candidates = []
        for start, finish, velocity, radius, restitution, identity, post, eligible in solids:
            circles.append((finish, radius))
            contact = swept_circle_contact(ball.position, end.position, start, finish, radius)
            if contact is None:
                continue
            fraction, _ = contact
            centre = _lerp(start, finish, fraction)
            point = _lerp(ball.position, end.position, fraction)
            incoming = integrate_free_ball(ball, remaining * fraction, config=config.ball).velocity
            normal = circle_normal(
                point, centre, circle_normal(tuple(-v for v in incoming), (0, 0))
            )
            approach = sum((v - surface) * n for v, surface, n in zip(incoming, velocity, normal))
            if dist(ball.position, start) >= radius - 1e-9 and approach >= -1e-10:
                continue
            candidates.append(
                _Solid(fraction, centre, velocity, radius, restitution, identity, post, eligible)
            )
        choices = [(c.fraction, 0 if c.post_index is not None else 3, c) for c in candidates]
        if pickup is not None:
            choices.append((pickup.fraction, 2, pickup))
        if crossing is not None:
            choices.append((crossing.fraction, 4, crossing))
        if not choices:
            return BallFlight(end, tuple(impacts))
        first = min(choice[0] for choice in choices)
        tied = [
            choice for choice in choices if isclose(choice[0], first, rel_tol=0.0, abs_tol=1e-12)
        ]
        _, _, winner = min(tied, key=lambda choice: choice[1])
        point = _lerp(ball.position, end.position, first)
        contact_ball = replace(
            integrate_free_ball(ball, remaining * first, config=config.ball), position=point
        )
        elapsed = min(1.0, elapsed + (1.0 - elapsed) * first)
        if isinstance(winner, PickupContact):
            return BallFlight(
                contact_ball, tuple(impacts), pickup=replace(winner, fraction=elapsed)
            )
        if isinstance(winner, BallExit):
            return BallFlight(
                contact_ball, tuple(impacts), crossing=replace(winner, fraction=elapsed)
            )
        group = [
            candidate
            for candidate in candidates
            if isclose(candidate.fraction, first, rel_tol=0.0, abs_tol=1e-12)
            and (not candidate.eligible or pickup is None or candidate is winner)
        ]
        if len(impacts) + len(group) > budget:
            # Defensive saturation: expose it, never hang or silently tunnel.
            final = project_outside_circles(
                point, tuple(circles), preferred=tuple(-v for v in ball.velocity)
            )
            stopped = replace(contact_ball, position=final, velocity=(0.0, 0.0))
            exit_after = detect_ball_exit(point, final, config=config)
            if exit_after is not None:
                exit_after = replace(exit_after, fraction=1.0)
            return BallFlight(stopped, tuple(impacts), crossing=exit_after, limited=True)
        current_circles = tuple(
            (_lerp(start, finish, first), radius) for start, finish, _, radius, *_ in solids
        )
        uncorrected = point
        point = project_outside_circles(
            point, current_circles, preferred=tuple(-v for v in ball.velocity)
        )
        contact_ball = replace(contact_ball, position=point)
        group.sort(key=lambda c: (c.post_index is None, c.centre, c.radius))
        touched = {c.player for c in group if c.player is not None}
        last_touch = (
            next(iter(touched))
            if len(touched) == 1
            else None
            if touched
            else ball.last_touch_player
        )
        for solid in group:
            normal = circle_normal(point, solid.centre)
            outgoing = reflect_velocity(
                contact_ball.velocity, solid.velocity, normal, solid.restitution
            )
            impacts.append(
                BallImpact(
                    elapsed,
                    point,
                    contact_ball.velocity,
                    outgoing,
                    solid.player,
                    solid.post_index,
                    last_touch,
                )
            )
            contact_ball = replace(contact_ball, velocity=outgoing)
        ball = replace(contact_ball, last_touch_player=last_touch)
        if not isclose(recent_fraction, elapsed, rel_tol=0.0, abs_tol=1e-12):
            recent = []
        recent_fraction = elapsed
        by_object = {(c.player, c.post_index): c for c in recent + group}
        recent = list(by_object.values())
        opposed = None
        for index, first_solid in enumerate(recent):
            first_normal = circle_normal(point, first_solid.centre)
            for second_solid in recent[index + 1 :]:
                second_normal = circle_normal(point, second_solid.centre)
                if sum(a * b for a, b in zip(first_normal, second_normal)) < -1 + 1e-7:
                    opposed = first_solid, second_solid, first_normal
        if opposed is not None:
            first_solid, second_solid, normal = opposed
            fixed = next((c for c in (first_solid, second_solid) if c.post_index is not None), None)
            anchor = (
                fixed.velocity
                if fixed is not None
                else tuple((a + b) / 2 for a, b in zip(first_solid.velocity, second_solid.velocity))
            )
            normal_speed = sum((v - a) * n for v, a, n in zip(ball.velocity, anchor, normal))
            tangent = tuple(v - normal_speed * n for v, n in zip(ball.velocity, normal))
            settled = integrate_free_ball(
                replace(ball, velocity=tangent), (1 - elapsed) * PHYSICS_DT, config=config.ball
            )
            escaped = project_outside_circles(
                settled.position, tuple(circles), preferred=tuple(-v for v in ball.velocity)
            )
            all_touchers = {c.player for c in recent if c.player is not None}
            last_touch = (
                next(iter(all_touchers))
                if len(all_touchers) == 1
                else None
                if all_touchers
                else ball.last_touch_player
            )
            settled = replace(settled, position=escaped, last_touch_player=last_touch)
            exit_after = detect_ball_exit(point, escaped, config=config)
            if exit_after is not None:
                exit_after = replace(exit_after, fraction=1.0)
            return BallFlight(
                settled,
                tuple(impacts),
                crossing=exit_after,
                constraint=BallConstraint(escaped, ball.velocity, settled.velocity),
            )
        # Initial-overlap correction may itself cross a boundary; classify that
        # correction before any subsequent free segment can lose the exit.
        corrected_exit = detect_ball_exit(uncorrected, point, config=config)
        if corrected_exit is not None:
            return BallFlight(
                ball, tuple(impacts), crossing=replace(corrected_exit, fraction=elapsed)
            )
