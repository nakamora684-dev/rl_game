"""Sliding player contact, solid posts and bounded coupled overlap correction."""

from dataclasses import replace
from math import hypot

from game.config import PitchConfig, SimulationConfig
from game.geometry import swept_circle_contact
from game.player import PlayerState, constrain_to_pitch
from game.rebounds import circle_normal, project_outside_circles, reflect_velocity
from game.rules import goalpost_centres

_MAX_PLAYER_POST_CONTACTS = 16
_MAX_COUPLED_PROJECTIONS = 64


def _normal(
    a: tuple[float, float], b: tuple[float, float], fallback: tuple[float, float]
) -> tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    distance = hypot(dx, dy)
    # Exact coincidence has no geometric normal. Split equally along a fixed axis.
    return (dx / distance, dy / distance) if distance else fallback


def _project_direction(
    position: tuple[float, float], direction: tuple[float, float], pitch: PitchConfig
) -> tuple[float, float]:
    """Remove only components pointing through an already active wall constraint."""
    projected = []
    for coordinate, component, extent in zip(position, direction, (pitch.width, pitch.height)):
        blocked = (coordinate <= pitch.player_radius and component < 0.0) or (
            coordinate >= extent - pitch.player_radius and component > 0.0
        )
        projected.append(0.0 if blocked else component)
    return tuple(projected)


def resolve_player_contact(
    previous_a: PlayerState,
    previous_b: PlayerState,
    player_a: PlayerState,
    player_b: PlayerState,
    *,
    pitch: PitchConfig,
) -> tuple[PlayerState, PlayerState]:
    """Sweep bodies, remove closing normal motion equally, retain tangent motion.

    Correct residual overlap after pitch constraints, redistributing correction
    when one player is pinned at a wall. Facing/stamina/timers are unchanged.
    This is one equal-mass contact over linear snapshot paths, not a rigid-body
    solver with repeated sub-tick wall/contact impulses or player friction.
    """
    diameter = 2 * pitch.player_radius
    fallback = (1.0, 0.0) if pitch.width >= pitch.height else (0.0, 1.0)
    contact = swept_circle_contact(
        previous_a.position, player_a.position, previous_b.position, player_b.position, diameter
    )
    if contact is None:
        return player_a, player_b
    fraction, _ = contact
    da = tuple(end - start for start, end in zip(previous_a.position, player_a.position))
    db = tuple(end - start for start, end in zip(previous_b.position, player_b.position))
    ca = tuple(start + fraction * delta for start, delta in zip(previous_a.position, da))
    cb = tuple(start + fraction * delta for start, delta in zip(previous_b.position, db))
    normal = _normal(ca, cb, fallback)
    closing_displacement = max(0.0, sum((a - b) * n for a, b, n in zip(da, db, normal)))
    closing_speed = max(
        0.0, sum((a - b) * n for a, b, n in zip(player_a.velocity, player_b.velocity, normal))
    )
    remaining = 1.0 - fraction
    player_a = constrain_to_pitch(
        replace(
            player_a,
            position=tuple(
                c + remaining * (d - 0.5 * closing_displacement * n)
                for c, d, n in zip(ca, da, normal)
            ),
            velocity=tuple(v - 0.5 * closing_speed * n for v, n in zip(player_a.velocity, normal)),
        ),
        pitch,
    )
    player_b = constrain_to_pitch(
        replace(
            player_b,
            position=tuple(
                c + remaining * (d + 0.5 * closing_displacement * n)
                for c, d, n in zip(cb, db, normal)
            ),
            velocity=tuple(v + 0.5 * closing_speed * n for v, n in zip(player_b.velocity, normal)),
        ),
        pitch,
    )
    for _ in range(16):
        distance = hypot(*(b - a for a, b in zip(player_a.position, player_b.position)))
        if distance >= diameter - 1e-10:
            break
        normal = _normal(player_a.position, player_b.position, fallback)
        depth = diameter - distance
        away_a = _project_direction(player_a.position, tuple(-n for n in normal), pitch)
        away_b = _project_direction(player_b.position, normal, pitch)
        gain = sum((b - a) * n for a, b, n in zip(away_a, away_b, normal))
        if gain == 0.0:
            raise ValueError("Pitch constraints cannot separate these overlapping player bodies.")
        correction = depth / gain
        player_a = constrain_to_pitch(
            replace(
                player_a,
                position=tuple(p + correction * n for p, n in zip(player_a.position, away_a)),
            ),
            pitch,
        )
        player_b = constrain_to_pitch(
            replace(
                player_b,
                position=tuple(p + correction * n for p, n in zip(player_b.position, away_b)),
            ),
            pitch,
        )
    else:
        raise ValueError("Player overlap correction did not converge within the iteration limit.")
    # A wall may have cancelled one player's share of the contact impulse. Remove
    # the remaining closing velocity from the unpinned body, not into the wall.
    normal = _normal(player_a.position, player_b.position, fallback)
    closing_speed = max(
        0.0, sum((a - b) * n for a, b, n in zip(player_a.velocity, player_b.velocity, normal))
    )
    away_a = _project_direction(player_a.position, tuple(-n for n in normal), pitch)
    away_b = _project_direction(player_b.position, normal, pitch)
    gain = sum((b - a) * n for a, b, n in zip(away_a, away_b, normal))
    if closing_speed and gain:
        player_a = constrain_to_pitch(
            replace(
                player_a,
                velocity=tuple(
                    v + closing_speed / gain * n for v, n in zip(player_a.velocity, away_a)
                ),
            ),
            pitch,
        )
        player_b = constrain_to_pitch(
            replace(
                player_b,
                velocity=tuple(
                    v + closing_speed / gain * n for v, n in zip(player_b.velocity, away_b)
                ),
            ),
            pitch,
        )
    return player_a, player_b


def _player_posts(
    previous: PlayerState, player: PlayerState, config: SimulationConfig
) -> tuple[PlayerState, bool]:
    """Sweep and slide through the remaining proposed displacement, not extra ticks."""
    radius = config.pitch.player_radius + config.rules.post_radius
    circles = tuple((centre, radius) for centre in goalpost_centres(config))
    pitch = config.pitch
    bounds = (
        pitch.player_radius,
        pitch.width - pitch.player_radius,
        pitch.player_radius,
        pitch.height - pitch.player_radius,
    )
    start, end, velocity = previous.position, player.position, player.velocity
    for _ in range(_MAX_PLAYER_POST_CONTACTS):
        contacts = []
        for centre, _ in circles:
            contact = swept_circle_contact(start, end, centre, centre, radius)
            if contact is None:
                continue
            fraction, _ = contact
            point = tuple(a + fraction * (b - a) for a, b in zip(start, end))
            normal = circle_normal(point, centre, (-1.0, 0.0) if centre[0] else (1.0, 0.0))
            closing = sum((b - a) * n for a, b, n in zip(start, end, normal))
            if (
                hypot(start[0] - centre[0], start[1] - centre[1]) < radius - 1e-9
                or closing < -1e-10
            ):
                contacts.append((fraction, centre, point, normal))
        if not contacts:
            end = project_outside_circles(end, circles, bounds=bounds)
            return constrain_to_pitch(
                replace(player, position=end, velocity=velocity), pitch
            ), False
        _, _, point, normal = min(contacts)
        point = project_outside_circles(
            point, circles, preferred=tuple(a - b for a, b in zip(start, end)), bounds=bounds
        )
        remaining = tuple(b - p for b, p in zip(end, point))
        tangent = reflect_velocity(remaining, (0.0, 0.0), normal, 0.0)
        velocity = reflect_velocity(velocity, (0.0, 0.0), normal, 0.0)
        start = point
        end_player = constrain_to_pitch(
            replace(
                player, position=tuple(p + d for p, d in zip(point, tangent)), velocity=velocity
            ),
            pitch,
        )
        end, velocity = end_player.position, end_player.velocity
    start = project_outside_circles(start, circles, bounds=bounds)
    return replace(player, position=start, velocity=(0.0, 0.0)), True


def resolve_players(
    previous_a: PlayerState,
    previous_b: PlayerState,
    player_a: PlayerState,
    player_b: PlayerState,
    *,
    config: SimulationConfig,
) -> tuple[PlayerState, PlayerState, bool]:
    """Resolve body/post sweeps, then mutually consistent static contact constraints."""
    player_a, player_b = resolve_player_contact(
        previous_a, previous_b, player_a, player_b, pitch=config.pitch
    )
    player_a, limited_a = _player_posts(previous_a, player_a, config)
    player_b, limited_b = _player_posts(previous_b, player_b, config)
    radius = config.pitch.player_radius + config.rules.post_radius
    posts = goalpost_centres(config)
    for _ in range(_MAX_COUPLED_PROJECTIONS):
        separated = (
            hypot(*(b - a for a, b in zip(player_a.position, player_b.position)))
            >= 2 * config.pitch.player_radius - 1e-9
        )
        clear_posts = all(
            hypot(p.position[0] - c[0], p.position[1] - c[1]) >= radius - 1e-9
            for p in (player_a, player_b)
            for c in posts
        )
        if separated and clear_posts:
            return player_a, player_b, limited_a or limited_b
        player_a, player_b = resolve_player_contact(
            player_a, player_b, player_a, player_b, pitch=config.pitch
        )
        player_a, limited = _player_posts(player_a, player_a, config)
        limited_a |= limited
        player_b, limited = _player_posts(player_b, player_b, config)
        limited_b |= limited
    raise ValueError("Coupled player/post constraints did not converge.")
