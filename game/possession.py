"""Attachment and swept pickup helpers; timed steals are orchestrated in simulation."""

from dataclasses import dataclass, replace
from math import cos, isclose, sin

from game.ball import BallState
from game.config import BallConfig, SimulationConfig
from game.geometry import swept_circle_contact
from game.player import PlayerId, PlayerState

_DEFAULT_BALL = BallConfig()
_DEFAULT_CONFIG = SimulationConfig()


@dataclass(frozen=True)
class PickupContact:
    owner: PlayerId
    fraction: float
    distance: float


def find_pickup_contact(
    ball_start: BallState,
    ball_end: BallState,
    player_a_start: PlayerState,
    player_b_start: PlayerState,
    player_a_end: PlayerState,
    player_b_end: PlayerState,
    *,
    config: SimulationConfig = _DEFAULT_CONFIG,
) -> PickupContact | None:
    """Choose first eligible swept contact, then nearest on simultaneous contact.

    Exact ties (within floating-point tolerance) stay free rather than favouring
    an identity. Eligibility uses lockout before end-of-tick timer advancement.
    Players control free balls at any speed; this performs no physical bounce.
    """
    if ball_start.possession_owner is not None or ball_end.possession_owner is not None:
        raise ValueError("Pickup selection requires a free ball throughout the sweep.")
    radius = config.pitch.player_radius + config.ball.radius
    contacts = []
    for owner, previous, current in (
        (PlayerId.A, player_a_start, player_a_end),
        (PlayerId.B, player_b_start, player_b_end),
    ):
        if current.pickup_lockout_ticks:
            continue
        contact = swept_circle_contact(
            ball_start.position, ball_end.position, previous.position, current.position, radius
        )
        if contact is not None:
            contacts.append((owner, *contact))
    if not contacts:
        return None
    if len(contacts) == 1:
        return PickupContact(*contacts[0])
    (_, time_a, distance_a), (_, time_b, distance_b) = contacts
    if not isclose(time_a, time_b, rel_tol=0.0, abs_tol=1e-12):
        return PickupContact(*(contacts[0] if time_a < time_b else contacts[1]))
    if isclose(distance_a, distance_b, rel_tol=1e-12, abs_tol=1e-9):
        return None
    return PickupContact(*(contacts[0] if distance_a < distance_b else contacts[1]))


def select_pickup_owner(
    ball_start: BallState,
    ball_end: BallState,
    player_a_start: PlayerState,
    player_b_start: PlayerState,
    player_a_end: PlayerState,
    player_b_end: PlayerState,
    *,
    config: SimulationConfig = _DEFAULT_CONFIG,
) -> PlayerId | None:
    """Convenience wrapper; shared stepping also needs contact fraction for rule ordering."""
    contact = find_pickup_contact(
        ball_start,
        ball_end,
        player_a_start,
        player_b_start,
        player_a_end,
        player_b_end,
        config=config,
    )
    return contact.owner if contact is not None else None


def sync_controlled_ball(
    ball: BallState,
    player_a: PlayerState,
    player_b: PlayerState,
    *,
    config: BallConfig = _DEFAULT_BALL,
) -> BallState:
    """Synchronize the cached attachment using its sole owner, never player velocity.

    The carrier snapshots must already contain this tick's movement, bounds, and
    facing updates. Do not pitch-clamp the ball: shared rules handle crossings.
    """
    if ball.possession_owner is None:
        raise ValueError("Controlled attachment requires a possession owner.")
    carrier = player_a if ball.possession_owner == PlayerId.A else player_b
    x, y = carrier.position
    distance = config.dribble_offset
    return replace(
        ball,
        position=(x + distance * cos(carrier.facing), y + distance * sin(carrier.facing)),
    )
