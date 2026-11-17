"""Readable engine event stamps, without sorting or inventing transition events."""

from collections.abc import Iterable

from game.config import PHYSICS_HZ
from game.simulation import SimulationEvent


def format_timeline(events: Iterable[SimulationEvent]) -> str:
    """Preserve emission order; seconds are stored tick/60, not exact contact time.

    Kicks use start ticks; most other events use completed ticks. Keep sub-tick
    fractions explicit instead of implying every stamp is an exact physical time.
    """
    lines = [
        "Event timeline (engine emission order; stamp_s = stored tick / 60)",
        (
            "Kicks use start ticks; other transitions usually use completed ticks. "
            "Contact fractions are shown separately."
        ),
        " seq   stamp_s  tick  event                 details",
    ]
    count = 0
    for event in events:
        details = []
        if event.cause is not None:
            details.append(f"cause={event.cause}")
        if event.kind in ("possession_changed", "ball_kicked"):
            previous = event.previous_owner.value if event.previous_owner is not None else "free"
            new = event.new_owner.value if event.new_owner is not None else "free"
            details.append(f"owner={previous}->{new}")
        for label, value in (
            ("scorer", event.scoring_player),
            ("receiver", event.restart_owner),
            ("last_touch", event.last_touch_player),
            ("player", event.contact_player),
            ("post", event.post_index),
            ("fraction", event.tick_fraction),
            ("collision", event.collision_index),
            ("ball", event.ball_position),
            ("velocity_before", event.velocity_before),
            ("velocity_after", event.velocity_after),
        ):
            if value is not None:
                details.append(f"{label}={value}")
        if event.kind == "out_of_bounds" and event.last_touch_player is None:
            details.append("last_touch=unknown receiver=neutral")
        lines.append(
            f"{count:4d}  {event.tick / PHYSICS_HZ:8.3f}  {event.tick:4d}  "
            f"{event.kind:<21} {' '.join(details)}".rstrip()
        )
        count += 1
    if count == 0:
        lines.append("(no events)")
    return "\n".join(lines)
