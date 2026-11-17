"""Renderer-independent swept contact between linear tick snapshots."""

from math import hypot, isclose, sqrt, ulp


def swept_circle_contact(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
    radius: float,
) -> tuple[float, float] | None:
    """First contact fraction/distance for two linearly swept tick snapshots.

    Work in relative coordinates: |r + fraction*d| <= combined radius. This is
    a straight-segment sweep, not exact friction/acceleration timing within a tick.
    Inputs are finite simulation snapshots and a positive combined radius.
    """
    rx, ry = (b - p for b, p in zip(first_start, second_start))
    dx, dy = (
        (b1 - b0) - (p1 - p0)
        for b0, b1, p0, p1 in zip(first_start, first_end, second_start, second_end)
    )
    scale = max(abs(rx), abs(ry), abs(dx), abs(dy), radius)
    x, y, vx, vy, r = (value / scale for value in (rx, ry, dx, dy, radius))
    c = x * x + y * y - r * r
    if c <= 0.0:
        return 0.0, hypot(rx, ry)
    a = vx * vx + vy * vy
    b = x * vx + y * vy
    if a == 0.0 or b >= 0.0:
        return None
    discriminant = b * b - a * c
    if discriminant < 0.0:
        if discriminant < -8 * ulp(max(b * b, a * c)):
            return None
        discriminant = 0.0
    fraction = c / (-b + sqrt(discriminant))
    if fraction > 1.0 and not isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return None
    fraction = min(1.0, fraction)
    return fraction, hypot(rx + fraction * dx, ry + fraction * dy)
