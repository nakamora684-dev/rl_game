"""Small collision response/projection helpers, without simulation clocks or rewards."""

from math import hypot, isclose, sqrt

Point = tuple[float, float]
Circle = tuple[Point, float]


def reflect_velocity(
    velocity: Point, surface_velocity: Point, normal: Point, restitution: float
) -> Point:
    """Reflect only closing normal relative velocity; retain tangential velocity."""
    closing = sum((v - surface) * n for v, surface, n in zip(velocity, surface_velocity, normal))
    if closing >= 0.0:
        return velocity
    return tuple(v - (1 + restitution) * closing * n for v, n in zip(velocity, normal))


def circle_normal(point: Point, centre: Point, fallback: Point = (1.0, 0.0)) -> Point:
    dx, dy = point[0] - centre[0], point[1] - centre[1]
    distance = hypot(dx, dy)
    return (dx / distance, dy / distance) if distance else fallback


def _intersections(first: Circle, second: Circle) -> list[Point]:
    (x, y), r = first
    (bx, by), other_r = second
    dx, dy = bx - x, by - y
    distance = hypot(dx, dy)
    if distance == 0.0 or distance > r + other_r or distance < abs(r - other_r):
        return []
    along = (r * r - other_r * other_r + distance * distance) / (2 * distance)
    height = sqrt(max(0.0, r * r - along * along))
    nx, ny = dx / distance, dy / distance
    return [
        (x + along * nx - height * ny, y + along * ny + height * nx),
        (x + along * nx + height * ny, y + along * ny - height * nx),
    ]


def project_outside_circles(
    point: Point,
    circles: tuple[Circle, ...],
    *,
    preferred: Point = (1.0, 0.0),
    bounds: tuple[float, float, float, float] | None = None,
) -> Point:
    """Nearest feasible point outside a small union of expanded solid circles.

    Candidate arc projections/intersections also resolve simultaneous overlaps;
    no A/B identity priority. Optional bounds are xmin,xmax,ymin,ymax. Tied
    solutions prefer the supplied escape direction, then a deterministic axis.
    """
    tolerance = 1e-9

    def feasible(candidate: Point) -> bool:
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            if not xmin <= candidate[0] <= xmax or not ymin <= candidate[1] <= ymax:
                return False
        return all(
            hypot(candidate[0] - c[0], candidate[1] - c[1]) >= r - tolerance for c, r in circles
        )

    if feasible(point):
        return point
    candidates = []
    fallback = circle_normal(preferred, (0.0, 0.0))
    for index, (centre, radius) in enumerate(circles):
        nx, ny = circle_normal(point, centre, fallback)
        candidates.append((centre[0] + radius * nx, centre[1] + radius * ny))
        for other in circles[index + 1 :]:
            candidates.extend(_intersections((centre, radius), other))
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            for x in (xmin, xmax):
                square = radius * radius - (x - centre[0]) ** 2
                if square >= 0.0:
                    height = sqrt(square)
                    candidates.extend(((x, centre[1] - height), (x, centre[1] + height)))
            for y in (ymin, ymax):
                square = radius * radius - (y - centre[1]) ** 2
                if square >= 0.0:
                    width = sqrt(square)
                    candidates.extend(((centre[0] - width, y), (centre[0] + width, y)))
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        candidates.append((min(xmax, max(xmin, point[0])), min(ymax, max(ymin, point[1]))))
        candidates.extend((x, y) for x in (xmin, xmax) for y in (ymin, ymax))
    candidates = [candidate for candidate in candidates if feasible(candidate)]
    if not candidates:
        raise ValueError("Solid geometry leaves no feasible nonpenetrating position.")
    distances = [hypot(p[0] - point[0], p[1] - point[1]) for p in candidates]
    nearest = min(distances)
    tied = [
        p
        for p, d in zip(candidates, distances)
        if isclose(d, nearest, rel_tol=1e-12, abs_tol=tolerance)
    ]
    return min(
        tied,
        key=lambda p: (
            -sum((v - start) * direction for v, start, direction in zip(p, point, preferred)),
            p,
        ),
    )
