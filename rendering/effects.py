"""Purely visual effects: ball trail, particles, screen shake, ball spin.

NO pygame and NO game rules: it only reads a GameState (events, ball) and keeps
its own cosmetic state. Pure Python so it is unit-tested without a display.
Random numbers come from a LOCAL random.Random seeded from the event tick, so
effects never touch the engine's randomness.
"""

from __future__ import annotations

import math
import random
from collections import deque

from agents.interface import EV_GOAL, EV_KICK, EV_STEAL, GameState
from rendering import theme

TELEPORT_DIST = 80.0  # ball jump bigger than this = restart: clear the trail


class BallTrail:
    def __init__(self, n: int = 10) -> None:
        self.pts: deque = deque(maxlen=n)

    def clear(self) -> None:
        self.pts.clear()

    def update(self, x: float, y: float, min_move: float = 1.5) -> None:
        if self.pts:
            lx, ly = self.pts[-1]
            moved = math.hypot(x - lx, y - ly)
            if moved > TELEPORT_DIST:
                self.pts.clear()
            elif moved < min_move:  # ball (almost) stopped: trail fades
                self.pts.popleft()
                return
        self.pts.append((x, y))

    def points(self) -> list[tuple[float, float, float]]:
        n = len(self.pts)
        return [(x, y, (i + 1) / n) for i, (x, y) in enumerate(self.pts)]


class Particles:
    MAX = 240

    def __init__(self) -> None:
        self.items: list[list] = []  # [x, y, vx, vy, life, max_life, color, size]

    def clear(self) -> None:
        self.items = []

    def burst(
        self,
        x: float,
        y: float,
        colors,
        n: int,
        speed: float,
        life: float,
        seed: int = 0,
        size: int = 3,
    ) -> None:
        rng = random.Random(seed)
        for _ in range(n):
            a, s = rng.uniform(0, 2 * math.pi), rng.uniform(0.3, 1.0) * speed
            self.items.append(
                [
                    x,
                    y,
                    s * math.cos(a),
                    s * math.sin(a),
                    life * rng.uniform(0.6, 1.0),
                    life,
                    rng.choice(colors),
                    size,
                ]
            )
        del self.items[: -self.MAX]

    def update(self, dt: float) -> None:
        drag = 0.9 ** (dt * 60.0)
        for p in self.items:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[2] *= drag
            p[3] *= drag
            p[4] -= dt
        self.items = [p for p in self.items if p[4] > 0]

    def view(self) -> list[tuple[float, float, float, tuple, int]]:
        return [(p[0], p[1], max(0.0, p[4] / p[5]), p[6], p[7]) for p in self.items]


class Shake:
    def __init__(self) -> None:
        self.amp, self.k = 0.0, 0

    def kick(self, amp: float) -> None:
        self.amp = max(self.amp, amp)

    def update(self, dt: float) -> None:
        self.amp *= 0.88 ** (dt * 60.0)
        self.k += 1
        if self.amp < 0.4:
            self.amp = 0.0

    def offset(self) -> tuple[int, int]:
        if self.amp == 0.0:
            return 0, 0
        return round(self.amp * math.sin(self.k * 2.1)), round(self.amp * math.cos(self.k * 1.7))


class Effects:
    def __init__(self) -> None:
        self.trail, self.particles, self.shake = BallTrail(), Particles(), Shake()
        self.spin = 0.0
        self._last: tuple[float, float] | None = None

    def clear(self) -> None:
        self.trail.clear()
        self.particles.clear()
        self.shake = Shake()
        self.spin, self._last = 0.0, None

    def update(self, st: GameState, dt: float) -> None:
        """Call once per rendered tick (the events of a tick are consumed here)."""
        bx, by = st.ball.x, st.ball.y
        for ev in st.events:
            if ev.kind == EV_GOAL and ev.player is not None:
                team = theme.TEAM[ev.player]
                self.particles.burst(
                    bx,
                    by,
                    [team, (255, 255, 255), theme.GOAL, theme.mix(team, (255, 255, 255), 0.5)],
                    60,
                    260.0,
                    1.2,
                    seed=ev.tick,
                    size=4,
                )
                self.shake.kick(7.0)
            elif ev.kind == EV_STEAL:
                self.particles.burst(
                    bx, by, [theme.CONTEST, (255, 255, 255)], 10, 120.0, 0.4, seed=ev.tick
                )
            elif ev.kind == EV_KICK:
                self.particles.burst(bx, by, [(255, 255, 255)], 5, 80.0, 0.25, seed=ev.tick, size=2)
        if st.possession is None:
            self.trail.update(bx, by)
            if self._last is not None:
                d = math.hypot(bx - self._last[0], by - self._last[1])
                if d < TELEPORT_DIST:
                    self.spin += d / theme.BALL_R
        else:
            self.trail.clear()
        self._last = (bx, by)
        self.particles.update(dt)
        self.shake.update(dt)

    def shake_offset(self) -> tuple[int, int]:
        return self.shake.offset()
