"""Optional, seeded start-position jitter (an ENGINE ADAPTER, not a physics change).

The engine reset is fully deterministic, so (deterministic policy + deterministic
opponent) replays the *same* match every time. Small start jitter gives the
variety the plan calls for ("randomize small starting variations once deterministic
reset works") while leaving all physics/rules untouched. Opt-in via
``--start-jitter`` (units; 0 = off). Needs sign-off from Members A/C.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from environment.engine import SimulationEngine
from game.simulation import reset as engine_reset


class JitteredSimulationEngine(SimulationEngine):
    def __init__(self, jitter: float = 0.0, seed: int = 0, config=None) -> None:
        super().__init__(config)
        if jitter < 0:
            raise ValueError("jitter must be >= 0")
        self.jitter = float(jitter)
        self._rng = np.random.default_rng(seed)

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, *, swap_sides: bool = False, initial_state=None):
        if initial_state is None and self.jitter > 0:
            base = engine_reset(config=self.config, swap_sides=swap_sides)
            pitch = self.config.pitch
            r = pitch.player_radius

            def shift(player):
                dx, dy = self._rng.uniform(-self.jitter, self.jitter, size=2)
                x = float(np.clip(player.position[0] + dx, 2 * r, pitch.width - 2 * r))
                y = float(np.clip(player.position[1] + dy, 2 * r, pitch.height - 2 * r))
                return replace(player, position=(x, y))

            initial_state = replace(
                base, player_a=shift(base.player_a), player_b=shift(base.player_b)
            )
        return super().reset(swap_sides=swap_sides, initial_state=initial_state)
