"""Flat <-> Gymnasium-Dict action conversion (numpy only; no torch).

Member D's learner works with one flat float vector per decision:

    [direction (0..8), effort (0..1), kick_trigger (0/1), kick_power (0..1)]

``FootballEnv`` wants the exact hybrid Dict from ``environment/spec.py``. This
module is the single place that converts between the two, so train / test /
observe all agree.
"""

from __future__ import annotations

import numpy as np

FLAT_ACTION_DIM = 4
IDX_DIRECTION, IDX_EFFORT, IDX_KICK, IDX_POWER = range(4)


def to_env_action(flat) -> dict:
    """Convert one flat action to the Dict accepted by ``FootballEnv.step``."""
    a = np.asarray(flat, dtype=np.float64).reshape(-1)
    if a.shape != (FLAT_ACTION_DIM,) or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected 4 finite values, got {a!r}")
    return {
        "direction": int(np.clip(round(a[IDX_DIRECTION]), 0, 8)),
        "effort": np.array([np.clip(a[IDX_EFFORT], 0.0, 1.0)], dtype=np.float32),
        "kick_trigger": int(a[IDX_KICK] > 0.5),
        "kick_power": np.array([np.clip(a[IDX_POWER], 0.0, 1.0)], dtype=np.float32),
    }


def random_flat_action(rng: np.random.Generator) -> np.ndarray:
    """Uniform random legal action (used by the 'random' reference agent)."""
    return np.array(
        [rng.integers(9), rng.random(), float(rng.random() < 0.1), rng.random()],
        dtype=np.float32,
    )


IDLE_FLAT_ACTION = np.zeros(FLAT_ACTION_DIM, dtype=np.float32)
