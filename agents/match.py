"""Engine-agnostic match loop (NO pygame). Used by observe.py, tests, and later test.py.

Graphical and headless modes both go through the same engine; the only difference
is the optional `on_step` hook (a renderer, a recorder...). The hook must not
change the state: tests/test_mode_equivalence.py checks this.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from agents.interface import PHASE_FINISHED, GameState


def reset_engine(engine, seed: int, swap_sides: bool = False) -> GameState:
    if swap_sides:
        return engine.reset(seed=seed, swap_sides=True)
    return engine.reset(seed=seed)


def run_match(
    engine,
    agents: Sequence,
    seed: int = 0,
    swap_sides: bool = False,
    on_step: Callable[[GameState], None] | None = None,
    max_ticks: int | None = None,
) -> GameState:
    """Play one match to the final whistle (or `max_ticks`). Returns the last state."""
    state = reset_engine(engine, seed, swap_sides)
    for a in agents:
        if hasattr(a, "reset"):
            a.reset()
    ticks = 0
    while state.phase != PHASE_FINISHED:
        state = engine.step([a.act(state) for a in agents])
        if on_step is not None:
            on_step(state)
        ticks += 1
        if max_ticks is not None and ticks >= max_ticks:
            break
    return state
