"""Record a match as the list of actions of both players, and play it back. NO pygame.

The engine is deterministic for a seed (tests/test_mode_equivalence.py), so the
same seed + side + actions reproduce the exact match. Format (JSON, version 1):
  {"version": 1, "seed": 3, "swap_sides": false, "meta": {...}, "actions": [[a0, a1], ...]}
with each action = [move_x, move_y, effort, kick, power] (full float precision, so the
replay is bit-exact; roughly 0.2-0.5 MB per 60 s match).
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.interface import Action, GameState

REPLAY_VERSION = 1


class ReplayRecorder:
    def __init__(self, seed: int, swap_sides: bool, **meta) -> None:
        self.seed, self.swap_sides, self.meta = seed, swap_sides, meta
        self.actions: list[list[list[float]]] = []

    def record(self, actions: list[Action]) -> None:
        self.actions.append([a.to_list() for a in actions])  # full precision: exact replay

    def to_dict(self) -> dict:
        return {
            "version": REPLAY_VERSION,
            "seed": self.seed,
            "swap_sides": self.swap_sides,
            "meta": self.meta,
            "actions": self.actions,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")))
        return path


def load_replay(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("version") != REPLAY_VERSION:
        raise ValueError(f"unsupported replay version {data.get('version')!r}")
    return data


class RecordingAgent:
    """Wraps an agent and logs every action it returns (one per tick)."""

    def __init__(self, agent, log: list) -> None:
        self.agent, self.log = agent, log

    def reset(self) -> None:
        self.log.clear()
        if hasattr(self.agent, "reset"):
            self.agent.reset()

    def act(self, state: GameState) -> Action:
        a = self.agent.act(state)
        self.log.append(a)
        return a


class ReplayAgent:
    """Plays back the actions of ONE player (index into each [a0, a1] pair)."""

    def __init__(self, data: dict, player: int) -> None:
        self.rows = [step[player] for step in data["actions"]]
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def act(self, state: GameState) -> Action:
        row = self.rows[self.i] if self.i < len(self.rows) else [0.0] * 5
        self.i += 1
        return Action.from_list(row)
