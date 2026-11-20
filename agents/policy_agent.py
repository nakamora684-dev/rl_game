"""Run a trained PPO checkpoint inside Member B's 60 Hz viewer/match loop (Member D).

``observe.py`` drives agents through ``agents.interface.Action`` and ``EngineAdapter``
(one physics tick per step). The trained policy, however, was trained on the 10 Hz
``FootballEnv`` with C's 50-feature observation and the hybrid action. To make the viewer
show *exactly* what ``train.py`` / ``test.py`` evaluate:

* ``PolicyAgent`` builds the observation with C's ``build_observation`` from the engine's
  native state, decides every 6 ticks, mirrors the action for the attack direction
  (``policy_to_world``) and returns a ``PolicyAction`` that carries the native PlayerAction.
* ``PolicyEngineAdapter`` is ``EngineAdapter`` that uses that native action as-is. B's own
  ``to_player_action`` remaps effort to 0.5..1 (jog..sprint), which would silently turn the
  policy's low-effort moves into jogging and make the viewer differ from evaluation.
  Kick timing still uses B's adapter logic (consumed on the first tick of a decision).
Nothing in game/, environment/ or B's adapter is modified.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from agents import interface as view
from agents.action_codec import to_env_action
from agents.engine_adapter import EngineAdapter, convert_events, to_player_action, to_view
from environment.contract import snapshot
from environment.observation import build_observation
from environment.spec import decode_action, policy_to_world
from game.actions import PlayerAction
from game.config import TICKS_PER_DECISION
from game.simulation import step_tick


@dataclass
class PolicyAction(view.Action):
    native: PlayerAction | None = None


class PolicyAgent:
    """``act(GameState) -> Action`` at 10 Hz for player ``index`` (0 = A, 1 = B)."""

    def __init__(self, policy, index: int, engine, every: int = TICKS_PER_DECISION) -> None:
        self.policy, self.index, self.engine, self.every = policy, index, engine, every
        self._cache: PolicyAction | None = None
        self._last = 0

    def reset(self) -> None:
        self._cache = None
        self.policy.reseed(0)

    def act(self, state: view.GameState) -> PolicyAction:
        if self._cache is None or state.tick - self._last >= self.every or state.tick < self._last:
            ms = snapshot(self.engine.raw, self.engine.config)
            obs = build_observation(ms, self.index)
            flat = np.asarray(self.policy.act(obs), dtype=np.float32)
            native = policy_to_world(decode_action(to_env_action(flat)), ms.attack_dirs[self.index])
            dx, dy = native.direction.vector
            self._cache = PolicyAction(
                move_x=dx, move_y=dy, effort=native.effort,
                kick=float(native.kick_trigger), power=native.kick_power, native=native,
            )
            self._last = state.tick
        return self._cache


class PolicyEngineAdapter(EngineAdapter):
    """EngineAdapter that honours ``PolicyAction.native`` (other agents unchanged)."""

    def step(self, actions: list[view.Action]) -> view.GameState:
        if len(actions) != 2:
            raise ValueError("Exactly one Action per player is required.")
        kicks = self._kick_requests(actions)
        native = []
        for action, (trigger, power) in zip(actions, kicks, strict=True):
            raw = getattr(action, "native", None)
            if raw is None:
                native.append(to_player_action(action, trigger, power))
            else:
                native.append(
                    replace(raw, kick_trigger=trigger,
                            kick_power=raw.kick_power if power is None else power)
                )
        result = step_tick(self.raw, native[0], native[1], config=self.config)
        self.raw = result.state
        return to_view(self.raw, convert_events(result.events))
