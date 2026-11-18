"""Gymnasium wrapper over the shared game, with exact hybrid controls.

No rendering, physics, reward tuning, or PPO-specific action distribution here.
Opponent providers use the same native state/PlayerId/PlayerAction as game.runner.
"""

from dataclasses import asdict
from typing import ClassVar

import gymnasium as gym
import numpy as np

from environment.contract import (
    PLAYER_IDS,
    GameEngine,
    MatchState,
    validate_state,
    validate_step_result,
)
from environment.engine import SimulationEngine
from environment.observation import build_observation
from environment.reward import COMPONENT_NAMES, RewardCalculator, RewardConfig
from environment.spec import (
    OBSERVATION_VERSION,
    decode_action,
    make_action_space,
    make_observation_space,
    policy_to_world,
)
from environment.termination import EndReason, check_termination, match_result
from game.actions import PlayerAction
from game.runner import ActionProvider, idle_provider


class FootballEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        engine: GameEngine | None = None,
        opponent: ActionProvider | None = None,
        reward_config: RewardConfig | None = None,
        debug: bool = True,
    ) -> None:
        super().__init__()
        self.engine = SimulationEngine() if engine is None else engine
        self.opponent = idle_provider if opponent is None else opponent
        self.rewarder = RewardCalculator(reward_config)
        self.debug = debug
        self.action_space = make_action_space()
        self.observation_space = make_observation_space()
        self._state: MatchState | None = None
        self._agent_idx = 0
        self._steps = 0
        self._done = True
        self._episode_components = {name: 0.0 for name in COMPONENT_NAMES}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "agent_idx" in options:
            idx = options["agent_idx"]
            if not isinstance(idx, (int, np.integer)) or idx not in (0, 1):
                raise ValueError("agent_idx must be integer 0 (A) or 1 (B).")
            self._agent_idx = int(idx)
        else:
            self._agent_idx = int(self.np_random.integers(2))
        swap_sides = options.get("swap_sides", False)
        if not isinstance(swap_sides, (bool, np.bool_)):
            raise TypeError("swap_sides must be a boolean.")
        # Only wrapper identity selection is random; current engine reset has no RNG.
        # Stateful/stochastic opponent providers must be reset/seeded by their owner.
        self._state = self.engine.reset(swap_sides=bool(swap_sides))
        if self.debug:
            validate_state(self._state)
        self.rewarder.reset()
        self._steps = 0
        self._done = False
        self._episode_components = {name: 0.0 for name in COMPONENT_NAMES}
        obs = build_observation(self._state, self._agent_idx)
        return obs, self._info((), None, EndReason.NONE, 0)

    def step(self, action):
        if self._done or self._state is None:
            raise RuntimeError("Call reset() before stepping or after episode completion.")
        prev, me = self._state, self._agent_idx
        agent_action = policy_to_world(decode_action(action), prev.attack_dirs[me])
        # The provider sees the SAME immutable pre-decision native snapshot, even
        # during restarts. It returns world-space controls; do not mirror it again.
        opp = 1 - me
        opp_action = self.opponent(self.engine.native_state, PLAYER_IDS[opp])
        if not isinstance(opp_action, PlayerAction):
            raise TypeError("Opponent must return a world-space PlayerAction.")
        actions = (agent_action, opp_action) if me == 0 else (opp_action, agent_action)
        result = self.engine.step(actions)
        if self.debug:
            validate_step_result(prev, result)
        self._state = result.state
        self._steps += 1
        reward, components = self.rewarder.compute(prev, result, me)
        for name, value in components.items():
            self._episode_components[name] += value
        obs = build_observation(result.state, me)
        if self.debug and not self.observation_space.contains(obs):
            raise ValueError("Observation outside declared space.")
        terminated, truncated, reason = check_termination(result.state, self._steps)
        self._done = terminated or truncated
        info = self._info(result.raw_events, components, reason, result.ticks_advanced)
        return obs, float(reward), terminated, truncated, info

    def _info(self, events, components, end_reason: EndReason, physics_ticks: int) -> dict:
        s = self._state
        info = {
            "agent_idx": self._agent_idx,
            "observation_version": OBSERVATION_VERSION,
            "tick": s.tick,
            "physics_ticks": physics_ticks,
            "score": s.score,
            "attack_dirs": s.attack_dirs,
            "phase": s.phase.value,
            "restart_ticks_remaining": s.restart_ticks_remaining,
            # Preserve all telemetry, causes, owners and collision fractions in order.
            "events": tuple(asdict(e) for e in events),
            "reward_components": components,
        }
        if end_reason is not EndReason.NONE:
            # Do not collide with SB3 Monitor's reserved "episode" key.
            info["end_reason"] = end_reason.value
            info["episode_stats"] = {
                "components": dict(self._episode_components),
                "steps": self._steps,
                "score": s.score,
                "match_result": match_result(s.score, self._agent_idx),
            }
        return info
