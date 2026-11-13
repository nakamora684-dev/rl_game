"""
environment/football_env.py -- Environnement Gymnasium pour l'agent RL (Member C, P4).

Rôle : relier le moteur (contrat GameEngine) à PPO.
  reset()  -> démarre un match, renvoie l'observation de l'agent ;
  step(a)  -> décode l'action, la convertit en repère monde, fait jouer l'adversaire,
              avance le moteur de 6 ticks, calcule observation / récompense / fin d'épisode.

Ce fichier ne contient AUCUNE physique et ne dépend pas de Pygame.

Côtés : le joueur 0 attaque toujours vers +x et le joueur 1 vers -x (attack_dirs = (1, -1)).
L'agent peut jouer l'indice 0 ou 1 (option "agent_idx" de reset, ou tirage aléatoire reproductible) :
c'est ainsi qu'on couvre les DEUX côtés à l'entraînement et à l'évaluation.
"""
from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np

from environment.contract import GameEngine, MatchState, Phase, validate_state, validate_step_result
from environment.observation import build_observation
from environment.reward import COMPONENT_NAMES, RewardCalculator, RewardConfig
from environment.spec import (
    FRAMES_PER_DECISION,
    DecodedAction,
    decode_action,
    make_action_space,
    make_observation_space,
    policy_to_world,
)
from environment.termination import EndReason, check_termination, match_result

# Un adversaire reçoit l'état complet et son indice, et renvoie une action en repère MONDE.
OpponentPolicy = Callable[[MatchState, int], DecodedAction]


def idle_opponent(state: MatchState, player_idx: int) -> DecodedAction:
    """Adversaire immobile (par défaut, tant que le baseline du Member B n'existe pas)."""
    return DecodedAction(move_x=0.0, move_y=0.0, effort=0.0, kick=False, kick_power=0.0)


class FootballEnv(gym.Env):
    metadata = {"render_modes": []}   # le rendu est dans rendering/, pas ici

    def __init__(
        self,
        engine: GameEngine,
        opponent: Optional[OpponentPolicy] = None,
        reward_config: Optional[RewardConfig] = None,
        debug: bool = True,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.opponent: OpponentPolicy = opponent or idle_opponent
        self.rewarder = RewardCalculator(reward_config)
        self.debug = debug   # True : valide le contrat moteur et l'observation à chaque pas (plus lent)

        self.action_space = make_action_space()            # Box [-1, 1]^5
        self.observation_space = make_observation_space()  # Box de dimension OBS_DIM

        self._state: Optional[MatchState] = None
        self._agent_idx = 0
        self._steps = 0
        self._done = True
        self._episode_components = {name: 0.0 for name in COMPONENT_NAMES}

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)   # initialise self.np_random (reproductible)
        options = options or {}

        # Côté de l'agent : imposé par options["agent_idx"], sinon tirage reproductible.
        if "agent_idx" in options:
            self._agent_idx = int(options["agent_idx"])
            if self._agent_idx not in (0, 1):
                raise ValueError("options['agent_idx'] doit valoir 0 ou 1")
        else:
            self._agent_idx = int(self.np_random.integers(2))

        # Graine du moteur dérivée de la nôtre : même seed => même épisode.
        engine_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._state = self.engine.reset(seed=engine_seed, attack_dirs=(1, -1))
        if self.debug:
            validate_state(self._state)

        self.rewarder.reset()
        self._steps = 0
        self._done = False
        self._episode_components = {name: 0.0 for name in COMPONENT_NAMES}

        obs = build_observation(self._state, self._agent_idx)
        return obs, self._info(events=(), components=None, end_reason=EndReason.NONE)

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self._done or self._state is None:
            raise RuntimeError("Épisode terminé : appelez reset() avant de rappeler step().")

        prev = self._state
        me = self._agent_idx
        opp = 1 - me

        # 1) Action de l'agent : [-1,1]^5 -> action légale -> repère monde.
        agent_action = policy_to_world(decode_action(action), prev.attack_dirs[me])

        # 2) Action de l'adversaire (déjà en repère monde).
        opp_action = self.opponent(prev, opp)
        if not isinstance(opp_action, DecodedAction):
            raise TypeError("L'adversaire doit renvoyer un DecodedAction")

        actions = [None, None]
        actions[me], actions[opp] = agent_action, opp_action

        # 3) Le moteur avance de 6 ticks en tenant les actions constantes.
        result = self.engine.step(tuple(actions), FRAMES_PER_DECISION)
        if self.debug:
            validate_step_result(prev, result)   # horloge, fenêtre d'événements, score/GOAL
        new = result.state
        if new.phase is Phase.RESTART:
            raise RuntimeError("Le moteur doit résoudre les remises en jeu en interne (contrat).")
        self._state = new
        self._steps += 1

        # 4) Récompense, observation, fin d'épisode.
        reward, components = self.rewarder.compute(prev, result, me)
        for k, val in components.items():
            self._episode_components[k] += val

        obs = build_observation(new, me)
        if self.debug and not self.observation_space.contains(obs):
            raise ValueError("Observation hors de l'espace déclaré")

        terminated, truncated, reason = check_termination(new, self._steps)
        self._done = terminated or truncated
        info = self._info(events=result.events, components=components, end_reason=reason)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------- info
    def _info(self, events, components, end_reason: EndReason) -> dict:
        s = self._state
        info = {
            "agent_idx": self._agent_idx,
            "tick": s.tick,
            "score": s.score,
            "events": tuple((e.type.name, e.tick, e.player) for e in events),
            "reward_components": components,   # composantes de CE pas (None au reset)
        }
        if end_reason is not EndReason.NONE:
            # Clé "episode_stats" (et non "episode") pour ne pas écraser celle du Monitor de Stable-Baselines3.
            info["end_reason"] = end_reason.value
            info["episode_stats"] = {
                "components": dict(self._episode_components),
                "steps": self._steps,
                "score": s.score,
                "match_result": match_result(s.score, self._agent_idx),
            }
        return info
