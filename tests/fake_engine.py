"""
tests/fake_engine.py -- Faux moteur qui respecte le contrat (environment/contract.py).

BUT : permettre au Member C de développer et tester le wrapper SANS attendre le vrai moteur.
Ce n'est PAS un moteur de football : la physique est minimale et les événements sont SCRIPTÉS.

Scénario déterministe (toujours le même) :
  tick  10 : le joueur 0 récupère le ballon libre   -> LOOSE_BALL_RECOVERY
  tick  20 : le joueur 1 vole le ballon             -> STEAL
  tick  30 : le joueur 1 marque                     -> GOAL (score 0-1), puis le ballon est libre au centre
  tick 3600: fin du match                           -> phase ENDED
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from environment.contract import (
    BallState,
    Event,
    EventType,
    MatchState,
    Phase,
    PlayerState,
    StepResult,
)
from environment.spec import (
    FRAMES_PER_DECISION,
    MAX_PHYSICS_TICKS,
    MAX_PLAYER_SPEED,
    PHYSICS_HZ,
    PITCH_H,
    PITCH_W,
    DecodedAction,
)

DT = 1.0 / PHYSICS_HZ


class FakeEngine:
    def __init__(self) -> None:
        self._state: Optional[MatchState] = None

    # -- API du contrat ------------------------------------------------------
    def reset(self, seed=None, attack_dirs=(1, -1)) -> MatchState:
        # Déterministe : la graine est ignorée (rien d'aléatoire ici).
        self._state = MatchState(
            tick=0,
            players=(
                PlayerState(200.0, 250.0, 0.0, 0.0, 0.0, 100.0),
                PlayerState(600.0, 250.0, 0.0, 0.0, math.pi, 100.0),
            ),
            ball=BallState(PITCH_W / 2, PITCH_H / 2, 0.0, 0.0),
            possession=None,
            last_toucher=None,
            score=(0, 0),
            attack_dirs=tuple(attack_dirs),
            phase=Phase.PLAYING,
            contest_progress=(0.0, 0.0),
            steal_cooldown=(0.0, 0.0),
        )
        return self._state

    def step(self, actions: tuple[DecodedAction, DecodedAction],
             n_ticks: int = FRAMES_PER_DECISION) -> StepResult:
        assert self._state is not None, "appeler reset() avant step()"
        s = self._state
        events: list[Event] = []
        advanced = 0

        for _ in range(n_ticks):
            if s.is_over:
                break
            s = replace(s, tick=s.tick + 1, players=self._move_players(s, actions))
            advanced += 1

            # --- événements scriptés ---
            if s.tick == 10:
                s = replace(s, possession=0, last_toucher=0)
                events.append(Event(EventType.LOOSE_BALL_RECOVERY, s.tick, player=0))
            elif s.tick == 20:
                s = replace(s, possession=1, last_toucher=1)
                events.append(Event(EventType.STEAL, s.tick, player=1, other=0))
            elif s.tick == 30:
                events.append(Event(EventType.GOAL, s.tick, player=1))
                s = replace(s, score=(s.score[0], s.score[1] + 1), possession=None,
                            ball=BallState(PITCH_W / 2, PITCH_H / 2, 0.0, 0.0))

            if s.tick >= MAX_PHYSICS_TICKS:
                s = replace(s, phase=Phase.ENDED)

        self._state = s
        return StepResult(state=s, events=tuple(events), ticks_advanced=advanced)

    # -- Physique minimale (juste pour que les états varient) -----------------
    @staticmethod
    def _move_players(s: MatchState, actions) -> tuple[PlayerState, PlayerState]:
        out = []
        for p, a in zip(s.players, actions):
            speed = MAX_PLAYER_SPEED * (0.5 + 0.5 * a.effort)
            vx, vy = a.move_x * speed, a.move_y * speed
            x = min(max(p.x + vx * DT, 0.0), PITCH_W)   # on reste dans le terrain
            y = min(max(p.y + vy * DT, 0.0), PITCH_H)
            # stamina : sprint (>0.5) consomme, sinon récupère
            stamina = p.stamina - 0.1 if a.effort > 0.5 else p.stamina + 0.05
            stamina = min(max(stamina, 0.0), 100.0)
            out.append(replace(p, x=x, y=y, vx=vx, vy=vy, stamina=stamina))
        return (out[0], out[1])
