"""
environment/contract.py  --  Contrat entre le moteur (game/, Member A) et le wrapper (environment/, Member C)

Ce fichier ne contient AUCUNE physique. Il définit seulement :
  * ce que le moteur doit EXPOSER (snapshots d'état en lecture seule),
  * quels ÉVÉNEMENTS il doit remonter (chacun une seule fois),
  * quelle INTERFACE il doit offrir (reset / step),
  * des fonctions de VALIDATION que le wrapper et les tests utilisent.

Le Member A implémente `GameEngine` dans game/. Le wrapper (P4) ne dépend que de ce contrat,
jamais des détails internes du moteur (et jamais de Pygame).

Conventions (à faire confirmer par le Member A) :
  * Repère MONDE : x dans [0, PITCH_W], y dans [0, PITCH_H], unités logiques.
  * Angles en radians, facing = atan2(dy, dx) dans le repère monde.
  * Joueurs indexés 0 et 1. Le joueur 0 commence à gauche (200,250), le joueur 1 à droite (600,250).
  * attack_dirs[i] = +1 si le joueur i attaque vers +x, -1 s'il attaque vers -x.
  * Les actions reçues par le moteur sont en repère MONDE (déjà passées par policy_to_world).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from environment.spec import (
    FRAMES_PER_DECISION,
    MAX_PHYSICS_TICKS,
    PITCH_H,
    PITCH_W,
    DecodedAction,
)

# ---------------------------------------------------------------------------
# 1. PHASES ET ÉVÉNEMENTS
# ---------------------------------------------------------------------------


class Phase(Enum):
    """Phase du match. Une seule source de vérité (pas de drapeaux séparés)."""
    PLAYING = "playing"   # jeu normal : la politique décide
    RESTART = "restart"   # remise en jeu (après but ou sortie) : pas de décision
    ENDED = "ended"       # match terminé (60 s écoulées)


class EventType(Enum):
    """Événements ponctuels. Le moteur en émet UN SEUL par occurrence réelle."""
    GOAL = "goal"                              # player = marqueur
    STEAL = "steal"                            # player = voleur, other = victime
    LOOSE_BALL_RECOVERY = "loose_ball_recovery"  # player = celui qui reprend un ballon libre
    OUT_OF_BOUNDS = "out_of_bounds"            # player = responsable (dernier toucheur)
    KICK = "kick"                              # player = tireur, value = puissance du tir [0,1]


@dataclass(frozen=True)
class Event:
    """Un événement daté. `tick` est le tick physique où il s'est produit."""
    type: EventType
    tick: int
    player: int                  # joueur principal (voir EventType pour le sens exact)
    other: Optional[int] = None  # joueur secondaire (ex. victime d'un vol)
    value: float = 0.0           # valeur annexe (ex. puissance d'un tir)


# ---------------------------------------------------------------------------
# 2. SNAPSHOTS D'ÉTAT (lecture seule)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerState:
    x: float
    y: float
    vx: float
    vy: float
    facing: float   # radians, repère monde
    stamina: float  # [0, 100]


@dataclass(frozen=True)
class BallState:
    x: float
    y: float
    vx: float       # le moteur peut mettre 0 quand le ballon est attaché ; le wrapper ne s'y fie pas
    vy: float


@dataclass(frozen=True)
class MatchState:
    """Photographie complète du match à un instant donné."""
    tick: int                                   # horloge unique : ticks physiques écoulés
    players: tuple[PlayerState, PlayerState]
    ball: BallState
    possession: Optional[int]                   # None (libre), 0 ou 1 : UN SEUL propriétaire
    last_toucher: Optional[int]                 # dernier joueur ayant touché le ballon
    score: tuple[int, int]                      # (buts du joueur 0, buts du joueur 1)
    attack_dirs: tuple[int, int]                # (+1, -1) ou (-1, +1)
    phase: Phase
    contest_progress: tuple[float, float]       # secondes de contact accumulées par le joueur i qui tente un vol
    steal_cooldown: tuple[float, float]         # secondes restantes avant que le joueur i puisse revoler

    @property
    def ticks_remaining(self) -> int:
        return max(0, MAX_PHYSICS_TICKS - self.tick)

    @property
    def time_remaining_frac(self) -> float:
        """Fraction de temps restant dans [0, 1] (1 = début du match)."""
        return self.ticks_remaining / MAX_PHYSICS_TICKS

    @property
    def is_over(self) -> bool:
        return self.phase is Phase.ENDED or self.tick >= MAX_PHYSICS_TICKS


@dataclass(frozen=True)
class StepResult:
    """Ce que le moteur renvoie après avoir avancé de `ticks_advanced` ticks."""
    state: MatchState
    events: tuple[Event, ...]   # TOUS les événements survenus pendant ces ticks, dans l'ordre
    ticks_advanced: int         # peut être < n_ticks si le match se termine en cours de route


# ---------------------------------------------------------------------------
# 3. INTERFACE DU MOTEUR (à implémenter par le Member A)
# ---------------------------------------------------------------------------


@runtime_checkable
class GameEngine(Protocol):
    """
    Ce que le wrapper attend du moteur.

    Règles du contrat :
      1. reset() est DÉTERMINISTE : même seed + mêmes attack_dirs => même état initial.
      2. step() avance de n_ticks ticks physiques en TENANT les deux actions constantes.
      3. Les phases RESTART sont résolues À L'INTÉRIEUR de step() : l'état renvoyé est PLAYING ou ENDED.
         (Hypothèse par défaut : la politique ne décide pas pendant une remise en jeu.)
      4. Chaque événement est émis UNE SEULE fois, avec son tick exact (même s'il survient au tick 3 sur 6).
      5. Le moteur ne dépend jamais de Pygame ni des récompenses.
    """

    def reset(
        self,
        seed: Optional[int] = None,
        attack_dirs: tuple[int, int] = (1, -1),
    ) -> MatchState:
        ...

    def step(
        self,
        actions: tuple[DecodedAction, DecodedAction],
        n_ticks: int = FRAMES_PER_DECISION,
    ) -> StepResult:
        ...


# ---------------------------------------------------------------------------
# 4. VALIDATIONS (utilisées par les tests et, en mode debug, par le wrapper)
# ---------------------------------------------------------------------------


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def validate_state(s: MatchState) -> None:
    """Lève ValueError si le snapshot est incohérent."""
    if s.tick < 0:
        raise ValueError(f"tick négatif : {s.tick}")
    for i, p in enumerate(s.players):
        if not _finite(p.x, p.y, p.vx, p.vy, p.facing, p.stamina):
            raise ValueError(f"joueur {i} : valeur non finie")
        if not (0.0 <= p.x <= PITCH_W and 0.0 <= p.y <= PITCH_H):
            raise ValueError(f"joueur {i} hors du terrain : ({p.x}, {p.y})")
        if not (0.0 <= p.stamina <= 100.0):
            raise ValueError(f"joueur {i} : stamina hors [0,100] : {p.stamina}")
    if not _finite(s.ball.x, s.ball.y, s.ball.vx, s.ball.vy):
        raise ValueError("ballon : valeur non finie")
    if s.possession not in (None, 0, 1):
        raise ValueError(f"possession invalide : {s.possession}")
    if s.last_toucher not in (None, 0, 1):
        raise ValueError(f"last_toucher invalide : {s.last_toucher}")
    if s.attack_dirs not in ((1, -1), (-1, 1)):
        raise ValueError(f"attack_dirs invalide : {s.attack_dirs}")
    if min(s.score) < 0:
        raise ValueError(f"score négatif : {s.score}")
    if min(s.contest_progress) < 0 or min(s.steal_cooldown) < 0:
        raise ValueError("minuteur de duel/cooldown négatif")


def validate_step_result(prev: MatchState, result: StepResult) -> None:
    """
    Vérifie la cohérence entre deux états successifs (protège contre les récompenses en double) :
      * l'horloge avance exactement de ticks_advanced,
      * chaque événement est daté dans la fenêtre (prev.tick, new.tick],
      * les événements sont triés par tick,
      * le nombre d'événements GOAL de chaque joueur = variation de son score.
    """
    new = result.state
    validate_state(new)

    if new.tick != prev.tick + result.ticks_advanced:
        raise ValueError(
            f"horloge incohérente : {prev.tick} + {result.ticks_advanced} != {new.tick}"
        )

    last_tick = prev.tick
    for e in result.events:
        if not (prev.tick < e.tick <= new.tick):
            raise ValueError(f"événement hors fenêtre : {e}")
        if e.tick < last_tick:
            raise ValueError("événements non triés par tick")
        last_tick = e.tick
        if e.player not in (0, 1):
            raise ValueError(f"événement avec joueur invalide : {e}")

    for i in (0, 1):
        goals = sum(1 for e in result.events if e.type is EventType.GOAL and e.player == i)
        if goals != new.score[i] - prev.score[i]:
            raise ValueError(
                f"joueur {i} : {goals} événements GOAL mais le score a varié de "
                f"{new.score[i] - prev.score[i]}"
            )
