"""
environment/termination.py -- Quand et pourquoi un épisode s'arrête.

DÉCISIONS (à valider avec le Member D) :
  * Le match se termine à 60 s simulées => `terminated = True` (fin naturelle du MDP).
    Justification : time_remaining fait partie de l'observation, donc la fin du match est prévisible
    par la politique ; ce n'est pas une coupure arbitraire (truncation).
  * Un but NE termine PAS l'épisode : le moteur fait une remise en jeu (section 4.2 du plan).
  * `truncated = True` seulement comme filet de sécurité si le moteur ne termine jamais le match
    (plus de MAX_DECISIONS pas).
"""
from __future__ import annotations

from enum import Enum

from environment.contract import MatchState
from environment.spec import MAX_DECISIONS


class EndReason(Enum):
    NONE = "none"                        # épisode en cours
    TIME_UP = "time_up"                  # coup de sifflet final
    SAFETY_TRUNCATION = "safety_truncation"  # filet de sécurité (bug du moteur probable)


def check_termination(state: MatchState, step_count: int) -> tuple[bool, bool, EndReason]:
    """Retourne (terminated, truncated, raison)."""
    if state.is_over:
        return True, False, EndReason.TIME_UP
    if step_count > MAX_DECISIONS:
        return False, True, EndReason.SAFETY_TRUNCATION
    return False, False, EndReason.NONE


def match_result(score: tuple[int, int], player_idx: int) -> str:
    """'win', 'draw' ou 'loss' du point de vue du joueur `player_idx`."""
    mine, theirs = score[player_idx], score[1 - player_idx]
    if mine > theirs:
        return "win"
    if mine < theirs:
        return "loss"
    return "draw"
