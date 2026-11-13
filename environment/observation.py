"""
environment/observation.py -- Construit l'observation PPO (vecteur de 38 valeurs) à partir d'un MatchState.

Principes (section 4.1 du plan) :
  * vecteur plat, normalisé, orienté "direction d'attaque" : la politique croit toujours attaquer vers +x ;
  * l'ORDRE des cases est celui de spec.OBS_FEATURES (contrat avec le Member D) ;
  * aucune information de "bookkeeping" de récompense n'est exposée (dernier toucheur, etc.).

Convention du ballon attaché : si quelqu'un contrôle le ballon (possession != None), ball_vx = ball_vy = 0.
"""
from __future__ import annotations

import math

import numpy as np

from environment.contract import MatchState
from environment.spec import (
    CONTEST_MAX_TIME,
    MAX_BALL_SPEED,
    MAX_PLAYER_SPEED,
    MAX_SCORE_DIFF,
    OBS_DIM,
    OBS_NAMES,
    PITCH_DIAG,
    PITCH_H,
    PITCH_W,
    STEAL_COOLDOWN_MAX,
    obs_bounds,
    to_policy_angle,
    to_policy_x,
)

STAMINA_MAX = 100.0
_LOW, _HIGH = obs_bounds()   # bornes déclarées dans spec.OBS_FEATURES


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _direction_and_distance(dx: float, dy: float) -> tuple[float, float, float]:
    """Vecteur (dx, dy) en unités logiques -> (direction unitaire x, direction unitaire y, distance normalisée [0,1])."""
    dist = math.hypot(dx, dy)
    if dist < 1e-6:                      # cible confondue avec le joueur : pas de direction définie
        return 0.0, 0.0, 0.0
    return dx / dist, dy / dist, _clip(dist / PITCH_DIAG, 0.0, 1.0)


def build_observation(state: MatchState, player_idx: int) -> np.ndarray:
    """
    Observation du joueur `player_idx` (0 ou 1) dans le repère "j'attaque vers +x".
    Retourne un tableau float32 de forme (OBS_DIM,).
    """
    if player_idx not in (0, 1):
        raise ValueError("player_idx doit valoir 0 ou 1")

    ad = state.attack_dirs[player_idx]        # +1 ou -1 : sens d'attaque de CE joueur
    me = state.players[player_idx]
    opp = state.players[1 - player_idx]
    half_w, half_h = PITCH_W / 2.0, PITCH_H / 2.0

    # Positions x dans le repère de la politique (miroir si ad == -1) ; y est inchangé.
    sx = to_policy_x(me.x, ad)
    ox = to_policy_x(opp.x, ad)
    bx = to_policy_x(state.ball.x, ad)

    v: dict[str, float] = {}

    # --- Joueur contrôlé ------------------------------------------------------
    v["self_x"] = (sx - half_w) / half_w
    v["self_y"] = (me.y - half_h) / half_h
    v["self_vx"] = _clip(me.vx * ad / MAX_PLAYER_SPEED, -1.0, 1.0)   # vitesse en x : signe miroité
    v["self_vy"] = _clip(me.vy / MAX_PLAYER_SPEED, -1.0, 1.0)
    th = to_policy_angle(me.facing, ad)
    v["self_face_sin"], v["self_face_cos"] = math.sin(th), math.cos(th)
    v["self_stamina"] = _clip(me.stamina / STAMINA_MAX, 0.0, 1.0)

    # --- Adversaire (relatif à moi) -------------------------------------------
    v["opp_rel_x"] = (ox - sx) / PITCH_W
    v["opp_rel_y"] = (opp.y - me.y) / PITCH_H
    v["opp_vx"] = _clip(opp.vx * ad / MAX_PLAYER_SPEED, -1.0, 1.0)
    v["opp_vy"] = _clip(opp.vy / MAX_PLAYER_SPEED, -1.0, 1.0)
    tho = to_policy_angle(opp.facing, ad)
    v["opp_face_sin"], v["opp_face_cos"] = math.sin(tho), math.cos(tho)
    v["opp_stamina"] = _clip(opp.stamina / STAMINA_MAX, 0.0, 1.0)

    # --- Ballon ---------------------------------------------------------------
    v["ball_rel_x"] = (bx - sx) / PITCH_W
    v["ball_rel_y"] = (state.ball.y - me.y) / PITCH_H
    attached = state.possession is not None
    v["ball_vx"] = 0.0 if attached else _clip(state.ball.vx * ad / MAX_BALL_SPEED, -1.0, 1.0)
    v["ball_vy"] = 0.0 if attached else _clip(state.ball.vy / MAX_BALL_SPEED, -1.0, 1.0)
    v["poss_self"] = 1.0 if state.possession == player_idx else 0.0
    v["poss_opp"] = 1.0 if state.possession == 1 - player_idx else 0.0
    v["poss_free"] = 1.0 if state.possession is None else 0.0        # one-hot : exactement un des trois vaut 1

    # --- Géométrie tactique : direction unitaire + distance vers 4 cibles -------
    own_goal = (0.0, half_h)           # dans le repère de la politique, mon but est à gauche...
    opp_goal = (PITCH_W, half_h)       # ...et le but adverse à droite.
    targets = {
        "ball": (bx, state.ball.y),
        "opp": (ox, opp.y),
        "own_goal": own_goal,
        "opp_goal": opp_goal,
    }
    for name, (tx, ty) in targets.items():
        dx, dy, dist = _direction_and_distance(tx - sx, ty - me.y)
        v[f"to_{name}_dx"], v[f"to_{name}_dy"], v[f"to_{name}_dist"] = dx, dy, dist

    # --- Contexte du match ----------------------------------------------------
    v["time_remaining"] = state.time_remaining_frac
    score_diff = state.score[player_idx] - state.score[1 - player_idx]
    v["score_diff"] = _clip(score_diff / MAX_SCORE_DIFF, -1.0, 1.0)

    # --- Interaction (vol de balle) --------------------------------------------
    # Un seul duel existe à la fois : on prend le max des deux progressions.
    # Savoir SI c'est moi qui vole ou qui suis volé se déduit de poss_self / poss_opp.
    progress = max(state.contest_progress)
    v["contest_active"] = 1.0 if progress > 0.0 else 0.0
    v["contest_progress"] = _clip(progress / CONTEST_MAX_TIME, 0.0, 1.0)
    v["steal_cooldown"] = _clip(state.steal_cooldown[player_idx] / STEAL_COOLDOWN_MAX, 0.0, 1.0)

    # --- Assemblage dans l'ordre EXACT de spec.OBS_FEATURES ---------------------
    if set(v) != set(OBS_NAMES):
        raise AssertionError(f"features incohérentes avec spec : écart = {set(v) ^ set(OBS_NAMES)}")
    vec = np.array([v[name] for name in OBS_NAMES], dtype=np.float32)
    if vec.shape != (OBS_DIM,) or not np.all(np.isfinite(vec)):
        raise ValueError("observation mal formée (forme ou valeur non finie)")
    return np.clip(vec, _LOW, _HIGH)


def describe_observation(obs: np.ndarray) -> dict[str, float]:
    """Associe chaque nom de feature à sa valeur (pratique pour déboguer et afficher)."""
    return {name: float(obs[i]) for i, name in enumerate(OBS_NAMES)}
