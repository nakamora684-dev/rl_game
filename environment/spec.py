"""
environment/spec.py  --  Spécification du MDP (Member C, phase P1)

Ce fichier est la SOURCE DE VÉRITÉ pour :
  * les constantes partagées (terrain, horloge),
  * l'espace d'actions (5 valeurs continues),
  * l'espace d'observation (vecteur plat normalisé, orienté "direction d'attaque").

Il ne contient AUCUNE physique et ne dépend ni de Pygame ni du moteur.
Les valeurs marquées [A?] sont des placeholders à confirmer avec le Member A.
Les formes (OBS_DIM, ACTION_DIM) sont à valider avec le Member D avant la P4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# 1. CONSTANTES PARTAGÉES (section 2 et 3 du plan)
# ---------------------------------------------------------------------------
PITCH_W = 800.0            # largeur logique du terrain (axe x, entre les buts)
PITCH_H = 500.0            # hauteur logique du terrain (axe y)
GOAL_W = 120.0             # ouverture du but, centrée sur les petits côtés

PHYSICS_HZ = 60                                   # fréquence de la physique
DECISION_HZ = 10                                  # fréquence de décision de la politique
FRAMES_PER_DECISION = PHYSICS_HZ // DECISION_HZ   # = 6 ticks physiques par action
MATCH_SECONDS = 60                                # durée d'un match simulé
MAX_PHYSICS_TICKS = MATCH_SECONDS * PHYSICS_HZ    # = 3600
MAX_DECISIONS = MAX_PHYSICS_TICKS // FRAMES_PER_DECISION  # = 600 pas max par épisode

# [A?] Bornes de normalisation. Le Member A fixera les vraies valeurs ;
# on s'en sert seulement pour ramener les vitesses dans [-1, 1].
MAX_PLAYER_SPEED = 200.0   # unités/seconde (placeholder)
MAX_BALL_SPEED = 600.0     # unités/seconde (placeholder)
CONTEST_MAX_TIME = 0.5     # secondes de contact nécessaires pour un vol (placeholder)
STEAL_COOLDOWN_MAX = 1.0   # secondes de délai après un vol (placeholder)
MAX_SCORE_DIFF = 3         # écart de score clippé dans l'observation

PITCH_DIAG = float(np.hypot(PITCH_W, PITCH_H))    # sert à normaliser les distances

# ---------------------------------------------------------------------------
# 2. ESPACE D'ACTIONS  (5 valeurs continues, section 2 / 4.2 du plan)
# ---------------------------------------------------------------------------
# DÉCISION DE DESIGN : la politique PPO voit un Box [-1, 1]^5 (symétrique).
# C'est la recommandation usuelle pour PPO avec une politique gaussienne.
# C'est le wrapper (decode_action) qui remet chaque valeur dans sa plage
# "physique" : effort, kick_trigger et kick_power vivent dans [0, 1].
ACTION_NAMES = ("move_x", "move_y", "effort", "kick_trigger", "kick_power")
ACTION_DIM = len(ACTION_NAMES)  # = 5

# DÉCISION DE DESIGN : le tir est déclenché si trigger > 0.5 (soit brut > 0).
# Le tir n'a d'effet que si le joueur contrôle le ballon (à confirmer avec A).
KICK_THRESHOLD = 0.5


@dataclass(frozen=True)
class DecodedAction:
    """Action légale, exprimée dans le REPÈRE DE LA POLITIQUE (attaque vers +x)."""
    move_x: float       # [-1, 1]
    move_y: float       # [-1, 1]  (norme du vecteur de déplacement <= 1)
    effort: float       # [0, 1]   0 = repos/récupération, 1 = sprint
    kick: bool          # True si le tir est déclenché ce pas
    kick_power: float   # [0, 1]   utilisé seulement si kick=True


def decode_action(raw) -> DecodedAction:
    """Convertit la sortie brute de la politique ([-1,1]^5) en action légale."""
    a = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)  # sécurité : clipping
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"Action attendue de forme ({ACTION_DIM},), reçu {a.shape}")

    # Déplacement : on limite la norme à 1 pour qu'aller en diagonale
    # ne soit pas plus rapide (sqrt(2)) qu'aller tout droit.
    move = a[:2].copy()
    norm = float(np.hypot(move[0], move[1]))
    if norm > 1.0:
        move /= norm

    # Remise à l'échelle [-1, 1] -> [0, 1] pour effort / trigger / power.
    effort = float((a[2] + 1.0) / 2.0)
    trigger = float((a[3] + 1.0) / 2.0)
    power = float((a[4] + 1.0) / 2.0)

    return DecodedAction(
        move_x=float(move[0]),
        move_y=float(move[1]),
        effort=effort,
        kick=trigger > KICK_THRESHOLD,
        kick_power=power,
    )


def policy_to_world(action: DecodedAction, attack_dir: int) -> DecodedAction:
    """
    Repasse l'action du repère "attaque vers +x" au repère réel du terrain.

    attack_dir = +1 : le joueur attaque vers +x  -> rien à changer.
    attack_dir = -1 : le joueur attaque vers -x  -> on inverse move_x.
    (Indispensable car l'observation est miroir-normalisée, voir plus bas.)
    """
    if attack_dir not in (1, -1):
        raise ValueError("attack_dir doit valoir +1 ou -1")
    return DecodedAction(
        move_x=action.move_x * attack_dir,
        move_y=action.move_y,
        effort=action.effort,
        kick=action.kick,
        kick_power=action.kick_power,
    )


# ---------------------------------------------------------------------------
# 3. ESPACE D'OBSERVATION  (section 4.1 du plan)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ObsFeature:
    name: str
    low: float
    high: float
    group: str


def _f(group: str, names: list[str], low: float, high: float) -> list[ObsFeature]:
    """Petit utilitaire pour déclarer plusieurs features de même plage."""
    return [ObsFeature(n, low, high, group) for n in names]


# L'ORDRE ci-dessous EST le contrat avec le Member D (ne pas le changer sans prévenir).
OBS_FEATURES: tuple[ObsFeature, ...] = tuple(
    # --- Joueur contrôlé (position absolue, repère attaque->+x) ---------------
    _f("self", ["self_x", "self_y"], -1.0, 1.0)             # (pos - centre) / demi-terrain
    + _f("self", ["self_vx", "self_vy"], -1.0, 1.0)         # vitesse / MAX_PLAYER_SPEED
    + _f("self", ["self_face_sin", "self_face_cos"], -1.0, 1.0)
    + _f("self", ["self_stamina"], 0.0, 1.0)                # stamina / 100
    # --- Adversaire (relatif au joueur contrôlé) ------------------------------
    + _f("opp", ["opp_rel_x", "opp_rel_y"], -1.0, 1.0)      # (opp - self) / dimensions terrain
    + _f("opp", ["opp_vx", "opp_vy"], -1.0, 1.0)
    + _f("opp", ["opp_face_sin", "opp_face_cos"], -1.0, 1.0)
    + _f("opp", ["opp_stamina"], 0.0, 1.0)
    # --- Ballon ---------------------------------------------------------------
    + _f("ball", ["ball_rel_x", "ball_rel_y"], -1.0, 1.0)
    + _f("ball", ["ball_vx", "ball_vy"], -1.0, 1.0)         # = 0 quand le ballon est attaché
    + _f("ball", ["poss_self", "poss_opp", "poss_free"], 0.0, 1.0)  # one-hot (3 catégories)
    # --- Géométrie tactique : direction unitaire (dx, dy) + distance ----------
    + _f("geom", ["to_ball_dx", "to_ball_dy"], -1.0, 1.0)
    + _f("geom", ["to_ball_dist"], 0.0, 1.0)                # distance / PITCH_DIAG
    + _f("geom", ["to_opp_dx", "to_opp_dy"], -1.0, 1.0)
    + _f("geom", ["to_opp_dist"], 0.0, 1.0)
    + _f("geom", ["to_own_goal_dx", "to_own_goal_dy"], -1.0, 1.0)
    + _f("geom", ["to_own_goal_dist"], 0.0, 1.0)
    + _f("geom", ["to_opp_goal_dx", "to_opp_goal_dy"], -1.0, 1.0)
    + _f("geom", ["to_opp_goal_dist"], 0.0, 1.0)
    # --- Contexte du match ----------------------------------------------------
    + _f("context", ["time_remaining"], 0.0, 1.0)           # ticks restants / 3600
    + _f("context", ["score_diff"], -1.0, 1.0)              # (mes buts - buts adv.) / MAX_SCORE_DIFF, clippé
    # --- État d'interaction (vol de balle) ------------------------------------
    + _f("interaction", ["contest_active"], 0.0, 1.0)       # 1 si un duel de vol est en cours
    + _f("interaction", ["contest_progress"], 0.0, 1.0)     # temps de contact / CONTEST_MAX_TIME
    + _f("interaction", ["steal_cooldown"], 0.0, 1.0)       # délai restant / STEAL_COOLDOWN_MAX
)

OBS_DIM = len(OBS_FEATURES)  # = 38
OBS_NAMES = tuple(f.name for f in OBS_FEATURES)
OBS_INDEX = {name: i for i, name in enumerate(OBS_NAMES)}  # accès par nom (tests / debug)

# Garde-fou : deux features ne doivent jamais porter le même nom.
assert len(set(OBS_NAMES)) == OBS_DIM, "Noms de features dupliqués dans OBS_FEATURES"


def obs_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Bornes basse/haute de l'observation (float32)."""
    low = np.array([f.low for f in OBS_FEATURES], dtype=np.float32)
    high = np.array([f.high for f in OBS_FEATURES], dtype=np.float32)
    return low, high


# ---------------------------------------------------------------------------
# 4. NORMALISATION / SYMÉTRIE  (attack-normalization, section 2 "Symmetry")
# ---------------------------------------------------------------------------
# CONVENTION : on miroite l'axe x quand le joueur attaque vers -x.
#   x' = PITCH_W - x    et    angle de regard θ' = π - θ    et    vx' = -vx
# Ainsi la politique voit toujours "mon but adverse est à droite (+x)".
def to_policy_x(x: float, attack_dir: int) -> float:
    """Position x (monde) -> position x dans le repère de la politique."""
    return x if attack_dir == 1 else PITCH_W - x


def to_policy_angle(theta: float, attack_dir: int) -> float:
    """Angle de regard (monde) -> angle dans le repère de la politique."""
    return theta if attack_dir == 1 else float(np.pi - theta)


# ---------------------------------------------------------------------------
# 5. CONSTRUCTION DES ESPACES GYMNASIUM
# ---------------------------------------------------------------------------
# Import paresseux : ce fichier reste importable (et testable) même sans gymnasium.
def make_action_space():
    from gymnasium import spaces
    return spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)


def make_observation_space():
    from gymnasium import spaces
    low, high = obs_bounds()
    return spaces.Box(low=low, high=high, dtype=np.float32)


if __name__ == "__main__":
    # Affichage lisible du contrat pour les revues avec A et D.
    print(f"ACTION_DIM = {ACTION_DIM} : {ACTION_NAMES}")
    print(f"OBS_DIM    = {OBS_DIM}\n")
    for i, f in enumerate(OBS_FEATURES):
        print(f"{i:2d}  {f.group:<12s} {f.name:<20s} [{f.low:+.0f}, {f.high:+.0f}]")
