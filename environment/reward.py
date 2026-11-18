"""
environment/reward.py -- Récompenses du point de vue d'un joueur (section 5 du plan).

IMPORTANT : ce sont des HYPOTHÈSES de départ ("experiment 0"), pas des garanties.
  * Les valeurs marquées [PLAN] viennent du plan.
  * Les valeurs marquées [TUNE] sont des placeholders : le plan dit "exact value to tune".
    Elles seront ajustées en P5/P6 avec des preuves (taux de victoire, différence de buts).

Principes :
  * Récompenses basées sur des ÉVÉNEMENTS ponctuels (jamais sur un drapeau persistant) ;
  * garde anti-doublon : un même événement ne peut pas être récompensé deux fois ;
  * shaping par POTENTIEL (Φ = position x du ballon dans le repère d'attaque, dans [0,1]) ;
  * chaque composante est renvoyée séparément pour TensorBoard et pour les sondes anti-"reward hacking".
"""

from __future__ import annotations

from dataclasses import dataclass

from environment.contract import Event, EventType, MatchState, StepResult
from environment.spec import to_policy_x
from environment.termination import match_result


@dataclass(frozen=True)
class RewardConfig:
    # --- Événements de but et de résultat ------------------------------------
    goal_for: float = 10.0  # [PLAN] but marqué
    goal_against: float = -10.0  # [PLAN] but encaissé
    win: float = 5.0  # [PLAN] victoire au coup de sifflet final
    draw: float = 0.0  # [PLAN] nul
    loss: float = -5.0  # [PLAN] défaite
    # --- Transitions de balle -------------------------------------------------
    steal: float = 1.0  # [PLAN] vol réussi (adversaire -> moi)
    loose_recovery: float = 0.5  # [PLAN] reprise d'un ballon libre
    steal_lost: float = -0.5  # [TUNE] je me fais voler le ballon
    out_of_bounds: float = -0.5  # [TUNE] je suis responsable d'une sortie de terrain
    possession_per_step: float = 0.0  # [PLAN] 0 au départ, pour éviter de récompenser le "stalling"
    # --- Shaping par potentiel ------------------------------------------------
    shaping_lambda: float = 2.0  # [PLAN] λ initial, valeur de test
    # γ du shaping. 1.0 => F = λ(Φ' - Φ) : somme télescopique, retour non actualisé lisible.
    # Pour une invariance stricte de la politique, l'aligner sur le γ de PPO (à discuter avec le Member D).
    shaping_gamma: float = 1.0
    # Si True, Φ(état terminal) = 0 (convention théorique de Ng et al.). False par défaut : plus lisible.
    zero_terminal_potential: bool = False


# Ordre fixe des composantes (utile pour les logs et les tests).
COMPONENT_NAMES: tuple[str, ...] = (
    "goal_for",
    "goal_against",
    "outcome",
    "steal",
    "steal_lost",
    "loose_recovery",
    "out_of_bounds",
    "possession",
    "progress_shaping",
)


def potential(state: MatchState, player_idx: int) -> float:
    """Φ(s) in [0,1] : position x du ballon dans le repère d'attaque de ce joueur (1 = au but adverse)."""
    ad = state.attack_dirs[player_idx]
    width = state.config.pitch.width
    return max(0.0, min(1.0, to_policy_x(state.ball.x, ad, width) / width))


class RewardCalculator:
    """À instancier une fois par environnement ; appeler reset() au début de chaque épisode."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.cfg = config or RewardConfig()
        self._seen: set[int] = set()  # identifiants uniques dans l'épisode courant

    def reset(self) -> None:
        self._seen.clear()

    def compute(
        self, prev: MatchState, result: StepResult, player_idx: int
    ) -> tuple[float, dict[str, float]]:
        """
        Récompense du pas (prev -> result.state) pour le joueur `player_idx`.
        Retourne (récompense totale, dict des composantes).
        """
        cfg = self.cfg
        me = player_idx
        new = result.state
        comp = {name: 0.0 for name in COMPONENT_NAMES}

        # 1) Événements ponctuels, avec garde anti-doublon ------------------------
        for e in result.events:
            self._guard_once(e)
            if e.type is EventType.GOAL:
                comp["goal_for" if e.player == me else "goal_against"] += (
                    cfg.goal_for if e.player == me else cfg.goal_against
                )
            elif e.type is EventType.STEAL:
                if e.player == me:
                    comp["steal"] += cfg.steal
                elif e.other == me:
                    comp["steal_lost"] += cfg.steal_lost
            elif e.type is EventType.LOOSE_BALL_RECOVERY:
                if e.player == me:
                    comp["loose_recovery"] += cfg.loose_recovery
            elif e.type is EventType.OUT_OF_BOUNDS and e.player == me:
                comp["out_of_bounds"] += cfg.out_of_bounds
            # EventType.KICK : volontairement sans récompense ni pénalité (un tir délibéré n'est pas une perte).

        # 2) Résultat final : une seule fois, au pas où le match se termine ---------
        if new.is_over and not prev.is_over:
            outcome = match_result(new.score, me)
            comp["outcome"] = {"win": cfg.win, "draw": cfg.draw, "loss": cfg.loss}[outcome]

        # 3) Possession par pas (0 par défaut) -------------------------------------
        if new.possession == me:
            comp["possession"] = cfg.possession_per_step

        # 4) Shaping par potentiel : F = λ (γ Φ(s') - Φ(s)) -------------------------
        phi_prev = potential(prev, me)
        phi_new = 0.0 if (new.is_over and cfg.zero_terminal_potential) else potential(new, me)
        comp["progress_shaping"] = cfg.shaping_lambda * (cfg.shaping_gamma * phi_new - phi_prev)

        return sum(comp.values()), comp

    def _guard_once(self, e: Event) -> None:
        """Lève une erreur si le même événement est présenté deux fois (bug moteur ou wrapper)."""
        key = e.event_id
        if key in self._seen:
            raise RuntimeError(f"Événement dupliqué (double récompense évitée) : {e}")
        self._seen.add(key)
