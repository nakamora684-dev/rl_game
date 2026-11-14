"""
tests/test_reward.py -- Vérifie environment/reward.py (événements, résultat final, shaping, anti-doublon).
Lancer :  python tests/test_reward.py     (ou : python -m pytest tests/test_reward.py)
"""
import os
import random
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from environment import spec
from environment.contract import BallState, Event, EventType, Phase, StepResult
from environment.reward import COMPONENT_NAMES, RewardCalculator, RewardConfig, potential
from fake_engine import FakeEngine

CFG = RewardConfig()


def base():
    return FakeEngine().reset()


def result_after(prev, events=(), **changes):
    """Fabrique un StepResult de 6 ticks à partir de prev, avec les changements d'état voulus."""
    new = replace(prev, tick=prev.tick + 6, **changes)
    return StepResult(state=new, events=tuple(events), ticks_advanced=6)


def reward(prev, res, idx=0, cfg=CFG):
    return RewardCalculator(cfg).compute(prev, res, idx)


def test_components_have_fixed_names():
    _, comp = reward(base(), result_after(base()))
    assert tuple(comp) == COMPONENT_NAMES


def test_quiet_step_gives_zero_reward():
    total, comp = reward(base(), result_after(base()))
    assert total == 0.0 and all(v == 0.0 for v in comp.values())


def test_goal_for_and_against():
    p = base()
    goal0 = Event(EventType.GOAL, p.tick + 3, player=0)
    r = result_after(p, [goal0], score=(1, 0))
    assert reward(p, r, idx=0)[1]["goal_for"] == CFG.goal_for          # +10
    assert reward(p, r, idx=1)[1]["goal_against"] == CFG.goal_against  # -10 pour l'autre joueur


def test_steal_and_loose_ball_recovery_and_steal_lost():
    p = base()
    steal = Event(EventType.STEAL, p.tick + 2, player=0, other=1)
    r = result_after(p, [steal], possession=0)
    assert reward(p, r, idx=0)[1]["steal"] == CFG.steal                # +1 pour le voleur
    assert reward(p, r, idx=1)[1]["steal_lost"] == CFG.steal_lost      # la victime est pénalisée une fois
    assert reward(p, r, idx=1)[1]["steal"] == 0.0                      # et ne reçoit PAS le +1

    rec = Event(EventType.LOOSE_BALL_RECOVERY, p.tick + 2, player=1)
    r2 = result_after(p, [rec], possession=1)
    assert reward(p, r2, idx=1)[1]["loose_recovery"] == CFG.loose_recovery
    assert reward(p, r2, idx=0)[1]["loose_recovery"] == 0.0


def test_out_of_bounds_penalizes_only_the_responsible_player():
    p = base()
    r = result_after(p, [Event(EventType.OUT_OF_BOUNDS, p.tick + 1, player=0)])
    assert reward(p, r, idx=0)[1]["out_of_bounds"] == CFG.out_of_bounds
    assert reward(p, r, idx=1)[1]["out_of_bounds"] == 0.0


def test_deliberate_kick_is_not_penalized():
    p = base()
    p = replace(p, possession=0)
    kick = Event(EventType.KICK, p.tick + 1, player=0, value=0.8)
    r = result_after(p, [kick], possession=None)       # le tir libère le ballon
    total, comp = reward(p, r, idx=0)
    assert total == 0.0 and comp["steal_lost"] == 0.0 and comp["out_of_bounds"] == 0.0


def test_final_outcome_win_draw_loss_exactly_once():
    p = replace(base(), tick=spec.MAX_PHYSICS_TICKS - 6)
    for score, expected in (((1, 0), CFG.win), ((0, 0), CFG.draw), ((0, 2), CFG.loss)):
        r = result_after(p, phase=Phase.ENDED, score=score)
        assert reward(p, r, idx=0)[1]["outcome"] == expected
    # Après la fin, un pas supplémentaire ne redonne pas le résultat.
    ended = replace(p, tick=spec.MAX_PHYSICS_TICKS, phase=Phase.ENDED, score=(1, 0))
    r_after = StepResult(state=ended, events=(), ticks_advanced=0)
    assert reward(ended, r_after, idx=0)[1]["outcome"] == 0.0


def test_duplicate_event_is_rejected():
    p = base()
    calc = RewardCalculator(CFG)
    goal = Event(EventType.GOAL, p.tick + 3, player=0)
    r = result_after(p, [goal], score=(1, 0))
    calc.compute(p, r, 0)
    try:
        calc.compute(p, r, 0)          # même événement présenté deux fois
    except RuntimeError:
        calc.reset()
        calc.compute(p, r, 0)          # après reset (nouvel épisode) : accepté
        return
    raise AssertionError("un événement dupliqué aurait dû être refusé")


def test_potential_is_attack_normalized():
    s = replace(base(), ball=BallState(600.0, 250.0, 0.0, 0.0))
    assert potential(s, 0) == 0.75        # le joueur 0 attaque +x : ballon à 600/800
    assert potential(s, 1) == 0.25        # le joueur 1 attaque -x : même ballon, potentiel opposé


def test_shaping_progress_and_symmetry():
    p = base()
    r = result_after(p, ball=BallState(600.0, 250.0, 0.0, 0.0))
    assert reward(p, r, idx=0)[1]["progress_shaping"] == CFG.shaping_lambda * 0.25    # +0.5
    assert reward(p, r, idx=1)[1]["progress_shaping"] == -CFG.shaping_lambda * 0.25   # -0.5


def test_shaping_telescopes_so_back_and_forth_farms_nothing():
    # Sonde anti-"reward hacking" n°2 : faire des allers-retours avec le ballon ne doit rien rapporter.
    rng = random.Random(0)
    calc = RewardCalculator(RewardConfig(shaping_gamma=1.0))
    state = base()
    start_phi = potential(state, 0)
    total_shaping = 0.0
    for _ in range(200):
        new_ball = BallState(rng.uniform(0, spec.PITCH_W), rng.uniform(0, spec.PITCH_H), 0.0, 0.0)
        r = result_after(state, ball=new_ball)
        _, comp = calc.compute(state, r, 0)
        total_shaping += comp["progress_shaping"]
        state = r.state
    # Somme télescopique : elle ne dépend que du point de départ et d'arrivée.
    assert abs(total_shaping - CFG.shaping_lambda * (potential(state, 0) - start_phi)) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
