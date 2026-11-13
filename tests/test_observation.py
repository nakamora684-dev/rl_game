"""
tests/test_observation.py -- Vérifie environment/observation.py.
Lancer :  python tests/test_observation.py     (ou : python -m pytest tests/test_observation.py)
"""
import math
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from environment import spec
from environment.contract import BallState, PlayerState
from environment.observation import build_observation, describe_observation
from fake_engine import FakeEngine

IDLE = spec.decode_action([0, 0, -1, -1, -1])


def initial_state():
    return FakeEngine().reset()


def mirror_state(s):
    """Miroir gauche/droite complet : x -> W-x, vx -> -vx, regard θ -> π-θ, côtés d'attaque inversés."""
    def mp(p):
        return replace(p, x=spec.PITCH_W - p.x, vx=-p.vx, facing=math.pi - p.facing)
    return replace(
        s,
        players=(mp(s.players[0]), mp(s.players[1])),
        ball=replace(s.ball, x=spec.PITCH_W - s.ball.x, vx=-s.ball.vx),
        attack_dirs=(-s.attack_dirs[0], -s.attack_dirs[1]),
    )


def busy_state():
    """Un état quelconque (non symétrique) pour les tests de miroir."""
    s = initial_state()
    return replace(
        s,
        tick=1200,
        players=(PlayerState(310.0, 120.0, 80.0, -30.0, 0.7, 64.0),
                 PlayerState(520.0, 380.0, -40.0, 60.0, 2.4, 91.0)),
        ball=BallState(450.0, 200.0, 150.0, -20.0),
        possession=None, last_toucher=1, score=(2, 1),
        contest_progress=(0.0, 0.2), steal_cooldown=(0.3, 0.0),
    )


def test_shape_dtype_finite_and_bounds():
    s = busy_state()
    low, high = spec.obs_bounds()
    for idx in (0, 1):
        o = build_observation(s, idx)
        assert o.shape == (spec.OBS_DIM,) and o.dtype == np.float32
        assert np.all(np.isfinite(o)) and np.all(o >= low) and np.all(o <= high)


def test_hand_computed_values_initial_state():
    d = describe_observation(build_observation(initial_state(), 0))
    assert d["self_x"] == -0.5 and d["self_y"] == 0.0                       # (200-400)/400
    assert math.isclose(d["opp_rel_x"], 0.5) and d["opp_rel_y"] == 0.0      # (600-200)/800
    assert math.isclose(d["ball_rel_x"], 0.25)                              # (400-200)/800
    assert d["self_face_cos"] == 1.0 and d["self_stamina"] == 1.0
    assert (d["poss_self"], d["poss_opp"], d["poss_free"]) == (0.0, 0.0, 1.0)
    assert d["time_remaining"] == 1.0 and d["score_diff"] == 0.0
    assert d["to_opp_goal_dx"] == 1.0 and d["to_own_goal_dx"] == -1.0       # but adverse à droite, mon but à gauche
    assert math.isclose(d["to_opp_goal_dist"], 600.0 / spec.PITCH_DIAG, rel_tol=1e-5)   # tolérance float32


def test_both_players_see_the_same_thing_at_kickoff():
    # Position initiale symétrique : joueur 0 (gauche, attaque +x) et joueur 1 (droite, attaque -x)
    # doivent recevoir exactement la même observation.
    s = initial_state()
    assert np.allclose(build_observation(s, 0), build_observation(s, 1), atol=1e-6)


def test_attack_direction_invariance_under_mirroring():
    # Même situation de jeu, vue de côtés opposés => observation identique.
    s = busy_state()
    sm = mirror_state(s)
    for idx in (0, 1):
        assert np.allclose(build_observation(s, idx), build_observation(sm, idx), atol=1e-5), idx


def test_possession_one_hot_and_attached_ball_velocity():
    s = busy_state()
    for owner, expected in ((None, (0, 0, 1)), (0, (1, 0, 0)), (1, (0, 1, 0))):
        d = describe_observation(build_observation(replace(s, possession=owner), 0))
        assert (d["poss_self"], d["poss_opp"], d["poss_free"]) == expected
        if owner is None:
            assert d["ball_vx"] != 0.0              # ballon libre : vitesse visible
        else:
            assert d["ball_vx"] == 0.0 and d["ball_vy"] == 0.0   # ballon attaché : vitesse masquée


def test_score_diff_sign_and_clipping():
    s = busy_state()                                   # score (2, 1)
    assert math.isclose(describe_observation(build_observation(s, 0))["score_diff"], 1 / 3, rel_tol=1e-5)
    assert math.isclose(describe_observation(build_observation(s, 1))["score_diff"], -1 / 3, rel_tol=1e-5)
    big = replace(s, score=(9, 0))
    assert describe_observation(build_observation(big, 0))["score_diff"] == 1.0   # clippé


def test_interaction_features():
    s = busy_state()      # contest_progress=(0, 0.2), steal_cooldown=(0.3, 0)
    d0 = describe_observation(build_observation(s, 0))
    d1 = describe_observation(build_observation(s, 1))
    assert d0["contest_active"] == 1.0 and d1["contest_active"] == 1.0          # duel visible des deux côtés
    assert math.isclose(d0["steal_cooldown"], 0.3 / spec.STEAL_COOLDOWN_MAX, rel_tol=1e-5)
    assert d1["steal_cooldown"] == 0.0                                          # chacun voit SON cooldown


def test_degenerate_target_on_player_gives_zero_direction():
    s = initial_state()
    s = replace(s, ball=BallState(s.players[0].x, s.players[0].y, 0.0, 0.0))    # ballon sur le joueur 0
    d = describe_observation(build_observation(s, 0))
    assert (d["to_ball_dx"], d["to_ball_dy"], d["to_ball_dist"]) == (0.0, 0.0, 0.0)


def test_invalid_player_idx():
    try:
        build_observation(initial_state(), 2)
    except ValueError:
        return
    raise AssertionError("player_idx=2 aurait dû être refusé")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
