"""
tests/test_spec.py -- Vérifie la cohérence de environment/spec.py.
Lancer avec :  python -m pytest tests/test_spec.py    (ou simplement : python tests/test_spec.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # racine du dépôt

from environment import spec


def test_clock_constants():
    # 60 Hz / 10 Hz = 6 ticks par décision ; 60 s -> 3600 ticks -> 600 décisions.
    assert spec.FRAMES_PER_DECISION == 6
    assert spec.MAX_PHYSICS_TICKS == 3600
    assert spec.MAX_DECISIONS == 600


def test_dimensions():
    assert spec.ACTION_DIM == 5
    assert spec.OBS_DIM == 38
    low, high = spec.obs_bounds()
    assert low.shape == high.shape == (spec.OBS_DIM,)
    assert np.all(low < high)


def test_decode_action_ranges():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        a = spec.decode_action(rng.uniform(-3, 3, size=5))  # valeurs hors plage -> clippées
        assert np.hypot(a.move_x, a.move_y) <= 1.0 + 1e-6    # pas de diagonale plus rapide
        assert 0.0 <= a.effort <= 1.0
        assert 0.0 <= a.kick_power <= 1.0


def test_decode_action_extremes():
    a = spec.decode_action([0, 0, -1, -1, -1])   # tout au minimum
    assert a.effort == 0.0 and a.kick_power == 0.0 and a.kick is False
    b = spec.decode_action([1, 1, 1, 1, 1])      # tout au maximum
    assert b.effort == 1.0 and b.kick is True
    assert abs(np.hypot(b.move_x, b.move_y) - 1.0) < 1e-6


def test_policy_to_world_flips_x_only():
    a = spec.decode_action([0.5, 0.3, 0.0, 1.0, 0.0])
    w = spec.policy_to_world(a, attack_dir=-1)
    assert w.move_x == -a.move_x and w.move_y == a.move_y and w.kick == a.kick
    assert spec.policy_to_world(a, attack_dir=1) == a


def test_symmetry_helpers():
    # Un joueur à x=200 attaquant vers -x voit x=600 dans le repère de la politique.
    assert spec.to_policy_x(200.0, -1) == 600.0
    assert spec.to_policy_x(200.0, +1) == 200.0
    # Regard vers +x (θ=0) devient regard vers -x (θ=π) après miroir, et inversement.
    assert np.isclose(spec.to_policy_angle(0.0, -1), np.pi)


def test_gymnasium_spaces_if_available():
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        return  # gymnasium non installé : on saute ce test
    assert spec.make_action_space().shape == (5,)
    assert spec.make_observation_space().shape == (spec.OBS_DIM,)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
