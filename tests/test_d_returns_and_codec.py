"""Member D: torch-free pieces (GAE, reward scaling, action codec, opponents, stats)."""

import numpy as np
import pytest

from agents.action_codec import FLAT_ACTION_DIM, random_flat_action, to_env_action
from agents.opponents import ChaserOpponent, direction_towards, make_opponent
from evaluation.metrics import MATCH_COLUMNS, bootstrap_ci, wilson_interval
from game.actions import Direction
from training.returns import RunningMeanStd, compute_gae


def _brute_force_gae(r, v, starts, nv, ns, g, lam, n):
    T = r.shape[0]
    out = np.zeros(T)
    for t in range(T):
        total, coef = 0.0, 1.0
        for k in range(t, T):
            next_start = ns[n] if k == T - 1 else starts[k + 1, n]
            next_val = nv[n] if k == T - 1 else v[k + 1, n]
            non_terminal = 1.0 - next_start
            total += coef * (r[k, n] + g * next_val * non_terminal - v[k, n])
            if non_terminal == 0:
                break
            coef *= g * lam
        out[t] = total
    return out


def test_gae_matches_brute_force_with_episode_boundaries():
    rng = np.random.default_rng(3)
    T, N = 30, 4
    r, v = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    starts = (rng.random((T, N)) < 0.15).astype(float)
    nv, ns = rng.normal(size=N), (rng.random(N) < 0.3).astype(float)
    adv, ret = compute_gae(r, v, starts, nv, ns, 0.97, 0.9)
    for n in range(N):
        assert np.allclose(adv[:, n], _brute_force_gae(r, v, starts, nv, ns, 0.97, 0.9, n))
    assert np.allclose(ret, adv + v)


def test_running_mean_std_matches_numpy():
    x = np.random.default_rng(0).normal(3.0, 2.0, size=4000)
    rms = RunningMeanStd()
    for chunk in np.array_split(x, 40):
        rms.update(chunk)
    assert rms.mean == pytest.approx(x.mean(), rel=1e-3)
    assert rms.var == pytest.approx(x.var(), rel=1e-3)


def test_to_env_action_produces_legal_hybrid_dict():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = to_env_action(random_flat_action(rng))
        assert set(a) == {"direction", "effort", "kick_trigger", "kick_power"}
        assert isinstance(a["direction"], int) and 0 <= a["direction"] <= 8
        assert a["effort"].shape == (1,) and a["effort"].dtype == np.float32
        assert a["kick_trigger"] in (0, 1) and 0.0 <= float(a["kick_power"][0]) <= 1.0


def test_to_env_action_rejects_bad_input():
    with pytest.raises(ValueError):
        to_env_action([1.0, 0.5, 0.0])
    with pytest.raises(ValueError):
        to_env_action([1.0, float("nan"), 0.0, 0.0])
    assert FLAT_ACTION_DIM == 4


def test_direction_towards_covers_eight_sectors():
    assert direction_towards(1, 0) is Direction.EAST
    assert direction_towards(-1, 0) is Direction.WEST
    assert direction_towards(0, -1) is Direction.NORTH
    assert direction_towards(0, 1) is Direction.SOUTH
    assert direction_towards(1, -1) is Direction.NORTHEAST
    assert direction_towards(-1, 1) is Direction.SOUTHWEST
    assert direction_towards(0, 0) is Direction.IDLE


def test_opponent_registry():
    assert isinstance(make_opponent("chaser"), ChaserOpponent)
    assert callable(make_opponent("idle"))
    with pytest.raises(ValueError):
        make_opponent("nope")


def test_wilson_and_bootstrap_intervals():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and 0.39 < lo < 0.41 and 0.59 < hi < 0.61
    assert wilson_interval(0, 0)[0] != wilson_interval(0, 0)[0]  # NaN for n=0
    lo, hi = bootstrap_ci([1.0] * 20)
    assert lo == hi == 1.0


def test_schema_has_no_duplicate_columns():
    assert len(MATCH_COLUMNS) == len(set(MATCH_COLUMNS))
