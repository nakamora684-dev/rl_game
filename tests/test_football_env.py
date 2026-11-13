"""
tests/test_football_env.py -- Vérifie le wrapper FootballEnv avec le faux moteur.
Lancer :  python tests/test_football_env.py     (nécessite : pip install gymnasium)

Quand le vrai moteur sera prêt, remplacer FakeEngine dans make_env().
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import gymnasium  # noqa: F401
    HAVE_GYM = True
except ImportError:
    HAVE_GYM = False

from environment import spec
from environment.reward import RewardConfig
from fake_engine import FakeEngine

if HAVE_GYM:
    from environment.football_env import FootballEnv


def make_env(**kw):
    return FootballEnv(FakeEngine(), **kw)       # <- à remplacer par le vrai moteur du Member A


def run_episode(env, agent_idx, action=None):
    action = np.zeros(spec.ACTION_DIM, dtype=np.float32) if action is None else action
    obs, info = env.reset(seed=0, options={"agent_idx": agent_idx})
    total, steps, last = 0.0, 0, None
    while True:
        obs, r, terminated, truncated, last = env.step(action)
        total += r
        steps += 1
        assert obs.shape == (spec.OBS_DIM,) and np.all(np.isfinite(obs))
        if terminated or truncated:
            return total, steps, terminated, truncated, last


def test_reset_returns_valid_observation_and_info():
    if not HAVE_GYM:
        return
    env = make_env()
    obs, info = env.reset(seed=1, options={"agent_idx": 0})
    assert env.observation_space.contains(obs)
    assert info["agent_idx"] == 0 and info["tick"] == 0 and info["score"] == (0, 0)


def test_full_episode_is_600_steps_and_terminated():
    if not HAVE_GYM:
        return
    total, steps, terminated, truncated, info = run_episode(make_env(), agent_idx=0)
    assert steps == spec.MAX_DECISIONS            # 600 décisions
    assert terminated and not truncated           # fin naturelle du match
    assert info["end_reason"] == "time_up" and info["tick"] == spec.MAX_PHYSICS_TICKS


def test_reward_scripted_scenario_both_sides():
    if not HAVE_GYM:
        return
    cfg = RewardConfig()
    # Scénario du faux moteur : J0 reprend le ballon (t10), J1 le vole (t20), J1 marque (t30), score final 0-1.
    # Le ballon reste au centre : le shaping vaut 0.
    total0, *_, info0 = run_episode(make_env(reward_config=cfg), agent_idx=0)
    expected0 = cfg.loose_recovery + cfg.steal_lost + cfg.goal_against + cfg.loss   # 0.5 -0.5 -10 -5
    assert abs(total0 - expected0) < 1e-6, (total0, expected0)
    assert info0["episode_stats"]["match_result"] == "loss"

    total1, *_, info1 = run_episode(make_env(reward_config=cfg), agent_idx=1)
    expected1 = cfg.steal + cfg.goal_for + cfg.win                                   # 1 + 10 + 5
    assert abs(total1 - expected1) < 1e-6, (total1, expected1)
    assert info1["episode_stats"]["match_result"] == "win"


def test_episode_components_sum_equals_total_reward():
    if not HAVE_GYM:
        return
    total, *_, info = run_episode(make_env(), agent_idx=0)
    assert abs(sum(info["episode_stats"]["components"].values()) - total) < 1e-6


def test_same_seed_gives_same_trajectory():
    if not HAVE_GYM:
        return
    def trajectory():
        env = make_env()
        obs, _ = env.reset(seed=42)
        out = [obs.copy()]
        for _ in range(20):
            obs, *_ = env.step(np.full(spec.ACTION_DIM, 0.3, dtype=np.float32))
            out.append(obs.copy())
        return np.stack(out)
    assert np.array_equal(trajectory(), trajectory())


def test_actions_are_mirrored_for_the_player_attacking_left():
    # Même action "avancer vers l'attaque" (move_x=+1 dans le repère de la politique) :
    # le joueur 0 doit aller vers +x, le joueur 1 vers -x (repère monde).
    if not HAVE_GYM:
        return
    forward = np.array([1.0, 0.0, 1.0, -1.0, -1.0], dtype=np.float32)
    for idx, sign in ((0, +1), (1, -1)):
        env = make_env()
        env.reset(seed=0, options={"agent_idx": idx})
        start = env._state.players[idx].x
        for _ in range(10):
            env.step(forward)
        moved = env._state.players[idx].x - start
        assert moved * sign > 0, (idx, moved)


def test_step_after_end_requires_reset():
    if not HAVE_GYM:
        return
    env = make_env()
    run_episode(env, agent_idx=0)
    try:
        env.step(np.zeros(spec.ACTION_DIM, dtype=np.float32))
    except RuntimeError:
        return
    raise AssertionError("step() après la fin aurait dû lever une erreur")


def test_gymnasium_env_checker():
    if not HAVE_GYM:
        return
    try:
        from gymnasium.utils.env_checker import check_env
    except ImportError:
        return
    check_env(make_env(), skip_render_check=True)


if __name__ == "__main__":
    if not HAVE_GYM:
        print("gymnasium n'est pas installé : faites  pip install gymnasium  puis relancez ce test.")
        sys.exit(1)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
