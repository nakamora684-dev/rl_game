"""Member D: evaluation pipeline against the REAL engine (needs the environment group)."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("gymnasium", reason="Install the locked environment dependency group.")
pd = pytest.importorskip("pandas")

from evaluation.metrics import MATCH_COLUMNS, SCHEMA_VERSION, summarize
from evaluation.policies import load_policy
from evaluation.runner import COMBOS, evaluate
from training.vec_env import EnvRunner, EnvSpec, VecFootball

pytestmark = pytest.mark.environment
SPEC = EnvSpec(opponent="chaser", randomize_sides=False)
SPEC_RB = EnvSpec(opponent="rule_based", randomize_sides=False)


def test_idle_agent_rows_match_schema_and_are_consistent():
    rows = evaluate(load_policy("idle"), SPEC, n_matches=4, seed_base=1_000_000,
                    agent_name="idle", policy_mode="stochastic")
    assert len(rows) == 4
    assert {(r["agent_idx"], r["swap_sides"]) for r in rows} == set(COMBOS)
    assert {r["attack_dir"] for r in rows} == {1, -1}  # both attack directions evaluated
    for r in rows:
        assert list(r) == MATCH_COLUMNS and r["schema_version"] == SCHEMA_VERSION
        assert r["decisions"] == 600 and r["end_reason"] == "time_up"
        assert r["goal_diff"] == r["goals_for"] - r["goals_against"]
        assert abs(r["poss_self_frac"] + r["poss_opp_frac"] + r["poss_free_frac"] - 1) < 1e-9
        assert r["idle_frac"] == 1.0 and r["sprint_frac"] == 0.0 and r["kick_attempts"] == 0


def test_episode_return_equals_sum_of_components():
    rows = evaluate(load_policy("random"), SPEC, n_matches=4, seed_base=1_000_000,
                    agent_name="random", policy_mode="stochastic")
    for r in rows:
        components = sum(v for k, v in r.items() if k.startswith("r_"))
        assert r["ep_return"] == pytest.approx(components, abs=1e-6)


def test_evaluation_is_reproducible_for_same_seed():
    kwargs = dict(n_matches=4, seed_base=1_000_000, agent_name="random", policy_mode="stochastic")
    a = evaluate(load_policy("random"), SPEC, **kwargs)
    b = evaluate(load_policy("random"), SPEC, **kwargs)
    assert a == b or all(
        {k: v for k, v in x.items() if v == v} == {k: v for k, v in y.items() if v == v}
        for x, y in zip(a, b, strict=True)
    )  # (NaN != NaN, so compare non-NaN entries)


def test_summary_counts_add_up():
    rows = evaluate(load_policy("heuristic"), SPEC, n_matches=4, seed_base=1_000_000,
                    agent_name="heuristic", policy_mode="stochastic")
    s = summarize(pd.DataFrame(rows))
    assert int(s.wins[0] + s.draws[0] + s.losses[0]) == 4
    assert 0.0 <= s.win_ci_low[0] <= s.win_rate[0] <= s.win_ci_high[0] <= 1.0


def test_vec_env_episode_stats_and_autoreset():
    vec = VecFootball(EnvSpec(opponent="chaser"), n_envs=2, seed=0)
    obs = vec.reset()
    assert obs.shape == (2, 50)
    import numpy as np

    action = np.zeros((2, 4), dtype=np.float32)
    episodes = []
    for _ in range(600):
        obs, rew, term, trunc, eps, finals = vec.step(action)
        episodes += [e for e in eps if e]
    assert len(episodes) == 2 and all(e["steps"] == 600 for e in episodes)
    assert obs.shape == (2, 50)  # auto-reset produced a fresh observation


def test_env_runner_randomizes_sides_with_seed():
    a, b = EnvRunner(EnvSpec(), 5), EnvRunner(EnvSpec(), 5)
    for _ in range(3):
        a.reset(), b.reset()
        assert a.env._agent_idx == b.env._agent_idx


def test_rule_based_can_be_evaluated_as_the_agent():
    rows = evaluate(load_policy("rule_based"), SPEC_RB, n_matches=4, seed_base=1_000_000,
                    agent_name="rule_based", policy_mode="stochastic")
    assert len(rows) == 4 and all(list(r) == MATCH_COLUMNS for r in rows)
    assert {r["attack_dir"] for r in rows} == {1, -1}
    assert all(r["decisions"] == 600 and r["kick_attempts"] >= 0 for r in rows)


def test_reward_override_parsing():
    from training.vec_env import parse_reward_overrides

    assert parse_reward_overrides(["steal=0", "loose_recovery=0.25"]) == {
        "steal": 0.0, "loose_recovery": 0.25}
    with pytest.raises(ValueError):
        parse_reward_overrides(["nonsense=1"])
