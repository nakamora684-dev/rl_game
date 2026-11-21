"""Member D: the 60 Hz viewer path (observe.py) must equal the 10 Hz training/eval path."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("gymnasium", reason="Install the locked environment dependency group.")

from agents.difficulty import make_baseline
from agents.match import run_match
from agents.policy_agent import PolicyAgent, PolicyEngineAdapter
from agents.provider import rule_based_provider
from agents.action_codec import to_env_action
from environment.football_env import FootballEnv
from evaluation.policies import load_policy

pytestmark = pytest.mark.environment


def _train_path(policy_name, swap):
    """FootballEnv (10 Hz decisions) with B's baseline as native provider."""
    policy = load_policy(policy_name)
    env = FootballEnv(opponent=rule_based_provider("normal"))
    policy.reseed(0)
    obs, _ = env.reset(options={"agent_idx": 0, "swap_sides": swap})
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(to_env_action(policy.act(obs)))
        done = term or trunc
    return env.engine.native_state


def _viewer_path(policy_name, swap):
    """observe.py path: 60 Hz EngineAdapter + PolicyAgent vs HeldAgent baseline."""
    engine = PolicyEngineAdapter()
    agents = [PolicyAgent(load_policy(policy_name), 0, engine), make_baseline(1, "normal")]
    run_match(engine, agents, seed=0, swap_sides=swap)
    return engine.raw


@pytest.mark.parametrize("policy_name", ["random", "heuristic"])
@pytest.mark.parametrize("swap", [False, True])
def test_viewer_run_equals_training_env_run(policy_name, swap):
    a, b = _train_path(policy_name, swap), _viewer_path(policy_name, swap)
    assert a == b  # identical final engine state: same physics, same decisions
