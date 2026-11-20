"""Member D: network + PPO + CLI smoke tests. Need the FULL project environment (torch)."""

import json

import pytest

torch = pytest.importorskip("torch", reason="Run `uv sync --locked` for the full stack.")
pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")

from agents.networks import ActorCritic, NetworkConfig, count_parameters
from agents.ppo import PPOConfig, PPOTrainer, load_checkpoint, save_checkpoint
from environment.spec import OBS_DIM
from training.logger import TrainingLogger
from training.vec_env import EnvSpec, VecFootball


def _obs(batch=5):
    return torch.rand(batch, OBS_DIM) * 2 - 1


def test_forward_shapes_and_legal_actions():
    model = ActorCritic()
    actions, logp, value = model.act(_obs(16))
    assert actions.shape == (16, 4) and logp.shape == (16,) and value.shape == (16,)
    assert torch.isfinite(logp).all() and torch.isfinite(value).all()
    assert ((actions[:, 0] >= 0) & (actions[:, 0] <= 8)).all()
    assert ((actions[:, 1] > 0) & (actions[:, 1] < 1)).all()
    assert set(actions[:, 2].tolist()) <= {0.0, 1.0}
    assert ((actions[:, 3] > 0) & (actions[:, 3] < 1)).all()


def test_log_prob_of_stored_action_matches_evaluate_actions():
    model = ActorCritic()
    obs = _obs(32)
    actions, logp, value = model.act(obs)
    new_logp, entropy, new_value = model.evaluate_actions(obs, actions)
    assert torch.allclose(logp, new_logp, atol=1e-5)
    assert torch.allclose(value, new_value, atol=1e-5)
    assert torch.isfinite(entropy).all()


def test_masking_effort_ignored_when_idle_and_power_ignored_without_kick():
    model = ActorCritic()
    obs = _obs(8)
    base = torch.tensor([[0.0, 0.3, 0.0, 0.3]] * 8)  # idle, no kick
    other = torch.tensor([[0.0, 0.9, 0.0, 0.9]] * 8)  # different effort/power, still idle/no kick
    lp_a, _, _ = model.evaluate_actions(obs, base)
    lp_b, _, _ = model.evaluate_actions(obs, other)
    assert torch.allclose(lp_a, lp_b)
    moving = torch.tensor([[3.0, 0.3, 0.0, 0.3]] * 8)
    moving2 = torch.tensor([[3.0, 0.9, 0.0, 0.3]] * 8)
    assert not torch.allclose(*(model.evaluate_actions(obs, a)[0] for a in (moving, moving2)))


def test_deterministic_action_is_repeatable():
    model = ActorCritic()
    obs = _obs(4)
    a1, _, _ = model.act(obs, deterministic=True)
    a2, _, _ = model.act(obs, deterministic=True)
    assert torch.equal(a1, a2)


def test_gradients_reach_every_parameter_group():
    model = ActorCritic()
    obs = _obs(64)
    actions, _, _ = model.act(obs)
    # make sure both moving and kicking branches are exercised
    actions[:, 0], actions[:, 2] = 3.0, 1.0
    logp, entropy, value = model.evaluate_actions(obs, actions)
    (logp.mean() + entropy.mean() + value.mean()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_shared_trunk_option_has_fewer_parameters():
    separate = count_parameters(ActorCritic(NetworkConfig(share_trunk=False)))
    shared = count_parameters(ActorCritic(NetworkConfig(share_trunk=True)))
    assert shared < separate


def test_checkpoint_roundtrip_and_incompatible_version_rejected(tmp_path):
    model = ActorCritic(NetworkConfig(hidden=64, n_res_blocks=1))
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, PPOConfig())
    loaded, ckpt = load_checkpoint(path)
    obs = _obs(3)
    assert torch.equal(model.act(obs, deterministic=True)[0], loaded.act(obs, deterministic=True)[0])
    bad = torch.load(path, weights_only=False)
    bad["obs_version"] = 1
    torch.save(bad, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="Retrain"):
        load_checkpoint(tmp_path / "old.pt")


def test_ppo_smoke_update_changes_weights_and_logs(tmp_path):
    cfg = PPOConfig(total_timesteps=64, n_envs=2, n_steps=32, n_epochs=2, minibatch_size=16,
                    device="cpu")
    model = ActorCritic(NetworkConfig(hidden=32, n_res_blocks=1))
    before = [p.detach().clone() for p in model.parameters()]
    vec = VecFootball(EnvSpec(opponent="chaser"), cfg.n_envs, seed=0)
    logger = TrainingLogger(tmp_path / "logs", tensorboard=False)
    trainer = PPOTrainer(model, vec, cfg, logger, tmp_path / "models", eval_fn=None, save_every=1)
    trainer.train()
    logger.close()
    assert any(not torch.equal(b, a) for b, a in zip(before, model.parameters(), strict=True))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert (tmp_path / "models" / "final.pt").exists()
    assert (tmp_path / "logs" / "updates.csv").exists()


def test_train_cli_end_to_end_then_test_cli(tmp_path, monkeypatch):
    import pandas as pd

    import importlib.util
    from pathlib import Path

    import train as train_cli

    # `import test` would collide with the stdlib `test` package, so load by path.
    spec = importlib.util.spec_from_file_location(
        "football_test_cli", Path(__file__).resolve().parents[1] / "test.py"
    )
    test_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_cli)

    monkeypatch.chdir(tmp_path)
    rc = train_cli.main([
        "--run-name", "smoke", "--opponent", "chaser", "--total-timesteps", "1200",
        "--n-envs", "2", "--n-steps", "300", "--minibatch-size", "150", "--n-epochs", "1",
        "--hidden", "32", "--n-res-blocks", "1", "--device", "cpu", "--eval-matches", "4",
        "--eval-every", "1", "--no-tensorboard",
    ])
    assert rc == 0
    cfg = json.loads((tmp_path / "logs/smoke/config.json").read_text())
    assert cfg["network"]["hidden"] == 32
    episodes = pd.read_csv(tmp_path / "logs/smoke/episodes.csv")
    assert {"ep_return", "goal_diff", "result"} <= set(episodes.columns)
    assert (tmp_path / "models/smoke/best.pt").exists()
    out = tmp_path / "results/eval_smoke.csv"
    rc = test_cli.main(["--agents", str(tmp_path / "models/smoke/final.pt"), "random",
                        "--opponent", "chaser", "--matches", "4", "--out", str(out)])
    assert rc == 0 and len(pd.read_csv(out)) == 8
