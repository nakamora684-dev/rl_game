"""Proximal Policy Optimization for the hybrid football action space (Member D).

Standard clipped PPO with GAE, written directly in PyTorch because SB3 PPO cannot
consume the environment's Dict action space (Discrete + Box mix). Choices:

* ``gamma=0.995`` -- a 600-decision match; the env's potential shaping uses gamma=1
  (see docs/environment.md): C/D must decide whether to align them (``--shaping-gamma``).
* Rewards are scaled by a running std of the discounted return (VecNormalize style)
  for the LEARNER only. Logged episode returns are always the RAW undiscounted sum.
* ``terminated`` at the final whistle -> no bootstrap (time remaining is observed);
  safety ``truncated`` -> bootstrap from the final observation's value.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from agents.networks import ActorCritic, NetworkConfig
from environment.spec import OBS_DIM, OBS_INDEX, OBSERVATION_VERSION
from training.returns import RunningMeanStd, compute_gae


@dataclass
class PPOConfig:
    total_timesteps: int = 3_000_000
    n_envs: int = 8
    n_workers: int = 0  # 0 = step envs in the main process; >0 = spawn worker processes
    n_steps: int = 512  # decisions per env per rollout (batch = n_envs * n_steps)
    n_epochs: int = 8
    minibatch_size: int = 512
    lr: float = 3e-4
    lr_final_frac: float = 0.1  # linear decay to lr * lr_final_frac
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03  # early-stop an update when approx KL exceeds 1.5x this; 0 = off
    normalize_advantages: bool = True
    normalize_rewards: bool = True
    reward_clip: float = 10.0
    seed: int = 0
    device: str = "auto"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(path, model: ActorCritic, ppo_cfg: PPOConfig, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "network_cfg": asdict(model.cfg),
            "ppo_cfg": asdict(ppo_cfg),
            "obs_dim": OBS_DIM,
            "obs_version": OBSERVATION_VERSION,
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path, device: str | torch.device = "cpu"):
    """Returns (model in eval mode, checkpoint dict). Rejects incompatible observations."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("obs_dim") != OBS_DIM or ckpt.get("obs_version") != OBSERVATION_VERSION:
        raise ValueError(
            f"{path}: checkpoint was trained on observation v{ckpt.get('obs_version')} "
            f"({ckpt.get('obs_dim')} features); current env is v{OBSERVATION_VERSION} "
            f"({OBS_DIM}). Retrain."
        )
    model = ActorCritic(NetworkConfig(**ckpt["network_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# -------------------------------------------------------------------- trainer
class PPOTrainer:
    def __init__(self, model, vec_env, cfg: PPOConfig, logger, run_dir, eval_fn=None,
                 eval_every: int = 20, save_every: int = 20):
        self.model, self.vec, self.cfg, self.log = model, vec_env, cfg, logger
        self.run_dir, self.eval_fn = Path(run_dir), eval_fn
        self.eval_every, self.save_every = eval_every, save_every
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        self.reward_rms = RunningMeanStd()
        self.rng = np.random.default_rng(cfg.seed)
        self.global_step = 0
        self.update_idx = 0
        self.best_key = (-1.0, -1e9)

    # -- resume support --
    def load_state(self, ckpt: dict) -> None:
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        self.update_idx = ckpt.get("update_idx", 0)
        if "reward_rms" in ckpt:
            self.reward_rms.load(ckpt["reward_rms"])

    def _extra(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_idx": self.update_idx,
            "reward_rms": self.reward_rms.state(),
        }

    def _save(self, name: str) -> None:
        save_checkpoint(self.run_dir / name, self.model, self.cfg, self._extra())

    # -- main loop --
    def train(self) -> None:
        cfg, dev, T, N = self.cfg, self.device, self.cfg.n_steps, self.cfg.n_envs
        batch = T * N
        total_updates = max(1, -(-cfg.total_timesteps // batch))
        obs = torch.as_tensor(self.vec.reset(), dtype=torch.float32, device=dev)
        next_start = torch.ones(N, device=dev)  # obs[t] is the first obs of an episode
        ret_acc = np.zeros(N, dtype=np.float64)
        start_time = time.time()
        step_at_start = self.global_step

        obs_buf = torch.zeros((T, N, OBS_DIM), device=dev)
        act_buf = torch.zeros((T, N, 4), device=dev)
        logp_buf, val_buf = torch.zeros((T, N), device=dev), torch.zeros((T, N), device=dev)
        rew_buf, start_buf = torch.zeros((T, N), device=dev), torch.zeros((T, N), device=dev)

        while self.update_idx < total_updates:
            frac = 1.0 - self.update_idx / total_updates
            lr = cfg.lr * (cfg.lr_final_frac + (1.0 - cfg.lr_final_frac) * frac)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.model.eval()
            for t in range(T):
                obs_buf[t], start_buf[t] = obs, next_start
                actions, logp, value = self.model.act(obs)
                act_buf[t], logp_buf[t], val_buf[t] = actions, logp, value
                nxt, rew, term, trunc, episodes, finals = self.vec.step(actions.cpu().numpy())
                self.global_step += N
                done = term | trunc
                if cfg.normalize_rewards:
                    ret_acc = ret_acc * cfg.gamma + rew
                    self.reward_rms.update(ret_acc)
                    scale = np.sqrt(self.reward_rms.var + 1e-8)
                    scaled = np.clip(rew / scale, -cfg.reward_clip, cfg.reward_clip)
                    ret_acc[done] = 0.0
                else:
                    scaled = rew.copy()
                for i in np.flatnonzero(trunc & ~term):  # time-limit bootstrap
                    final = torch.as_tensor(finals[i], dtype=torch.float32, device=dev)[None]
                    with torch.no_grad():
                        scaled[i] += cfg.gamma * float(self.model.value(final))
                rew_buf[t] = torch.as_tensor(scaled, dtype=torch.float32, device=dev)
                next_start = torch.as_tensor(done, dtype=torch.float32, device=dev)
                obs = torch.as_tensor(nxt, dtype=torch.float32, device=dev)
                for ep in episodes:
                    if ep is not None:
                        self.log.episode(ep, self.global_step)

            with torch.no_grad():
                next_value = self.model.value(obs)
            adv, returns = compute_gae(
                rew_buf, val_buf, start_buf, next_value, next_start, cfg.gamma, cfg.gae_lambda
            )

            stats = self._optimize(
                obs_buf.reshape(batch, OBS_DIM), act_buf.reshape(batch, 4),
                logp_buf.reshape(batch), adv.reshape(batch), returns.reshape(batch),
                val_buf.reshape(batch),
            )
            self.update_idx += 1
            elapsed = time.time() - start_time
            stats.update(self._behaviour(obs_buf, act_buf))
            stats["train/lr"] = lr
            stats["perf/steps_per_sec"] = (self.global_step - step_at_start) / max(elapsed, 1e-9)
            stats["perf/reward_scale"] = float(np.sqrt(self.reward_rms.var))
            self.log.update(stats, self.global_step)
            print(
                f"[update {self.update_idx}/{total_updates}] step={self.global_step} "
                f"sps={stats['perf/steps_per_sec']:.0f} kl={stats['train/approx_kl']:.4f} "
                f"ent={stats['train/entropy']:.2f} ev={stats['train/explained_variance']:.2f} "
                f"ma_ret={sum(self.log._returns) / max(1, len(self.log._returns)):.2f}",
                flush=True,
            )

            if self.save_every and self.update_idx % self.save_every == 0:
                self._save(f"ckpt_{self.global_step}.pt")
            self._save("latest.pt")
            if self.eval_fn and self.eval_every and self.update_idx % self.eval_every == 0:
                self._run_eval()
        self._save("final.pt")
        if self.eval_fn and not (self.eval_every and self.update_idx % self.eval_every == 0):
            self._run_eval()

    def _run_eval(self) -> None:
        self.model.eval()
        result = self.eval_fn(self.model, self.global_step)
        self.log.evaluation(result, self.global_step)
        key = (result["points"], result["goal_diff"])
        print(f"  [eval] points={result['points']:.3f} goal_diff={result['goal_diff']:.2f} "
              f"win_rate={result['win_rate']:.2f}", flush=True)
        if key > self.best_key:
            self.best_key = key
            self._save("best.pt")

    def _optimize(self, b_obs, b_act, b_logp, b_adv, b_ret, b_val) -> dict:
        cfg, n = self.cfg, b_obs.shape[0]
        self.model.train()
        kls, clips, pgs, vls, ents = [], [], [], [], []
        stop = False
        for _ in range(cfg.n_epochs):
            perm = torch.as_tensor(self.rng.permutation(n), device=self.device)
            epoch_kls = []
            for start in range(0, n, cfg.minibatch_size):
                mb = perm[start:start + cfg.minibatch_size]
                new_logp, entropy, value = self.model.evaluate_actions(b_obs[mb], b_act[mb])
                log_ratio = new_logp - b_logp[mb]
                ratio = log_ratio.exp()
                adv = b_adv[mb]
                if cfg.normalize_advantages:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(
                    -adv * ratio, -adv * ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range)
                ).mean()
                v_loss = 0.5 * ((value - b_ret[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    clips.append(((ratio - 1).abs() > cfg.clip_range).float().mean().item())
                epoch_kls.append(kl)
                kls.append(kl)
                pgs.append(pg_loss.item())
                vls.append(v_loss.item())
                ents.append(ent.item())
            if cfg.target_kl and np.mean(epoch_kls) > 1.5 * cfg.target_kl:
                stop = True
            if stop:
                break
        var_ret = b_ret.var()
        explained = float("nan") if var_ret < 1e-12 else float(1 - (b_ret - b_val).var() / var_ret)
        return {
            "train/policy_loss": float(np.mean(pgs)),
            "train/value_loss": float(np.mean(vls)),
            "train/entropy": float(np.mean(ents)),
            "train/approx_kl": float(np.mean(kls)),
            "train/clip_fraction": float(np.mean(clips)),
            "train/explained_variance": explained,
            "train/epochs_run": len(kls) / max(1, -(-n // cfg.minibatch_size)),
        }

    @staticmethod
    def _behaviour(obs_buf, act_buf) -> dict:
        """Cheap behaviour diagnostics straight from the rollout (not proof of skill)."""
        direction = act_buf[..., 0]
        moving = direction != 0
        return {
            "behaviour/kick_rate": float((act_buf[..., 2] > 0.5).float().mean()),
            "behaviour/idle_fraction": float((~moving).float().mean()),
            "behaviour/mean_effort_moving": float(act_buf[..., 1][moving].mean())
            if moving.any() else 0.0,
            "behaviour/mean_stamina": float(obs_buf[..., OBS_INDEX["self_stamina"]].mean()),
            "behaviour/possession_self": float(obs_buf[..., OBS_INDEX["poss_self"]].mean()),
        }
