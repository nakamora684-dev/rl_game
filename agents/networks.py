"""Customized actor-critic for the hybrid football action space (Member D).

Architecture ("GroupResNet"):

    obs (50) --split by semantic group (self / opp / ball / geom / context /
               interaction / phase, taken from environment.spec.OBS_FEATURES)
             --> one small Linear+act embedding PER GROUP
             --> concat --> Linear --> N pre-norm residual blocks --> LayerNorm
             --> heads

Actor and critic have SEPARATE encoders by default (``share_trunk=False``) so the
value loss cannot distort the policy features.

Hybrid action distribution (matches ``environment/spec.py``):

    direction   Categorical(9)       0 idle, 1 N ... 8 NW
    kick        Bernoulli            trigger
    effort      Beta(a, b) on [0,1]  only counts when direction != idle
    power       Beta(a, b) on [0,1]  only counts when kick == 1

Masking effort/power log-probs removes gradient noise from parameters the engine
ignores in that situation (engine: idle ignores effort; power only matters on a kick).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, Beta, Categorical

from environment.spec import OBS_FEATURES

BETA_EPS = 1e-4  # keep sampled Beta values strictly inside (0,1) for finite log-probs
_ACTIVATIONS = {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU}


@dataclass
class NetworkConfig:
    group_embed: int = 32  # embedding width per observation group
    hidden: int = 128  # trunk width
    n_res_blocks: int = 2  # residual blocks per encoder
    activation: str = "silu"  # silu | tanh | relu
    layer_norm: bool = True
    share_trunk: bool = False  # True: one encoder for actor and critic


def _group_indices() -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, feature in enumerate(OBS_FEATURES):
        groups.setdefault(feature.group, []).append(i)
    return groups


class _ResBlock(nn.Module):
    def __init__(self, width: int, act, layer_norm: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width) if layer_norm else nn.Identity()
        self.fc1, self.fc2, self.act = nn.Linear(width, width), nn.Linear(width, width), act()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class GroupEncoder(nn.Module):
    def __init__(self, cfg: NetworkConfig) -> None:
        super().__init__()
        act = _ACTIVATIONS[cfg.activation]
        groups = _group_indices()
        self.group_names = list(groups)
        for k, name in enumerate(self.group_names):
            self.register_buffer(f"idx{k}", torch.tensor(groups[name], dtype=torch.long))
        self.embeds = nn.ModuleList(
            nn.Sequential(nn.Linear(len(groups[n]), cfg.group_embed), act())
            for n in self.group_names
        )
        self.proj = nn.Linear(cfg.group_embed * len(groups), cfg.hidden)
        self.blocks = nn.Sequential(
            *[_ResBlock(cfg.hidden, act, cfg.layer_norm) for _ in range(cfg.n_res_blocks)]
        )
        self.out_norm = nn.LayerNorm(cfg.hidden) if cfg.layer_norm else nn.Identity()
        self.out_act = act()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        parts = [
            emb(obs[:, getattr(self, f"idx{k}")]) for k, emb in enumerate(self.embeds)
        ]
        h = self.proj(torch.cat(parts, dim=-1))
        return self.out_act(self.out_norm(self.blocks(h)))


def _init(module: nn.Module, gain: float) -> None:
    nn.init.orthogonal_(module.weight, gain)
    nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    def __init__(self, cfg: NetworkConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NetworkConfig()
        if self.cfg.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {tuple(_ACTIVATIONS)}")
        self.actor_enc = GroupEncoder(self.cfg)
        self.critic_enc = self.actor_enc if self.cfg.share_trunk else GroupEncoder(self.cfg)
        h = self.cfg.hidden
        self.dir_head = nn.Linear(h, 9)
        self.kick_head = nn.Linear(h, 1)
        self.effort_head = nn.Linear(h, 2)
        self.power_head = nn.Linear(h, 2)
        self.value_head = nn.Linear(h, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init(m, math.sqrt(2))
        for head in (self.dir_head, self.kick_head, self.effort_head, self.power_head):
            _init(head, 0.01)
        _init(self.value_head, 1.0)

    # ---- distributions -------------------------------------------------
    def _dists(self, obs: torch.Tensor):
        h = self.actor_enc(obs)
        eff = F.softplus(self.effort_head(h)) + 1.0  # alpha, beta >= 1 -> unimodal
        pow_ = F.softplus(self.power_head(h)) + 1.0
        return (
            Categorical(logits=self.dir_head(h)),
            Bernoulli(logits=self.kick_head(h).squeeze(-1)),
            Beta(eff[:, 0], eff[:, 1]),
            Beta(pow_[:, 0], pow_[:, 1]),
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_enc(obs)).squeeze(-1)

    @staticmethod
    def _log_prob_entropy(dists, actions: torch.Tensor):
        d_dir, d_kick, d_eff, d_pow = dists
        direction = actions[:, 0].long()
        effort = actions[:, 1].clamp(BETA_EPS, 1 - BETA_EPS)
        kick = actions[:, 2]
        power = actions[:, 3].clamp(BETA_EPS, 1 - BETA_EPS)
        moving = (direction != 0).float()
        kicking = (kick > 0.5).float()
        log_prob = (
            d_dir.log_prob(direction)
            + d_kick.log_prob(kick)
            + moving * d_eff.log_prob(effort)
            + kicking * d_pow.log_prob(power)
        )
        # Expected (detached) weights so entropy tracks what is actually used.
        w_move = 1.0 - d_dir.probs[:, 0].detach()
        w_kick = d_kick.probs.detach()
        entropy = (
            d_dir.entropy() + d_kick.entropy() + w_move * d_eff.entropy() + w_kick * d_pow.entropy()
        )
        return log_prob, entropy

    # ---- API used by PPO / evaluation ---------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Returns flat actions (B,4), log-probs (B,), values (B,).

        Stochastic sampling uses torch's global RNG (seed it with torch.manual_seed
        for reproducible evaluation).
        """
        dists = self._dists(obs)
        d_dir, d_kick, d_eff, d_pow = dists
        if deterministic:
            direction = d_dir.probs.argmax(-1)
            kick = (d_kick.probs > 0.5).float()
            effort, power = d_eff.mean, d_pow.mean
        else:
            direction, kick = d_dir.sample(), d_kick.sample()
            effort, power = d_eff.sample(), d_pow.sample()
        actions = torch.stack(
            [
                direction.float(),
                effort.clamp(BETA_EPS, 1 - BETA_EPS),
                kick,
                power.clamp(BETA_EPS, 1 - BETA_EPS),
            ],
            dim=-1,
        )
        log_prob, _ = self._log_prob_entropy(dists, actions)
        return actions, log_prob, self.value(obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Returns log-probs, entropy, values for stored actions (with gradients)."""
        log_prob, entropy = self._log_prob_entropy(self._dists(obs), actions)
        return log_prob, entropy, self.value(obs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["ActorCritic", "NetworkConfig", "count_parameters"]
