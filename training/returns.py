"""Torch-free return utilities (work on numpy arrays AND torch tensors)."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Parallel (Chan et al.) running mean/variance, used to scale rewards for the learner."""

    def __init__(self) -> None:
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean, batch_var, n = float(np.mean(x)), float(np.var(x)), x.size
        delta, total = batch_mean - self.mean, self.count + n
        self.mean += delta * n / total
        m2 = self.var * self.count + batch_var * n + delta**2 * self.count * n / total
        self.var, self.count = m2 / total, total

    def state(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load(self, state: dict) -> None:
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]


def compute_gae(rewards, values, starts, next_value, next_start, gamma: float, lam: float):
    """Generalised Advantage Estimation over a (T, N) rollout.

    ``starts[t]`` is 1 when obs[t] is the first observation of a new episode, i.e. the
    previous step ended an episode, so nothing bootstraps across that boundary.
    ``next_value`` / ``next_start`` describe the observation after the last step.
    Time-limit bootstraps are already folded into ``rewards`` by the caller.
    Returns (advantages, returns) with returns = advantages + values.
    """
    T = rewards.shape[0]
    advantages = rewards * 0.0
    last = rewards[0] * 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            non_terminal, next_val = 1.0 - next_start, next_value
        else:
            non_terminal, next_val = 1.0 - starts[t + 1], values[t + 1]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last = delta + gamma * lam * non_terminal * last
        advantages[t] = last
    return advantages, advantages + values
