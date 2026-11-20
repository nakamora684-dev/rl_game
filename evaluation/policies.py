"""Policies the evaluator can drive. All expose ``act(obs) -> flat action (4,)``.

torch is imported lazily so reference agents ('random', 'idle') run without it.
Evaluation NEVER updates weights (no optimizer here; torch.no_grad in the model).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.action_codec import IDLE_FLAT_ACTION, random_flat_action


class RandomPolicy:
    name = "random"

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return random_flat_action(self.rng)


class IdlePolicy:
    name = "idle"

    def reseed(self, seed: int) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return IDLE_FLAT_ACTION


class HeuristicPolicy:
    """Hand-written reference that uses ONLY the policy observation (same interface as PPO).

    Chase the ball; with the ball run at the goal and shoot inside ~230 units. A useful
    sanity floor: a trained agent should clearly beat it, and it exercises the
    attack-normalised observation on both sides of the pitch.
    """

    name = "heuristic"
    SHOOT_RANGE = 230.0
    DIAGONAL = 943.4  # hypot(800, 500): the observation's distance scale

    def reseed(self, seed: int) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        from agents.opponents import direction_towards
        from environment.spec import OBS_INDEX as ix

        if obs[ix["poss_self"]] > 0.5:
            dx, dy = obs[ix["to_opp_goal_dx"]], obs[ix["to_opp_goal_dy"]]
            shoot = obs[ix["to_opp_goal_dist"]] * self.DIAGONAL < self.SHOOT_RANGE
            return np.array([direction_towards(dx, dy), 0.7, float(shoot), 1.0], dtype=np.float32)
        dx, dy = obs[ix["to_ball_dx"]], obs[ix["to_ball_dy"]]
        return np.array([direction_towards(dx, dy), 0.7, 0.0, 0.0], dtype=np.float32)


class TorchPolicy:
    def __init__(self, model, name: str, deterministic: bool, device="cpu") -> None:
        self.model, self.name, self.deterministic, self.device = model, name, deterministic, device

    def reseed(self, seed: int) -> None:
        import torch

        torch.manual_seed(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]
            action, _, _ = self.model.act(x, deterministic=self.deterministic)
        return action[0].cpu().numpy()


def load_policy(spec: str, deterministic: bool = False, device: str = "cpu"):
    """``spec`` is 'random', 'idle', 'heuristic', or a checkpoint path (name = parent/stem)."""
    if spec == "random":
        return RandomPolicy()
    if spec == "idle":
        return IdlePolicy()
    if spec == "heuristic":
        return HeuristicPolicy()
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"Agent {spec!r} is not 'random', 'idle', 'heuristic' or a checkpoint file.")
    from agents.ppo import load_checkpoint

    model, _ = load_checkpoint(path, device)
    return TorchPolicy(model, f"{path.parent.name}/{path.stem}", deterministic, device)
