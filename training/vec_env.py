"""Environment construction + lightweight vectorisation for PPO.

``gymnasium.vector`` batches Dict action spaces awkwardly, so the learner uses a
flat ``(n_envs, 4)`` action array and this module converts to the real hybrid Dict
(``agents.action_codec``). Workers use the ``spawn`` start method so the same code
works on Windows, macOS and Linux (everything sent to a worker is picklable).
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from agents.action_codec import to_env_action
from agents.opponents import make_opponent
from environment.football_env import FootballEnv
from environment.reward import RewardConfig
from training.start_jitter import JitteredSimulationEngine


@dataclass(frozen=True)
class EnvSpec:
    """Picklable recipe for one environment (so workers can build their own)."""

    opponent: str = "chaser"
    start_jitter: float = 0.0
    randomize_sides: bool = True
    debug: bool = False  # debug=True validates every state: ~slower, use for tests only
    reward_overrides: dict = field(default_factory=dict)


def parse_reward_overrides(items) -> dict:
    """['steal=0', 'loose_recovery=0'] -> {'steal': 0.0, ...}; names are validated."""
    from dataclasses import fields

    valid = {f.name: f.type for f in fields(RewardConfig)}
    out: dict = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or name not in valid:
            raise ValueError(f"bad --reward item {item!r}; use name=value with name in {sorted(valid)}")
        out[name] = (value.lower() == "true") if name == "zero_terminal_potential" else float(value)
    return out


def build_env(spec: EnvSpec, seed: int = 0) -> FootballEnv:
    engine = JitteredSimulationEngine(jitter=spec.start_jitter, seed=seed)
    reward_cfg = RewardConfig(**spec.reward_overrides) if spec.reward_overrides else None
    return FootballEnv(
        engine=engine,
        opponent=make_opponent(spec.opponent),
        reward_config=reward_cfg,
        debug=spec.debug,
    )


class EnvRunner:
    """One env with automatic reset, side randomisation and episode summaries."""

    def __init__(self, spec: EnvSpec, seed: int) -> None:
        self.spec, self.seed = spec, seed
        self.env = build_env(spec, seed)
        self.rng = np.random.default_rng(seed)
        self._first = True
        self.ep_return = 0.0

    def reset(self) -> np.ndarray:
        options = {}
        if self.spec.randomize_sides:
            options = {
                "agent_idx": int(self.rng.integers(2)),
                "swap_sides": bool(self.rng.integers(2)),
            }
        else:
            options = {"agent_idx": 0, "swap_sides": False}
        opp = self.env.opponent
        if hasattr(opp, "reset"):
            opp.reset()
        self.env.engine.seed(int(self.rng.integers(2**31)))
        obs, _ = self.env.reset(seed=self.seed if self._first else None, options=options)
        self._first = False
        self.ep_return = 0.0
        return obs

    def step(self, flat_action):
        obs, reward, terminated, truncated, info = self.env.step(to_env_action(flat_action))
        self.ep_return += reward
        episode, final_obs = None, None
        if terminated or truncated:
            stats = info["episode_stats"]
            idx = info["agent_idx"]
            score = stats["score"]
            episode = {
                "ep_return": self.ep_return,  # raw, undiscounted total reward
                "steps": stats["steps"],
                "score_for": score[idx],
                "score_against": score[1 - idx],
                "goal_diff": score[idx] - score[1 - idx],
                "result": stats["match_result"],
                "end_reason": info["end_reason"],
                "agent_idx": idx,
                "attack_dir": info["attack_dirs"][idx],
                **{f"r_{k}": float(v) for k, v in stats["components"].items()},
            }
            final_obs = obs
            obs = self.reset()
        return obs, float(reward), bool(terminated), bool(truncated), episode, final_obs


class _Group:
    """A list of runners stepped sequentially (used in-process or inside a worker)."""

    def __init__(self, spec: EnvSpec, seeds: list[int]) -> None:
        self.runners = [EnvRunner(spec, s) for s in seeds]

    def reset(self):
        return np.stack([r.reset() for r in self.runners])

    def step(self, actions):
        return [r.step(a) for r, a in zip(self.runners, actions, strict=True)]


def _worker(conn, spec, seeds):  # pragma: no cover - runs in a child process
    group = _Group(spec, seeds)
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send(group.reset())
            elif cmd == "step":
                conn.send(group.step(data))
            elif cmd == "close":
                break
    finally:
        conn.close()


class VecFootball:
    """Vectorised training env. ``n_workers=0`` runs everything in this process."""

    def __init__(self, spec: EnvSpec, n_envs: int, seed: int = 0, n_workers: int = 0) -> None:
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        self.n_envs = n_envs
        seeds = [seed + i for i in range(n_envs)]
        n_workers = min(n_workers, n_envs)
        self._chunks = [list(c) for c in np.array_split(np.arange(n_envs), max(n_workers, 1))]
        if n_workers <= 0:
            self._local = _Group(spec, seeds)
            self._procs, self._pipes = [], []
        else:
            self._local = None
            ctx = mp.get_context("spawn")
            self._pipes, self._procs = [], []
            for chunk in self._chunks:
                parent, child = ctx.Pipe()
                proc = ctx.Process(
                    target=_worker, args=(child, spec, [seeds[i] for i in chunk]), daemon=True
                )
                proc.start()
                child.close()
                self._pipes.append(parent)
                self._procs.append(proc)

    def reset(self) -> np.ndarray:
        if self._local:
            return self._local.reset()
        for p in self._pipes:
            p.send(("reset", None))
        return np.concatenate([p.recv() for p in self._pipes])

    def step(self, actions: np.ndarray):
        """Returns obs, rewards, terminated, truncated, episodes(list|None), final_obs(list|None)."""
        if self._local:
            results = self._local.step(actions)
        else:
            for p, chunk in zip(self._pipes, self._chunks, strict=True):
                p.send(("step", actions[chunk]))
            results = [r for p in self._pipes for r in p.recv()]
        obs, rew, term, trunc, eps, finals = zip(*results, strict=True)
        return (
            np.stack(obs).astype(np.float32),
            np.array(rew, dtype=np.float32),
            np.array(term),
            np.array(trunc),
            list(eps),
            list(finals),
        )

    def close(self) -> None:
        for p in self._pipes:
            try:
                p.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
        self._pipes, self._procs = [], []

