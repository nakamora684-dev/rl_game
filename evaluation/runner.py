"""Run a policy against an opponent and record one metrics row per match."""

from __future__ import annotations

import numpy as np

from agents.action_codec import to_env_action
from evaluation.metrics import MatchRecorder
from training.vec_env import EnvSpec, build_env

COMBOS = ((0, False), (1, False), (0, True), (1, True))  # (agent_idx, swap_sides)
# agent_idx x swap_sides covers both attack directions AND both player identities.


def play_match(policy, env, *, seed: int, agent_idx: int, swap_sides: bool, meta: dict) -> dict:
    """One full match. ``env`` is a FootballEnv built by training.vec_env.build_env."""
    policy.reseed(seed)
    if hasattr(env.engine, "seed"):
        env.engine.seed(seed)  # seeds optional start jitter (engine reset itself is deterministic)
    opp = env.opponent
    if hasattr(opp, "reset"):
        opp.reset()
    obs, _ = env.reset(seed=seed, options={"agent_idx": agent_idx, "swap_sides": swap_sides})
    recorder = MatchRecorder(agent_idx, env.engine.config)
    while True:
        prev = env._state  # read-only MatchState projection of the shared engine
        native = getattr(policy, "act_native", None)  # e.g. the rule-based baseline as agent
        raw = (native(env.engine.native_state, agent_idx, prev.attack_dirs[agent_idx])
               if native else policy.act(obs))
        action = np.asarray(raw, dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(to_env_action(action))
        recorder.observe(prev, action, info, env._state)
        if terminated or truncated:
            return recorder.finalize(
                info, meta={**meta, "seed": seed, "agent_idx": agent_idx,
                            "swap_sides": swap_sides}
            )


def evaluate(policy, spec: EnvSpec, *, n_matches: int, seed_base: int, agent_name: str,
             policy_mode: str, progress=None) -> list[dict]:
    """Balanced over the four (identity, attack-direction) combos; same seeds for every agent."""
    rows = []
    env = build_env(spec, seed_base)
    for i in range(n_matches):
        agent_idx, swap = COMBOS[i % len(COMBOS)]
        rows.append(
            play_match(
                policy, env, seed=seed_base + i, agent_idx=agent_idx, swap_sides=swap,
                meta={"schema_version": 1, "agent": agent_name, "opponent": spec.opponent,
                      "match_id": i, "policy_mode": policy_mode},
            )
        )
        if progress:
            progress(i + 1, n_matches)
    return rows


def quick_eval_fn(spec: EnvSpec, n_matches: int, seed_base: int):
    """Callback for train.py: deterministic policy, returns headline numbers."""
    from evaluation.policies import TorchPolicy

    def run(model, global_step: int) -> dict:
        import torch

        policy = TorchPolicy(model, "training", deterministic=True,
                             device=next(model.parameters()).device)
        rng_state = torch.get_rng_state()  # evaluation must not disturb training randomness
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            rows = evaluate(policy, spec, n_matches=n_matches, seed_base=seed_base,
                            agent_name="training", policy_mode="deterministic")
        finally:
            torch.set_rng_state(rng_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        n = len(rows)
        return {
            "points": sum(r["points"] for r in rows) / n,
            "win_rate": sum(r["result"] == "win" for r in rows) / n,
            "goal_diff": sum(r["goal_diff"] for r in rows) / n,
            "ep_return": sum(r["ep_return"] for r in rows) / n,
        }

    return run
