# Training, evaluation and release tooling (Member D)

Status: implemented and unit-tested; **no PPO training run has been executed yet** — no learning
or performance claim is made. Reward weights are Member C's experiment-0 values, untouched.

## Quick start

```powershell
uv sync --locked                       # full stack (PyTorch, TensorBoard, ...)
uv run --locked pytest tests/test_d_*.py -q

# 1. smoke-test the whole pipeline in about a minute (placeholder opponent)
uv run python train.py --run-name smoke --opponent chaser --total-timesteps 20000 --n-envs 4

# 2. real run against Member B's baseline (agents/provider.py)
uv run python train.py --run-name exp0 --opponent rule_based --n-workers 4
uv run tensorboard --logdir logs

# 3. evaluate (held-out seeds, both attack directions) and plot
uv run python test.py --agents models/exp0/best.pt random idle heuristic --opponent rule_based --matches 100
uv run jupyter lab evaluate.ipynb      # set RUN = "exp0" in the first cell
```

## What was added

| Path | Purpose |
|---|---|
| `agents/networks.py` | Custom `ActorCritic` ("GroupResNet"): one embedding per observation group (self/opp/ball/geom/context/interaction/phase, from `OBS_FEATURES`) → residual trunk → heads. Separate actor/critic encoders by default. |
| `agents/ppo.py` | Clipped PPO + GAE in PyTorch, checkpoints (`load_checkpoint` rejects observation-version mismatches), resume. |
| `agents/action_codec.py` | Flat 4-vector ⇄ the env's hybrid Dict action. Single conversion point for train/test/observe. |
| `agents/opponents.py` | Opponent registry: `idle`, `chaser` (**temporary stand-in**), `rule_based` (Member B's class). |
| `training/vec_env.py` | `EnvSpec`, per-episode auto-reset, random sides, episode summaries, sync or `spawn`-worker vectorisation (Windows-safe). |
| `training/returns.py` | Torch-free GAE and running return statistics (verified against a brute-force reference). |
| `training/logger.py` | TensorBoard + CSV (`episodes.csv`, `updates.csv`, `evals.csv`). |
| `training/start_jitter.py` | Optional seeded start-position jitter (engine adapter; no physics change). |
| `evaluation/` | `metrics.py` (CSV schema v1, per-match recorder, Wilson/bootstrap intervals), `runner.py`, `policies.py`. |
| `agents/policy_agent.py` | Runs a checkpoint inside B's 60 Hz viewer (`observe.py --p0 ppo --model ...`); verified to reproduce the training/eval path exactly (same final engine state). |
| `train.py`, `test.py`, `evaluate.ipynb`, `configs/ppo_default.json` | Entry points, notebook, default config. |
| `tests/test_d_*.py` | Unit/integration tests (torch tests skip without PyTorch). |

## Action distribution (why a custom PPO)

SB3's PPO cannot take the env's mixed Dict space. The policy samples
`direction ~ Categorical(9)`, `kick ~ Bernoulli`, `effort ~ Beta`, `power ~ Beta` (shape ≥ 1,
so always in [0,1] and never clipped/rejected by the env). Effort's log-prob counts only when
`direction != idle`, and power's only when `kick == 1`, because the engine ignores them otherwise
(`game/stamina.py`, `game/simulation.py`). Nothing is discretised.

## Training/evaluation conventions

* Logged `rollout/ep_return` is the **raw undiscounted** total episode reward (+ moving average).
  Reward scaling (running return std) affects only the learner.
* `terminated` at the whistle → no bootstrap; `truncated` (safety only) → bootstrap from the final obs.
* Training sides are randomised per episode (identity and attack direction).
* `test.py` cycles the four (identity × attack-direction) combos, uses the same seeds for every agent,
  and never updates weights. Training env seeds are `< 100 000`; periodic eval uses `500 000+`;
  `test.py` defaults to `1 000 000+`.
* Periodic evaluation (deterministic policy) saves `best.pt` by `(points, goal_diff)`.
* Per-match CSV columns are fixed in `evaluation/metrics.MATCH_COLUMNS` (schema v1).

## Open items that need other members

1. **Member B** — integrated: `--opponent rule_based[_easy|_hard]` uses `agents.provider.rule_based_provider`.
   Observation: against it, a hand-written obs-only reference agent (`heuristic`) won 17/20 vs `normal`,
   18/20 vs `hard` but only 7/20 vs `easy` (20 matches each, jittered starts) — the presets are not
   monotonic in strength and the baseline is beatable by a very simple strategy; worth a look before
   claiming "PPO beats the baseline" means much. Report results against all presets.
2. **Members A/C** — the engine reset and the rule-based opponent are deterministic, so a deterministic
   policy replays identical matches (only 4 distinct ones). `test.py` therefore defaults to *stochastic*
   actions and offers `--start-jitter`; please approve or replace with an engine-level seeded reset.
3. **Member C** — (a) PPO `gamma=0.995` vs the env's shaping `gamma=1.0`: decide whether to align
   (`train.py --shaping-gamma`). (b) Reward-hack probe: the steal/recovery loop is farmable against the
   placeholder opponent (a reference agent logged ~33 steals per match with positive return and
   goal difference ≤ 0). Re-check against the real baseline. (c) `evaluation` reads `env._state`; a public
   read-only accessor would be cleaner.
4. **Everyone** — fill in "Machine specs" in the README after the first benchmark; measure steps/second
   (`perf/steps_per_sec` in TensorBoard) before choosing `--total-timesteps`.

## Reward exploit found in exp0 (steal / loose-ball loop)

Observed in `observe.py`: the trained agent steals, immediately kicks the ball away, the baseline
recovers the loose ball, and the agent steals again (event log: `Steal BLUE -> Kick BLUE -> Loose ball
won RED`, repeated within a second). In `environment/reward.py` a steal pays +1, a kick costs nothing
and the opponent's recovery costs the agent nothing, so each cycle nets about +1 (plan section 5,
probe 1). Rising return with no goals is the signature.

Experiment-1 setting (remove the farmable terms; goals, outcome, progress shaping and the steal-lost
penalty stay):

    uv run python train.py --run-name exp1 --opponent rule_based --n-envs 8 --n-workers 4 --start-jitter 25 --reward steal=0 loose_recovery=0

Do NOT resume exp0 (the policy has learned the loop). Evaluate exp1 with the same `--reward` flags only
if you want comparable `ep_return` columns; outcome metrics are unaffected by reward settings. If a
steal bonus is wanted later, pair it with a penalty for losing the ball right after (Member C).
`test.py --agents rule_based rule_based_hard ...` evaluates Member B's baseline as an agent in the same CSV.
