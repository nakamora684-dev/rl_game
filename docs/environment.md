# Real-engine Gymnasium environment — contract version 2

The student approved preserving hybrid controls, exposing actual interaction/restart
state, and fixing compatibility **without retuning rewards**. `game/` is unchanged.
The wrapper has no renderer and is **not yet directly trainable with standard SB3 PPO**.

## Reproduce without the training/graphics stack

```powershell
uv sync --locked --only-group environment
uv run --locked --only-group environment pytest -m "not full_stack" -q
uv run --locked --only-group environment ruff check game environment tests
uv run --locked --only-group environment ruff format --check game environment tests
```

This group installs the already-declared NumPy/Gymnasium requirements plus engine
test tools, not Torch/Pygame/SB3. Engine-only runs visibly skip tests needing optional
packages rather than treating missing Gymnasium checks as passes.

## Four integration steps

1. **Adapter:** `environment/engine.py::SimulationEngine` owns one immutable native
   state and delegates every decision to `game.simulation.step_decision`. Its
   read-only `MatchState` projection is not a second simulation. Reset clears the
   adapter's event IDs. An explicit adapter `initial_state` is for resumed/scenario
   inspection and must use matching config, just as in the headless runner.
2. **Actions/observations:** replace the obsolete continuous five-value action with
   the approved hybrid Dict; publish the 50-feature observation version below.
3. **Events/episodes:** preserve restart decisions and all native telemetry; classify
   only genuine gameplay recoveries/steals for rewards. Keep experimental weights.
4. **Tests:** use real-engine initial scenarios, exact shared-step comparisons,
   both identities/orientations, full matches and Gymnasium's environment checker.

## Actions: policy frame versus world frame

```python
import numpy as np
from environment.football_env import FootballEnv

env = FootballEnv()  # real engine, idle diagnostic opponent by default
obs, info = env.reset(seed=42, options={"agent_idx": 0, "swap_sides": False})
obs, reward, terminated, truncated, info = env.step({
    "direction": 3,  # EAST in policy frame: towards the attacking goal
    "effort": np.array([0.7], dtype=np.float32),
    "kick_trigger": 0,
    "kick_power": np.array([0.0], dtype=np.float32),
})
```

| Dict key | Gymnasium space | Engine meaning |
|---|---|---|
| `direction` | `Discrete(9)` | 0 idle, 1 N, 2 NE, 3 E, 4 SE, 5 S, 6 SW, 7 W, 8 NW |
| `effort` | `Box(0,1,shape=(1,),float32)` | Continuous movement effort |
| `kick_trigger` | `Discrete(2)` | One request, consumed on the first physics tick |
| `kick_power` | `Box(0,1,shape=(1,),float32)` | Continuous power; triggered zero still kicks at 50 units/s |

The wrapper maps left-attacking policy directions with a horizontal mirror, not
angle rounding. Effort/power remain continuous, not discretized. Invalid/nonfinite
actions are rejected, not silently clipped. Facing controls the kick; there is no
independent aiming or charging. The engine handles cooldown and failed-request
discard/no-retry semantics. `ACTION_DIM=4` counts **heterogeneous components**, not
a flat four-number Box. Old five-value actions/checkpoints are incompatible.

**Member D:** standard Stable-Baselines3 PPO does not accept this mixed Dict action
space. Agree a suitable mixed-action policy/distribution/integration with C before
training. This change deliberately does not implement it or alter `train.py`.

## Opponent and orientation handoff to B/C/D

`FootballEnv(opponent=provider)` accepts the **same** provider signature as
`game.runner`: `(SimulationState, PlayerId) -> PlayerAction`. It receives the
immutable native pre-decision state and returns **world-space** controls. The
wrapper never mirrors the opponent action a second time. The idle default is an
inspection controller, not the evaluation baseline.

Index 0 is A and index 1 is B, independently of side. `agent_idx` can be fixed to
0/1 or chosen by the wrapper's seeded RNG. `swap_sides=True` uses the engine's
side-swapped reset. Evaluate both orientations explicitly. Current engine resets
are deterministic and have no seed argument. Fresh deterministic providers yield
repeatable trajectories; callers must reset/seed any private state/RNG in supplied
opponents themselves. The wrapper does not silently reseed a teammate's controller.

## Observation version 2 — 50 float32 values

`spec.OBS_NAMES` is the authoritative feature order. `info["observation_version"]`
is 2. The first 35 features retain C's original order; indices 35 onward replace
the old, nonexistent per-player "steal cooldown" model.

| Indices | Features |
|---|---|
| 0–6 | Self position, velocity, facing `(sin,cos)`, stamina |
| 7–13 | Opponent relative position, velocity, facing `(sin,cos)`, stamina |
| 14–20 | Ball relative position/free velocity, self/opponent/free possession one-hot |
| 21–32 | Unit direction and distance to ball, opponent, own goal, attacking goal |
| 33–34 | Remaining time and clipped score difference |
| 35–37 | Contest active, contest progress, possession protection |
| 38–41 | Self/opponent kick cooldown, then self/opponent pickup lockout |
| 42–45 | Playing/goal restart/boundary restart/finished phase one-hot |
| 46 | Restart countdown |
| 47–49 | Restart receiver: self/opponent/neutral (all zero outside restart) |

Positions, player velocity, goal geometry and timers use the **actual adapter
configuration**, not stale placeholder pitch/speed/contest constants. The default
steal duration is 18 ticks, protection 30, kick/pickup timers 12 and restart 60.
All timer features are fractions of their configured duration, clipped to [0,1].
Zero-duration configurable timers use a safe denominator of one. Ball velocity
uses an explicit 600-unit/s observation scale, **not** a physics speed cap; extreme
velocities saturate. Score difference retains C's clipping at ±3 goals. Attached
ball free velocity is zero. Last toucher is bookkeeping, not an observation;
the public restart receiver is exposed because it affects the next transition.
Normalization/clipping means this vector is not claimed to be a lossless Markov
state under arbitrary extreme/custom configurations.

**Member D:** update network input shapes; old 38-feature checkpoints are not
compatible. This is a student-approved interface change, not proof of team sign-off.

## Timing, events and rewards

- Every `step` advances up to six ticks. A normal full match is still 600 calls /
  3,600 ticks. Query the opponent during restarts; never fast-forward a restart.
- Goals do not terminate the episode. Match time continues through restarts. The
  final natural whistle gives `terminated=True`; terminal stepping requires reset.
- If a restart ends partway through a decision, held movement resumes, but a kick
  rejected on its first tick is not retried. Motion/stamina freeze during restarts.
- Reward projections map `goal` using **scoring_player**, including own goals;
  `out_of_bounds` uses last toucher, which can be unknown. Unknown means no blamed
  player/penalty, never A-first responsibility.
- Only `possession_changed/cause=recovery` earns loose-recovery reward and only
  `cause=steal` earns steal reward. Setup, kickoff, boundary placement and post loss
  are not recoveries. A kick is not automatically a penalized turnover.
- Keep native timestamps/order: kicks use pre-integration tick; goals/acquisitions
  use completed tick. Validation allows these conventions without rewriting stamps.
- Episode-local monotonically assigned event IDs distinguish genuine same-type,
  same-tick repeated contacts/acquisitions. Replaying an ID raises a duplicate error.
- `info["events"]` holds **all** original events as field dictionaries, including
  causes, ownership, collision fraction/index, positions, velocities and diagnostic
  `physics_limit`/`ball_constraint` events. Reward projections do not drop telemetry.
- `reward_components` reports this step. Terminal `episode_stats.components` sums
  to the undiscounted return. Actual TensorBoard/Monitor wiring remains D's task;
  the wrapper does not overwrite Monitor's reserved `episode` key.

Weights remain C's experimental defaults: goals ±10, result +5/0/−5, steal +1,
recovery +0.5, stolen possession −0.5, blamed boundary −0.5, possession rate 0,
shaping λ=2, γ=1, terminal-potential zeroing disabled. No tuning was authorized.
Before training, C/D must revisit discount/terminal-potential consistency, incentives
from free-ball progress/restarts and lose/recover loops. Passing compatibility tests
does **not** validate reward optimality or establish that exploits are impossible.

## Evidence and limits

`tests/test_contract.py` compares the adapter's first decision for all 50 existing
diagnostic starting scenarios with a direct shared-engine decision, preserving raw
events and native state exactly. This is not a rerun of each scenario's full audit.
`tests/test_football_env.py` also compares a complete seeded wrapper match with the
headless runner, accounting for the float32 Gym action boundary. The old scripted
`tests/fake_engine.py` has been replaced by real-engine tests; scenario test engines
only override the starting state, never physics or event generation.

No renderer, trained agent, PPO distribution, baseline, benchmark or visual equivalence
is implemented/proven here. Physics retains its documented bounded-sweep limitations.

Executed on 17 Nov: the locked environment-group `pytest -m "not full_stack" -q`
run passed **692 tests and 36 subtests**, with the full-stack smoke test deselected.
Ruff check/format checks passed across `game environment tests` (51 formatted files).
After an exact engine-only sync removed NumPy/Gymnasium, the same test selection
passed **622 tests and 36 subtests**, with four optional-dependency modules visibly
skipped and the full-stack smoke test deselected. See `docs/next_session.md` for
the exact commands. The project virtual environment was restored to the approved
lightweight environment group.
