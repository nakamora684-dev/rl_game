# Member A: movement and simulation foundation

This is a renderer-independent physics foundation, **not a complete football game**.
All runtime code uses the Python standard library. The reproducible engine tool setup
is documented in the root README; no manual global package installation is required.

## What exists

- `game/actions.py`: typed nine-direction movement intent and continuous effort.
- `game/config.py`: immutable, validated physics/pitch configuration and time constants.
- `game/player.py`: player snapshots, acceleration, idle braking, smooth facing, pitch bounds.
- `game/stamina.py`: effort-dependent consumption/recovery and gradual fatigue factors.
- `game/ball.py`: authoritative free-ball state and constant-deceleration motion.
- `game/simulation.py`: fresh reset, side swapping, shared two-player stepping, clock, and one-shot finish.
- `python -m game`: headless movement/clock inspection (no display); add `--stamina` for a sprint/rest CSV trace.

No kick processing, possession, goals, score, player-player collision,
renderer, Gymnasium wrapper, or PPO encoding is implemented yet.

## Coordinates and units

Positions are continuous logical units, **x right and y down**. Velocity is units/s;
acceleration and braking are units/s². Facing is radians: zero east, positive clockwise.
Direction IDs: 0 idle, 1 N, 2 NE, 3 E, 4 SE, 5 S, 6 SW, 7 W, 8 NW.
Diagonals have unit length; idle is exactly `(0, 0)`.
The physical forward vector is `(cos(facing), sin(facing))`. C may order observation
features as `(sin, cos)`; that is a feature ordering, not a geometric `(x,y)` vector.
The final observation and attack-normalization contract is not implemented here.

The full planned action is direction + effort + binary kick + power. The current
`MovementAction` implements only direction/effort, rather than silently accepting
and ignoring unimplemented kicks. Numerical adapters must explicitly use `Direction(id)`.
Effort outside `[0,1]` or non-finite inputs are rejected, not silently clipped.

## Starting tuning, not frozen parameters

| Parameter | Value | Interpretation |
|---|---:|---|
| Pitch | 800 × 500 | Logical dimensions |
| Player radius | 15 | Center kept at least this far inside each boundary |
| Base maximum speed | 160 units/s | Five seconds per 800 units at full-stamina cruising speed, ignoring acceleration/radius |
| Base acceleration limit | 320 units/s² | Rest to full speed in 0.5 s / 30 ticks when fatigue is held off |
| Idle braking | 40 units/s² | Student's provisional choice, not retuned to match new speed |
| Facing turn rate | 2π radians/s | Provisional configurable rate; quarter-turn takes 0.25 s |

At full speed, the current idle braking takes **4 seconds** to stop. This needs play
testing before freezing. The earlier nine-tick stop applies to initial speed 6, not 160.

## Player update

First update stamina from this tick's active requested effort, then compute fatigued
speed/acceleration limits as described below. The following movement steps use those
limits, not the unmodified base configuration. Never scale the existing velocity by fatigue.

1. For non-idle effort above zero, compute `target_velocity = direction * effort * max_speed`.
2. Move velocity toward that target by at most `acceleration * dt`, without overshoot.
   Lower effort reduces the target gradually; it does not instantly clamp current velocity.
   The vector limit is the same for diagonal movement and reversal.
3. Idle or zero effort uses constant braking instead. Scale both velocity components
   together, clamping speed to zero. A relative roundoff tolerance handles exact stop ticks.
4. Update position using the **new** velocity (semi-implicit Euler). This slightly
   underestimates continuous stopping distance, including no travel on the final stop tick.
5. A non-idle direction turns facing along the shortest arc at the configured rate,
   even at zero effort. Idle always preserves facing. An exact 180° tie turns counterclockwise.
   Facing is independent of sliding velocity; it is not an extra aiming action.
6. With a pitch supplied, clamp the center within the player radius. Cancel only outward
   normal velocity; retain tangential sliding and inward motion. There is no wall bounce.

Effort scales target speed, not the active acceleration limit in this initial model.
Fatigue scales both the speed target and active acceleration limit. At full stamina,
active steering/reduction uses 320, whereas idle coasting uses braking 40.
This deliberately makes tiny positive effort brake harder than effort exactly zero;
the discontinuity is tested and should be revisited during movement/stamina tuning.
Idle ignores unused effort and counts as rest for stamina. Existing velocities can
temporarily exceed a lowered fatigue target; they approach it gradually instead of
being teleported to the cap. The isolated `step_idle_player` helper remains a
stamina-free braking primitive; use `step_player` for actual player updates.

## Stamina: approved experiment-0 settings

`StaminaConfig` records the configurable starting settings, not a claim of optimal tuning:

| Setting | Value |
|---|---:|
| Maximum stamina | 100 |
| Full-effort drain | 5 points/s, approximately 20 s from full to empty |
| Idle recovery | 100/15 points/s, approximately 15 s from empty to full |
| Neutral active effort | 0.5 (neither drain nor recovery) |
| Empty-stamina speed/acceleration factors | 0.5 / 0.5 |

For active effort `e` above neutral `n`, the rate is `-drain_rate * (e-n)/(1-n)`.
At or below neutral it is `recovery_rate * (n-e)/n`. Idle always uses `e=0`.
Stamina advances by `rate * dt` and is clamped to `[0,100]`; minor roundoff near
an exact-duration endpoint is normal. Driving against a wall still consumes requested
effort, and coasting at idle restores energy. There is no kick cost.

The two performance factors are linear: `minimum_factor + (1-minimum_factor) * stamina/100`.
At stamina 50, both defaults give 0.75: base speed 160 becomes target limit 120,
and acceleration 320 becomes limit 240. A player already moving at 160 therefore
slows toward 120 by at most 4 units/s per tick, rather than instantly losing 40.
Braking, facing, and turn rate are not scaled. Each tick derives limits from the
original base configuration, so fatigue factors never compound into the configuration.

The sprint/rest inspection command logs one-second samples for a single player in
open space (no pitch clamp), using the same `step_player` function as the simulation:

```powershell
uv run --locked --only-group engine python -m game --stamina
```

Expected checkpoints verified by tests: stamina/speed limit are 95/156 at second 1,
50/120 at second 10, approximately 0/80 at second 20; fifteen subsequent seconds
of rest restore stamina to 100. This log is inspection evidence, not a playability benchmark.

## Free ball: first A3 increment

`BallConfig` uses the approved starting reference speed **300 units/s** and constant
deceleration **100 units/s²**. This reference roll takes 3 s to stop and travels 450
units; a weaker 150-unit/s roll stops after 1.5 s and 112.5 units. Reference speed
is not a speed cap, maximum kick speed, or a kick-power mapping.

`BallState` stores continuous position/velocity. It is free-only for now: ownership,
last touch, and controlled attachment will be introduced together with their rules.
Free-ball position is authoritative state, not derived from a player.

For each fixed tick, preserve velocity direction, subtract `deceleration * dt` from
speed without reversing, and integrate average speed over the moving portion of
the tick. If stopping occurs early, the ball travels until that instant then rests.
This analytic constant-deceleration integration differs intentionally from the
player's approximate velocity-first position update; it preserves the 450-unit roll.
Zero deceleration means constant-velocity coasting. The ball is **not pitch-clamped**:
crossing an edge currently emits no goal/out-of-bounds event because those rules do
not exist yet. No tunnelling-safe collision/goal detection is claimed.

The shared simulation resets the ball at pitch center, advances it once per playing
tick, and freezes it during the existing restart/finished scaffolds. Players do not
pick it up or collide with it yet. Both `step_tick` and `step_decision` use the same update.

```powershell
uv run --locked --only-group engine python -m game --ball
```

This prints a four-second roll inspection CSV. At seconds 1/2/3, position x is
250/400/450 and speed is 200/100/0, from initial position `(0,0)`. Tests also cover
partial-tick stopping, diagonal direction, weaker rolls, zero friction, reset,
six-tick equivalence, custom configuration, edge crossing, and immutable state.

## Time, phases, events, and reset

- `step_tick` advances exactly `1/60` simulated second, independent of wall time.
- `step_decision` holds both movement actions for up to **six separate ticks** (10 Hz).
  It aggregates events from every tick and stops early at the final whistle.
- `StepResult.physics_ticks` counts all advanced simulation ticks, including frozen
  restart ticks. It is not a count of ticks with player displacement.
- The 60-second match clock counts **3,600 ticks, including restart ticks**.
  Remaining time is derived from that integer tick. Finished states never advance again.
- The `match_finished` event appears once at tick 3,600, not on subsequent calls.
- Both players read the same input snapshot; physics never imports rendering/training libraries.
- `reset()` creates independent immutable state at `(200,250)` / `(600,250)`.
  It also resets a stationary free ball at pitch center `(400,250)`.
  `reset(swap_sides=True)` swaps spawn/facing and attack orientations. No randomness is
  used yet, so there is no seed argument or RNG state to reset at this stage.

Restart phases are **testable scaffolding only**: no rule currently creates them.
They freeze both player snapshots and decrement an integer countdown while time runs.
That currently includes stamina: the placeholder restart freeze neither drains nor
recovers it. This preserves existing scaffold semantics; actual football restart
energy recovery remains a decision to confirm before implementing restart rules.
The last countdown tick stays frozen; motion resumes next tick. A held movement action
can resume within the same six-tick block. Actual restart placement/ownership, whether
policies are queried then, and kick request expiry remain decisions for the next phases.

## Small example

```python
from game.actions import Direction, MovementAction
from game.simulation import reset, step_decision

state = reset()
result = step_decision(state, MovementAction(Direction.EAST, effort=1.0))
print(result.state.simulation_tick)       # 6
print(result.state.player_a.velocity)     # approximately (31.953333, 0)
print(result.state.player_a.position)     # approximately (201.864593, 250)
print(result.state.player_a.stamina)      # approximately 99.5
```

Run `uv run --locked --only-group engine python -m game` for a complete clock demo:
the first five decisions accelerate A, then A coasts to rest. Inspect speed 158 and
stamina 97.5 at tick 30, then zero speed/full stamina at tick 270. There should be
one terminal event at tick 3,600.

Automated tests cover normalized directions, effort, reversal, braking, facing,
boundary sliding, immutable inputs, reset isolation, side swapping, six-tick
equivalence, restart timing, finite replay, and exactly-once termination.
Stamina tests additionally cover rate interpolation, both clamps, linear fatigue,
no momentum scaling, no compounded configuration, wall effort, recovery while coasting,
two-player symmetry, decision/tick integration, reset, and the CSV inspection command.
Earlier movement-equation tests explicitly disable fatigue through configuration
to isolate their numerical expectations; integration/stamina tests exercise the defaults.
These are deterministic unit checks, not evidence that gameplay is realistically tuned.
Visual inspection with B: later overlay velocity/facing arrows and check that idle
coasting preserves orientation, turning follows intent, and wall contact permits sliding.
