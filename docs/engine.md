# Member A: movement and simulation foundation

This is a renderer-independent physics foundation, **not a complete football game**.
All runtime code uses the Python standard library. The reproducible engine tool setup
is documented in the root README; no manual global package installation is required.

## What exists

- `game/actions.py`: movement intent and full `PlayerAction` (direction/effort/trigger/power).
- `game/config.py`: immutable, validated physics/pitch configuration and time constants.
- `game/player.py`: player snapshots, acceleration, idle braking, smooth facing, pitch bounds.
- `game/stamina.py`: effort-dependent consumption/recovery and gradual fatigue factors.
- `game/ball.py`: authoritative owner/last-touch state and free-ball motion.
- `game/geometry.py`: reusable straight-segment swept circle contact.
- `game/collisions.py`: equal-mass sliding player contact, solid posts and coupled correction.
- `game/rebounds.py`: relative normal reflection and nonpenetrating circle-union projection.
- `game/ball_contacts.py`: ordered pickup/post/player/exit processing and bounded multi-impact flights.
- `game/possession.py`: controlled attachment and swept automatic free-ball pickup.
- `game/rules.py`: whole-ball goal/boundary classification and explicit restart plans.
- `game/simulation.py`: shared movement/kick/pickup/steal/rules updates, score, restarts, clock, reset.
- `game/runner.py`: headless full-match loop with replaceable action providers and complete event collection.
- `python -m game`: headless movement/clock inspection (no display); add `--stamina` for a sprint/rest CSV trace.
- `python -m game --match`: full idle match; `--match scripted` uses fixed test actions, not a baseline opponent.

Locked-player and goalpost rebounds now share the simulation core. A headless match
runner is implemented; no renderer, Gymnasium wrapper or PPO encoding is implemented
here. Visual validation, adapter integration, tuning and further numerical hardening remain.

## Coordinates and units

Positions are continuous logical units, **x right and y down**. Velocity is units/s;
acceleration and braking are units/s². Facing is radians: zero east, positive clockwise.
Direction IDs: 0 idle, 1 N, 2 NE, 3 E, 4 SE, 5 S, 6 SW, 7 W, 8 NW.
Diagonals have unit length; idle is exactly `(0, 0)`.
The physical forward vector is `(cos(facing), sin(facing))`. C may order observation
features as `(sin, cos)`; that is a feature ordering, not a geometric `(x,y)` vector.
The final observation and attack-normalization contract is not implemented here.

`PlayerAction` implements direction + effort + binary kick + power as an immutable
subclass of `MovementAction`. Existing movement-only calls remain compatible and
request no kicks. Numerical adapters must explicitly use `Direction(id)`. Effort
and power outside `[0,1]` or nonfinite inputs are rejected, not silently clipped.
The trigger must be integer/bool 0 or 1, not an arbitrary continuous value. This
engine action is not an SB3 encoding; C/D still own that unresolved interface.

## Starting tuning, not frozen parameters

| Parameter | Value | Interpretation |
|---|---:|---|
| Pitch | 800 × 500 | Logical dimensions |
| Player radius | 15 | Center kept at least this far inside each boundary |
| Base maximum speed | 160 units/s | Five seconds per 800 units at full-stamina cruising speed, ignoring acceleration/radius |
| Base acceleration limit | 320 units/s² | Rest to full speed in 0.5 s / 30 ticks when fatigue is held off |
| Idle braking | 40 units/s² | Student's provisional choice, not retuned to match new speed |
| Facing turn rate | 2π radians/s | Provisional configurable rate; quarter-turn takes 0.25 s |
| Kick speed | 50–300 units/s | Approved starting linear mapping, not frozen tuning |
| Kick cooldown | 12 ticks / 0.2 s | Per player; simulation time runs during restarts |
| Ball radius | 7 units | Pickup contact at player radius + ball radius = 22 units |
| Locked-player ball restitution | 0.5 | Normal relative rebound; no player recoil |
| Goalpost radius / ball restitution | 4 units / 0.7 | Inner edges preserve the clear goal opening |
| Ball impact budget | 32 per tick | Shared across free flights/control-loss cycles; explicit saturation telemetry |
| Kicker pickup lockout | 12 ticks / 0.2 s | Separate from kick cooldown; only the kicker is blocked |
| Player contact | Equal mass, no bounce | Closing normal motion removed; tangential motion retained |
| Steal range / contest | 35 units / 18 ticks | Player-centre distance, 0.3 s of consecutive eligible ticks |
| Acquisition protection | 30 ticks / 0.5 s | After recovery, steal, kickoff, or owned boundary restart |
| Goal opening | 120 units | Centred; whole radius-7 ball must fit inside |
| Restart duration | 60 ticks / 1 s | Clock/timers run; motion/stamina frozen |

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

`BallState` stores continuous position, free-ball velocity, one `possession_owner`,
and a separate `last_touch_player`. When owner is `None`, position is authoritative
free-ball state, not derived from a player. `step_free_ball` rejects a controlled
ball instead of accidentally integrating attachment and free motion together.

For each fixed tick, preserve velocity direction, subtract `deceleration * dt` from
speed without reversing, and integrate average speed over the moving portion of
the tick. If stopping occurs early, the ball travels until that instant then rests.
This analytic constant-deceleration integration differs intentionally from the
player's approximate velocity-first position update; it preserves the 450-unit roll.
Zero deceleration means constant-velocity coasting. The ball is **not pitch-clamped**:
the standalone `step_free_ball` primitive only integrates motion. Shared simulation
stepping adds swept pickup and whole-ball goal/out-of-bounds rules, described below.
Do not call the roll primitive alone as a complete gameplay tick.

The shared simulation resets the ball at pitch center and freezes it during
restart/finished phases. Within each playing tick, `integrate_free_ball` integrates
analytic rolling over finite nonnegative durations between ordered contacts.
`step_free_ball` remains its fixed-tick wrapper. Shared stepping adds eligible pickup,
solid rebounds and exits; clocks/timers still advance once, not once per impact.
Both `step_tick` and `step_decision` use the same update.

```powershell
uv run --locked --only-group engine python -m game --ball
```

This prints a four-second roll inspection CSV. At seconds 1/2/3, position x is
250/400/450 and speed is 200/100/0, from initial position `(0,0)`. Tests also cover
partial-tick stopping, diagonal direction, weaker rolls, zero friction, reset,
six-tick equivalence, custom configuration, edge crossing, and immutable state.

## Controlled possession and dribbling

The student-approved attachment direction is **facing, not velocity**. After both
players' movement, facing, and pitch constraints are updated, the simulation chooses
exactly one ball update: free integration when owner is `None`, otherwise attachment.
For carrier `(x,y)`, facing angle `theta`, and offset `d`:

`ball_position = (x + d*cos(theta), y + d*sin(theta))`.

`BallConfig.dribble_offset` defaults to **22 logical units**, a provisional
center-to-center distance. It must be finite and positive. With the new student-approved
ball radius 7 and player radius 15, offset 22 places the attached ball at surface contact.
These are configurable starting values, not frozen optimal geometry.
The attached position is a synchronized snapshot of this derived value, not an
independent moving state. Rendering reads it but never maintains another attachment.
The ball remains ahead of facing when the player is stationary or sliding sideways.
It is not pitch-clamped, even if the offset puts it beyond an edge; shared goal/boundary
rules detect and resolve full-ball exits rather than hiding the crossing by clamping.

`PlayerId.A` / `PlayerId.B` are stable identities, not left/right sides. The ball
stores one owner and the separate last toucher; player possession flags are derived
with `state.has_possession(PlayerId.A)`. A controlled ball must have zero **free-ball**
velocity and its last toucher must match its owner. Zero here is an explicit storage
convention: carrier motion changes ball position; it does not mean the attached ball
is stationary in world coordinates. A free ball may retain a last toucher after release.

`assign_possession(state, PlayerId.A)` is an explicit **setup/test helper**. It can
move the ball from anywhere to the carrier, so it is not an agent action or a legal
pickup/steal implementation. It attaches immediately, clears free velocity, updates
last touch, and preserves the players' stamina/motion and simulation tick. It returns
a `StepResult` with zero advanced ticks and one `possession_changed` event carrying
`previous_owner`, `new_owner`, and `cause="setup"`. C must not reward setup events as
steals or loose-ball recoveries. Same-owner assignment emits no new transition; a
changed offset can still resynchronize position. No event repeats merely for dribbling.
Assignments are rejected during restart/finished phases. Restart phases freeze the
existing attachment, and reset clears all possession/last-touch history.

```python
from game.player import PlayerId
from game.simulation import assign_possession, reset, step_decision

state = assign_possession(reset(), PlayerId.A).state
assert state.has_possession(PlayerId.A)
assert not state.has_possession(PlayerId.B)
state = step_decision(state).state
assert state.ball.position == (222.0, 250.0)
```

```powershell
uv run --locked --only-group engine python -m game --dribble
```

The CSV samples ticks 0/6/24/84: initial attachment, eastward movement, north-facing
eastward coasting, then rest. At tick 24, velocity still points east, but the ball
is 22 units north of the carrier. Tests cover both owners, side swapping, all-facing
attachment geometry, custom offset, no independent ball motion, setup event idempotence,
last-touch persistence, reset, restart/finish freezes, and full-match deterministic replay.
Direct kick release, kick actions, automatic recovery, collision response, stealing,
and possession protection are described below.

## Direct kick release: physics operation, not a full action

`release_kick(state, PlayerId.A, launch_speed=300.0)` returns a zero-tick `StepResult`.
It requires a playing phase and that the named player owns the ball. It synchronizes
the current attachment, then atomically clears ownership and sets free velocity to
`launch_speed * (cos(facing), sin(facing))`, retaining the kicker as last toucher.
The kicker's pickup lockout is initialized to 12 ticks (configurable); player motion,
stamina, phase, and clock are unchanged. The next `step_tick` or
`step_decision` integrates the now-free ball using the existing friction.

This low-level API takes an explicit **finite positive world-space speed** in units/s;
it does not add carrier velocity. Thus even a sideways-coasting player's shot follows
facing exactly. It has no reference-speed cap. Zero/negative/nonfinite speed, a free
ball, the wrong owner, and non-playing phases raise errors; these exceptions are
programmer guards, not gameplay failed-request handling. The helper does not check or
start kick cooldown and is **not an agent action API**; use shared stepping for gameplay.
Its explicit speed is not capped by `KickConfig`. Tests/setup may use it to isolate
release physics; gameplay actions use the approved mapping below.

One `ball_kicked` event records the unchanged tick, `previous_owner=kicker`,
`new_owner=None`, and `cause="kick"`. This event already describes the possession
transition; no duplicate `possession_changed` event is emitted. It is not a recovery,
steal, guaranteed shot on goal, or automatically penalized turnover. Reward interpretation
belongs to C. Ordinary free-motion ticks emit no repeated kick. A second release
without possession raises an error.

```powershell
uv run --locked --only-group engine python -m game --kick
```

The demo explicitly uses the **reference roll speed**, not a maximum-power shot, and
moves the opponent to `(600,400)` so the roll is not intercepted.
From `(222,250)` it travels to x=472/622/672 at 1/2/3 seconds, stopping after 450 units.
Tests cover both owners, cardinal/diagonal facing, unchanged player/stamina snapshots,
side swapping, custom attachment, invalid release guards, next-tick friction, no
duplicate event, and deterministic terminal stepping. For later visual inspection,
overlay facing and velocity arrows and verify that release detaches along facing.

## Kick actions: approved first-tick request semantics

`KickConfig(min_speed=50, max_speed=300, cooldown_ticks=12)` is configurable starting
tuning. The linear mapping is `speed = min_speed + (max_speed-min_speed)*kick_power`.
Trigger 0 never kicks, regardless of power. Trigger 1 with power 0 produces a 50-unit/s
kick, not a no-op. Power 0.5 produces 175; power 1 produces 300. With deceleration 100,
these stop after 0.5/1.75/3 seconds and travel 12.5/153.125/450 units. Linear speed is
**not** linear distance: stopping distance is `speed²/(2*deceleration)` when friction
is positive. The maximum equals the reference roll speed by the student's new choice,
not because the reference field imposes a speed cap. Weak kicks are very short, while
the strongest kick cannot cover the whole 800-unit pitch; validate these gameplay tradeoffs.

`step_decision` accepts either `PlayerAction` or movement-only actions for each player.
It attempts each requested kick **only on the first physics tick**, after updating both
players' movement/facing and before free-ball integration. Only the current owner, in
playing phase and with zero kick cooldown at the start of that tick, may kick. A
successful kick integrates friction/motion in that same tick and emits one `ball_kicked`
event at the pre-integration tick, consistent with the direct release operation. The
other five ticks retain movement but strip kick requests. Missing possession, cooldown,
or a tick starting in restart discards the request without events or a new cooldown.
If a restart/cooldown expires later within the decision, the kick is not retried.
Every new decision may request a kick, even reusing an action whose trigger stays 1;
there is no hidden rising-edge, pending request, or action-object mutation.

`step_tick` is the low-level one-tick interface: every call with a triggered `PlayerAction`
is a **fresh request**. Callers holding actions for six ticks must strip the trigger
themselves after the first tick, or simply use `step_decision`. `step_player` and stamina
helpers process movement only; they do not attempt kicks or advance interaction timers.

`PlayerState.kick_cooldown_ticks` is explicit, initialized/reset to zero. A successful
kick initializes it to 12. Each advanced simulation tick decrements it, including the
release tick and restart ticks, clamped at zero. A kick at tick t therefore yields 11
at state tick t+1, 6 at t+6, and 0 at t+12; another kick can succeed starting at t+12
if possession has been recovered. A request made one tick too early is discarded,
even if the timer reaches zero before the next decision. The opponent has a separate
timer. Finished states remain frozen. This timer is **not** a pickup/steal lockout;
the kicker pickup lockout is a separate field. Setup assignment preserves timers and cannot
bypass the gameplay kick guard. No kick stamina cost or fatigue-dependent power is added.

```python
from game.actions import PlayerAction
from game.player import PlayerId
from game.simulation import assign_possession, reset, step_decision

state = assign_possession(reset(), PlayerId.A).state  # Test setup, not gameplay pickup.
result = step_decision(state, PlayerAction(kick_trigger=1, kick_power=0.5))
assert result.state.simulation_tick == 6
assert result.state.ball.possession_owner is None
assert result.state.player_a.kick_cooldown_ticks == 6
assert [event.kind for event in result.events] == ["ball_kicked"]
```

```powershell
uv run --locked --only-group engine python -m game --kick-actions
```

This CSV compares independently reset powers 0/0.5/1 after six ticks. Speeds are
40/165/290 (friction has already acted for 0.1 s), positions x=226.5/239/251.5, and
each run emits one kick. Tests additionally cover updated facing, both owners,
simultaneous requests, cooldown boundaries, failed-request expiry, restart transitions,
custom tuning, movement-only compatibility, reset/replay, and a kick at the final tick
without losing the terminal event. Core pickup/steal/rules now exist, but visual tuning
and integration/hardening remain. Coordinate this `PlayerAction` contract and
the observable cooldown field with B/C/D before their adapters are implemented.

## Automatic free-ball pickup: first contact, not stealing

The approved initial rule controls a free ball **at any speed** on contact. Contact
distance is `config.pitch.player_radius + config.ball.radius`, currently 15+7=22 units.
`select_pickup_owner` checks both eligible players without mutating input snapshots.
It sweeps relative movement between the ball/player start and end positions, rather
than checking only endpoints. A fast ball crossing through a player is therefore not
missed just because it ends beyond them. First contact wins. Simultaneous contacts
use the nearer player at contact; an exact tie stays free, avoiding identity bias.
Floating-point ties use fraction tolerance 1e-12 and distance tolerance 1e-9 logical
units (plus 1e-12 relative); these are numerical tolerances, not gameplay reach margins.

For relative starting displacement `r`, relative tick displacement `d`, combined
radius `R`, contact satisfies `|r + fraction*d|² <= R²` for fraction in `[0,1]`.
This is a **linear snapshot sweep**, not an exact friction/acceleration trajectory
inside the tick. Free-ball endpoints still use analytic friction. Pickup then attaches
the ball at the winner's updated end-of-tick position/facing and zeroes free velocity;
it does not simulate contact impulses or residual free motion. Player-pitch clamping
is likewise represented by the segment between snapshots. Goal/boundary precedence
now compares contact and exit fractions on the same sweep; player-player contact has
its own response below. These are not exact accelerated sub-tick trajectories or a
full multi-impact rigid-body solver.

A recovered ball sets one owner and matching last toucher. It emits one
`possession_changed` event with `previous_owner=None`, `new_owner=winner`, and
`cause="recovery"`, at the **completed tick** (`old simulation_tick + 1`). It is not
`cause="setup"` or a steal from controlled possession. C owns reward arithmetic and
must distinguish these causes. Subsequent attached ticks emit no repeated recovery.
Simultaneous ties and failed eligibility emit no recovery event; solid contacts may
emit collision events instead. Ordinary unimpeded free motion emits no event.

`PlayerState.pickup_lockout_ticks` is separate from `kick_cooldown_ticks`. Successful
release (including the low-level `release_kick` helper) initializes only the kicker's
pickup lockout from `PossessionConfig.kicker_pickup_lockout_ticks`, default 12.
The opponent remains eligible, even in the release tick. Shared stepping evaluates
pickup before decrementing timers; a kick at tick t leaves 11 at t+1 and 0 at t+12,
allowing renewed pickup starting with the tick at t+12. Both timers run during
restarts, freeze when finished, and reset to zero for both players. The lockout is
not renewed by recovery. A player whose kick cooldown remains active may recover
but cannot kick until that separate cooldown expires. No pickup is processed on
restart ticks, including the last countdown tick.

For a ball already free at tick start, sweep both players' old→updated positions.
For a new kick, release occurs after player motion, so the pickup sweep begins at
the release attachment and uses the already-updated player positions; it must not
retroactively sweep the shot through pre-release player movement. Recovery follows
kick processing: a first-tick request that failed for lack of possession is **not**
retried if pickup succeeds later that tick or decision. A kick and opponent recovery
can both occur in one tick; a final-tick recovery and `match_finished` are both kept.

```powershell
uv run --locked --only-group engine python -m game --pickup
```

The default setup demo kicks east from A: tick 1 is free with A's pickup lockout 11;
at tick 12 it is zero; B intercepts at tick 98, controlling at `(578,250)` with one
`cause="recovery"` event. Tick 104 shows no repeat. In contrast, `--kick` moves B out
of the path to isolate the 450-unit roll. The movement/clock demo now may also emit
recovery events; its `terminal_events` field counts only `match_finished`, not all events.
Tests cover both owners, exact contact and near misses, tangency, high-speed crossing,
moving-player sweeps, earliest contact, simultaneous ties, lockouts and interception,
restart/finish behavior, reset and deterministic replay. Visual inspection with B:
overlay contact circles and sweep segments, then check an interception at maximum speed.
Stealing, player-player response, goals and restarts are described below. Pickup-locked
players remain ineligible to trap but are solid ball obstacles, as described next.

## Ball rebounds, solid posts and controlled post loss

Approved starting restitution is **0.5 off pickup-locked players** and **0.7 off posts**.
Only closing normal velocity is reflected, in the surface's reference frame:
`v_after = v_before - (1+e) * dot(v_before-v_surface, normal) * normal`.
Tangential velocity is retained. Separating/tangent surface touch does not repeatedly
bounce. Players receive no ball-impact recoil. Moving-player surface velocity uses
resolved physical player velocity, not snapshot displacement divided by `dt`: geometric
depenetration must not create rebound energy. Snapshot paths still determine contact
geometry. Freshly kicked balls use stationary updated release snapshots and zero
surface velocities, consistent with the existing post-movement kick convention.
Injected initial body/post overlaps are repaired before movement/pickup sweeps are
constructed, without advancing time or stamina. Ordinary movement/contact resolution
then follows; correction is not swept as a moving bat through the ball.
Eligible players still trap at any speed instead of rebounding. Exact contested
pickup ties stay free and resolve solid contact without choosing an A-first owner.

`RulesConfig.post_radius=4` places post centres at `(0,186)`, `(0,314)`, `(800,186)`,
`(800,314)`. Their inner edges preserve the 120-unit clear opening (190..310).
Ball/post centre contact distance is 11; player/post distance is 19. Player bodies
slide against posts without bounce and remain inside the original rectangular bounds.
Body/post sweeps allow 16 contacts, then report `physics_limit`, cause
`player_post_budget`, if exhausted; coupled body/post overlap projection allows 64
iterations and raises on nonconvergence rather than publishing overlapping bodies.

For each remaining ball flight, analytic rolling supplies endpoints; linear relative
sweeps find the earliest post, player or strict exit. At fraction ties within 1e-12,
the priority is **post, eligible pickup, solid player, strict exit**. Rebounds continue
for the remainder of the tick, allowing post→pickup or post→goal in one tick.
Genuine player solid contact updates last toucher; post contact preserves it.
Simultaneous ambiguous player touches set it to `None`, allowing a neutral boundary
restart instead of inventing responsibility. Depenetration alone adds no launch energy.

When attachment motion/turning hits a post before an exit, release possession with
`possession_changed`, `cause="post_collision"`, and the former owner's pickup lockout
initialized to 12 (11 in the completed snapshot). Do not emit a kick or start kick
cooldown. Impact velocity is physical carrier velocity plus facing-offset displacement
divided by `dt`, excluding positional correction; stored controlled velocity remains
zero. Recovery/steal attachment jumps use incoming or
carrier-derived velocity, never teleport displacement. Steal acquisition occurs at
tick end; recovery attachment uses the pickup fraction. Obstructed attachment can
therefore lose control immediately, clearing acquisition protection/progress.

`BallConfig.max_impacts_per_tick=32` is shared across free-flight/control-loss cycles.
Exhaustion stops/project-clears the ball and emits `physics_limit`, cause `impact_budget`;
repeated obstructed acquisition has a separate bounded `attachment_contact_budget` guard.
Normal head-on squeezing between opposing surfaces uses `ball_constraint`,
`cause="opposed_contacts"`, rather than repeatedly rebounding at zero elapsed time.
It settles normal motion against a fixed post (or paired moving surfaces), retains
tangent motion, then projects clear of end-snapshot obstacles and consumes the tick.
This is a bounded geometric settling approximation, not an exact multi-impact path;
it does not resolve every intervening contact during that settling remainder.

`ball_collision` events carry completed tick, `tick_fraction` in [0,1], zero-based
per-tick `collision_index`, `contact_player` or `post_index`, contact position, last
toucher, and `velocity_before`/`velocity_after`. Post indices follow the four-centre
order above. Constraint/limit events expose exceptional resolutions; they are not
kicks, recoveries or reward arithmetic. Log them during later match/visual validation.

```powershell
uv run --locked --only-group engine python -m game --contacts
```

This isolated one-tick CSV shows locked-player rebound, post rebound, controlled post
loss/settling and post→goal. It disables ball friction to isolate restitution and is
**not a match runner**. Deterministic tests include high speeds, multiple impacts,
moving surfaces, post-adjoining stress, last-touch responsibility and final whistle.
The model still uses straight snapshot segments (including turning attachment), not
exact accelerated/curved time-of-impact or a professional rigid-body solver. Custom
geometry can be infeasible and raise; no claim of exhaustive numerical safety is made.

## Player-player contact and stealing

After independent player movement and pitch constraints, `resolve_player_contact`
sweeps the two radius-15 circles. At contact it removes closing normal displacement
and velocity with an equal-mass, zero-restitution response, retaining tangential
sliding. It prevents straight-path cross-through even if the proposed endpoints have
swapped sides. Residual overlap is corrected along available directions; wall-blocked
components are removed instead of pushing either player through a pitch edge. A pinned
player's lost share is handled by the other player's available correction. Exact
coincidence has no geometric normal: use a deterministic longest-pitch-axis split.
Normal nondegenerate contacts are identity-symmetric; no player wins by evaluation order.
Facing, stamina and interaction timers are not modified by the collision helper itself.

This is a snapshot sweep plus bounded position correction (16 iterations, numerical
overlap tolerance 1e-10), not a continuous multi-impact wall/body solver. Impossible
separation under a pathological custom pitch raises an error rather than silently
publishing overlapping bodies. Tests include a seeded 300-case extreme-speed/bounds/
identity-swap regression and a dedicated two-different-walls corner case. These are
executed tests, not a claim of exhaustive collision correctness or realistic tuning.

`PossessionConfig.steal_distance=35` measures player-centre distance **after contact
resolution**. With no acquisition protection, each playing tick within range adds one
to `SimulationState.steal_progress_ticks`, representing the current owner's defender.
At `steal_contest_ticks=18`, possession changes directly from carrier to defender,
attachment/last toucher are synchronized, progress is cleared, and one
`possession_changed` event with `cause="steal"` records previous/new owner and completed
tick t+1. It is not a loose-ball recovery. No tackle action, facing advantage, stamina
modifier, or steal reward arithmetic is added. Leaving range, restarting, kicking,
or a different-owner setup assignment clears progress rather than pausing it. Same-owner
setup assignment remains idempotent and does not erase a live contest/protection timer.

`SimulationState.possession_protection_ticks` belongs to the single current carrier,
not a duplicated per-player ownership flag. Recovery, steal, kickoff and owned boundary
restart initialize it to 30 **at the completed acquisition snapshot**. The following
30 simulation ticks decrement protection without accumulating steal progress; contest
can begin on the next eligible tick. This differs intentionally from kick/pickup timers
that start before release-tick motion and count that release tick. Protection runs during
restarts, freezes at finish, and clears on reset/release/changed setup owner. Free balls
cannot carry stale contest/protection. Acquiring protection does not prevent kicks.

```powershell
uv run --locked --only-group engine python -m game --steal
```

The static 35-unit contest shows progress 17 at tick 17, one A→B steal at tick 18
with protection 30, protection zero at tick 48, then B→A only at tick 66 after another
18 eligible ticks. Tests cover both owners, broken proximity, protection after recovery,
kick interruption, phase transitions, terminal events, custom config and replay. Later
visual inspection should show non-overlapping sliding bodies and overlay steal/protection
timers. Only eligible free-ball contacts are trapped; locked-player/post rebounds
use the separate ordered ball-contact model above.

## Goals, out of bounds, score and controlled restarts

`RulesConfig` uses the approved **120-unit** centred openings and **60-tick** restart
duration. `score_a`/`score_b` are authoritative nonnegative integers and reset to zero.
Score by attack orientation, not last toucher: A scores at the right goal when its
attack direction is +1 and at the left when -1. An own-goal last touch still credits
the appropriate opposing attacker. A goal does not end a match; the 3,600-tick clock does.

`detect_ball_exit` classifies the earliest strict whole-ball exit along the same linear
snapshot path used for pickup. With radius 7, thresholds are x=-7/807 and y=-7/507.
The ball must move **beyond**, not merely reach, a threshold. For a side goal, the
entire ball must fit the goal opening: default centre y lies in the inclusive interval
197..303, inside the opening 190..310. Solid posts are processed before later exits;
an unobstructed miss is out of bounds. Simultaneous corner exits are out of bounds. Already fully outside
setup snapshots are classified at fraction zero so a bad setup does not roll forever.

Free-ball pickup and exit fractions are compared before applying either transition.
Earlier contact controls the ball; a later exit on the abandoned free path is cancelled.
At an exact fraction tie (1e-12 numerical tolerance), contact wins because reaching the
clearance threshold alone is not a strict exit. After recovery, attachment movement is
also checked for exit. Controlled attachment is checked before a timed steal can erase
an already crossed line, and again after steal attachment if it moves the ball outside.
These comparisons are repeated after rebounds on each remaining linear flight;
they are not exact accelerated within-tick goal times or all possible contact trajectories.

**Approved gameplay limitation:** with player centres clamped inside radius 15 and
dribble offset 22, a controlled ball reaches x=807 or -7 but cannot move strictly beyond
by ordinary dribbling alone. Default goals therefore require a free-ball kick. Larger
custom attachment offsets can cross and are detected. Supporting normal dribble goals
would require an explicitly revised goal-mouth/player-boundary design, not weakening
the strict whole-ball rule silently. The student explicitly retained kick-only goals
when approving posts/rebounds; goal-mouth player entry is not pending implementation.

A confirmed goal emits one `goal` event; other exits emit one `out_of_bounds` event.
Events include completed tick, scoring player where relevant, last toucher, crossing
position, previous/new owner and restart receiver. Score/phase updates are atomic.
The ball stops at the crossing position, becomes free, retains last toucher, and clears
contest/protection. An explicit `RestartPlan` stores receiver and exit information.
Restart phases do not re-detect the same exit, preventing duplicate goals/turnovers.

During the next 60 ticks, player/ball motion and **stamina freeze**, while match clock,
kick cooldown, pickup lockout and protection timers advance. The scoring tick itself
was a playing tick, so its ordinary movement/stamina update still applies. On the last
countdown tick, apply restart placement but do not process gameplay actions until the
following tick. Clear velocities and interaction timers; preserve score, orientation,
stamina and simulation tick. Owned restarts start fresh 30-tick acquisition protection.

- Goal: conceder gets controlled possession with the ball at pitch centre. Carrier is
  placed one dribble offset behind centre, facing their attacking direction; opponent
  returns to their own-half spawn. Emit `possession_changed`, `cause="kickoff"`.
- Known last-touch out of bounds: opponent receives possession near the exit. Carrier
  centre is inset/clamped to the legal player rectangle, facing inward from that side;
  attachment consequently sits safely infield. Responsible player returns to own-half
  spawn near their goal. Emit `cause="boundary_restart"`, not a recovery reward signal.
- Unknown last toucher: neither player is blamed. Reset to spawn positions and a free
  centre ball; emit `restart_completed`, `cause="neutral_restart"`.

With default geometry, e.g. a right-edge restart puts receiver centre x=785, facing
left, ball x=763. Near-exit placement deliberately respects body/ball geometry rather
than leaving the ball fully outside. B/C should use explicit restart events and causes
instead of inferring turnover/reward from a teleport. General event snapshots now also
include last-touch/ball-position data for setup, kicks, recovery and steals.

```powershell
uv run --locked --only-group engine python -m game --rules
```

This compares an on-target shot and an off-target shot from `(804,y)` at speed 300:
tick 1 starts goal/out-of-bounds restart, tick 31 still freezes, tick 61 completes the
kickoff/boundary restart, and tick 67 shows play resumed without a duplicate exit event.
Tests cover mouth edges, strict threshold touches, both sides/orientations, own-goal
credit, all four boundaries, unknown touch, fast shots, pickup ordering, repeated scores,
restart request expiry, preserved stamina and final-whistle ordering. A goal and the
terminal event can both occur on the last tick; a match can also finish during a restart.
At finish, the pending restart is cleared and no future kickoff/goal event is emitted.

## Time, phases, events, and reset

- `step_tick` advances exactly `1/60` simulated second, independent of wall time.
- `step_decision` holds movement for up to **six separate ticks** (10 Hz), consuming
  each kick request only on the first tick.
  It aggregates events from every tick and stops early at the final whistle.
- `StepResult.physics_ticks` counts all advanced simulation ticks, including frozen
  restart ticks. It is not a count of ticks with player displacement.
- The 60-second match clock counts **3,600 ticks, including restart ticks**.
  Remaining time is derived from that integer tick. Finished states never advance again.
- The `match_finished` event appears once at tick 3,600, not on subsequent calls.
- Both players read the same input snapshot; physics never imports rendering/training libraries.
- `reset()` creates independent immutable state at `(200,250)` / `(600,250)`.
  It also resets a stationary free ball at pitch center `(400,250)`, clearing owner/last toucher
  and both players' interaction timers.
  `reset(swap_sides=True)` swaps spawn/facing and attack orientations. No randomness is
  used yet, so there is no seed argument or RNG state to reset at this stage.

For custom pitches whose usual quarter-width spawns overlap, reset tries vertical
quarter-height spawns, then opposite inset corners. Body/post constraints are checked
before returning state; impossible or unsupported layouts raise a reset error, not a
later first-tick failure. On the audited 50×100 pitch this yields `(25,25)` / `(25,75)`;
facing/attack directions remain east/west, and side swapping exchanges identities.
The fallback is not a general arbitrary-geometry packing solver.

Goal and out-of-bounds rules now create real restart phases with an explicit plan, as
described above. Tests can still construct a plan-free countdown/freeze fixture for
isolated phase timing; that is not a legal football restart API. A held movement action
can resume within the six-tick block when a real restart completes, but its first-tick
kick request has already expired. Whether the RL wrapper queries policies during these
phases remains C's interface decision; shared stepping itself always uses simulation time.

## Headless full-match runner (17 Nov)

`game.runner.run_match(provider_a=idle_provider, provider_b=idle_provider, *,
initial_state=None, config=...)` runs to the engine's final whistle. It creates a fresh
state with `reset(config=config)` when no state is supplied. For side swapping, supply
`reset(swap_sides=True, config=config)`; for a resumed match, supply its current state
and matching configuration. Configuration is not stored in the state and cannot be
automatically inferred or checked for a match.

Each provider is a callable `(state: SimulationState, player: PlayerId) -> PlayerAction`.
The runner calls A then B with the **same immutable pre-step snapshot**, once per
decision interval, **including during restarts** (student-approved runner contract).
This exposes full engine state for inspection, not a finalized RL observation. C still
owns the Gymnasium policy-query/observation contract. Providers must return a full
`PlayerAction`; movement-only actions, dictionaries and numeric encodings are not
silently converted. Provider/engine exceptions propagate rather than returning a
misleading partial success. Keep private controller RNG/script state independent and
reset it before replay; the engine itself has no seed/randomized reset.

All physics goes through `step_decision`. Intervals span up to six ticks starting at
the supplied state's tick, even for an off-grid resumed state. A final interval may
be shorter. Restarts still consume clock time; held movement can resume when a restart
expires inside a decision, but its discarded first-tick kick cannot retry. No provider
is queried after finish or for already-finished input.

The immutable `MatchResult` contains:

- `final_state`: authoritative score, orientation and finished state; no duplicate score store.
- `events`: every engine event generated during this invocation, in original emission
  order with untouched timestamps. Do not sort by tick: kick timestamps use different
  conventions from completed-tick events. Earlier/setup events are not reconstructed.
- `physics_ticks`: actual advanced ticks, including frozen restarts.
- `decision_count`: actual calls to `step_decision`, not a hard-coded 600.

A fresh match normally returns 3,600 ticks / 600 decisions; a resumed match counts only
remaining work. Already-finished input returns the same state, no events and zero counts.
No full state/action trajectory is stored; event collection uses memory proportional
to event count and is for inspection, not yet an optimized training/logging interface.

```python
from game.actions import Direction, PlayerAction
from game.player import PlayerId
from game.runner import run_match
from game.simulation import SimulationState


def scripted_action(state: SimulationState, player: PlayerId) -> PlayerAction:
    # A short fixed test action, not a tactical opponent.
    if player == PlayerId.A and state.simulation_tick < 60:
        return PlayerAction(Direction.EAST, effort=0.5)
    return PlayerAction()


result = run_match(scripted_action, scripted_action)
print(result.final_state.score_a, result.final_state.score_b)
```

```powershell
uv run --locked --only-group engine python -m game --match
uv run --locked --only-group engine python -m game --match scripted
uv run --locked --only-group engine python -m game --match scripted --swap-sides
```

The CLI preserves all existing inspection modes. Its JSON summary includes scores,
ticks/decisions, event counts, and counts/causes for `ball_collision`, `ball_constraint`
and `physics_limit`, including zero counts. The Python result retains their complete
event records; the CLI prints counts only. The fixed script moves B north for the first
15 decisions, moves A in its attacking direction at effort 0.75 for the first 45,
requests one full-power kick at decision 45, then idles. It is not human input, an
adaptive rule-based opponent, PPO, or a claim of realistic gameplay. No wall-clock
pacing or throughput benchmark is performed; renderer equivalence remains unverified.

## Event timeline and scenario audit (17 Nov)

```powershell
uv run --locked --only-group engine python -m game --match scripted --timeline
uv run --locked --only-group engine python -m game.scenarios --list
uv run --locked --only-group engine python -m game.scenarios --output results/physics_scenarios --report docs/physics_scenario_report.md
```

`--timeline` requires `--match`. It writes a readable event timeline to **stderr**,
leaving the existing JSON summary alone on stdout. `game/timeline.py` preserves
emission order and adds a zero-based sequence number, stored tick/60 stamp, kind,
cause, possession transitions, scorer/receiver and available collision details.
Kicks use start-tick timestamps; most transitions use completed ticks. Stamps are
not exact physical contact times; fractional contacts remain explicit. Timeline
output does not repair inconsistent engine event fields.

`game/scenarios.py` provides `ScenarioAgent`: a fixed-action schedule or independently
seeded random action provider for **diagnostics only**, not Member B's baseline or
an RL policy. Its 50 named scenarios cover movement/stamina, bodies, rolling/kicks,
request expiry, pickups/steals, posts/rebounds, goals/restarts, final ticks, deliberate
saturation and custom reset geometry. Four seeded action-combination cases run for
the whole match; many targeted edge probes run for only one tick. They are **not
50 full matches**, subjective player experience or exhaustive physics coverage.

Each case runs twice with freshly initialized controllers. The harness inspects
every advanced tick, strips kicks after the first tick of each decision, compares
complete decision blocks with `step_decision`, and hashes the state/action/event
trajectory for deterministic replay. The cadence check shares engine code; it is
not an independent continuous-physics oracle. Initial setup may inject positions,
velocities, owner/timers or compressed restart countdowns; the controller cannot
modify state during play. Random actions do not imply randomized engine resets.

The output directory contains `results.json`, `report.md` and per-case event
timelines. JSON records configs, initial/final states, agent plans, expected/actual
checks, errors, first-failure context, every event and replay hashes. `--scenario NAME`
selects a case; `--report PATH` also writes a shareable report. Existing `.gitignore`
ignores `results/`, so regenerate or explicitly choose which artifacts to share.
Audit exit code **1 means at least one FAIL**, with evidence still saved. WARN marks
explicit physics-limit telemetry or a declared model-hardening concern, not a silent
success. State/event failures and exceptions remain FAIL, even if deterministic.

The original 17 Nov audit recorded **43 PASS, 2 WARN, 5 FAIL**. Subsequent student-requested
repairs fixed restart-event last-touch snapshots, separated positional correction from
rebound energy and added narrow reset placement. The repaired audit recorded
**49 PASS, 1 WARN, 0 FAIL**, with 50/50 identical replays; only deliberate one-impact
budget saturation warns. Restart acquisition events now describe the resulting ball's
owner/last toucher, while original exit events retain the responsible toucher's data.
The warning/telemetry detectors remain active and synthetic regression tests exercise
them. See `docs/physics_fixes.md` for the original findings/repair rationale and
`docs/physics_scenario_report.md` for current evidence. No constants, dependencies,
renderer or reward logic changed; snapshot/settling approximations remain.

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
