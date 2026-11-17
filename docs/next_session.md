# Next session: full-match runner

> **Latest: runner-audit repairs (17 Nov).** The student requested fixes using the
> scenario report. Restart-event last touch now comes from the resulting ball;
> corrected body positions are not used as rebound velocity; initial overlap repair
> occurs before movement sweeps; controlled post-loss energy excludes carrier
> correction. Narrow reset uses separated fallback spawns before publishing state.
> No tuning constants or dependencies changed. See `docs/physics_fixes.md` for the
> baseline and rationale. New raw evidence is in `results/physics_scenarios_fixed/`,
> preserving the original artifacts. The regenerated audit reports **49 PASS, 1 WARN,
> 0 FAIL; 50/50 identical replays**. The remaining WARN deliberately exhausts a
> one-impact budget. The older findings below are historical, not unresolved work.
> Visual validation, renderer equivalence and throughput are still unmeasured.

### Repair verification (17 Nov 2026)

```powershell
uv run --locked --only-group engine pytest -m "not full_stack" -q
# 562 passed, 1 deselected, 36 subtests passed
uv run --locked --only-group engine ruff check game tests
# All checks passed
uv run --locked --only-group engine ruff format --check game tests
# 38 files already formatted
uv lock --check
# 179 packages resolved; lock consistent
uv run --locked --only-group engine python -m game.scenarios --output results/physics_scenarios_fixed --report docs/physics_scenario_report.md
# 49 PASS, 1 deliberate-budget WARN, 0 FAIL; 50/50 identical replays; exit 0
uv run --locked --only-group engine python -m game --match scripted --timeline
# A 1-0 B, tick 3600, 600 decisions; kickoff last_touch=B; no physics limits
```

Added 25 regression cases; the synthetic correction-energy detector test now also
checks the stronger zero-speed expectation fails on injected bad energy. Full-stack
dependencies remain excluded. `graphify-out/` still describes an earlier snapshot;
its `.needs_update` marker requests a later code/document refresh. Use current source
and the report for these repairs. No commits or pushes were made.

> **Latest: timeline/scenario audit implemented (17 Nov).** Use
> `python -m game --match scripted --timeline` for a readable timeline on stderr.
> The diagnostic `ScenarioAgent` and 50 probes are in `game/scenarios.py`; see
> `docs/physics_scenario_report.md` for the executed audit, not just a proposed plan.
> Corrected results: **43 PASS, 2 WARN, 5 FAIL; 50/50 deterministic replays**.
> Four FAILs share one stale restart-event last-toucher defect; one concerns an
> unsupported narrow reset layout. WARNs expose coincident-body correction energy
> and deliberately reduced impact-budget saturation. No physics was silently fixed.
>
> Next: discuss and authorize a small fix for restart event telemetry, then separately
> consider correction-derived ball energy and narrow reset/config handling. Do not
> treat four occurrences of one telemetry problem as four independent physics bugs.
> The initial body-order oracle was corrected to isolate the ball: legal restart
> teleportation/lane changes must not be misclassified as collision tunnelling.
>
> Verification: `uv run --locked --only-group engine pytest -m "not full_stack" -q`
> returned **537 passed, 1 deselected, 36 subtests passed**; Ruff check and format
> checks passed (37 files). The full 50-case audit command returned exit code 1
> intentionally because it detected FAILs; it still saved raw JSON and timelines
> under `results/physics_scenarios/`. A passing test suite is not an all-pass audit.

> **17 Nov update:** the first headless runner is now implemented in `game/runner.py`,
> with `--match [idle|scripted]` inspection and `--swap-sides`. Both providers are
> queried every decision interval, including restarts, as approved by the student.
> See `docs/engine.md` for the implemented API. The original handoff below describes
> the pre-runner boundary; its statements that the runner is unstarted are historical.
> Rendering, real controllers, Gymnasium/rewards/PPO and throughput/visual validation
> remain outside this increment. Existing uncommitted work must still be preserved.

### Runner verification (17 Nov 2026)

Executed after the runner increment:

```powershell
uv run --locked --only-group engine pytest -m "not full_stack" -q
# 497 passed, 1 deselected, 36 subtests passed
uv run --locked --only-group engine ruff check game tests
# All checks passed
uv run --locked --only-group engine ruff format --check game tests
# 33 files already formatted
uv lock --check
# 179 packages resolved; lock consistent
git diff --check
# Passed; Windows LF/CRLF warnings only
uv run --locked --only-group engine python -m game --match scripted --swap-sides
# A 1-0 B; tick 3600; 600 decisions; one kick, goal and whistle; no physics limits
```

There are 36 new runner/CLI test cases. Tests include same-snapshot provider calls,
restart querying/expiry, deterministic seeded action scripts, both orientations,
custom config, exact terminal/partial decisions, event order/diagnostics and invalid
provider output. The seeded scripts randomize actions only, not engine resets. The
full-stack test remains excluded; these checks are not throughput or visual evidence.

Suggested next step: inspect the runner and script with the student, then coordinate
controller/rendering integration with B and the action-provider contract with C/D.
Choose that next increment explicitly rather than silently implementing their modules.

## Scope and working agreement

The agreed remaining physics increment is implemented; the **full-match runner has
not been started**. Next session should design and implement that runner, not replace
the engine or silently build B/C/D's renderer, Gymnasium adapter, rewards, PPO encoding
or proper rule-based baseline. Ask implementation questions if uncertain; save learning
questions for session end. The student authorized coding and larger testable increments.
Use local tools/free assistant routes; do not activate subscriptions or paid fallback
agents without approval. No new dependencies were needed for this increment.

Read `AGENTS.md` (especially section 17), `docs/engine.md` and current tests first.
The working tree has substantial **uncommitted** work from several physics increments;
do not revert it. No commit or push was made. Inspect `git status --short` before editing.

## Stable engine API and runner constraints

- `reset(*, swap_sides=False, config=...)` creates immutable `SimulationState`.
  There is currently no engine randomness/seed argument. Runner-side seeded action
  scripts can provide reproducible scenarios without claiming randomized resets.
- `step_decision(state, action_a, action_b, config=...)` returns `StepResult` containing
  `state`, aggregated `events`, and actual `physics_ticks`. It holds movement for up
  to six 60 Hz ticks (10 Hz decisions), stopping at `MatchPhase.FINISHED`.
- `step_tick` treats each trigger as a fresh request. Prefer `step_decision` to avoid
  accidentally repeating a kick on all six held ticks. Each new decision can request
  again, even with the same action object. Failed requests are discarded, not retried.
- `PlayerAction(direction, effort, kick_trigger, kick_power)` extends `MovementAction`.
  Direction IDs: idle/N/NE/E/SE/S/SW/W/NW = 0..8; use `Direction(id)`. Effort/power
  are continuous [0,1]; trigger binary. Speed is 50+250*power, facing-based, no charging.
  Kick cooldown and pickup lockout are distinct 12-tick per-player fields.
- A full match is 3,600 simulation ticks (normally 600 six-tick calls), including
  restarts. Never use wall time to determine results. Restarts freeze motion/stamina
  but advance clock/timers; finished stepping is idempotent with no repeated events.
- State exposes score, attack orientation, phase/restart plan/countdown, ball owner
  and last toucher, player motion/facing/stamina/timers, steal progress and protection.
  Use owner as the sole possession source. Rendering, when added, must read this state.
- `assign_possession` and `release_kick` are zero-tick setup/low-level helpers, not
  runner policy actions. Do not farm setup events or bypass guards during normal play.
- Existing `python -m game` modes are inspection commands, not a general runner.
  Do not overwrite the empty teammate entry points to make them appear finished.

## Approved physics to preserve

- Movement/stamina, analytic free-ball rolling, controlled facing attachment, kicks,
  swept any-speed recovery, equal-mass sliding bodies, timed stealing/protection,
  strict whole-ball goals, last-touch exits, controlled restarts and final whistle.
- Locked-player ball normal restitution 0.5, no recoil; eligible players trap instead.
  Solid radius-4 posts use restitution 0.7; centres are `(0,186)`, `(0,314)`,
  `(800,186)`, `(800,314)`, preserving the 120-unit clear opening.
- Controlled post contact loses possession, starts pickup lockout, but emits no kick
  and starts no kick cooldown. Use attachment motion/turning for impact velocity;
  instantaneous pickup/steal attachment must not create teleport-derived launch energy.
- Rectangular player bounds remain: **kick-only goals are intentional**. The student
  rejected goal-mouth player entry; do not reopen it as unfinished physics.
- Ordered ball flights handle multiple impacts, post→pickup and post→goal. Exact ties:
  post, eligible pickup, solid player, strict exit. Player solid contact updates last
  toucher; posts preserve it; ambiguous simultaneous touches can leave it unknown.

## Diagnostics and known limitations

- `ball_collision`: cause `goalpost`/`player_contact`, completed tick, fraction,
  per-tick collision index, player/post identity, position, last toucher and velocities.
- `ball_constraint`, cause `opposed_contacts`: normal opposed-contact settling, not a
  kick/recovery. `physics_limit`: `impact_budget`, `attachment_contact_budget` or
  `player_post_budget`; surface these in runner diagnostics rather than dropping them.
- Ball impact budget 32 is shared across flights/control-loss cycles within a tick.
  Saturation stops/project-clears. Player/post sweeps allow 16 contacts, coupled
  projections 64; infeasible/nonconverging geometry may raise explicitly.
- Linear snapshot sweeps are not exact accelerated/curved time-of-impact. Opposed
  settling consumes the remainder with a tangent roll plus endpoint projection,
  rather than resolving every intervening collision. Visual validation and sustained
  play are still needed; unit tests do not prove realistic tuning or complete safety.
- No renderer/environment equivalence, headless throughput benchmark or evaluation
  results are claimed. Reward arithmetic remains outside `game/`.

## Verified checks (11 Nov 2026)

```powershell
uv run --locked --only-group engine pytest -m "not full_stack" -q
# 461 passed, 1 deselected, 36 subtests passed
uv run --locked --only-group engine ruff check game tests
# All checks passed
uv run --locked --only-group engine ruff format --check game tests
# 31 files already formatted
uv lock --check
# 179 packages resolved; lock consistent
git diff --check
# Passed; Windows LF/CRLF warnings only
uv run --locked --only-group engine python -m game --contacts
# Locked-player vx=-150; post vx=-840; controlled loss/settling; post then goal
```

The excluded `full_stack` import test requires graphics/ML dependencies. Python stays
pinned at 3.12.14; use the locked `engine` group, not an unrecorded global/pip install.
The contact CLI disables friction to isolate restitution; it is not performance evidence.

## Suggested first runner increment (proposal, not an agreed interface)

Agree a small headless runner API with the student: two action providers, optional
initial state/config, and a result with final score/state and complete event trace.
Decide explicitly whether providers are queried during restarts. Preserve 10 Hz
decision semantics and simulated time; avoid adding a renderer or baseline strategy
at the same time. A scripted deterministic provider is a test fixture, not the team's
rule-based evaluation opponent.

First tests: idle match reaches exactly tick 3,600 with one whistle; scripted actions
replay identically; goals/restarts do not end early; all sub-tick events survive;
already-finished input does not query providers or advance; both attacking sides work.
Record wall-clock throughput separately only if actually measured.
