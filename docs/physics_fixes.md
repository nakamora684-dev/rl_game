# Runner-audit physics repairs (17 Nov 2026)

## Baseline retained

The original `docs/physics_scenario_report.md` recorded **43 PASS, 2 WARN, 5 FAIL**,
with 50/50 identical replays. Four failures shared one restart-event telemetry defect;
the fifth was the accepted 50×100 reset layout. A stationary coincident-body fixture
also generated 1348.33 units/s of ball speed. Determinism did not make those results
correct. Original ignored artifacts remain in `results/physics_scenarios/`; repaired
runs use `results/physics_scenarios_fixed/` instead of overwriting that evidence.
The scenario report is regenerated for the repaired engine; this note preserves the
original findings and explains the changes.

## Repairs

1. **Restart event snapshots.** `_complete_restart` now reads event last toucher from
   the resulting ball, not the pre-restart ball. Owned acquisition therefore reports
   the receiver as both owner and last toucher. Neutral restart reports neither.
   Original goal/out-of-bounds events still retain the toucher responsible for the exit.
   No score, receiver, timer or reward rule changed.

2. **Correction is geometry, not velocity.** Initial body/post overlaps are repaired
   before movement/pickup sweeps are constructed. Rebound impulses use resolved physical
   player velocity rather than positional displacement divided by `dt`, which included
   depenetration. Snapshots still determine contact geometry; the two are deliberately
   separate. Newly kicked balls retain the stationary release-snapshot convention.
   Controlled post-loss launch velocity uses physical carrier translation plus
   facing-offset motion, excluding carrier positional correction. Genuine movement
   and turning can still transfer energy; stationary correction cannot invent it.

3. **Narrow reset placement.** Default left/right quarter-pitch spawns are unchanged.
   If they overlap, reset tries vertical quarter-pitch spawns, then opposite inset
   corners. It checks body/post constraints before returning state. A pitch too small
   for two bodies is rejected during reset with a clear error, rather than returning
   an overlapping state that fails on the first tick. The 50×100 configuration now
   starts at `(25,25)` / `(25,75)` with unchanged east/west facing and deterministic
   identity swapping. This is a spawn fallback, not a general-purpose packing solver
   or a redesign of attacking sides; unusual post configurations can still be rejected.

## Verification and scope

`tests/test_audit_regressions.py` first reproduced the findings against the old code
(20 failures, 4 passes), then checks the six affected audit probes, stationary free
and controlled correction, real moving-player impulse, narrow side-swapped reset,
early infeasible-reset rejection and restart event/state agreement across both sides,
both touchers/unknown touch, goals and boundaries. Existing audit telemetry and
model-warning detectors remain active; synthetic bad telemetry/energy tests still
verify that the audit flags them. Scenario 14 now also explicitly asserts zero ball
speed, and scenario 50 explicitly checks separated initial bodies. Its historical
`50_unsupported_narrow_reset` identifier is retained for traceability.

The repaired 50-probe run returned **49 PASS, 1 WARN, 0 FAIL**, 50/50 identical replays.
Only scenario 45 warns: a deliberately one-impact budget emits `impact_budget` and
stops the ball. That safety diagnostic is preserved, not suppressed. See
`docs/physics_scenario_report.md` for measured per-case evidence and
`docs/next_session.md` for final engine check commands/results.

No dependencies, physics constants, kick-only-goal decision, stamina tuning, rewards,
renderer, baseline controller or PPO interfaces changed. Snapshot sweeps and bounded
settling remain approximations; this catalog is not exhaustive correctness, visual
playability, renderer equivalence or throughput evidence.
