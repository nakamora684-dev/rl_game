# Physics scenario audit

**50 distinct scenarios: 49 PASS, 1 WARN, 0 FAIL.**

Each scenario runs twice with fresh deterministic controllers. Every advanced physics tick is inspected; complete six-tick blocks and terminal partial blocks are also compared with `step_decision`. Replay compares state/action/event trajectory hashes. Replays and cadence checks are not counted as additional scenarios.

PASS means only that the declared expectations and snapshot invariants held. WARN means explicit limit telemetry or a marked model-hardening concern occurred. FAIL can mean a state/event inconsistency, a failed expectation or an exception. Identical repeated failures are still failures, even if replay is deterministic.

Python: 3.12.14. Primary-pass inspected ticks: 23794. Identical replays: 50/50.

## Findings

- No violated checks in this catalog; this is not exhaustive correctness evidence.
- **Deliberate saturation in 45_deliberate_budget_saturation:** `{'impact_budget': 1}`.

Baseline audit and engine repair history: `docs/physics_fixes.md`. This report records the current executed probes, not the pre-fix results.

## Physics/tuning feedback

- The reference roll and stopping probes check analytic distance, while player coast uses semi-implicit integration. Their distances should not be compared as if they used the same integrator; inspect the measured values below.
- Four-second idle braking at speed 160 and much stronger braking at tiny positive effort are existing tuning choices. These need human visual play-testing, not a claim of physical realism from passing assertions.
- Controlled possession may overlap the defender's ball-contact circle. This is a hybrid attachment/steal model; it is not an independent free-ball collision failure.
- Explicit high-speed, reduced-budget and narrow-pitch fixtures are adversarial tests, not normal gameplay settings. Do not interpret their velocities as action kick powers.

## Scenario results

| Scenario | Status | Ticks | Decisions | Goals | Collisions | Constraints | Limits | Replay |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 01_idle_full_match | PASS | 3600 | 600 | 0 | 0 | 0 | 0 | same |
| 02_normalized_diagonal | PASS | 60 | 10 | 0 | 0 | 0 | 0 | same |
| 03_velocity_reversal | PASS | 240 | 40 | 0 | 0 | 0 | 0 | same |
| 04_idle_braking | PASS | 240 | 40 | 0 | 0 | 0 | 0 | same |
| 05_zero_effort_turn | PASS | 30 | 5 | 0 | 0 | 0 | 0 | same |
| 06_sprint_against_wall | PASS | 1200 | 200 | 0 | 0 | 0 | 0 | same |
| 07_empty_stamina_speed | PASS | 120 | 20 | 0 | 0 | 0 | 0 | same |
| 08_sprint_then_recover | PASS | 2100 | 350 | 0 | 0 | 0 | 0 | same |
| 09_neutral_effort | PASS | 600 | 100 | 0 | 0 | 0 | 0 | same |
| 10_tiny_effort_brakes_harder | PASS | 6 | 1 | 0 | 0 | 0 | 0 | same |
| 11_head_on_bodies | PASS | 240 | 40 | 0 | 0 | 0 | 0 | same |
| 12_tangential_slide | PASS | 60 | 10 | 0 | 0 | 0 | 0 | same |
| 13_high_speed_body_crossing | PASS | 1 | 1 | 0 | 0 | 0 | 0 | same |
| 14_coincident_bodies | PASS | 1 | 1 | 0 | 2 | 0 | 0 | same |
| 15_body_post_wall_pressure | PASS | 120 | 20 | 0 | 0 | 0 | 0 | same |
| 16_reference_roll | PASS | 180 | 30 | 0 | 0 | 0 | 0 | same |
| 17_partial_tick_stop | PASS | 1 | 1 | 0 | 0 | 0 | 0 | same |
| 18_frictionless_roll | PASS | 60 | 10 | 0 | 0 | 0 | 0 | same |
| 19_turn_then_kick | PASS | 1 | 1 | 0 | 0 | 0 | 0 | same |
| 20_midpower_stop_distance | PASS | 105 | 18 | 0 | 0 | 0 | 0 | same |
| 21_pickup_does_not_retry_kick | PASS | 6 | 1 | 0 | 0 | 0 | 0 | same |
| 22_cooldown_expiry_no_retry | PASS | 6 | 1 | 0 | 0 | 0 | 0 | same |
| 23_restart_expiry_no_retry | PASS | 6 | 1 | 0 | 0 | 0 | 0 | same |
| 24_extreme_swept_pickup | PASS | 1 | 1 | 0 | 0 | 0 | 0 | same |
| 25_exact_pickup_tie | PASS | 6 | 1 | 0 | 0 | 0 | 0 | same |
| 26_steal_then_protected_return | PASS | 66 | 11 | 0 | 0 | 0 | 0 | same |
| 27_contest_interrupted | PASS | 60 | 10 | 0 | 0 | 0 | 0 | same |
| 28_protection_exact_boundary | PASS | 48 | 8 | 0 | 0 | 0 | 0 | same |
| 29_kick_clears_contest | PASS | 1 | 1 | 0 | 0 | 0 | 0 | same |
| 30_recovery_post_loss_no_energy | PASS | 1 | 1 | 0 | 1 | 0 | 0 | same |
| 31_locked_player_rebound | PASS | 1 | 1 | 0 | 1 | 0 | 0 | same |
| 32_post_restitution | PASS | 1 | 1 | 0 | 1 | 0 | 0 | same |
| 33_four_rebounds_one_tick | PASS | 1 | 1 | 0 | 4 | 0 | 0 | same |
| 34_post_then_pickup | PASS | 1 | 1 | 0 | 1 | 0 | 0 | same |
| 35_post_then_goal | PASS | 1 | 1 | 1 | 1 | 0 | 0 | same |
| 36_strict_threshold_touch | PASS | 6 | 1 | 0 | 0 | 0 | 0 | same |
| 37_inclusive_mouth_edge | PASS | 1 | 1 | 1 | 0 | 0 | 0 | same |
| 38_swapped_own_goal | PASS | 1 | 1 | 1 | 0 | 0 | 0 | same |
| 39_known_touch_boundary_restart | PASS | 66 | 11 | 0 | 0 | 0 | 0 | same |
| 40_unknown_touch_neutral_restart | PASS | 66 | 11 | 0 | 0 | 0 | 0 | same |
| 41_full_restart_freeze | PASS | 67 | 12 | 1 | 0 | 0 | 0 | same |
| 42_final_tick_goal | PASS | 1 | 1 | 1 | 0 | 0 | 0 | same |
| 43_off_grid_terminal_resume | PASS | 7 | 2 | 0 | 0 | 0 | 0 | same |
| 44_opposed_contact_settling | PASS | 1 | 1 | 0 | 2 | 1 | 0 | same |
| 45_deliberate_budget_saturation | WARN | 1 | 1 | 0 | 1 | 0 | 1 | same |
| 46_random_default | PASS | 3600 | 600 | 0 | 0 | 0 | 0 | same |
| 47_random_swapped | PASS | 3600 | 600 | 0 | 0 | 0 | 0 | same |
| 48_random_near_posts | PASS | 3600 | 600 | 1 | 0 | 0 | 0 | same |
| 49_random_low_stamina | PASS | 3600 | 600 | 0 | 0 | 0 | 0 | same |
| 50_unsupported_narrow_reset | PASS | 6 | 1 | 0 | 0 | 0 | 0 | same |

## Per-scenario evidence and feedback

### 01_idle_full_match — PASS

Exact 60-second match and one whistle

- Advanced 3600/3600 requested ticks; final phase `finished`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: match_finished/* count = 1; expected `1`, observed `1`.
- PASS expectation: ends at tick 3600; expected `3600`, observed `3600`.
- No violated checks in this probe; this does not establish realistic tuning.

### 02_normalized_diagonal — PASS

Diagonal target does not multiply speed by sqrt(2)

- Advanced 60/60 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 80/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 431.014 units.
- PASS expectation: diagonal speed = 80; expected `80.0`, observed `80.0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 03_velocity_reversal — PASS

Eastward run then bounded westward reversal

- Advanced 240/240 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 80; maximum snapshot-observed player/ball speeds 158/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 140 units.
- PASS expectation: final x velocity is negative; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 04_idle_braking — PASS

Open-space coast from 160 at braking 40

- Advanced 240/240 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 160/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 218.559 units.
- PASS expectation: stops at 4 seconds; expected `0`, observed `0.0`.
- PASS expectation: semi-implicit travel = 318 2/3; expected `318.6666666666667`, observed `318.6666666666674`.
- Measured stop uses player semi-implicit integration, not the ball's analytic integration. Four-second idle stopping is an approved tuning concern.

### 05_zero_effort_turn — PASS

Facing turns north without driving acceleration

- Advanced 30/30 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: faces north; expected `-1.5707963267948966`, observed `-1.5707963267948966`.
- PASS expectation: no speed; expected `0`, observed `0.0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 06_sprint_against_wall — PASS

Requested effort drains stamina despite blocked motion

- Advanced 1200/1200 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 1.31881e-12; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 208.17 units.
- PASS expectation: stamina depleted after 20 seconds; expected `0`, observed `1.3188061753766078e-12`.
- PASS expectation: wall x unchanged; expected `785`, observed `785.0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 07_empty_stamina_speed — PASS

Fatigue halves full-effort target speed

- Advanced 120/120 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 0; maximum snapshot-observed player/ball speeds 80/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 367.518 units.
- PASS expectation: empty target = 80; expected `80`, observed `80.0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 08_sprint_then_recover — PASS

20-second sprint followed by 15 seconds idle

- Advanced 2100/2100 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 1.31881e-12; maximum snapshot-observed player/ball speeds 158/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 140.001 units.
- PASS expectation: sprint reaches empty; expected `0`, observed `1.3188061753766078e-12`.
- PASS expectation: rest restores full stamina; expected `100`, observed `100.0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 09_neutral_effort — PASS

Effort 0.5 neither consumes nor regenerates

- Advanced 600/600 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 50; maximum snapshot-observed player/ball speeds 60/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 140 units.
- PASS expectation: neutral stamina stays 50; expected `50`, observed `50.0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 10_tiny_effort_brakes_harder — PASS

Compare coasting with tiny active target

- Advanced 6/6 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 160/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 290 units.
- PASS expectation: tiny active effort brakes more than idle; expected `True`, observed `True`.
- Known tuning discontinuity: idle uses braking 40; tiny positive active effort uses acceleration up to 320. Do not mistake it for smooth effort-dependent braking.

### 11_head_on_bodies — PASS

Symmetric approach must not pass through

- Advanced 240/240 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 80; maximum snapshot-observed player/ball speeds 158/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 0 units.
- PASS expectation: A remains left of B; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 12_tangential_slide — PASS

Touching bodies retain lateral sliding

- Advanced 60/60 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 80/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 6.58435e-05 units.
- PASS expectation: A slides down rather than freezing; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 13_high_speed_body_crossing — PASS

Artificial extreme velocities must not tunnel

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 30000/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 0 units.
- PASS expectation: body order preserved; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 14_coincident_bodies — PASS

Deterministic split of exactly coincident fixture

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 0 units.
- PASS expectation: coincidence resolved along pitch long axis; expected `True`, observed `True`.
- PASS expectation: stationary correction adds no ball speed; expected `0`, observed `0.0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 15_body_post_wall_pressure — PASS

A second body pushes near a solid goalpost and wall

- Advanced 120/120 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 90; maximum snapshot-observed player/ball speeds 21.3111/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 0 units.
- PASS expectation: A stays left of post clearance; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 16_reference_roll — PASS

300-unit/s roll stops after 3 seconds and 450 units

- Advanced 180/180 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/300 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 470 units.
- PASS expectation: final x = 550 after 450-unit reference roll; expected `550`, observed `549.9999999999998`.
- PASS expectation: reference ball stopped; expected `0`, observed `0.0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 17_partial_tick_stop — PASS

Very slow ball stops partway through one tick

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0.5 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 470 units.
- PASS expectation: final x = 100.00125 after partial-tick stop; expected `100.00125`, observed `100.00125`.
- No violated checks in this probe; this does not establish realistic tuning.

### 18_frictionless_roll — PASS

Zero friction preserves velocity and linear travel

- Advanced 60/60 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/300 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 470 units.
- PASS expectation: one-second x = 400; expected `400`, observed `400.0`.
- PASS expectation: speed remains 300; expected `300`, observed `300.0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 19_turn_then_kick — PASS

First-tick facing update changes release direction

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/298.333 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 397.2 units.
- PASS expectation: ball_kicked/* count = 1; expected `1`, observed `1`.
- PASS expectation: northward component after east-to-north turn; expected `True`, observed `True`.
- A kick during a turn follows that tick's partially turned facing, not the requested direction immediately.

### 20_midpower_stop_distance — PASS

Power 0.5 maps to speed 175, not half the maximum distance

- Advanced 105/105 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/173.333 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 397.2 units.
- PASS expectation: ball_kicked/* count = 1; expected `1`, observed `1`.
- PASS expectation: final x = 375.125 after 153.125-unit roll; expected `375.125`, observed `375.1250000000002`.
- Kick power is linear in speed, but stopping distance is quadratic in speed.

### 21_pickup_does_not_retry_kick — PASS

No possession at request, recovery later in the decision

- Advanced 6/6 requested ticks; final phase `playing`, score [0, 0], owner `A`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: ball_kicked/* count = 0; expected `0`, observed `0`.
- PASS expectation: possession_changed/recovery count = 1; expected `1`, observed `1`.
- No violated checks in this probe; this does not establish realistic tuning.

### 22_cooldown_expiry_no_retry — PASS

Cooldown expires inside a held decision

- Advanced 6/6 requested ticks; final phase `playing`, score [0, 0], owner `A`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: ball_kicked/* count = 0; expected `0`, observed `0`.
- PASS expectation: cooldown reaches zero; expected `0`, observed `0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 23_restart_expiry_no_retry — PASS

Compressed real restart ends mid-decision; kick stays discarded

- Advanced 6/6 requested ticks; final phase `playing`, score [1, 0], owner `B`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 21.3333/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 192 units.
- PASS expectation: ball_kicked/* count = 0; expected `0`, observed `0`.
- PASS expectation: possession_changed/kickoff count = 1; expected `1`, observed `1`.
- PASS expectation: held movement resumes; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 24_extreme_swept_pickup — PASS

Ball crosses receiver and boundary within one tick; pickup wins first

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `A`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/60000 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 220 units.
- PASS expectation: possession_changed/recovery count = 1; expected `1`, observed `1`.
- PASS expectation: goal/* count = 0; expected `0`, observed `0`.
- PASS expectation: A traps the swept ball; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 25_exact_pickup_tie — PASS

Exactly simultaneous equal-distance pickups stay free

- Advanced 6/6 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 14 units.
- PASS expectation: possession_changed/* count = 0; expected `0`, observed `0`.
- PASS expectation: no A-first owner; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 26_steal_then_protected_return — PASS

Consecutive 18-tick steals separated by 30 protected ticks

- Advanced 66/66 requested ticks; final phase `playing`, score [0, 0], owner `A`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 0 units.
- PASS expectation: possession_changed/steal count = 2; expected `2`, observed `2`.
- PASS expectation: steals at ticks 18 and 66; expected `[18, 66]`, observed `[18, 66]`.
- Controlled attachment can geometrically overlap the defender; the hybrid model resolves control via the steal timer, not a free-ball bounce.

### 27_contest_interrupted — PASS

Leaving range clears, rather than pauses, steal progress

- Advanced 60/60 requested ticks; final phase `playing`, score [0, 0], owner `A`.
- Minimum stamina 99; maximum snapshot-observed player/ball speeds 63.8267/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 5 units.
- PASS expectation: possession_changed/steal count = 0; expected `0`, observed `0`.
- PASS expectation: 12 initial contest ticks; expected `12`, observed `12`.
- PASS expectation: contest cleared; expected `0`, observed `0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 28_protection_exact_boundary — PASS

30 following ticks block progress before 18 eligible ticks

- Advanced 48/48 requested ticks; final phase `playing`, score [0, 0], owner `B`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 0 units.
- PASS expectation: first steal exactly tick 48; expected `True`, observed `True`.
- PASS expectation: no progress while protected; expected `0`, observed `0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 29_kick_clears_contest — PASS

A kick interrupts a nearly completed steal

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/298.333 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 5 units.
- PASS expectation: ball_kicked/* count = 1; expected `1`, observed `1`.
- PASS expectation: possession_changed/steal count = 0; expected `0`, observed `0`.
- PASS expectation: free-ball contest cleared; expected `0`, observed `0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 30_recovery_post_loss_no_energy — PASS

Recovery attachment into a post loses control without teleport-derived energy

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 551.532 units.
- PASS expectation: possession_changed/recovery count = 1; expected `1`, observed `1`.
- PASS expectation: possession_changed/post_collision count = 1; expected `1`, observed `1`.
- PASS expectation: ball_kicked/* count = 0; expected `0`, observed `0`.
- PASS expectation: no invented speed; expected `0`, observed `0.0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 31_locked_player_rebound — PASS

Locked player is solid and becomes last toucher

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/300 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: ball_collision/player_contact count = 1; expected `1`, observed `1`.
- PASS expectation: normal speed reflected at restitution 0.5; expected `-150`, observed `-150.0`.
- PASS expectation: last touch changes to A; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 32_post_restitution — PASS

Post bounce preserves last touch and uses restitution 0.7

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/1200 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: ball_collision/goalpost count = 1; expected `1`, observed `1`.
- PASS expectation: post vx = -840; expected `-840`, observed `-840.0`.
- PASS expectation: post preserves B last touch; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 33_four_rebounds_one_tick — PASS

Multiple locked-player impacts consume the remaining tick

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/6000 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 20 units.
- PASS expectation: ball_collision/* count = 4; expected `4`, observed `4`.
- PASS expectation: four-bounce vx = 375; expected `375`, observed `375.0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 34_post_then_pickup — PASS

Post bounce can precede a recovery within one tick

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `B`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/6000 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 552.959 units.
- PASS expectation: post collision before recovery; expected `True`, observed `True`.
- PASS expectation: B recovers; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 35_post_then_goal — PASS

Post turns off-target high-speed shot into a goal

- Advanced 1/1 requested ticks; final phase `goal_restart`, score [1, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/6000 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: post then goal event order; expected `True`, observed `True`.
- Goal events agree with score changes at every observed physics tick.

### 36_strict_threshold_touch — PASS

Touching x=807 alone is not a strict whole-ball exit

- Advanced 6/6 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: goal/* count = 0; expected `0`, observed `0`.
- PASS expectation: out_of_bounds/* count = 0; expected `0`, observed `0`.
- No violated checks in this probe; this does not establish realistic tuning.

### 37_inclusive_mouth_edge — PASS

Whole ball fits exactly at lower legal centre y=197

- Advanced 1/1 requested ticks; final phase `goal_restart`, score [1, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/300 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: goal/* count = 1; expected `1`, observed `1`.
- PASS expectation: out_of_bounds/* count = 0; expected `0`, observed `0`.
- Goal events agree with score changes at every observed physics tick.

### 38_swapped_own_goal — PASS

Credit attack direction, not the last toucher

- Advanced 1/1 requested ticks; final phase `goal_restart`, score [0, 1], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/300 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: B scores despite A last touch; expected `1`, observed `1`.
- Goal events agree with score changes at every observed physics tick.

### 39_known_touch_boundary_restart — PASS

A's top exit awards an inset restart to B

- Advanced 66/66 requested ticks; final phase `playing`, score [0, 0], owner `B`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/300 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 278.585 units.
- PASS expectation: out_of_bounds/* count = 1; expected `1`, observed `1`.
- PASS expectation: possession_changed/boundary_restart count = 1; expected `1`, observed `1`.
- PASS expectation: B owns restart; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 40_unknown_touch_neutral_restart — PASS

Unknown responsibility creates free centre restart without blame

- Advanced 66/66 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/300 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: restart_completed/neutral_restart count = 1; expected `1`, observed `1`.
- PASS expectation: neutral centre ball; expected `True`, observed `True`.
- No violated checks in this probe; this does not establish realistic tuning.

### 41_full_restart_freeze — PASS

Goal followed by all 60 frozen restart ticks and play resumption

- Advanced 67/67 requested ticks; final phase `playing`, score [1, 0], owner `B`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/300 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 192 units.
- PASS expectation: goal/* count = 1; expected `1`, observed `1`.
- PASS expectation: possession_changed/kickoff count = 1; expected `1`, observed `1`.
- PASS expectation: restart completes at tick 61; expected `True`, observed `True`.
- Goal events agree with score changes at every observed physics tick.

### 42_final_tick_goal — PASS

Final-tick goal and whistle both survive; no future kickoff

- Advanced 1/1 requested ticks; final phase `finished`, score [1, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/300 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: goal then whistle; expected `True`, observed `True`.
- PASS expectation: pending restart cleared; expected `True`, observed `True`.
- Goal events agree with score changes at every observed physics tick.

### 43_off_grid_terminal_resume — PASS

Seven remaining ticks span one full and one partial decision

- Advanced 7/7 requested ticks; final phase `finished`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 370 units.
- PASS expectation: match_finished/* count = 1; expected `1`, observed `1`.
- PASS expectation: two decision queries; expected `2`, observed `2`.
- No violated checks in this probe; this does not establish realistic tuning.

### 44_opposed_contact_settling — PASS

Normal squeezing settles while tangential roll survives

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/6000.13 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 14 units.
- PASS expectation: ball_constraint/opposed_contacts count = 1; expected `1`, observed `1`.
- PASS expectation: physics_limit/* count = 0; expected `0`, observed `0`.
- PASS expectation: tangential speed retained; expected `40`, observed `40.0`.
- Opposed-contact settling used a bounded approximation; visual inspection is still needed.

### 45_deliberate_budget_saturation — WARN

Artificial one-impact budget reports saturation explicitly

- Advanced 1/1 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/6000 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 20 units.
- PASS expectation: physics_limit/impact_budget count = 1; expected `1`, observed `1`.
- PASS expectation: saturated ball stops; expected `0`, observed `0.0`.
- Explicit physics-limit saturation occurred; inspect trace, not a silent success.
- Deliberately reduced budget, not a failure at the default 32-impact budget.

### 46_random_default — PASS

Both agents issue seeded random actions for a full match

- Advanced 3600/3600 requested ticks; final phase `finished`, score [0, 0], owner `free`.
- Minimum stamina 96.7803; maximum snapshot-observed player/ball speeds 131.187/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 313.643 units.
- PASS expectation: match_finished/* count = 1; expected `1`, observed `1`.
- Random probes test combinations, not tactical quality; a low score does not itself imply bad physics.

### 47_random_swapped — PASS

Both agents issue seeded random actions for a full match

- Advanced 3600/3600 requested ticks; final phase `finished`, score [0, 0], owner `free`.
- Minimum stamina 96.7803; maximum snapshot-observed player/ball speeds 131.187/206.438 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 156.508 units.
- PASS expectation: match_finished/* count = 1; expected `1`, observed `1`.
- Random probes test combinations, not tactical quality; a low score does not itself imply bad physics.

### 48_random_near_posts — PASS

Both agents issue seeded random actions for a full match

- Advanced 3600/3600 requested ticks; final phase `finished`, score [1, 0], owner `free`.
- Minimum stamina 97.1726; maximum snapshot-observed player/ball speeds 102.487/280.948 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 30.173 units.
- PASS expectation: match_finished/* count = 1; expected `1`, observed `1`.
- Goal events agree with score changes at every observed physics tick.
- Random probes test combinations, not tactical quality; a low score does not itself imply bad physics.

### 49_random_low_stamina — PASS

Both agents issue seeded random actions for a full match

- Advanced 3600/3600 requested ticks; final phase `finished`, score [0, 0], owner `free`.
- Minimum stamina 0; maximum snapshot-observed player/ball speeds 76.2647/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 282.079 units.
- PASS expectation: match_finished/* count = 1; expected `1`, observed `1`.
- Random probes test combinations, not tactical quality; a low score does not itself imply bad physics.

### 50_unsupported_narrow_reset — PASS

Historical 50x100 reset failure: fallback spawns must be separated and advance normally

- Advanced 6/6 requested ticks; final phase `playing`, score [0, 0], owner `free`.
- Minimum stamina 100; maximum snapshot-observed player/ball speeds 0/0 units/s (includes injected initial velocities).
- Minimum completed-snapshot body clearance: 20 units.
- PASS expectation: accepted configuration produces 7 snapshots including initial; expected `7`, observed `7`.
- PASS expectation: reset bodies are already separated; expected `True`, observed `True`.
- Historical scenario ID retained. Narrow reset now uses vertical fallback spawns; this does not prove support for every custom pitch/post layout.

## Reproduction and artifacts

```powershell
uv run --locked --only-group engine python -m game.scenarios --output results/physics_scenarios
# Single case:
uv run --locked --only-group engine python -m game.scenarios --scenario 45_deliberate_budget_saturation --output results/physics_budget_probe
```

The output directory contains `results.json` (initial/final state, first-failure context, config, agent plans, expectations, errors, all events and replay hashes), `report.md`, and one event timeline per scenario. `--report PATH` additionally writes this report at PATH. Audit exit status is 1 if any scenario FAILs, otherwise 0; reports are written before exit. Generated `results/` artifacts are ignored by the existing repository policy.

## Limitations

The catalog contains 50 named probes, not 50 full matches: horizons range from one tick to a full 3,600-tick match. Setup fixtures may inject positions, velocities, ownership, timers and compressed restarts. Seeded agents randomize actions only; resets are not randomized. This diagnostic controller is not the team's evaluation opponent and is not an RL policy. There is no subjective experience, visual observation, wall-clock throughput benchmark or reward evaluation. Linear snapshot sweeps, turning chords and bounded settling are not an exact accelerated rigid-body solver. The cadence check shares engine code and cannot independently prove physics correctness. State snapshots are inspected after ticks; assertions do not reconstruct every instant within a tick. Event timeline seconds use stored tick stamps; fractions remain explicit. No physics constants or mechanics were changed by this audit.
