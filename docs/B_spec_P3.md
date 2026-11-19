> **Integration note (18 Nov):** this document was written against the old mock engine. The mock is gone: `agents/engine_adapter.py` now runs Member A's real engine. Read `docs/B_integration.md` first; mock-specific statements below are historical.

# Member B - P3 spec: rules + rendering, playable match, baseline v3

Status: delivered on the MOCK engine; to be re-validated on A's real engine (section 8).
Interface: still v2 (no field change). Tests: 108, no display needed.

## 1. Goal of P3 (plan)
A graphical match is playable, the baseline can finish a match, and rendering does not own
physics. Graphical and headless use the same simulation, baseline and metrics.

## 2. What was built
| Block | Files | Result |
|---|---|---|
| Baseline v3 | `agents/config.py`, `agents/difficulty.py`, `agents/rule_based.py` | Corner aiming, goal-side defence, bounded anticipation, stamina-aware sprint, easy/normal/hard presets. |
| Metrics + tools | `agents/metrics.py`, `tools/baseline_stats.py` | Per-match CSV (stable schema), paired seeds in both directions, W/D/L and goal difference with 95% CI. |
| Rendering | `rendering/theme.py, pitch.py, sprites.py, effects.py, hud.py, renderer.py` | Static pitch drawn once, players with stamina ring, ball trail, goal confetti, screen shake, new HUD. |
| Play UX | `rendering/flow.py, screens.py, controls.py, replay.py`, `session.py`, `play.py` | Menu, pause, full-time stats, key remap, gamepad, screenshot (F12), exact replays. |
| Mock engine v2 | `rendering/mock_engine.py` | Goals, steals, out-of-bounds, restarts, one-shot events (so the baseline can be tested). |

Architecture rule kept: the renderer only reads `GameState`. Effects, flashes and shake are
cosmetic state inside the renderer and never feed back into the engine.

## 3. Baseline v3 (rule order)
0. Not in the playing phase: idle. 1. I have the ball: kick if aligned and the facing line enters
the goal mouth, else dribble toward the aim corner (swerve only if the opponent is ahead/beside,
away from touchlines). 2. Opponent has the ball: contest if close or already goal-side, else
stand goal-side `block_dist` in front of the carrier. 3. Ball free: run to where it will be.
4. Sprint only if stamina allows (hysteresis) and the run is long enough to matter; when losing or
drawing in the last 10 s the reserve is spent. `agent.last_rule` names the rule that fired.
The kick direction is the player's facing (there is no kick-direction action), so "aiming" means
turning toward the corner before kicking. Same View as the RL agent, no privileged field.

Presets (generated from the code; all change thresholds only, never physics or information):

| Parameter | easy | normal | hard | Meaning |
|---|---|---|---|---|
| `shoot_dist` | 150.0 | 220.0 | 260.0 | max distance to the aim point to shoot |
| `face_tol` | 0.6 | 0.35 | 0.25 | max angle (rad) between facing and aim line |
| `aim_corners` | False | True | True | aim at the goal corner far from the opponent |
| `snap_shot` | False | True | True | shoot early when pressed |
| `opp_avoid_dist` | 60.0 | 90.0 | 110.0 | swerve when the opponent is closer |
| `goal_side` | False | True | True | defend goal-side before contesting |
| `contest_dist` | 70.0 | 70.0 | 80.0 | closer than this: go for the steal |
| `lead_time_max` | 0.0 | 0.4 | 0.6 | max ball anticipation (s) |
| `sprint_reserve` | 0.5 | 0.3 | 0.25 | stop sprinting under this stamina share |
| `sprint_resume` | 0.7 | 0.5 | 0.45 | resume sprinting above this |
| `endgame_all_in` | False | True | True | spend the reserve late when not winning |

`normal` is the evaluation baseline. easy/hard are for demos and an optional PPO curriculum.
Strength ordering of the presets is NOT validated yet: the mock has no keeper or contact physics.

## 4. Human play
You are BLUE (player 0). Move: arrows / ZQSD / WASD. Sprint: Shift. Kick: hold Space to charge,
release to kick. P or Esc: pause. F1: debug overlay. F12: screenshot. Menu: difficulty and side.
`--difficulty easy|normal|hard`, `--side left|right`, `--swap` (flip side at each rematch),
`--seed`, `--no-menu`, `--controls keys.json`. A replay is saved at full time in
`results/replays/`; `observe.py --replay FILE --headless` reproduces the same score.

## 5. Evaluation tooling (shared with D)
`agents/metrics.py::ROW_FIELDS` is the CSV schema. Per player: goals, shots (kick whose facing line
crosses the opponent mouth), kicks, steals, recoveries, out_of_bounds, possession_share,
attack_half, idle_frac, sprint_frac, stamina_mean/end. Match: seed, swapped, agent names, ticks,
scores, winner, duplicate_events (must be 0). D can reuse `agents.baseline_stats.play_paired` and
`MatchRecorder` in test.py so PPO-vs-baseline rows have the same columns.

## 6. Assumptions A must confirm
| Topic | Assumption in my code | Check |
|---|---|---|
| Velocity unit of GameState | units per physics update (`VEL_PER_TICK = True`) | `test_velocity_unit_matches_displacement`; flip the flag if per second |
| Events | each event once, on the tick it happens | `duplicate_events == 0` in every CSV row |
| Restart | `phase == restart`, `tick` keeps counting, restart_owner gets the ball | contract tests |
| `STEAL_RANGE` | display only value in the debug overlay | replace with the real constant |
| Determinism | same seed + actions = same match (replays rely on it) | `test_replay_reproduces_the_exact_match` on the real engine |

## 7. For C and D
C: build the PPO observation from the same `View` fields as the baseline; the baseline is idle in
`restart`, so the policy should be too. D: `make_baseline(index, difficulty)`; use held-out
seeds (the tool defaults to 1000+) and report the sample size and CI printed by the tool.

## 8. Integration checklist when game/engine.py arrives (P3-1)
1. `uv run pytest`: contract, mode-equivalence and velocity tests now run on the real engine.
2. `uv run python play.py` and play one full match.
3. `uv run python -m agents.baseline_stats --seeds 25`: check goals happen, `duplicate_events` is 0,
   idle fraction is low and out-of-bounds is not looping. If the baseline cannot score, tune
   `agents/config.py` (P5) rather than changing the engine.
4. Fix `STEAL_RANGE` in the debug overlay; flip `VEL_PER_TICK` if test 1 says so.

## 9. Manual visual check (5 min, never done in the sandbox)
- Window opens, pitch stripes and nets visible, nets fit in the margin.
- Stamina ring turns amber then red, tired player looks darker, bar flashes under 25%.
- Goal: confetti + shake + "GOAL!" once; restart pill shows the owner; full-time panel shows stats.
- Pause freezes everything (no flash replay); menu keys work; F12 writes a PNG.
- Try `--scale 1.0` and `--scale 1.5`; if text overlaps, adjust `rendering/theme.py` / `hud.py`.

## 10. Known limits
Pygame never opened here (fake pygame in tests). Mock physics is not the real physics, so goal
counts and preset strength mean nothing until the real engine is used. Gamepad untested on hardware.
Replays need the same engine and baseline version. Baseline tuning and symmetry sign-off are P5.
