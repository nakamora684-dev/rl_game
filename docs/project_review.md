# Project review map

This is the human entry point for the Graphify project graph. Labels distinguish
implemented code, approved starting settings, proposals, and planned team work.
The graph is an index of source evidence, not proof that every planned feature exists.

## Source authority and history

- `football_drl_final_project_plan.docx` is the original team plan and assignment context.
- `AGENTS.md` records the newer student-approved decisions and implementation status.
- `docs/engine.md` documents current engine behavior and its tests.
- `README.md`, `pyproject.toml`, `uv.lock`, and `.python-version` describe reproducible setup.
- `game/` and `tests/` are the current implementation and executable checks.
- The original plan's continuous x/y movement action is superseded by the approved
   nine-direction + continuous effort/power + binary kick interface. The engine
   implements this as `PlayerAction`; the hybrid PPO encoding is still open.
- The plan's pitch drawing uses y-up; the approved engine convention is x-right/y-down.
- The graph must preserve these historical differences rather than silently merging them.

## Implemented: Member A foundation

- Immutable player, ball, movement-action, configuration, and simulation snapshots.
- 60 Hz physics; each decision advances up to six individual ticks (10 Hz).
- Both players use identical movement and stamina equations.
- Nine normalized movement directions including idle, continuous effort, gradual
  target-velocity acceleration, smooth facing, constant idle braking, and pitch bounds.
- Stamina consumption/recovery and gradual speed/acceleration fatigue, not momentum scaling.
- Free-ball constant-deceleration motion, including partial-tick stopping; no pitch clamp.
- Controlled possession: one owner, separate last toucher, facing-based attachment,
  derived possession flags, and explicit setup events (not gameplay pickups or steals).
- Direct kick release with explicit speed, facing-based free velocity, retained last
  toucher, and one `ball_kicked` event.
- `PlayerAction` integration: linear power mapping, first-tick-only requests discarded
  on failure/restart, and explicit per-player kick cooldown advancing in simulation time.
- Automatic free-ball pickup: swept first contact at any speed, nearest on simultaneous
  contact, exact ties stay free; one recovery event, separate 12-tick kicker pickup lockout.
- Swept equal-mass player contact without bounce, tangential sliding, wall-aware overlap correction.
- Locked-player ball rebounds (restitution 0.5, no recoil), solid radius-4 posts
  (ball restitution 0.7), and sliding body/post constraints preserving the clear opening.
- Controlled post contact releases without kick/cooldown, starts pickup lockout, and
  uses real attachment motion rather than teleport energy. Bounded multi-impact flights
  emit collision, opposed-contact settling and saturation diagnostics.
- Automatic stealing: 35-unit centre range, 18 consecutive eligible ticks; recovery/steal
  and owned restarts grant 30-tick acquisition protection. Events distinguish recovery/steal/setup.
- Strict whole-ball swept goals and boundary exits, orientation-correct scoring, and
  60-tick restarts: conceder centre kickoff or last-touch opponent boundary restart.
- Unknown-touch exits restart neutrally; score/clock/stamina/sides survive restarts, and
  match time continues while motion/stamina freeze. Final-tick goals do not erase the whistle.
- Reset, side swapping, 60-second clock running through restarts, and one-shot finish.
- Headless movement, stamina, and ball inspection commands. No graphical inspection yet.
- Runner-audit repairs (17 Nov): restart event/state last-touch agreement, no rebound
  energy from stationary correction, and separated narrow reset spawns. The repaired
  catalog has 49 PASS, 1 deliberate-budget WARN, 0 FAIL, with 50/50 identical replays;
  see `docs/physics_fixes.md` and `docs/physics_scenario_report.md`.

## Approved starting settings: still subject to tuning

| Setting | Starting value / interpretation |
|---|---|
| Coordinates | x right, y down; facing zero east, positive clockwise |
| Movement intent | Idle, N, NE, E, SE, S, SW, W, NW; diagonals normalized |
| Effort / kick power | Continuous `[0,1]`; binary kick request once per decision |
| Speed / acceleration | 160 units/s base speed; 320 units/s² base acceleration |
| Player idle braking | 40 units/s²; 4 s to stop from base speed, needs play testing |
| Stamina | 100 maximum; 5/s full-effort drain; 100/15 per second idle recovery |
| Neutral effort | 0.5; no drain/recovery for active movement at this effort |
| Empty-stamina performance | 50% speed/acceleration; braking and facing unaffected |
| Free-ball roll | 300 units/s reference, 100 units/s² deceleration, 3 s / 450 units |
| Dribble attachment | Follows facing, not velocity; provisional 22-unit offset |
| Kick power | Linear speed 50–300 units/s; triggered power 0 means minimum-speed kick |
| Kick cooldown | 12 ticks / 0.2 s per player, including restart time; separate from pickup/steal rules |
| Ball radius / pickup contact | Radius 7; contact distance 15+7=22, checked with relative snapshot sweeps |
| Kicker pickup lockout | 12 ticks / 0.2 s, separate from kick cooldown; opponent may intercept immediately |
| Player contact | Equal mass, zero bounce; preserve tangential sliding |
| Ball / locked player | Normal relative restitution 0.5; no player recoil; eligible players trap |
| Goalposts | Radius 4, ball restitution 0.7; preserve clear opening 120 |
| Impact budget | 32 ball impacts per tick, shared across control-loss cycles; explicit limit event |
| Steal contest / protection | Range 35, 18 eligible ticks; 30 protected ticks after acquisition |
| Goals / restart | Centred width 120, strict whole-ball clearance; 60 frozen-motion/stamina ticks |
| Clock | 3,600 ticks; time runs during restarts; goals will not end the match |

Pitch size 800×500, player radius 15, and turn rate 2π radians/s remain configurable
implementation defaults. Maximum kick speed now separately defaults to 300 by student approval;
the reference roll speed is not itself a speed cap.

## Planned: remaining validation and integration

1. Validate kick tuning visually and coordinate the full action/cooldown contract with B/C/D.
2. Validate contact/steal/rules tuning visually and coordinate state/events with B/C.
3. Build the full-match runner in a new session; see `docs/next_session.md` for the handoff.
4. Further multi-impact/numerical hardening and replay equivalence, especially observing
   constraint/limit diagnostics under sustained play. No goal-mouth player entry is planned.
5. Renderer/environment equivalence, throughput measurement, and engine report section.

Restarts now implement football placement/ownership, not just countdown scaffolding.
Default strict clearance plus current player bounds means ordinary dribbling cannot
score; kick-only goals were explicitly retained by the student. Goalpost/locked-player
rebounds are implemented with bounded snapshot sweeps, not exact accelerated TOI.
Rendering/RL integration, throughput measurement and visual validation
remain outstanding; do not equate passing engine tests with complete delivery.

## Planned: team ownership and interfaces

| Owner | Area | Main handoff |
|---|---|---|
| A | Shared simulation and physics/event tests | State, actions, fixed stepping, trustworthy transitions |
| B | Pygame, human controls, rule-based baseline | Reads engine state; never owns physics |
| C | Gymnasium observation/reward/action adapter | Computes rewards outside engine; agrees hybrid action encoding with D |
| D | PPO/custom MLP, logging, evaluation/release | TensorBoard undiscounted returns, CSV, notebook, report |

PPO is the chosen baseline, not an implemented learner. Final observation features,
reward weights, network architecture, training budget, and baseline behavior are open.
Root play/train/test/observe scripts are scaffolds. The evaluation notebook/report
remain deliverables, not existing completed results.

## Assignment and validation context

Delivery deadline: 31 December 2026; proposed feature freeze: 19 December. The plan
requires graphical/headless modes, customized network, a documented rule-based
baseline, TensorBoard episode returns, multi-metric comparison, CSV, `evaluate.ipynb`,
README/uv setup, and PDF report/slides. Four-person team approval may need confirmation.

Evaluation should use held-out seeds and both attacking sides; proposed sample size
is about 100 matches per comparison if feasible. Outcomes and goal difference matter
more than rising shaped reward. Steals and loose-ball recovery must remain distinct.

Engine tests run with `uv run --locked --only-group engine pytest -m "not full_stack"`.
The full-stack import test is intentionally excluded without graphics/ML dependencies.
Deterministic unit tests do not demonstrate realistic tuning, successful training,
or completed integration. See actual command results rather than inferred test success.
