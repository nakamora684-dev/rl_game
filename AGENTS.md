# AGENTS.md — 1v1 Football Deep Reinforcement Learning

> Repository-level instructions for AI coding agents, especially when assisting **Member A (Game Engine & Physics Lead)**. Read this file before proposing or changing code. This file is a project guide, not a substitute for the project specification or the student's own understanding.

## 0. Mission and working philosophy

Build a **2D top-down 1v1 football game** with a **single deterministic shared simulation core** usable in headless reinforcement-learning training and graphical Pygame gameplay. A DRL agent will be trained using **PPO as the initial/baseline DRL algorithm**, then compared with a rule-based opponent.

**The human is learning. Do not treat this as an autonomous code-generation job.** The student owns the decisions, implementation understanding, tests, and explanations. Prefer a mentor/pair-programmer workflow:

1. Ask what the student understands and what they intend to implement when starting a new substantial mechanic.
2. Explain the relevant physics/software concept, invariants, alternatives, and trade-offs in plain language.
3. Propose a **small, independently testable step**, not an entire subsystem.
4. By default, **do not write implementation code until explicitly requested**. The student has previously preferred design guidance without code; this file does not override a new explicit request for code.
5. When code is requested, provide the smallest useful change, explain each important choice, and suggest a test the student should write or run.
6. When reviewing the student's code, first identify the root cause, ask a guiding question where useful, and give hints before offering a complete fix, unless the student requests a direct fix.
7. Never invent test results, benchmarks, physics constants, or completed features. Distinguish **agreed design**, **proposed defaults**, and **open questions**.
8. Ask the student to explain a newly implemented mechanic back in their own words, especially at phase milestones. Encourage debugging, experimentation, and reading docs.
9. Do not modify unrelated modules, add dependencies, or restructure the repository without explaining and obtaining approval.
10. Respect other team members' ownership; coordinate interface changes rather than silently changing their modules.

## 1. Assignment and delivery constraints

The course project requires a game-based RL environment with:

- Interactive graphical mode and **headless mode** for fast training/testing.
- A trained autonomous DRL agent; **PPO is our chosen baseline DRL algorithm**.
- A **customized neural-network architecture** even if using Stable-Baselines3.
- TensorBoard logging of **total undiscounted reward per episode**; use a moving average if noisy.
- A documented **rule-based baseline** and a justified, multi-metric comparison against the RL agent.
- Per-match evaluation results in **CSV** and plots/tables in `evaluate.ipynb`.
- A final PDF presentation/report covering game, MDP, DRL algorithm, customized network, hyperparameters, baseline, implementation/dependencies/how to run, results, references, and original contribution(s).
- Required root files: `train.py`, `test.py`, `play.py`, `observe.py`, `evaluate.ipynb`, `README.md`, `pyproject.toml`, `uv.lock` (uv project).
- Deadline: **31 December 2026**. Our team has **four members**; if the assignment specifies a different team size, confirm approval with the instructor.

## 2. Team ownership and collaboration

| Member | Lead area | Responsibilities |
|---|---|---|
| **A — current student** | **Simulation engine & physics** | Core state, player movement, stamina, ball dynamics, collisions, possession/steal mechanics, match rules, deterministic reset, physics/event tests. |
| B | Pygame and rule-based gameplay | Renderer/HUD, human controls, rule-based opponent, visual inspection. |
| C | RL environment & reward | Gymnasium-compatible interface, observations/actions, reward bookkeeping, termination, reward exploit tests. |
| D | PPO, evaluation, delivery | Custom PPO policy/value network, training/logging/checkpoints, CSV evaluation, notebook, integration/release coordination. |

**Interfaces to agree before coding:** A↔B rendering reads state but never owns physics; A↔C step/reset/events/state representation; B↔C opponent action interface; C↔D action/observation shapes and episode semantics. All members review the integrated game weekly. Each member documents and tests their own area.

## 3. Agreed gameplay design

### Pitch and match

- **2D top-down football**; visual inspiration was provided by the student (YouTube video `https://www.youtube.com/watch?v=5pLDyW1sCQQ`). Do not claim the video has been inspected unless it actually has.
- Initial **proposed** logical pitch: **800 × 500 units**, centered goal openings **120 units** wide on left/right short boundaries, player radius **15 units**. These are implementation defaults to validate, not hard course requirements.
- Initial **proposed** spawn positions: player A `(200,250)`, player B `(600,250)`, ball `(400,250)`.
- Match length: **60 seconds of simulated time**. Higher score wins; ties are draws.
- A goal occurs when the ball **fully crosses** the appropriate goal line inside the opening. Register the event once, then enter a controlled restart; scoring does **not** automatically terminate the match.
- Out of bounds: the **last player to touch** is responsible, loses the ball, and the opponent restarts from the exit location; responsible player resets near their own goal. Starting the receiver in controlled possession is a **proposed default**, to confirm.
- Swap attacking sides/orientations between episodes and evaluate both sides to avoid left/right overfitting.

### Player movement and stamina

- Both players use identical movement/physics rules.
- **Acceleration-based movement** with eight-direction + idle input, continuous effort, smooth turning, deceleration and persistent facing direction when stopped. Logical coordinates use **x right, y down**; positions and velocities remain continuous.
- Movement intensity/effort is continuous and changes speed/effort; high effort consumes stamina, low effort/rest regenerates it.
- Stamina begins at **100/100**, persists through possession changes, and gradually reduces speed and acceleration when low. Initially it does **not** directly affect kick accuracy, turning rate, tackling, or kick cost.
- **Student-approved stamina starting settings:** full-effort drain 5 points/s (20 s from full to empty), idle recovery 100/15 points/s (15 s from empty to full), neutral active effort 0.5, and linear speed/acceleration factors down to 50% at empty. Rates interpolate linearly on either side of neutral. Idle counts as rest regardless of unused effort; requested non-idle effort costs energy even against a wall. Update stamina first each physics tick, then derive target speed/acceleration from the base configuration; do not multiply current velocity by fatigue. These are approved experiment-0 settings to test, not frozen optimal tuning.
- **Approved starting braking model:** constant idle braking, clamped at zero to prevent reversal. **40 logical units/second²** is the student's provisional braking setting, to validate against actual running speeds. Idle preserves facing even while coasting.
- No numerical acceleration, turn-rate, speed, stamina-consumption or recovery constants have been finalized. Propose measurable values, test behaviour, and document them before freezing.
- **5 Nov implementation defaults (student delegated speed/acceleration selection):** maximum speed 160 units/s (800 units / 5 s cruising), acceleration limit 320 units/s² (rest to full speed in 0.5 s), provisional turn rate 2π radians/s. These are configurable starting values, not frozen tuning. Effort scales target speed; velocity approaches it with bounded acceleration. A non-idle direction turns facing even at zero effort; idle preserves it. At speed 160, idle braking 40 takes 4 s to stop and still needs validation.

### Ball, possession and stealing

- **Hybrid ball model:** in controlled possession the ball follows a small offset in front of the carrier; on kick it becomes free with power-dependent initial velocity and friction; either player may recover a free ball.
- **Approved free-ball starting tuning:** constant deceleration 100 units/s², with reference speed 300 units/s stopping after 3 s and travelling 450 units. Weaker rolls stop sooner; reference speed does not finalize maximum kick speed/power mapping. Free-ball position is stored; only controlled attachment can be derived from a carrier. The first A3 slice integrates average speed over each tick's moving portion, including partial-tick stopping, and does not clamp the ball to the pitch.
- Kick direction follows the player's **facing direction**. There is no independent aim direction. Kick trigger and kick power are selected in the **same decision step** (no charging mechanic).
- **Steal/contact contest:** a defender must remain close for a short interval before taking possession; cooldown prevents immediate possession toggling. Exact thresholds/timings remain to be tuned.
- Keep **one authoritative possession owner** (`A`, `B`, or `none`); derive player-level possession indicators. Track last toucher separately.
- Handle ball crossing, goal lines, and high-speed kicks robustly (e.g., swept checks/substeps if needed). Test tunnelling.

## 4. Time model and action contract

- Physics simulation: **60 Hz** → fixed `dt = 1/60` simulated seconds.
- Policy/opponent decisions: **10 Hz** → one decision every **6 physics updates**. Hold the last action between decisions.
- The match clock **runs during restarts**. A full 60-second match has **3,600 physics updates**, spanning 600 six-tick decision intervals; whether policies are queried during restarts remains an interface decision. The match can finish during a restart.
- **Simulation time, not wall-clock time**, determines match duration. Headless training advances as fast as compute allows; visible Pygame mode is paced for human play.
- **Student-approved hybrid action** contract (supersedes the earlier five-continuous-value proposal in the Word plan; communicate this change to B/C/D):

| Component | Type / range | Meaning |
|---|---|---|
| `direction` | Discrete, 9 choices | Idle, N, NE, E, SE, S, SW, W, NW; engine IDs are 0 through 8 in this order |
| `effort` | Continuous `[0, 1]` | Movement intensity / stamina expenditure |
| `kick_trigger` | Binary `0` or `1` | At most one kick request consumed per decision, not per held physics tick |
| `kick_power` | Continuous `[0, 1]` | Instantaneous shot strength if kick is requested |

- Direction vectors are normalized to length 1, except idle `(0,0)`. Idle supplies no driving acceleration; constant braking slows existing velocity without changing facing.
- **Important unresolved interface details:** exact kick-attempt timing, failed-request handling, phase-change handling, and cooldown duration. A first-tick attempt discarded on failure is a proposal, not yet approved. Each new decision may issue a new request; enforce valid-possession/cooldown guards.
- This hybrid action needs an explicit encoding or custom distribution for Stable-Baselines3 PPO; a generic structured mixed action is not directly supported. C/D must agree on this before building the RL interface. Do not silently discretize effort/power or round a continuous direction value.

## 5. Internal state and event invariants — Member A's main contract

### Store (authoritative, persistent)

- For each player: position `(x,y)`, velocity `(vx,vy)`, facing orientation, stamina, any active contact/steal progress and cooldown timers.
- Ball: position `(x,y)`, velocity when free, possession owner (one source of truth), last toucher.
- Match: score, attack orientation, authoritative simulation tick, phase (`playing`, `goal_restart`, `out_of_bounds_restart`, `finished` or equivalent), relevant restart owner/timer.
- Event transition history only where needed: previous possession owner, one-shot goal/out-of-bounds/steal/recovery events; previous potential/progress only if reward computation requires it.
- Any additional state required by actual physics must be explicit, not hidden in rendering or global mutable variables.

### Derive instead of duplicating

- Remaining match time from authoritative simulation tick (unless an explicit clock-pause policy requires a different representation).
- Per-player `has_ball` from the single ball possession owner.
- Fixed dribble offset from carrier position/facing (store it only if dynamic).
- Distances, directions, normalized progress, and other geometric features from positions and orientation.
- Do not store persistent goal/out-of-bounds Boolean flags that accidentally cause duplicate events on successive updates.

### Critical invariants

1. At most **one** player controls the ball at a time.
2. A controlled ball follows its owner; a free ball evolves according to its own physics.
3. The ball cannot be both controlled and independently kicked/free in the same state.
4. A confirmed goal, boundary crossing, steal or recovery is emitted **exactly once** per transition.
5. Restart phases prevent duplicate goals/turnovers; restart position, velocity, possession, cooldown, and clock policies are explicit.
6. All mechanics work identically in headless and rendered modes because both use the **same** simulation core.
7. Identical seed + initial state + action sequence should yield identical simulated state trajectories (within floating-point tolerance) on the same implementation/runtime.
8. Position/velocity/stamina values remain finite; stamina stays in `[0,100]`; ball/player state never becomes internally contradictory.
9. Reset removes leftover event flags, timers, possession history, and other prior-episode state.
10. A goal line opening and an out-of-bounds boundary are handled consistently at corners/edges.

## 6. PPO observation: design proposal, not yet user-confirmed

Member C owns the Gymnasium observation, but Member A must expose sufficient engine state for it. Start with a **flat, normalized numeric vector**, preferably **player-relative and attack-normalized**. The student has reviewed the internal state but has **not yet explicitly confirmed** the final observation choices.

Candidate features:

- Self: normalized pitch position, velocity, facing represented as `(sin θ, cos θ)`, stamina.
- That `(sin θ, cos θ)` is a candidate feature order, not a geometric direction vector. In the y-down engine, physical forward is `(cos θ, sin θ)` with zero east and positive clockwise; agree the final ordering/attack normalization with C.
- Opponent: relative position, velocity, facing, stamina, possession indicator.
- Ball: relative position, free-ball velocity (with clear convention when attached), possession category (`self`, `opponent`, `free`).
- Tactical geometry: relative distances/directions to ball, opponent, and goals (some may be redundant; assess rather than blindly adding all).
- Match: normalized remaining time and score difference (candidate default).
- Contact/cooldown and restart phase/timers **if they affect the next transition and are not otherwise inferable**.

Do **not** automatically expose last toucher, previous possession owner, reward components, or prior progress as observation features: those are engine bookkeeping, unless a field genuinely represents observable, decision-relevant game information.

## 7. Rewards: owner is Member C; Member A supplies reliable events

**Never silently finalize or tune the reward function on behalf of the student or Member C.** The following are **experimental starting values discussed earlier**, not guaranteed optimal weights:

| Signal | Tentative value | Event semantics |
|---|---:|---|
| Score goal | +10 | Once per goal |
| Concede goal | −10 | Once per conceded goal |
| Match win | +5 | Once at terminal whistle |
| Match draw | 0 | Neutral terminal outcome signal |
| Match loss | −5 | Once at terminal whistle |
| Steal from opponent-controlled possession | +1 | Once on confirmed transition |
| Recover loose ball | +0.5 | Once on free → controlled transition |
| Per-step possession duration | 0 initially | Avoid rewarding stalling |
| Attacking ball progress | Potential-based shaping; `λ≈2` proposed | Must be consistent across transitions, not reset to generate artificial rewards |
| Turnover / boundary penalty | Context-dependent; **not finalized** | Avoid double-counting and penalizing every legitimate kick |

The student's reward objectives: match outcomes, goals, ball recovery, controlled possession, turnover penalties, and progress with the ball toward the opponent's goal. **Controlled possession** means attached/dribbled ball; free ball is not controlled. Distinguish steals from loose-ball recoveries. Agent must not farm rewards by deliberately losing/recovering possession, kicking needlessly, moving back-and-forth, stalling, or forcing boundaries.

**Member A's task:** expose trustworthy transition events, with explicit previous/new owners and event causes. Member C computes reward. Keep reward arithmetic **out of the physics engine**.

## 8. Architecture and ownership boundaries

Suggested project layout (coordinate names with team; do not assume files already exist):

```text
football-drl/
├── AGENTS.md
├── game/               # Member A: state, physics, player, ball, rules, reset
├── environment/        # Member C: RL wrapper, observations, rewards
├── rendering/          # Member B: Pygame visuals and human input
├── agents/             # Member B: rule-based baseline
├── models/             # Member D: checkpoints/configs
├── logs/               # Member D: TensorBoard
├── results/            # Member D: CSV and plots
├── train.py             # Member D
├── test.py              # Member D
├── play.py              # Member B
├── observe.py           # Members B/D
├── evaluate.ipynb       # Member D
├── README.md            # Shared; release coordinator D
├── pyproject.toml
└── uv.lock
```

Design a **pure, renderer-independent simulation API**. No Pygame calls, event loops, sleeping, display objects, or wall-clock timing in `game/`. Avoid importing training frameworks in `game/`. Use seeded randomness only through controlled simulation state/configuration. The rendering layer reads state; the RL wrapper calls the same simulation; agents submit actions through a documented interface.

Do not pick a class hierarchy prematurely. Prefer simple, testable data structures and clearly separated state transition functions. Document units (logical distance, seconds, speed, acceleration), coordinate orientation, collision radii, and order of updates.

## 9. Update order — proposed for discussion and tests

A possible deterministic physics tick sequence is:

1. If the match is in a restart/terminal phase, advance the appropriate phase/timer; do not apply ordinary gameplay updates when prohibited.
2. Apply the held actions to both players; update effort, acceleration, velocity, facing and stamina.
3. Resolve player–pitch and player–player interactions.
4. Process kick requests subject to possession/cooldown guards.
5. Update controlled-ball attachment **or** integrate free-ball motion and friction.
6. Resolve free-ball pickup / steal contest and associated cooldowns.
7. Detect swept goal/boundary crossings and apply exactly one authoritative match transition.
8. Emit structured events, update the simulation tick, and publish a coherent new state.

This order is **provisional**: evaluate edge cases such as a kick and boundary crossing in the same tick, simultaneous pickup, or a collision that changes ball ownership. Write tests before committing to the final order.

## 10. Member A: phased roadmap and learning checkpoints

| Phase | Student-owned tasks | Evidence / checkpoint |
|---|---|---|
| **A0 — Foundation** | Read design; choose coordinate conventions, units, fixed-step model, game-state schema, events, reset/seed strategy; agree interfaces with B/C. | Student can explain `dt`, 60 Hz vs 10 Hz, deterministic stepping, and state invariants. |
| **A1 — Player physics** | Implement positions, acceleration/deceleration, normalized diagonal input, smooth facing, pitch bounds. | Tests: zero input, full effort, diagonal speed, turning, stopping, boundary collision. |
| **A2 — Stamina** | Implement consumption/recovery and smooth speed/acceleration reduction. | Plot or log stamina/speed over sustained sprint and rest; explain why behavior is gradual. |
| **A3 — Ball dynamics** | Implement free-ball velocity/friction, controlled attachment, kick trigger/power/facing. | Tests: direction, kick power monotonicity, friction, pickup, no phantom repeated kick. |
| **A4 — Contact/possession** | Implement owner transitions, contest progress, cooldown, pickup rules. | Tests: no double ownership, contested steals, recovery, cooldown and no rapid toggling. |
| **A5 — Match rules** | Goal detection, out-of-bounds last-touch, score, reset/restart phases, 60 s clock. | Tests: one-shot goal, edge cases, responsible last toucher, proper restart, end of match. |
| **A6 — Integration** | Expose stable simulation API; support B renderer and C environment; benchmark headless speed. | Same seeded action sequence produces equivalent state trajectories with/without renderer. |
| **A7 — Hardening** | Fix collision tunnelling, numerical instability, reproducibility issues; document engine. | Repeatable regression tests and a short engine design note. |

**Overall proposed team schedule:** P1 design 24–30 Oct; P2 simulation 27 Oct–13 Nov; P3 rules/rendering 3–20 Nov; P4 RL wrapper 10–24 Nov; P5 baseline/tests 17–28 Nov; P6 PPO 24 Nov–12 Dec; P7 evaluation 5–19 Dec; P8 release 14–30 Dec. These are planning targets, not proof of completed work. As of 3 Nov, revisit any missed dates with the team; prioritize a working simulation over schedule cosmetics. Freeze new features around 19 Dec and reserve 20–30 Dec for stabilization before the 31 Dec deadline.

## 11. Test-first checklist for every Member A change

Before implementing a new mechanic, the AI should help the student write down:

- **Inputs:** state, action, config, fixed timestep, seed.
- **Expected outputs:** next state, event(s), phase changes.
- **Invariants:** what must always be true.
- **Boundary cases:** zero/maximum effort, zero movement, pitch edges, overlapping players, simultaneous contacts, high-speed shots, restarts, end of match.
- **One tiny manual example:** values small enough to reason about without executing code.
- **One automated test** and one visual inspection idea (when relevant).

After implementation, report only tests actually executed, with command, outcome, and limitations. Prefer deterministic unit tests to visual intuition. Do not claim physics is realistic solely because it looks plausible; measure it.

## 12. Evaluation context (so Member A designs for instrumentation)

Member D will evaluate against the rule-based agent on held-out seeds and both attacking directions, ideally ~100 matches per comparison if feasible, with sample size and uncertainty reported. Metrics include win/draw/loss, goal difference, goals for/against, shots/conversion, controlled possession, turnovers by cause, steals versus loose-ball recoveries, attacking progress, stamina use, kick/idle frequency, out-of-bounds loops, and computational throughput.

Member A must provide reliable **event identifiers/types, time, player/ball positions, possession transitions, scores, stamina, and termination reasons** so these metrics can be computed without guessing from rendering. TensorBoard requires **undiscounted episode return**; Member D owns logging, but event correctness starts in the engine.

## 13. How an AI agent should respond to Member A

Use this pattern unless the student explicitly requests a different one:

**Concept → Decision → Student attempt → Review → Test → Next step.**

- **Concept:** concise explanation with a small numerical or football example.
- **Decision:** state what is agreed, what is a proposal, and which trade-off matters.
- **Student attempt:** invite a sketch, pseudocode, data structure, or own implementation first.
- **Review:** inspect the student's attempt for invariants, performance, edge cases, and integration issues. Explain *why* a change is necessary.
- **Test:** specify expected behaviour and how to verify it.
- **Next step:** advance only one manageable increment.

Do not overwhelm with an entire repository of code. For difficult topics (collision resolution, numerical integration, swept intersections, contact state machines), teach the underlying ideas and show how to reason about them. If asked to generate code, be transparent about what was generated and require the student to understand and test it. Do not make irreversible changes or commit/push code without explicit permission.

### Useful questions the agent should ask before substantial implementation

- What are the exact inputs, outputs, and invariants for this mechanic?
- Which parameters are measured/configurable rather than hard-coded?
- What is the simplest edge case that could break this design?
- How can we prove this runs identically headless and rendered?
- What does Member B/C need from this interface?
- Can you explain the physical meaning and units of each equation?

## 14. Open decisions: do not present as settled

- Exact physics parameters (speed, acceleration, drag/friction, turn rate, ball speed, player/ball radii, contact distance, contest duration, cooldowns, stamina rates).
- Detailed goal restart sequence and out-of-bounds placement/ownership conventions; the clock running during restarts is approved.
- Exact kick-request attempt/expiry semantics and kick cooldown; binary trigger and once-per-decision consumption are approved.
- Exact normalized observation features (player-relative attack-normalized is a recommendation, not yet confirmed).
- Final reward weights and turnover penalties; potential-based shaping implementation and its assumptions.
- PPO network architecture and hyperparameters, training budget, curriculum/self-play extensions.
- Team-specific repository conventions and instructor approval for four members if required.

## 15. Project reference and authority

The team's detailed planning document is `football_drl_final_project_plan.docx` (provided in the ChatGPT conversation). If a local copy is present, read it for additional tables and diagrams. This `AGENTS.md` captures the design discussions and Member A guidance but should be kept synchronized when the team explicitly approves changes. **Do not treat proposals as fixed facts** or silently resolve disagreements between the document and newer team decisions. Ask the student to confirm material changes.

## 16. Implemented foundation and reproducible engine workflow (5 Nov)

- `game/` now has immutable player/action/config/state snapshots, normalized direction mapping, target-velocity acceleration, constant idle braking, smooth facing, player-pitch bounds, side-swapping reset, one-tick/six-tick stepping, a 60-second running clock, and a one-shot terminal event. See `docs/engine.md` for precise semantics and limitations.
- Stamina consumption/recovery and speed/acceleration effects are now implemented in `game/stamina.py` and integrated through `step_player`/simulation. `python -m game --stamina` logs a 20 s sprint / 15 s rest trace. The isolated `step_idle_player` helper still does braking only; do not use it as a complete player tick.
- Free-ball state/motion is now implemented in `game/ball.py` and shared simulation stepping; reset places it stationary at pitch center. `python -m game --ball` prints the reference roll CSV. Ownership/last touch will be added with possession rules; all current balls are free.
- This is **not a complete football simulation**: no score/goals, kicks, possession, ball/player-player collisions, RL wrapper, or rendering yet. Restart phases are a countdown/freeze scaffold, not implemented football restart rules. They currently freeze players/stamina/ball too; actual restart recovery policy still needs confirmation. Do not claim all Member A phases are complete.
- Python is pinned in `.python-version`; engine tools are declared in the `engine` dependency group and locked in `uv.lock`. Reproduce with `uv sync --locked --only-group engine`.
- Engine checks: `uv run --locked --only-group engine pytest -m "not full_stack"`; lint/format checks: `uv run --locked --only-group engine ruff check game tests` and `uv run --locked --only-group engine ruff format --check game tests`. The `full_stack` smoke test needs the full project install and is explicitly excluded here.
- Headless inspection: `uv run --locked --only-group engine python -m game`. Do not modify B/C/D entry-point scaffolds just to make them appear complete. Every environment dependency change must be reflected in `pyproject.toml`/`uv.lock`, and interpreter changes in `.python-version`; never rely on unrecorded global or manual pip installs.
