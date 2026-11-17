# Review guide and coverage

Start with `docs/project_review.md`. Implemented, approved starting, proposed, and planned concepts are labelled separately. The original plan is historical and does not override newer decisions.

Coverage: local AST for 40 detected code/config files; current Markdown sources including the next-session handoff; original Word text/tables and all three previously reviewed embedded diagrams; repository file/config/dependency inventory. Word conversion used standard-library ZIP/XML because the optional Office converter was absent. Empty entry-point scripts are explicitly scaffolds. Current physics semantic facts were refreshed locally from reviewed source changes and student approvals, without additional agent calls. Historical plan evidence was retained.

The graph is an evidence index, not a completeness proof. Automated checks executed during this build: 461 engine tests and 36 subtests passed; one full-stack dependency test intentionally excluded; Ruff lint/format, lock consistency and git diff checks passed. See graph_health.json for extraction diagnostics and graph_build.json for provenance.

**Graph health warning:** undirected_same_endpoint_collapsed_edges=327. Parallel edges are coalesced visually, but all relation/source evidence is retained on graph.json edges; raw extraction is also retained locally for audit.

# Graph Report - rl_game  (2026-11-11)

## Corpus Check
- Corpus is ~35,399 words - fits in a single context window. You may not need a graph.

## Summary
- 754 nodes · 3206 edges · 27 communities (18 shown, 9 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 320 edges (avg confidence: 0.95)
- Token usage: AST 0 LLM tokens; semantic agent token counts were not exposed by the harness (unknown, not zero). Used free model routes; no additional API key configured.

## Community Hubs (Navigation)
- Stamina and fatigue
- Automatic ball recovery
- Shared simulation and rules
- Original team plan
- Ball rebounds and constraints
- Free-ball dynamics
- Engine inspection and tests
- Goals and restarts
- Engine inspection and tests
- Design status and roadmap
- Design status and roadmap
- Design status and roadmap
- Dependencies and reproducibility
- Design status and roadmap
- CI and validation
- Player movement and controls
- Engine inspection and tests
- Design status and roadmap
- Design status and roadmap
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests

## God Nodes (most connected - your core abstractions)
1. `reset()` - 146 edges
2. `PlayerId` - 133 edges
3. `step_tick()` - 130 edges
4. `[External module] pytest (third-party)` - 128 edges
5. `PlayerState` - 87 edges
6. `assign_possession()` - 82 edges
7. `BallState` - 76 edges
8. `MovementAction` - 75 edges
9. `step_decision()` - 74 edges
10. `Direction` - 72 edges

## Surprising Connections (you probably didn't know these)
- `[Planned] Held-out-seed multi-match validation` --semantically_similar_to--> `[Planned] Evaluation metrics and instrumentation`  [INFERRED] [semantically similar]
  docs/project_review.md → AGENTS.md
- `[Source] AGENTS.md` --references--> `[Source] game/ball_contacts.py`  [EXTRACTED]
  AGENTS.md → game/ball_contacts.py
- `[Source] AGENTS.md` --references--> `[Source] game/ball.py`  [EXTRACTED]
  AGENTS.md → game/ball.py
- `[Source] AGENTS.md` --references--> `[Source] game/possession.py`  [EXTRACTED]
  AGENTS.md → game/possession.py
- `[Source] AGENTS.md` --references--> `[Source] game/stamina.py`  [EXTRACTED]
  AGENTS.md → game/stamina.py

## Import Cycles
- Verified against directed local Python AST imports: none found.
- Graphify's undirected relationship view produced apparent cycles; those are not treated as confirmed Python import cycles. See graph_health.json for the verified import edges.


## Hyperedges (group relationships)
- **Pitch entities (pitch, goals, players, ball)** — graphify_out_source_previews_image1_pitch, graphify_out_source_previews_image1_left_goal_area, graphify_out_source_previews_image1_right_goal_area, graphify_out_source_previews_image1_player_a, graphify_out_source_previews_image1_player_b, graphify_out_source_previews_image1_ball [EXTRACTED 1.00]
- **All eight proposed schedule phases overlap consecutively** — graphify_out_source_previews_image3_p1_design_interfaces, graphify_out_source_previews_image3_p2_core_simulation, graphify_out_source_previews_image3_p3_rules_rendering, graphify_out_source_previews_image3_p4_gymnasium_observations_rewards, graphify_out_source_previews_image3_p5_rule_based_baseline_tests, graphify_out_source_previews_image3_p6_ppo_training_custom_network, graphify_out_source_previews_image3_p7_evaluation_diagnostics, graphify_out_source_previews_image3_p8_integration_report_demo [EXTRACTED 1.00]
- **Implemented Member A engine foundation** — agents_fixed_timestep, agents_acceleration_movement, agents_stamina_implementation, docs_engine_free_ball_integration, agents_side_swapping, agents_match_clock [EXTRACTED 1.00]
- **Pending match-rule transition chain** — agents_goal_detection, agents_out_of_bounds_last_touch, agents_steal_contest, agents_restart_scaffold [INFERRED 0.75]
- **Baseline comparison evaluation** — graphify_out_source_previews_original_project_plan_evaluation_protocol, graphify_out_source_previews_original_project_plan_held_out_seeds, graphify_out_source_previews_original_project_plan_rule_based_opponent, graphify_out_source_previews_original_project_plan_reward_hacking [INFERRED 0.85]
- **Goal restart reward chain** — graphify_out_source_previews_original_project_plan_match_duration, graphify_out_source_previews_original_project_plan_goal_restart, graphify_out_source_previews_original_project_plan_one_shot_goal_event, graphify_out_source_previews_original_project_plan_reward_weights [INFERRED 0.85]
- **Hybrid action contract and its open encoding** — agents_hybrid_action, agents_superseded_five_continuous, agents_kick_request_semantics, agents_sb3_action_encoding [INFERRED 0.85]
- **PPO training reward loop** — graphify_out_source_previews_original_project_plan_ppo_baseline, graphify_out_source_previews_original_project_plan_continuous_five_action, graphify_out_source_previews_original_project_plan_reward_weights, graphify_out_source_previews_original_project_plan_training_diagnostics [INFERRED 0.85]
- **Reward shaping evaluation** — graphify_out_source_previews_original_project_plan_reward_weights, graphify_out_source_previews_original_project_plan_potential_shaping, graphify_out_source_previews_original_project_plan_evaluation_protocol [INFERRED 0.85]
- **Shared core match loop** — graphify_out_source_previews_original_project_plan_shared_simulation_core, graphify_out_source_previews_original_project_plan_physics_update, graphify_out_source_previews_original_project_plan_policy_decision, graphify_out_source_previews_original_project_plan_match_duration [INFERRED 0.85]

## Communities (27 total, 9 thin omitted)

### Community 0 - "Stamina and fatigue"
Cohesion: 0.06
Nodes (90): Direction, MovementAction, MovementConfig, StaminaConfig, PlayerState, step_idle_player(), step_player(), performance_factors() (+82 more)

### Community 1 - "Automatic ball recovery"
Cohesion: 0.09
Nodes (98): PlayerAction, KickConfig, PlayerId, assign_possession(), release_kick(), reset(), step_decision(), step_tick() (+90 more)

### Community 2 - "Shared simulation and rules"
Cohesion: 0.08
Nodes (51): [External module] dataclasses (Python stdlib), [External module] enum (Python stdlib), _normal(), _project_direction(), resolve_player_contact(), resolve_players(), PitchConfig, SimulationConfig (+43 more)

### Community 3 - "Original team plan"
Cohesion: 0.06
Nodes (65): [Original diagram] A attacks to the right (toward x=800), [Original diagram] B attacks to the left (toward x=0), [Original diagram] Axes x 0-800 right, y 0-500 up (y-up diagram), [Original diagram] Ball spawn (400, 250) with center circle, [Original diagram] Center circle around ball at pitch center, [Original diagram] Goal opening ~120 units (approximate, from title), [Original diagram] Left goal area (orange) at x=0, approx y 160-340, [Original diagram] Proposed logical pitch 800 x 500 units (+57 more)

### Community 4 - "Ball rebounds and constraints"
Cohesion: 0.09
Nodes (35): advance_free_ball(), BallConstraint, BallFlight, BallImpact, first_post_contact(), _lerp(), _Solid, _player_posts() (+27 more)

### Community 5 - "Free-ball dynamics"
Cohesion: 0.13
Nodes (39): BallState, integrate_free_ball(), step_free_ball(), BallConfig, [Source] game/ball.py, [Source] tests/test_ball_contacts.py, [Source] tests/test_ball.py, flight() (+31 more)

### Community 6 - "Engine inspection and tests"
Cohesion: 0.14
Nodes (31): [External module] argparse (Python stdlib), [External module] csv (Python stdlib), inspect_ball(), inspect_contacts(), inspect_dribbling(), write_snapshot(), inspect_kick(), inspect_kick_actions() (+23 more)

### Community 7 - "Goals and restarts"
Cohesion: 0.14
Nodes (28): RulesConfig, BoundarySide, detect_ball_exit(), RestartPlan, MatchPhase, [Source] game/rules.py, [Source] tests/test_rules.py, shot() (+20 more)

### Community 8 - "Engine inspection and tests"
Cohesion: 0.10
Nodes (18): PossessionConfig, [External module] itertools (Python stdlib), [Source] game/config.py, [Source] tests/test_stealing.py, contest(), test_custom_contest_and_protection_settings(), test_free_ball_cannot_have_contest_or_protection_and_reset_clears_them(), test_full_contact_match_replays_with_one_event_per_spaced_steal() (+10 more)

### Community 9 - "Design status and roadmap"
Cohesion: 0.14
Nodes (25): [Approved and implemented] Clock and timers run during restarts, [Implemented] 60 Hz physics and 10 Hz held decisions, [Approved and implemented] Strict whole-ball goals and scoring, [Planned] Headless and graphical modes, [Approved starting] Idle braking 40 units/s², [Implemented] Headless simulation core with contact and match rules, [Implemented] 60-second match clock (3,600 ticks), [Implemented] One-shot transitions and final whistle (+17 more)

### Community 10 - "Design status and roadmap"
Cohesion: 0.11
Nodes (20): [Planned] Evaluation metrics and instrumentation, [Approved starting] Hybrid action: 9-direction + effort + binary kick + power, [Approved and implemented] First-tick kick requests and 12-tick cooldown, [Planned] PPO baseline with customized network, [Planned] Rule-based baseline comparison, [Open] Stable-Baselines3 hybrid action encoding, [Verified] 461 engine tests and 36 subtests, [Documented] Source authority and history (+12 more)

### Community 11 - "Design status and roadmap"
Cohesion: 0.20
Nodes (21): [Implemented] Acceleration-based nine-direction movement, [Implemented] Swept pickup ordered against whole-ball exits, [Implemented] Exclusive free or controlled ball update, [Approved and implemented] Locked-player and goalpost ball rebounds, [Implemented] Collision, constraint and saturation telemetry, [Implemented] Facing-based controlled possession and dribbling, [Approved and implemented] Controlled post loss without a kick, [Implemented] Zero free-ball velocity while attached (+13 more)

### Community 12 - "Dependencies and reproducibility"
Cohesion: 0.28
Nodes (20): rl-game, [Declared dependency] graphifyy 0.9.103, [Declared dependency] gymnasium 1.4.0, [Declared dependency] ipykernel 7.4.0, [Declared dependency] jupyter 1.1.1, [Declared dependency] matplotlib 3.11.2, [Declared dependency] numpy 2.5.4, [Declared dependency] pandas 3.1.0 (+12 more)

### Community 13 - "Design status and roadmap"
Cohesion: 0.29
Nodes (10): [Implemented] Facing-based kick with linear power mapping, [Approved starting] Free-ball 300 units/s, 100 units/s², 3 s / 450 units, [Implemented] Controlled/free ball, kick, recovery and stealing, [Approved and implemented] Last-touch boundary restart, [Implemented] One authoritative owner and separate last toucher, [Approved and implemented] 35-unit, 18-tick steal contest, [Implemented] Analytic free-ball roll plus shared swept rules, [Implemented] Swept sliding player contact and corner correction (+2 more)

### Community 14 - "CI and validation"
Cohesion: 0.38
Nodes (9): jobs, test, name, on, run, uses, runs-on, steps (+1 more)

### Community 16 - "Engine inspection and tests"
Cohesion: 0.48
Nodes (4): [External module] pathlib (Python stdlib), [Source] tests/test_smoke.py, test_game_does_not_import_pygame(), test_imports()

### Community 17 - "Design status and roadmap"
Cohesion: 0.40
Nodes (5): [Proposed] Flat normalized observation vector, [Implemented core] Explicit state, scores, possession timers and restart plan, [Documented] Five-continuous-value action superseded, [Approved starting] x-right/y-down coordinates, [Documented] Plan y-up and x/y-action superseded

### Community 18 - "Design status and roadmap"
Cohesion: 0.50
Nodes (5): [Implemented] Locked engine dependency group, [Implemented] full_stack test excluded from engine runs, [Implemented] Python 3.12.14 pin, [Documented] Deterministic tests do not prove tuning or integration, [Implemented] uv locked setup

## Knowledge Gaps
- **19 isolated node(s):** `[Verified] 461 engine tests and 36 subtests`, `[Planned] Feature Freeze Nineteen December`, `[Original plan] Free Ball Physics`, `[Proposed] Goal Opening One Hundred Twenty Units`, `[Original plan] Low Stamina Speed Reduction` (+14 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 32 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `[Source] AGENTS.md` connect `Design status and roadmap` to `Stamina and fatigue`, `Original team plan`, `Ball rebounds and constraints`, `Free-ball dynamics`, `Design status and roadmap`, `Design status and roadmap`, `Dependencies and reproducibility`, `Design status and roadmap`, `Design status and roadmap`, `Design status and roadmap`?**
  _High betweenness centrality (0.269) - this node is a cross-community bridge._
- **Are the 114 inferred relationships involving `PlayerId` (e.g. with `BallState` and `advance_free_ball()`) actually correct?**
  _`PlayerId` has 114 INFERRED edges - model-reasoned connections that need verification._
- **What connects `[Verified] 461 engine tests and 36 subtests`, `[Planned] Feature Freeze Nineteen December`, `[Original plan] Free Ball Physics` to the rest of the system?**
  _19 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Stamina and fatigue` be split into smaller, more focused modules?**
  _Cohesion score 0.0626031746031746 - nodes in this community are weakly interconnected._
- **Why does `[Source] docs/engine.md` connect `Design status and roadmap` to `Stamina and fatigue`, `Shared simulation and rules`, `Ball rebounds and constraints`, `Free-ball dynamics`, `Goals and restarts`, `Engine inspection and tests`, `Design status and roadmap`, `Design status and roadmap`, `Design status and roadmap`, `Player movement and controls`?**
  _High betweenness centrality (0.194) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `step_tick()` (e.g. with `MovementAction` and `PlayerAction`) actually correct?**
  _`step_tick()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Should `Automatic ball recovery` be split into smaller, more focused modules?**
  _Cohesion score 0.09222359372244754 - nodes in this community are weakly interconnected._