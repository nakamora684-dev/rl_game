# Review guide and coverage

Start with `docs/project_review.md`. Implemented, approved starting, proposed, and planned concepts are labelled separately. The original plan is historical and does not override newer decisions.

Coverage: local AST for 24 detected code/config files; all four current Markdown sources; original Word text/tables and all three embedded diagrams; repository file/config/dependency inventory. Word conversion used standard-library ZIP/XML because the optional Office converter was absent. Empty entry-point scripts are explicitly scaffolds.

The graph is an evidence index, not a completeness proof. Automated checks executed during this build: 194 engine tests passed; full-stack dependency test intentionally excluded; Ruff lint and formatting passed. See graph_health.json for extraction diagnostics and graph_build.json for provenance.

**Graph health warning:** undirected_same_endpoint_collapsed_edges=123. Parallel edges are coalesced visually, but all relation/source evidence is retained on graph.json edges; raw extraction is also retained locally for audit.

# Graph Report - rl_game  (2026-11-07)

## Corpus Check
- Corpus is ~13,463 words - fits in a single context window. You may not need a graph.

## Summary
- 409 nodes · 1310 edges · 19 communities (11 shown, 8 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 107 edges (avg confidence: 0.94)
- Token usage: AST 0 LLM tokens; semantic agent token counts were not exposed by the harness (unknown, not zero). Used free model routes; no additional API key configured.

## Community Hubs (Navigation)
- Original team plan
- Design status and roadmap
- Player movement and controls
- Free-ball dynamics
- Simulation clock and reset
- Stamina and fatigue
- Player movement and controls
- Dependencies and reproducibility
- Engine inspection and tests
- CI and validation
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests
- Engine inspection and tests

## God Nodes (most connected - your core abstractions)
1. `[Source] football_drl_final_project_plan.docx` - 67 edges
2. `[Source] AGENTS.md` - 60 edges
3. `[External module] pytest (third-party)` - 59 edges
4. `PlayerState` - 56 edges
5. `MovementAction` - 55 edges
6. `Direction` - 53 edges
7. `reset()` - 40 edges
8. `step_decision()` - 32 edges
9. `[Source] tests/test_stamina.py` - 31 edges
10. `step_player()` - 28 edges

## Surprising Connections (you probably didn't know these)
- `[Planned] Held-out-seed multi-match validation` --semantically_similar_to--> `[Planned] Evaluation metrics and instrumentation`  [INFERRED] [semantically similar]
  docs/project_review.md → AGENTS.md
- `[Source] AGENTS.md` --references--> `[Source] game/ball.py`  [EXTRACTED]
  AGENTS.md → game/ball.py
- `[Source] AGENTS.md` --references--> `[Source] game/stamina.py`  [EXTRACTED]
  AGENTS.md → game/stamina.py
- `[Source] docs/engine.md` --references--> `[Source] game/actions.py`  [EXTRACTED]
  docs/engine.md → game/actions.py
- `[Source] docs/engine.md` --references--> `[Source] game/ball.py`  [EXTRACTED]
  docs/engine.md → game/ball.py

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

## Communities (19 total, 8 thin omitted)

### Community 0 - "Original team plan"
Cohesion: 0.05
Nodes (65): [Original diagram] A attacks to the right (toward x=800), [Original diagram] B attacks to the left (toward x=0), [Original diagram] Axes x 0-800 right, y 0-500 up (y-up diagram), [Original diagram] Ball spawn (400, 250) with center circle, [Original diagram] Center circle around ball at pitch center, [Original diagram] Goal opening ~120 units (approximate, from title), [Original diagram] Left goal area (orange) at x=0, approx y 160-340, [Original diagram] Proposed logical pitch 800 x 500 units (+57 more)

### Community 1 - "Design status and roadmap"
Cohesion: 0.07
Nodes (58): [Implemented] Acceleration-based nine-direction movement, [Approved starting] Match clock runs through restarts, [Implemented] Locked engine dependency group, [Planned] Evaluation metrics and instrumentation, [Planned] Facing-based kick with power mapping, [Implemented] 60 Hz physics and 10 Hz held decisions, [Approved starting] Free-ball 300 units/s, 100 units/s², 3 s / 450 units, [Implemented] full_stack test excluded from engine runs (+50 more)

### Community 2 - "Player movement and controls"
Cohesion: 0.15
Nodes (34): [External module] enum (Python stdlib), Direction, MovementAction, PitchConfig, [External module] math (Python stdlib), [Source] game/actions.py, [Source] tests/test_player_controls.py, step_player() (+26 more)

### Community 3 - "Free-ball dynamics"
Cohesion: 0.13
Nodes (27): [External module] dataclasses (Python stdlib), BallState, step_free_ball(), BallConfig, MovementConfig, [Source] game/ball.py, [Source] game/config.py, [Source] tests/test_ball.py (+19 more)

### Community 4 - "Simulation clock and reset"
Cohesion: 0.16
Nodes (28): SimulationConfig, MatchPhase, reset(), SimulationEvent, SimulationState, step_decision(), step_tick(), StepResult (+20 more)

### Community 5 - "Stamina and fatigue"
Cohesion: 0.17
Nodes (33): StaminaConfig, step_player(), performance_factors(), update_stamina(), _validate_stamina(), [External module] pytest (third-party), [Source] game/stamina.py, [Source] tests/test_stamina.py (+25 more)

### Community 6 - "Player movement and controls"
Cohesion: 0.17
Nodes (9): constrain_to_pitch(), PlayerState, step_idle_player(), [Source] game/player.py, [Source] tests/test_player_motion.py, DirectionTests, IdleBrakingTests, trajectory() (+1 more)

### Community 7 - "Dependencies and reproducibility"
Cohesion: 0.14
Nodes (27): rl-game, [Declared dependency] graphifyy 0.9.103, [Declared dependency] gymnasium 1.4.0, [Declared dependency] ipykernel 7.4.0, [Declared dependency] jupyter 1.1.1, [Declared dependency] matplotlib 3.11.2, [Declared dependency] numpy 2.5.4, [Declared dependency] pandas 3.1.0 (+19 more)

### Community 8 - "Engine inspection and tests"
Cohesion: 0.25
Nodes (14): [External module] argparse (Python stdlib), [External module] csv (Python stdlib), inspect_ball(), inspect_stamina(), main(), [External module] io (Python stdlib), [External module] json (Python stdlib), [Source] game/__main__.py (+6 more)

### Community 9 - "CI and validation"
Cohesion: 0.38
Nodes (9): jobs, test, name, on, run, uses, runs-on, steps (+1 more)

### Community 10 - "Engine inspection and tests"
Cohesion: 0.48
Nodes (4): [External module] pathlib (Python stdlib), [Source] tests/test_smoke.py, test_game_does_not_import_pygame(), test_imports()

## Knowledge Gaps
- **28 isolated node(s):** `[Requirement] Assignment Required Outputs`, `[Planned] Stable-Baselines3 Action Encoding`, `[Original plan] Shared Simulation Core`, `[Proposed] Pitch Proposal Eight Hundred By Five Hundred`, `[Proposed] Goal Opening One Hundred Twenty Units` (+23 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 39 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `[Source] AGENTS.md` connect `Design status and roadmap` to `Original team plan`, `Free-ball dynamics`, `Stamina and fatigue`, `Dependencies and reproducibility`?**
  _High betweenness centrality (0.415) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `PlayerState` (e.g. with `SimulationState` and `test_acceleration_reaches_same_speed_in_every_direction()`) actually correct?**
  _`PlayerState` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `[Requirement] Assignment Required Outputs`, `[Planned] Stable-Baselines3 Action Encoding`, `[Original plan] Shared Simulation Core` to the rest of the system?**
  _28 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Original team plan` be split into smaller, more focused modules?**
  _Cohesion score 0.05081585081585081 - nodes in this community are weakly interconnected._
- **Why does `[Source] football_drl_final_project_plan.docx` connect `Original team plan` to `Design status and roadmap`?**
  _High betweenness centrality (0.275) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `MovementAction` (e.g. with `step_player()` and `step_decision()`) actually correct?**
  _`MovementAction` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Should `Design status and roadmap` be split into smaller, more focused modules?**
  _Cohesion score 0.06646825396825397 - nodes in this community are weakly interconnected._