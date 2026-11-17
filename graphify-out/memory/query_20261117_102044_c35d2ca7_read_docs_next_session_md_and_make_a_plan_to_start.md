---
type: "query"
date: "2026-11-17T10:20:44.176970+00:00"
question: "read docs/next_session.md and make a plan to start building the full game match runner"
contributor: "graphify"
outcome: "useful"
source_nodes: ["step_decision()", "test_clock_ends_after_exactly_600_decision_intervals()"]
---

# Q: read docs/next_session.md and make a plan to start building the full game match runner

## Answer

Expanded via graph vocabulary: runner decision restart events finished deterministic diagnostics. Graph traversal identified step_decision and existing restart/terminal tests; verified current docs and source directly. Proposed plan, not implemented: add game/runner.py with two action providers reading the same immutable state, optional supplied initial state and matching configuration, delegate all physics to step_decision, preserve every returned event in engine order, return final state and actual advanced tick/decision counts. Agree restart querying explicitly; recommend querying every decision boundary including restarts for the first headless runner without finalizing the RL adapter contract. Add deterministic idle/scripted match tests, partial terminal and restart boundary tests, then an opt-in match CLI with diagnostics. Keep rendering/rewards/PPO/baseline and teammate entry points out of scope.

## Outcome

- Signal: useful

## Source Nodes

- step_decision()
- test_clock_ends_after_exactly_600_decision_intervals()