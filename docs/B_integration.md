# Member B integration (v2 of main): B's code on top of A's engine and C's contract

Nothing in `game/`, `environment/`, `docs/environment.md` or any other team file was changed.
Two bridges, both pure translation:

* `agents/engine_adapter.py` : **60 Hz** (one physics tick per step) for the renderer, `play.py`, `observe.py`.
* `agents/provider.py` : the baseline as a native `game.runner.ActionProvider`
  `(SimulationState, PlayerId) -> PlayerAction`, i.e. what `FootballEnv(opponent=...)` expects.

C's `environment/engine.py::SimulationEngine` stays the 10 Hz training adapter (it cannot be rendered per
tick). `tests/test_provider.py` proves the three paths agree: provider run == renderer run (same final
engine state), and B's event classification == C's `SimulationEngine` events (also accepted by C's
`validate_step_result`).

## What was added

| Path | Role |
|---|---|
| `agents/engine_adapter.py` | `EngineAdapter` (reset / step, one physics tick per step), `to_view`, `to_player_action`, `convert_events` |
| `agents/interface.py` | B-side **read-only view** of the state (`GameState`, `Action`, `Event`); constants are read from `game/config.py` |
| `agents/rule_based.py`, `difficulty.py`, `config.py`, `observation_view.py`, `base.py`, `match.py`, `metrics.py`, `baseline_stats.py` | rule-based baseline (3 presets), 10 Hz wrapper, metrics, CSV tool |
| `rendering/*.py` | Pygame renderer, HUD, screens, effects, controls, replay, `session.py` (human vs bot, no pygame) |
| `play.py`, `observe.py` | the two scaffold entry points, now implemented |
| `tests/test_*.py` (16 new files) | B tests, all run on A's real engine |
| `docs/B_*.md` | B specs/report (P1-P3) + this note |

## Vocabulary map (A -> B view)

| A (`game/`) | C (`environment/contract.py`) | B view (`agents/interface.py`) |
|---|---|---|
| `player_a`, `player_b` (`PlayerId.A/B`) | `players[0]`, `players[1]` | same |
| `position`, `velocity` (units/s), `facing`, `stamina` | `x, y, vx, vy, facing, stamina` | same |
| `ball.possession_owner` | `possession` | same |
| `score_a`, `score_b` | `score` | same |
| `simulation_tick` | `tick` | same |
| `attack_direction_a` / `_b` | `attack_dirs` | same |
| `MatchPhase.PLAYING / GOAL_RESTART / OUT_OF_BOUNDS_RESTART / FINISHED` | `phase` (`Phase` = `MatchPhase`) | `"playing" / "restart" / "finished"` (the two restarts are merged) |
| `steal_progress_ticks` | `steal_progress_ticks` | same (`contest_active` = derived property) |
| `restart_ticks_remaining`, `restart_plan.owner` | same two names | same (`restart_timer` in seconds = derived property) |
| `ball.last_touch_player`, `possession_protection_ticks` | `last_toucher`, `possession_protection_ticks` | same |
| events `goal`, `out_of_bounds`, `ball_kicked`, `possession_changed` (cause `steal` / `recovery`) | `EventType.GOAL / OUT_OF_BOUNDS / KICK / STEAL / LOOSE_BALL_RECOVERY` | `Event(kind, player, tick, other, event_id)`; `kind` strings = the `EventType` values |
| `PlayerAction(direction, effort, kick_trigger, kick_power)` | world-space `PlayerAction` (providers) | `Action(move_x, move_y, effort, kick, power)` inside B only; converted by `to_player_action` |

## Decisions B made to fit A's contract (to validate)

1. **Directions**: B's continuous move vector is quantised to the 8 compass directions (idle under 0.25).
2. **Effort**: A's `effort` is the speed command (0 = brake). B's `effort` 0..1 is mapped to A's 0.5..1
   (0 = jog at the stamina-neutral 0.5, 1 = sprint). Idle = A's idle.
3. **Kicks**: consumed on the first tick of a 6-tick decision (A's rule). A held trigger is NOT repeated (rising
   edge only). A human press in the middle of a decision is latched to the next boundary (<= 0.08 s).
4. **Restarts**: the baseline is queried during restarts like any decision, and returns idle (A's runner contract).
5. **Opponent provider**: the baseline's sprint/jog mapping (decision 2) stays inside B; the provider returns plain
   `PlayerAction` values, so C never sees B's `Action`.
6. **Constants** (pitch, goal width, radii, posts, Hz) are read from `game/config.py` / `game/rules.py`.
7. **Seeds**: A's `reset` has no randomness, so `seed` is accepted and ignored.

## How to run

```
uv sync --locked
uv run pytest -m "not full_stack" -q
uv run python play.py                      # you (blue) vs the baseline
uv run python observe.py --headless --episodes 4 --swap
uv run python -m agents.baseline_stats --seeds 5 --out results/baseline.csv
```

## Plugging the baseline into C's environment

```python
from agents.provider import rule_based_provider
env = FootballEnv(opponent=rule_based_provider("normal"))   # also: "easy", "hard"
```
Build a fresh provider (or call `.reset()`) before every replay: it keeps private sprint state.
