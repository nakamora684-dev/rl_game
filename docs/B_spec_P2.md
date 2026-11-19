> **Integration note (18 Nov):** this document was written against the old mock engine. The mock is gone: `agents/engine_adapter.py` now runs Member A's real engine. Read `docs/B_integration.md` first; mock-specific statements below are historical.

# Member B - P2 spec: interface freeze, fair baseline, integration

Status: DRAFT to validate with A (engine) and C (observation). Version: interface v2.

## 1. Goal of P2
Run rendering and the rule-based agent on the REAL engine, with no change to the
simulation, and prove that graphical and headless modes are the same simulation.

## 2. Interface v1 (game/interface.py)
Added to `GameState`, all with defaults (old code keeps working):
| Field | Meaning |
|---|---|
| `events: list[Event]` | One-shot events of THIS tick only: goal, out_of_bounds, steal, recover, kick. Never repeated on later ticks. |
| `restart_timer: float` | Seconds left in a restart phase. |
| `restart_owner: int \| None` | Player who gets the ball after the restart. |
`INTERFACE_VERSION = 2`: bump it whenever a field changes.
v2 adds `reset(seed, swap_sides=False)` to the engine protocol. `swap_sides=True` mirrors the
start so player 0 attacks toward -x (needed to evaluate both directions, plan section 7).

Rules for A: emit each event once; `possession` is the single source of truth.
Rules for C: reward events are read from `events`, never inferred from state diffs.

## 3. Decisions proposed for the P1 open questions
| Question | Proposal (to confirm) |
|---|---|
| Does kick power depend on stamina? | No. Plan: kicking costs no stamina, no effect on accuracy. |
| Who owns `observe.py`? | B: rendering and replay loop. D: loading a PPO checkpoint. |
| Policy during restarts? | Paused: the baseline returns an idle action in `restart`. C must match this. |

## 4. Fair baseline
- `agents/observation_view.py`: attack-normalized `View` (always attacks toward +x,
  only x mirrored). No events, no timers, no last-touch: nothing privileged.
- `rule_based.py` decides on the `View` only, then un-mirrors x. Symmetric by construction.
- Stamina uses hysteresis (stop under 30%, resume over 50%) to avoid flicker.
- C should build the PPO observation from the same `View` fields so both agents see the same info.

## 5. Tests added (tests/test_baseline_fairness.py)
View normalization, possession categories, mirror symmetry, idle on restart,
stamina hysteresis, no hidden fields, contract defaults.

## 6. Rendering, loop and tools (batch 2)
- `rendering/hud.py`: scoreboard, clock, stamina bars, kick-power bar, restart banner,
  full-time banner, and `FlashTracker` (goal/out/steal flashes from one-shot events).
- `rendering/debug_overlay.py` (F1): possession ring, steal contest, velocity vectors, tick/phase/events.
  `STEAL_RANGE` is display-only: A must confirm the real value.
- `agents/base.py::HeldAgent`: baseline decides at 10 Hz (1 decision held for 6 ticks), like the RL agent.
  Note for A: a held `kick=1` lasts 6 ticks, so the engine must only kick when the player owns the ball.
- `agents/match.py::run_match`: ONE match loop for headless and graphical. The renderer is only an `on_step` hook.
- `play.py` (`--seed`, `--scale`), `observe.py` (`--headless`, `--episodes`, `--swap`).
  `load_agent("ppo")` is a stub for Member D.
- The renderer reads `events` once per `render()` call: call it every physics tick (as play.py does)
  or flashes can be missed.

## 7. Tests
27 tests, no display needed. `test_render_readonly.py` uses a fake pygame.
`test_interface_contract.py` and `test_mode_equivalence.py` run on the mock now and
automatically also on `game.engine.Game` once A delivers it.

## 8. Not verified here
The real pygame window was NOT opened (pygame was not installable in this session).
Run `uv sync && uv run python play.py` once and check the visuals yourself.
