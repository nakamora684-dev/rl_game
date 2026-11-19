# Member B - P1 spec: visuals, controls, interfaces

Owner: Member B. Status: DRAFT to validate with A, C, D by 30 Oct.

## 1. What B owns
`play.py`, `rendering/`, `agents/rule_based.py`, baseline description.

## 2. Visual spec
- Top-down 2D, modern flat look (reference video cited in the report if used).
- Pitch 800x500 logical units, scaled only in the renderer. Striped grass, white
  lines, yellow goal mouths (120 wide) centered on the short edges.
- HUD: score (center), time left (center), stamina bar per player (top corners),
  kick-power bar for the human.
- Debug overlay (F1): possession ring, velocity vectors.
- Renderer is READ-ONLY: it never changes the game state or applies rules.

## 3. Human controls
| Action | Keys |
|---|---|
| Move x/y | Arrows, ZQSD or WASD |
| Effort (sprint) | Shift (hold) |
| Kick + power | Space: hold to charge, release to kick (0.8 s = full power) |
| Debug / reset / quit | F1 / R / Esc |

## 4. Needs from Member A (to confirm)
1. `game/engine.py` exposes `Game` with `reset(seed) -> GameState` and
   `step([Action, Action]) -> GameState` (ONE physics update, 1/60 s).
2. `GameState` fields as in `game/interface.py` (positions, facing, stamina,
   possession owner, score, tick, phase, attack_dir, contest_active).
3. Headless and graphical modes call the same `Game`.

## 5. Needs from Member C
- The rule-based agent reads the same information as the RL observation (no
  privileged state). Confirm the observation fields.
- `Action` order: move_x, move_y, effort, kick, power (same as the RL action).

## 6. Rule-based agent: first draft
1. Opponent has ball: contest the steal. 2. Ball free: go to ball.
3. I have the ball: dribble to goal, swerve around the opponent.
4. Close to goal and facing it: kick. 5. Sprint only above a stamina reserve.

## 7. Open questions
- Does kick power also depend on stamina? (plan says kicking costs none)
- Who owns `observe.py`? (not assigned in the plan; B proposes the rendering part)
- Is the policy active during restarts or paused? (plan default: paused)
