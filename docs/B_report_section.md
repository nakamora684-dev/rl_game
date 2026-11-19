# Report section - Member B: visual mode, human play and rule-based baseline

(Draft text for the PDF report. Items in [brackets] are filled after P5/P7 with real-engine results.)

## 1. Visual mode
The graphical mode is a Pygame view of the same simulation used for headless training. The renderer
receives a `GameState` and draws it; it contains no gameplay rules, so the game cannot behave
differently on screen than in training. The pitch (stripes, markings, boxes, nets) is drawn once and
reused every frame. Cosmetic effects (ball trail, goal confetti, screen shake) are driven by the
one-shot events of the engine and never change the state. The HUD shows the score, the clock and the
stamina of both players. Visual inspiration: [cite the supplied video, https://www.youtube.com/watch?v=5pLDyW1sCQQ].

## 2. Human play
The human controls the blue player with the same five controls as the RL agent: movement (2 axes),
effort, kick and kick power. Kick power is charged by holding the key and released to kick. A menu
selects difficulty and starting side, pause is available at any time, and a full-time panel shows
shots, steals, recoveries, possession, sprint time and remaining stamina. Matches can be replayed
exactly from a saved action log, which also helps to inspect agent behaviour.

## 3. Stamina in the interface (original contribution, visual part)
Stamina is shown by a ring around each player (green, amber, red), by a bar in the HUD that flashes
below 25%, and by a darker body colour when a player is tired. This makes the mechanic visible to a
human player and to anyone watching a trained agent manage its energy. The rule is the same for the
human, the baseline and the RL agent: sprinting consumes stamina and low stamina reduces speed.

## 4. Rule-based baseline
Design goals: transparent (a short rule list), fair (same mechanics and same information as the RL
agent) and deterministic (reproducible evaluation).

| Priority | Situation | Behaviour |
|---|---|---|
| 0 | Restart phase | Idle (decisions are paused) |
| 1 | Baseline has the ball | Shoot if aligned with the aim corner and the shot would enter the goal; otherwise dribble toward the corner far from the opponent, avoiding the opponent and the touchlines |
| 2 | Opponent has the ball | Contest the steal when close; otherwise stand between the carrier and the own goal |
| 3 | Ball is free | Run to the predicted ball position (bounded anticipation) |
| 4 | Stamina | Sprint only when the run is long enough; stop below 30%, resume above 50%; spend the reserve in the last 10 s when not winning |

Fairness: the baseline reads only an attack-normalized view (positions, velocities, stamina, possession,
score, time left). It has no access to events, timers or the opponent's intentions, and it acts at 10 Hz
like the RL agent. Mirror symmetry is tested on random states for all difficulty levels. Three presets
(easy, normal, hard) change thresholds only. The evaluation baseline is `normal`.

## 5. Verification
108 automated tests run without a display: symmetry and legality of baseline actions, rule-level
behaviour, full matches in both directions, event counting, metrics and CSV schema, replay exactness,
flow of menu/pause, and a read-only check of the renderer. The same contract and mode-equivalence tests
run on the real engine. [Baseline-vs-baseline statistics on the real engine: n = __ matches, goal
difference __ +/- __ (95% CI), idle fraction __, out-of-bounds per match __.]

## 6. Limitations
The baseline is hand-designed and not optimal: it never passes (1v1), and its shooting accuracy depends
on the real physics. Presets are not yet ranked on the real engine. [Fill after P5.]

## 7. Slide outline (3 slides)
1. Game and visual mode: screenshot, HUD, stamina ring, same simulation for GUI and headless.
2. Baseline: the rule table above, fairness (same View), presets.
3. Verification and diagnostics: test summary, baseline-vs-baseline CSV metrics, limitations.
