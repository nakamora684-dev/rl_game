"""Per-match metrics for the plan's evaluation protocol (section 7). NO pygame.

Used now by tools/baseline_stats.py and the tests; Member D can reuse it in test.py
so the CSV schema is the same for baseline-vs-baseline and PPO-vs-baseline.

Everything is computed from the public GameState (+ the one-shot `events`):
  goals/steals/recoveries/out_of_bounds/kicks : counted from events (once each)
  shots           : a kick whose facing line crosses the opponent goal mouth
  possession_share: share of PLAYING ticks where the player owns the ball
  attack_half     : share of playing ticks with the ball in the opponent half
  idle_frac       : share of playing ticks where the player moved < IDLE_DIST
  sprint_frac     : share of playing ticks with effort >= 0.5
  stamina_mean/end: stamina in [0,1]
"""

from __future__ import annotations

import math

from agents.interface import (
    EV_GOAL,
    EV_KICK,
    EV_OUT,
    EV_RECOVER,
    EV_STEAL,
    PHASE_FINISHED,
    PHASE_PLAYING,
    PITCH_W,
    STAMINA_MAX,
    GameState,
    goal_mouth,
)

IDLE_DIST = 0.05  # logical units moved in one tick
SPRINT_EFFORT = 0.5
PLAYER_FIELDS = (
    "goals",
    "shots",
    "kicks",
    "steals",
    "recoveries",
    "out_of_bounds",
    "possession_share",
    "attack_half",
    "idle_frac",
    "sprint_frac",
    "stamina_mean",
    "stamina_end",
)
MATCH_FIELDS = (
    "seed",
    "swapped",
    "p0_agent",
    "p1_agent",
    "ticks",
    "end_reason",
    "score_p0",
    "score_p1",
    "goal_diff_p0",
    "winner",
    "duplicate_events",
)
ROW_FIELDS = MATCH_FIELDS + tuple(f"p{i}_{f}" for i in (0, 1) for f in PLAYER_FIELDS)


def is_shot(state: GameState, player: int) -> bool:
    """Kick whose facing line reaches the opponent goal line inside the mouth."""
    p = state.players[player]
    attacks = state.attack_dirs[player]
    goal_x = PITCH_W if attacks > 0 else 0.0
    dx = goal_x - p.x
    cos, sin = math.cos(p.facing), math.sin(p.facing)
    if dx * cos <= 0.0 or abs(cos) < 1e-9:
        return False  # facing away from the goal line
    y_hit = p.y + dx * sin / cos
    _, y_top, y_bot = goal_mouth(1 if attacks > 0 else 0)
    return y_top <= y_hit <= y_bot


class _Recording:
    """Wraps an agent and records the effort of every tick in the playing phase."""

    def __init__(self, agent, rec: MatchRecorder, index: int) -> None:
        self.agent, self.rec, self.index = agent, rec, index

    def reset(self) -> None:
        if hasattr(self.agent, "reset"):
            self.agent.reset()

    def act(self, state: GameState):
        a = self.agent.act(state)
        if state.phase == PHASE_PLAYING:
            self.rec.sprint[self.index] += 1 if a.effort >= SPRINT_EFFORT else 0
        return a


class MatchRecorder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        z = lambda: [0, 0]
        self.playing = 0
        self.goals, self.shots, self.kicks, self.steals = z(), z(), z(), z()
        self.recoveries, self.oob, self.poss, self.attack = z(), z(), z(), z()
        self.idle, self.sprint = z(), z()
        self.stamina_sum = [0.0, 0.0]
        self.stamina_end = [1.0, 1.0]
        self.duplicates = 0
        self._seen: set[tuple] = set()
        self._prev: list[tuple[float, float]] | None = None

    def wrap(self, agent, index: int) -> _Recording:
        return _Recording(agent, self, index)

    def on_step(self, st: GameState) -> None:
        for e in st.events:
            key = (e.kind, e.player, e.tick)
            if key in self._seen:
                self.duplicates += 1
                continue
            self._seen.add(key)
            if e.player is None:
                continue
            if e.kind == EV_GOAL:
                self.goals[e.player] += 1
            elif e.kind == EV_KICK:
                self.kicks[e.player] += 1
                self.shots[e.player] += 1 if is_shot(st, e.player) else 0
            elif e.kind == EV_STEAL:
                self.steals[e.player] += 1
            elif e.kind == EV_RECOVER:
                self.recoveries[e.player] += 1
            elif e.kind == EV_OUT:
                self.oob[e.player] += 1
        pos = [(p.x, p.y) for p in st.players]
        if st.phase == PHASE_PLAYING:
            self.playing += 1
            for i in (0, 1):
                self.poss[i] += 1 if st.possession == i else 0
                bx_right = st.ball.x > PITCH_W / 2
                self.attack[i] += 1 if bx_right == (st.attack_dirs[i] > 0) else 0
                self.stamina_sum[i] += st.players[i].stamina / STAMINA_MAX
                if self._prev is not None:
                    moved = math.hypot(pos[i][0] - self._prev[i][0], pos[i][1] - self._prev[i][1])
                    self.idle[i] += 1 if moved < IDLE_DIST else 0
        for i in (0, 1):
            self.stamina_end[i] = st.players[i].stamina / STAMINA_MAX
        self._prev = pos

    def summary(
        self,
        final: GameState,
        seed: int = 0,
        swapped: bool = False,
        names: tuple[str, str] = ("p0", "p1"),
    ) -> dict:
        n = max(1, self.playing)
        gd = final.score[0] - final.score[1]
        row = dict(
            seed=seed,
            swapped=int(swapped),
            p0_agent=names[0],
            p1_agent=names[1],
            ticks=final.tick,
            end_reason="time" if final.phase == PHASE_FINISHED else "truncated",
            score_p0=final.score[0],
            score_p1=final.score[1],
            goal_diff_p0=gd,
            winner="p0" if gd > 0 else "p1" if gd < 0 else "draw",
            duplicate_events=self.duplicates,
        )
        for i in (0, 1):
            row.update(
                {
                    f"p{i}_goals": self.goals[i],
                    f"p{i}_shots": self.shots[i],
                    f"p{i}_kicks": self.kicks[i],
                    f"p{i}_steals": self.steals[i],
                    f"p{i}_recoveries": self.recoveries[i],
                    f"p{i}_out_of_bounds": self.oob[i],
                    f"p{i}_possession_share": self.poss[i] / n,
                    f"p{i}_attack_half": self.attack[i] / n,
                    f"p{i}_idle_frac": self.idle[i] / n,
                    f"p{i}_sprint_frac": self.sprint[i] / n,
                    f"p{i}_stamina_mean": self.stamina_sum[i] / n,
                    f"p{i}_stamina_end": self.stamina_end[i],
                }
            )
        return row
