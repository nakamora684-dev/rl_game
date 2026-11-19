"""Rule-based opponent (v3).

Fairness rule (plan, section 9.1): same physics/stamina as the RL agent and NO
privileged information. It only reads a `View` (agents/observation_view.py),
which is attack-normalized, so the logic is written ONCE for "attack toward +x"
and is symmetric by construction. The x-axis is un-mirrored on output.

Rules, in priority order (every number lives in agents/config.py):
  0. Not in the playing phase -> idle (decisions are paused on restarts).
  1. I have the ball:
       a. aligned with the aim point and close enough -> KICK (full/close power)
       b. otherwise dribble toward the aim point (corner farther from the
          opponent), swerving away from the opponent / the touchline.
  2. Opponent has the ball:
       a. close (or already goal-side) -> run at the carrier to contest a steal
       b. far -> stand goal-side, `block_dist` in front of the carrier
  3. Ball is free -> run to where the ball will be (bounded anticipation).
  4. Sprint only if stamina allows (hysteresis) AND it is useful.

`last_rule` holds the name of the rule that fired (debug overlay, tests, stats).
Kick direction is the facing of the player: there is no kick-direction action.
"""

from __future__ import annotations

import math

from agents.config import BaselineConfig, to_per_second
from agents.interface import GOAL_W, PITCH_H, PITCH_W, Action, GameState
from agents.observation_view import OPP, SELF, View, _wrap, make_view


def _norm(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    return (0.0, 0.0) if n < 1e-9 else (dx / n, dy / n)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class RuleBasedAgent:
    def __init__(self, index: int, config: BaselineConfig | None = None) -> None:
        self.index = index
        self.cfg = config or BaselineConfig()
        self._tired = False  # stamina hysteresis (avoids flicker)
        self.last_rule = "idle"

    def reset(self) -> None:
        self._tired = False
        self.last_rule = "idle"

    def act(self, state: GameState) -> Action:
        return self.act_view(make_view(state, self.index))

    def act_view(self, v: View) -> Action:
        if not v.playing:  # rule 0
            self.last_rule = "idle"
            return Action()
        a = self._decide(v)
        if v.mirrored:  # back to real x
            a = Action(-a.move_x, a.move_y, a.effort, a.kick, a.power)
        return a.clipped()

    # ------------------------------------------------------------------ stamina
    def _sprint_ok(self, v: View) -> bool:
        c = self.cfg
        late = c.endgame_all_in and v.time_left < c.endgame_secs and v.score_diff <= 0
        reserve = c.endgame_reserve if late else c.sprint_reserve
        resume = c.endgame_resume if late else c.sprint_resume
        if v.me.stamina < reserve:
            self._tired = True
        elif v.me.stamina > resume:
            self._tired = False
        return not self._tired

    # ------------------------------------------------------------------ decision
    def _decide(self, v: View) -> Action:
        sprint_ok = self._sprint_ok(v)
        if v.possession == SELF:
            return self._carry(v, sprint_ok)
        if v.possession == OPP:
            return self._defend(v, sprint_ok)
        return self._chase_free(v, sprint_ok)

    def _run_to(self, v: View, tx: float, ty: float, sprint: bool) -> Action:
        mx, my = _norm(tx - v.me.x, ty - v.me.y)
        return Action(mx, my, 1.0 if sprint else 0.0)

    def _chase_free(self, v: View, sprint_ok: bool) -> Action:
        c, me = self.cfg, v.me
        d = math.hypot(v.ball_x - me.x, v.ball_y - me.y)
        t = min(c.lead_time_max, d / c.ref_speed)
        tx = _clamp(v.ball_x + to_per_second(v.ball_vx) * t, 0.0, PITCH_W)
        ty = _clamp(v.ball_y + to_per_second(v.ball_vy) * t, 0.0, PITCH_H)
        self.last_rule = "chase"
        return self._run_to(v, tx, ty, sprint_ok and d > c.sprint_min_dist)

    def _defend(self, v: View, sprint_ok: bool) -> Action:
        c, me, opp = self.cfg, v.me, v.opp
        d = math.hypot(opp.x - me.x, opp.y - me.y)
        behind_ball = me.x < opp.x  # between opp and my goal
        if not c.goal_side or d <= c.contest_dist or (behind_ball and d <= 2 * c.contest_dist):
            self.last_rule = "contest"
            return self._run_to(v, opp.x, opp.y, sprint_ok and d > c.sprint_min_dist)
        ux, uy = _norm(0.0 - opp.x, PITCH_H / 2 - opp.y)  # toward my own goal
        tx, ty = opp.x + ux * c.block_dist, opp.y + uy * c.block_dist
        self.last_rule = "block"
        far = math.hypot(tx - me.x, ty - me.y) > 80.0
        return self._run_to(v, tx, ty, sprint_ok and far)

    def _on_target(self, v: View) -> bool:
        """True if the line along the player's facing enters the goal mouth."""
        me = v.me
        cos, sin = math.cos(me.facing), math.sin(me.facing)
        if cos <= 1e-6 or v.goal_x - me.x <= 0.0:
            return False
        y_hit = me.y + (v.goal_x - me.x) * sin / cos
        return abs(y_hit - PITCH_H / 2) <= GOAL_W / 2 - self.cfg.post_margin / 2

    def _carry(self, v: View, sprint_ok: bool) -> Action:
        c, me, opp = self.cfg, v.me, v.opp
        off = GOAL_W / 2 - c.post_margin
        aim_y = PITCH_H / 2
        if c.aim_corners:  # corner far from opponent
            aim_y += off if opp.y < PITCH_H / 2 else -off
        ax, ay = _norm(v.goal_x - me.x, aim_y - me.y)
        dist_aim = math.hypot(v.goal_x - me.x, aim_y - me.y)
        d_opp = math.hypot(opp.x - me.x, opp.y - me.y)
        diff = abs(_wrap(math.atan2(aim_y - me.y, v.goal_x - me.x) - me.facing))
        pressed = d_opp < c.opp_avoid_dist

        # a kick must really send the ball into the mouth (facing line check)
        on_target = self._on_target(v)
        shoot = dist_aim < c.shoot_dist and diff < c.face_tol and on_target
        if (
            not shoot
            and c.snap_shot
            and d_opp < 40.0
            and on_target
            and dist_aim < 1.2 * c.shoot_dist
            and diff < 2 * c.face_tol
        ):
            shoot = True
        if shoot:
            self.last_rule = "shoot"
            power = 1.0 if dist_aim > 150.0 else c.close_power
            return Action(ax, ay, 0.0, 1.0, power)

        mx, my = ax, ay
        # swerve only if the opponent is ahead/beside; a chaser BEHIND is outrun
        if pressed and opp.x > me.x - 20.0:  # swerve away from opponent
            side = 1.0 if me.y >= opp.y else -1.0
            if me.y < c.wall_margin:
                side = 1.0
            elif me.y > PITCH_H - c.wall_margin:
                side = -1.0
            mx, my = _norm(ax * 0.5, ay * 0.5 + side)
        breakaway = opp.x < me.x - 30.0 and dist_aim > c.shoot_dist
        sprint = sprint_ok and (pressed or breakaway) and dist_aim > 0.8 * c.shoot_dist
        self.last_rule = "carry"
        return Action(mx, my, 1.0 if sprint else 0.0)
