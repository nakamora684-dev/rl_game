"""Per-match metrics, CSV schema and statistics (numpy/pandas only).

One row per match. Every counter is computed from the engine's one-shot events or
from decision-sampled state, so each event is counted once.

Definitions (also stated in the report):
* shot            - an agent kick whose straight-line path (kick position, agent facing,
                    speed from kick power) reaches the opponent goal line within the goal
                    opening (+15 units tolerance) before friction stops the ball.
* possession frac - ticks with the ball attached to a player / all ticks (restarts included).
* attacking half  - attack-normalised x > half the pitch width.
* sprint frac     - decisions that move with effort above the stamina neutral effort.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

MATCH_COLUMNS = [
    "schema_version", "agent", "opponent", "match_id", "seed", "agent_idx", "swap_sides",
    "attack_dir", "policy_mode", "result", "points", "goals_for", "goals_against", "goal_diff",
    "ep_return", "decisions", "end_reason",
    "shots", "goals_per_shot", "kicks", "kick_attempts",
    "steals_won", "loose_recoveries", "opp_steals", "opp_loose_recoveries",
    "lost_to_steal", "lost_boundary", "lost_other", "out_of_bounds_total",
    "poss_self_frac", "poss_opp_frac", "poss_free_frac",
    "attack_half_ball_frac", "attack_half_agent_frac", "ball_x_attack_mean",
    "stamina_mean", "stamina_min", "stamina_end", "sprint_frac",
    "idle_frac", "mean_effort_moving", "x_reversal_rate",
    "r_goal_for", "r_goal_against", "r_outcome", "r_steal", "r_steal_lost",
    "r_loose_recovery", "r_out_of_bounds", "r_possession", "r_progress_shaping",
]


class MatchRecorder:
    """Feed it (prev_state, flat_action, step_info, new_state) after every decision."""

    def __init__(self, agent_idx: int, config) -> None:
        self.me, self.cfg = agent_idx, config
        self.me_id = "A" if agent_idx == 0 else "B"
        self.ticks = 0
        self.c = dict.fromkeys(
            ["shots", "kicks", "kick_attempts", "steals_won", "loose_recoveries", "opp_steals",
             "opp_loose_recoveries", "lost_to_steal", "lost_boundary", "lost_other",
             "out_of_bounds_total"], 0)
        self.t = dict.fromkeys(
            ["poss_self", "poss_opp", "poss_free", "ball_att_half", "agent_att_half",
             "ball_x_sum"], 0.0)
        self.stamina: list[float] = []
        self.decisions = 0
        self.sprint = self.idle = self.moving = 0
        self.effort_sum = 0.0
        self.reversals = 0
        self._prev_fwd = 0

    # --- geometry helpers (attack-normalised) ---
    def _norm_x(self, x: float, attack_dir: int) -> float:
        return x if attack_dir == 1 else self.cfg.pitch.width - x

    def _is_shot(self, event: dict, prev, power: float, attack_dir: int) -> bool:
        x, y = event["ball_position"]
        facing = prev.players[self.me].facing
        cos_f, sin_f = math.cos(facing), math.sin(facing)
        if cos_f * attack_dir <= 1e-6:
            return False
        goal_x = self.cfg.pitch.width if attack_dir == 1 else 0.0
        dist = (goal_x - x) / cos_f
        y_at_goal = y + sin_f * dist
        speed = self.cfg.kick.speed_for_power(power)
        decel = self.cfg.ball.deceleration
        reach = math.inf if decel <= 0 else speed**2 / (2 * decel)
        half_opening = self.cfg.rules.goal_width / 2 + 15.0
        return dist <= reach and abs(y_at_goal - self.cfg.pitch.height / 2) <= half_opening

    def observe(self, prev, flat_action, info: dict, new) -> None:
        ticks = max(1, int(info["physics_ticks"]))
        self.ticks += ticks
        self.decisions += 1
        me, ad = self.me, new.attack_dirs[self.me]
        direction, effort, trigger, power = (
            int(round(flat_action[0])), float(flat_action[1]),
            float(flat_action[2]) > 0.5, float(flat_action[3]),
        )
        # --- own behaviour ---
        if direction == 0:
            self.idle += 1
        else:
            self.moving += 1
            self.effort_sum += effort
            if effort > self.cfg.stamina.neutral_effort:
                self.sprint += 1
        self.c["kick_attempts"] += int(trigger)
        fwd = 1 if direction in (2, 3, 4) else -1 if direction in (6, 7, 8) else 0
        if fwd and self._prev_fwd and fwd == -self._prev_fwd:
            self.reversals += 1
        self._prev_fwd = fwd
        # --- state sampling (weighted by ticks) ---
        if new.possession == me:
            self.t["poss_self"] += ticks
        elif new.possession is None:
            self.t["poss_free"] += ticks
        else:
            self.t["poss_opp"] += ticks
        half = self.cfg.pitch.width / 2
        bx = self._norm_x(new.ball.x, ad)
        self.t["ball_att_half"] += ticks * (bx > half)
        self.t["agent_att_half"] += ticks * (self._norm_x(new.players[me].x, ad) > half)
        self.t["ball_x_sum"] += ticks * bx / self.cfg.pitch.width
        self.stamina.append(new.players[me].stamina / 100.0)
        # --- one-shot events ---
        for e in info["events"]:
            kind, cause = e["kind"], e["cause"]
            prev_owner, new_owner = e["previous_owner"], e["new_owner"]
            if kind == "ball_kicked" and prev_owner == self.me_id:
                self.c["kicks"] += 1
                if self._is_shot(e, prev, power, prev.attack_dirs[me]):
                    self.c["shots"] += 1
            elif kind == "out_of_bounds":
                self.c["out_of_bounds_total"] += 1
                if e["last_touch_player"] == self.me_id:
                    self.c["lost_boundary"] += 1
            elif kind == "possession_changed":
                if cause == "steal":
                    if new_owner == self.me_id:
                        self.c["steals_won"] += 1
                    elif prev_owner == self.me_id:
                        self.c["lost_to_steal"] += 1
                        self.c["opp_steals"] += 1
                elif cause == "recovery":
                    key = "loose_recoveries" if new_owner == self.me_id else "opp_loose_recoveries"
                    self.c[key] += 1
                if prev_owner == self.me_id and cause != "steal":
                    self.c["lost_other"] += 1

    def finalize(self, info: dict, *, meta: dict) -> dict:
        stats = info["episode_stats"]
        idx = info["agent_idx"]
        gf, ga = stats["score"][idx], stats["score"][1 - idx]
        result = stats["match_result"]
        total = max(1, self.ticks)
        shots = self.c["shots"]
        row = {
            "schema_version": SCHEMA_VERSION, **meta,
            "attack_dir": info["attack_dirs"][idx], "result": result,
            "points": {"win": 1.0, "draw": 0.5, "loss": 0.0}[result],
            "goals_for": gf, "goals_against": ga, "goal_diff": gf - ga,
            "ep_return": sum(stats["components"].values()),
            "decisions": self.decisions, "end_reason": info["end_reason"],
            **self.c,
            "goals_per_shot": (gf / shots) if shots else float("nan"),
            "poss_self_frac": self.t["poss_self"] / total,
            "poss_opp_frac": self.t["poss_opp"] / total,
            "poss_free_frac": self.t["poss_free"] / total,
            "attack_half_ball_frac": self.t["ball_att_half"] / total,
            "attack_half_agent_frac": self.t["agent_att_half"] / total,
            "ball_x_attack_mean": self.t["ball_x_sum"] / total,
            "stamina_mean": float(np.mean(self.stamina)) if self.stamina else float("nan"),
            "stamina_min": float(np.min(self.stamina)) if self.stamina else float("nan"),
            "stamina_end": self.stamina[-1] if self.stamina else float("nan"),
            "sprint_frac": self.sprint / max(1, self.decisions),
            "idle_frac": self.idle / max(1, self.decisions),
            "mean_effort_moving": self.effort_sum / self.moving if self.moving else 0.0,
            "x_reversal_rate": self.reversals / max(1, self.decisions - 1),
            **{f"r_{k}": float(v) for k, v in stats["components"].items()},
        }
        missing = [c for c in MATCH_COLUMNS if c not in row]
        if missing:
            raise KeyError(f"Row is missing schema columns: {missing}")
        return {c: row[c] for c in MATCH_COLUMNS}


# ------------------------------------------------------------------ statistics
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(x, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarize(df: pd.DataFrame, by=("agent",)) -> pd.DataFrame:
    """Headline table per group: W/D/L, win-rate CI, goal diff CI, key behaviour means."""
    rows = []
    for key, g in df.groupby(list(by), sort=False):
        n = len(g)
        wins, draws, losses = (int((g.result == r).sum()) for r in ("win", "draw", "loss"))
        w_lo, w_hi = wilson_interval(wins, n)
        gd_lo, gd_hi = bootstrap_ci(g.goal_diff)
        key = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(by, key, strict=True)), "matches": n, "wins": wins, "draws": draws,
            "losses": losses, "win_rate": wins / n, "win_ci_low": w_lo, "win_ci_high": w_hi,
            "points": g.points.mean(), "goals_for": g.goals_for.mean(),
            "goals_against": g.goals_against.mean(), "goal_diff": g.goal_diff.mean(),
            "goal_diff_ci_low": gd_lo, "goal_diff_ci_high": gd_hi,
            "shots": g.shots.mean(), "poss_self_frac": g.poss_self_frac.mean(),
            "attack_half_ball_frac": g.attack_half_ball_frac.mean(),
            "stamina_mean": g.stamina_mean.mean(), "sprint_frac": g.sprint_frac.mean(),
            "idle_frac": g.idle_frac.mean(), "ep_return": g.ep_return.mean(),
        })
    return pd.DataFrame(rows)
