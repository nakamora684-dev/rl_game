"""Version-2, attack-normalized observations from read-only engine projections."""

import math

import numpy as np

from environment.contract import MatchState, Phase
from environment.spec import (
    BALL_SPEED_SCALE,
    MAX_SCORE_DIFF,
    OBS_DIM,
    OBS_NAMES,
    obs_bounds,
    to_policy_angle,
    to_policy_x,
)

_LOW, _HIGH = obs_bounds()


def _fraction(ticks: int, duration: int) -> float:
    return ticks / max(1, duration)


def build_observation(state: MatchState, player_idx: int) -> np.ndarray:
    if player_idx not in (0, 1):
        raise ValueError("player_idx must be 0 or 1.")
    config = state.config
    width, height = config.pitch.width, config.pitch.height
    diagonal = math.hypot(width, height)
    speed_scale = max(1.0, config.movement.max_speed)
    me, opp = state.players[player_idx], state.players[1 - player_idx]
    ad = state.attack_dirs[player_idx]
    sx = to_policy_x(me.x, ad, width)
    ox = to_policy_x(opp.x, ad, width)
    bx = to_policy_x(state.ball.x, ad, width)
    self_angle, opp_angle = to_policy_angle(me.facing, ad), to_policy_angle(opp.facing, ad)
    free = state.possession is None
    v = {
        "self_x": (sx - width / 2) / (width / 2),
        "self_y": (me.y - height / 2) / (height / 2),
        "self_vx": me.vx * ad / speed_scale,
        "self_vy": me.vy / speed_scale,
        "self_face_sin": math.sin(self_angle),
        "self_face_cos": math.cos(self_angle),
        "self_stamina": me.stamina / 100,
        "opp_rel_x": (ox - sx) / width,
        "opp_rel_y": (opp.y - me.y) / height,
        "opp_vx": opp.vx * ad / speed_scale,
        "opp_vy": opp.vy / speed_scale,
        "opp_face_sin": math.sin(opp_angle),
        "opp_face_cos": math.cos(opp_angle),
        "opp_stamina": opp.stamina / 100,
        "ball_rel_x": (bx - sx) / width,
        "ball_rel_y": (state.ball.y - me.y) / height,
        "ball_vx": state.ball.vx * ad / BALL_SPEED_SCALE if free else 0.0,
        "ball_vy": state.ball.vy / BALL_SPEED_SCALE if free else 0.0,
        "poss_self": float(state.possession == player_idx),
        "poss_opp": float(state.possession == 1 - player_idx),
        "poss_free": float(free),
        "time_remaining": state.time_remaining_frac,
        "score_diff": (state.score[player_idx] - state.score[1 - player_idx]) / MAX_SCORE_DIFF,
        "contest_active": float(state.steal_progress_ticks > 0),
        "contest_progress": _fraction(
            state.steal_progress_ticks, config.possession.steal_contest_ticks
        ),
        "possession_protection": _fraction(
            state.possession_protection_ticks, config.possession.protection_ticks
        ),
        "restart_remaining": _fraction(state.restart_ticks_remaining, config.rules.restart_ticks),
    }
    for prefix, player in (("self", me), ("opp", opp)):
        v[f"{prefix}_kick_cooldown"] = _fraction(
            player.kick_cooldown_ticks, config.kick.cooldown_ticks
        )
        v[f"{prefix}_pickup_lockout"] = _fraction(
            player.pickup_lockout_ticks, config.possession.kicker_pickup_lockout_ticks
        )
    for phase in Phase:
        v[f"phase_{phase.value}"] = float(state.phase is phase)
    restarting = state.phase in (Phase.GOAL_RESTART, Phase.OUT_OF_BOUNDS_RESTART)
    v["restart_self"] = float(restarting and state.restart_owner == player_idx)
    v["restart_opp"] = float(restarting and state.restart_owner == 1 - player_idx)
    v["restart_neutral"] = float(restarting and state.restart_owner is None)
    for name, (tx, ty) in {
        "ball": (bx, state.ball.y),
        "opp": (ox, opp.y),
        "own_goal": (0.0, height / 2),
        "opp_goal": (width, height / 2),
    }.items():
        dx, dy = tx - sx, ty - me.y
        distance = math.hypot(dx, dy)
        v[f"to_{name}_dx"] = dx / distance if distance > 1e-6 else 0.0
        v[f"to_{name}_dy"] = dy / distance if distance > 1e-6 else 0.0
        v[f"to_{name}_dist"] = distance / diagonal
    if set(v) != set(OBS_NAMES):
        raise AssertionError(f"Observation/schema mismatch: {set(v) ^ set(OBS_NAMES)}")
    vec = np.array([v[name] for name in OBS_NAMES], dtype=np.float32)
    if vec.shape != (OBS_DIM,) or not np.all(np.isfinite(vec)):
        raise ValueError("Malformed or nonfinite observation.")
    return np.clip(vec, _LOW, _HIGH)


def describe_observation(obs: np.ndarray) -> dict[str, float]:
    return {name: float(obs[i]) for i, name in enumerate(OBS_NAMES)}
