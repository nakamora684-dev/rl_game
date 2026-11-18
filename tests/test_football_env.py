"""Real-engine integration: no scripted physics or invented gameplay events."""

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy", reason="Install the locked environment dependency group.")
pytest.importorskip("gymnasium", reason="Install the locked environment dependency group.")

from environment.contract import Phase
from environment.engine import SimulationEngine
from environment.football_env import FootballEnv
from environment.reward import RewardConfig
from environment.spec import OBS_DIM
from game.actions import Direction, PlayerAction
from game.ball import BallState
from game.config import MATCH_TICKS
from game.player import PlayerId
from game.runner import run_match
from game.scenarios import AgentPlan, ScenarioAgent
from game.simulation import assign_possession, reset

pytestmark = pytest.mark.environment


def raw_action(direction=Direction.IDLE, effort=0.0, kick=0, power=0.0):
    return {
        "direction": int(direction),
        "effort": np.array([effort], dtype=np.float32),
        "kick_trigger": kick,
        "kick_power": np.array([power], dtype=np.float32),
    }


class ScenarioEngine(SimulationEngine):
    """Only override starting layout; every step uses the real shared physics."""

    def __init__(self, initial):
        super().__init__()
        self.initial = initial

    def reset(self, *, swap_sides=False, initial_state=None):
        return super().reset(initial_state=self.initial if initial_state is None else initial_state)


def scenario_env(initial, **kwargs):
    return FootballEnv(ScenarioEngine(initial), **kwargs)


def run_episode(env, agent_idx=0):
    env.reset(seed=0, options={"agent_idx": agent_idx})
    total, steps = 0.0, 0
    while True:
        obs, reward, terminated, truncated, info = env.step(raw_action())
        assert obs.shape == (OBS_DIM,) and env.observation_space.contains(obs)
        total += reward
        steps += 1
        if terminated or truncated:
            return total, steps, terminated, truncated, info


@pytest.mark.parametrize("idx", [0, 1])
@pytest.mark.parametrize("swap", [False, True])
def test_reset_and_forward_controls_cover_both_identities_and_both_orientations(idx, swap):
    env = FootballEnv()
    obs, info = env.reset(seed=1, options={"agent_idx": idx, "swap_sides": swap})
    assert env.observation_space.contains(obs)
    assert info["agent_idx"] == idx and info["tick"] == 0 and info["score"] == (0, 0)
    assert info["attack_dirs"] == ((-1, 1) if swap else (1, -1))
    start = env.engine.native_state
    for _ in range(10):
        env.step(raw_action(Direction.EAST, 1))
    native = env.engine.native_state
    before = start.player_a if idx == 0 else start.player_b
    after = native.player_a if idx == 0 else native.player_b
    assert (after.position[0] - before.position[0]) * info["attack_dirs"][idx] > 0


def test_full_episode_600_steps_and_components_sum_to_undiscounted_return():
    total, steps, terminated, truncated, info = run_episode(FootballEnv())
    assert steps == 600 and terminated and not truncated
    assert info["end_reason"] == "time_up" and info["tick"] == MATCH_TICKS
    assert info["episode_stats"]["match_result"] == "draw"
    assert sum(info["episode_stats"]["components"].values()) == pytest.approx(total)
    assert "episode" not in info  # reserved for the downstream Monitor


def test_goal_does_not_terminate_or_fast_forward_and_opponent_is_queried_during_restarts():
    calls = []

    def opponent(state, player):
        calls.append((state.simulation_tick, state.phase, player))
        return PlayerAction(kick_trigger=1, kick_power=1)

    initial = replace(reset(), ball=BallState((806.9, 250), (60, 0), last_touch_player=PlayerId.A))
    env = scenario_env(initial, opponent=opponent)
    env.reset(options={"agent_idx": 0})
    _, _, terminated, truncated, info = env.step(raw_action())
    assert not terminated and not truncated
    assert info["tick"] == 6 and info["restart_ticks_remaining"] == 55
    assert info["phase"] == "goal_restart"
    assert info["reward_components"]["goal_for"] == 10
    assert [e["kind"] for e in info["events"]] == ["goal"]
    assert info["events"][0]["scoring_player"] is PlayerId.A
    for _ in range(9):
        env.step(raw_action())
    assert env.engine.native_state.restart_ticks_remaining == 1
    _, _, _, _, info = env.step(raw_action())  # kickoff at tick 61, then five playing ticks
    assert info["tick"] == 66 and info["phase"] == "playing"
    assert env.engine.native_state.ball.possession_owner is PlayerId.B
    assert info["reward_components"]["loose_recovery"] == 0
    assert info["reward_components"]["steal"] == 0
    assert all(e["kind"] != "ball_kicked" for e in info["events"])
    assert len(calls) == 11
    assert all(p is PlayerId.B for _, _, p in calls)
    assert any(phase is Phase.GOAL_RESTART for _, phase, _ in calls)


@pytest.mark.parametrize("power,speed", [(0, 50), (0.5, 175), (1, 300)])
def test_kick_trigger_power_and_facing_semantics(power, speed):
    initial = assign_possession(reset(), PlayerId.A).state
    env = scenario_env(initial)
    env.reset(options={"agent_idx": 0})
    _, _, _, _, info = env.step(raw_action(kick=1, power=power))
    assert [(e["kind"], e["tick"]) for e in info["events"]] == [("ball_kicked", 0)]
    native = env.engine.native_state
    assert native.ball.velocity[0] == pytest.approx(speed - 10)
    assert native.ball.velocity[1] == 0  # east facing, no independent aiming
    assert native.player_a.kick_cooldown_ticks == native.player_a.pickup_lockout_ticks == 6


def test_recovery_and_steal_receive_distinct_components():
    rec = replace(reset(), ball=BallState((200, 250)))
    env = scenario_env(rec)
    env.reset(options={"agent_idx": 0})
    info = env.step(raw_action(kick=1))[4]
    assert info["reward_components"]["loose_recovery"] == 0.5
    assert info["reward_components"]["steal"] == 0
    assert [e["cause"] for e in info["events"]] == ["recovery"]
    native = reset()
    native = replace(native, player_b=replace(native.player_b, position=(235, 250)))
    native = assign_possession(native, PlayerId.A).state
    native = replace(native, steal_progress_ticks=17)
    env = scenario_env(native)
    env.reset(options={"agent_idx": 1})
    info = env.step(raw_action())[4]
    assert info["reward_components"]["steal"] == 1
    assert info["reward_components"]["loose_recovery"] == 0


@pytest.mark.parametrize("toucher,penalty", [(None, 0), (PlayerId.A, -0.5)])
def test_neutral_and_blamed_boundary_restarts_do_not_reward_placement(toucher, penalty):
    initial = replace(reset(), ball=BallState((400, -6.9), (0, -60), last_touch_player=toucher))
    env = scenario_env(initial)
    env.reset(options={"agent_idx": 0})
    info = env.step(raw_action())[4]
    assert info["reward_components"]["out_of_bounds"] == penalty
    for _ in range(10):
        info = env.step(raw_action())[4]
        assert info["reward_components"]["loose_recovery"] == 0
        assert info["reward_components"]["steal"] == 0


def test_collision_metadata_and_post_control_loss_are_preserved_without_fake_kick():
    initial = reset()
    initial = replace(
        initial, player_a=replace(initial.player_a, position=(765, 186), velocity=(180, 0))
    )
    initial = assign_possession(initial, PlayerId.A).state
    env = scenario_env(initial)
    env.reset(options={"agent_idx": 0})
    info = env.step(raw_action())[4]
    assert any(e["cause"] == "post_collision" for e in info["events"])
    hit = next(e for e in info["events"] if e["kind"] == "ball_collision")
    assert hit["tick_fraction"] is not None and hit["post_index"] is not None
    assert hit["velocity_before"] is not None and hit["velocity_after"] is not None
    assert all(e["kind"] != "ball_kicked" for e in info["events"])


def test_last_tick_goal_and_outcome_awarded_together_and_step_after_finish_rejected():
    native = replace(
        reset(),
        simulation_tick=MATCH_TICKS - 1,
        ball=BallState((806.9, 250), (60, 0), last_touch_player=PlayerId.A),
    )
    env = scenario_env(native)
    env.reset(options={"agent_idx": 0})
    _, _, terminated, truncated, info = env.step(raw_action())
    assert terminated and not truncated and info["physics_ticks"] == 1
    assert info["reward_components"]["goal_for"] == 10
    assert info["reward_components"]["outcome"] == 5
    assert [e["kind"] for e in info["events"]] == ["goal", "match_finished"]
    with pytest.raises(RuntimeError):
        env.step(raw_action())


def test_reset_clears_reward_ids_cooldowns_score_and_termination():
    env = FootballEnv()
    first, info = env.reset(seed=7)
    for _ in range(20):
        env.step(raw_action(Direction.EAST, 1, 1, 1))
    again, new_info = env.reset(seed=7)
    assert np.array_equal(first, again) and info == new_info
    assert env.rewarder._seen == set()
    assert env._steps == 0 and not env._done


def test_native_opponent_gets_pre_step_snapshot_and_world_action_is_not_mirrored():
    calls = []

    def opponent(state, player):
        calls.append((state, player))
        return PlayerAction(Direction.WEST, 1)

    env = FootballEnv(opponent=opponent)
    env.reset(options={"agent_idx": 1, "swap_sides": True})
    initial = env.engine.native_state
    env.step(raw_action(Direction.EAST, 1))
    assert calls == [(initial, PlayerId.A)]
    assert calls[0][0] is initial
    assert env.engine.native_state.player_a.velocity[0] < 0
    assert env.engine.native_state.player_b.velocity[0] > 0


def test_invalid_opponent_and_agent_actions_do_not_advance_physics():
    env = FootballEnv(opponent=lambda state, player: object())
    env.reset()
    initial = env.engine.native_state
    with pytest.raises(TypeError):
        env.step(raw_action())
    assert env.engine.native_state is initial
    with pytest.raises(ValueError):
        env.step(np.zeros(5, dtype=np.float32))
    assert env.engine.native_state is initial


def test_same_seed_fresh_native_opponents_give_identical_sustained_trajectories():
    def trajectory():
        opponent = ScenarioAgent(AgentPlan(seed=14), 0)
        env = FootballEnv(opponent=opponent)
        env.action_space.seed(6)
        obs, _ = env.reset(seed=42)
        states, observations, infos = [], [obs], []
        for _ in range(100):
            obs, _, _, _, info = env.step(env.action_space.sample())
            states.append(env.engine.native_state)
            observations.append(obs)
            infos.append(info)
        return states, np.stack(observations), infos

    states, obs, infos = trajectory()
    again_states, again_obs, again_infos = trajectory()
    assert states == again_states and infos == again_infos and np.array_equal(obs, again_obs)


def test_full_seeded_match_matches_headless_runner_with_world_space_provider():
    plan_a, plan_b = AgentPlan(seed=21), AgentPlan(seed=22)
    a, b = ScenarioAgent(plan_a, 0), ScenarioAgent(plan_b, 0)
    env = FootballEnv(opponent=b)
    env.reset(seed=0, options={"agent_idx": 0})  # A attacks right: policy == world
    raw_events = []
    while True:
        action = a(env.engine.native_state, PlayerId.A)
        _, _, terminated, truncated, info = env.step(
            raw_action(action.direction, action.effort, action.kick_trigger, action.kick_power)
        )
        raw_events.extend(info["events"])
        if terminated or truncated:
            break
    # Match the float32 Gym action boundary for exact equality with the runner.
    fresh_a = ScenarioAgent(plan_a, 0)

    def provider_a(state, player):
        action = fresh_a(state, player)
        return replace(
            action,
            effort=float(np.float32(action.effort)),
            kick_power=float(np.float32(action.kick_power)),
        )

    expected = run_match(provider_a, ScenarioAgent(plan_b, 0))
    assert env.engine.native_state == expected.final_state
    assert [(e["kind"], e["tick"]) for e in raw_events] == [
        (e.kind, e.tick) for e in expected.events
    ]
    assert info["episode_stats"]["steps"] == expected.decision_count == 600


def test_experimental_reward_settings_are_not_retuned():
    cfg = RewardConfig()
    assert (cfg.goal_for, cfg.goal_against, cfg.win, cfg.draw, cfg.loss) == (10, -10, 5, 0, -5)
    assert (cfg.steal, cfg.loose_recovery, cfg.steal_lost, cfg.out_of_bounds) == (
        1,
        0.5,
        -0.5,
        -0.5,
    )
    assert cfg.shaping_gamma == 1 and not cfg.zero_terminal_potential


def test_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(FootballEnv(), skip_render_check=True)
