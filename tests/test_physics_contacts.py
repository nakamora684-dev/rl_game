import math
import random
from dataclasses import replace

import pytest

from game.actions import Direction, MovementAction
from game.ball import BallState
from game.config import MATCH_TICKS, BallConfig, PossessionConfig, SimulationConfig
from game.player import PlayerId, PlayerState
from game.rules import goalpost_centres
from game.simulation import MatchPhase, assign_possession, reset, step_decision, step_tick


def test_controlled_post_hit_knocks_free_without_kick_or_normal_case_saturation():
    state = reset()
    state = replace(
        state, player_a=replace(state.player_a, position=(765.0, 186.0), velocity=(180.0, 0.0))
    )
    state = assign_possession(state, PlayerId.A).state
    result = step_tick(state)
    assert result.state.ball.possession_owner is None
    assert result.state.player_a.pickup_lockout_ticks == 11
    assert result.state.player_a.kick_cooldown_ticks == 0
    assert result.events[0].cause == "post_collision"
    assert any(event.cause == "goalpost" for event in result.events)
    assert all(event.kind not in ("ball_kicked", "physics_limit") for event in result.events)
    assert math.dist(result.state.ball.position, (800.0, 186.0)) >= 11 - 1e-9
    assert math.dist(result.state.ball.position, result.state.player_a.position) >= 22 - 1e-9


def test_turning_only_attachment_has_real_post_impact_velocity():
    state = reset()
    state = replace(state, player_a=replace(state.player_a, position=(773.0, 198.0)))
    state = assign_possession(state, PlayerId.A).state
    assert state.ball.velocity == (0.0, 0.0)
    result = step_tick(state, MovementAction(Direction.NORTH, 0.0))
    assert result.state.ball.possession_owner is None
    assert result.events[0].cause == "post_collision"
    post_hit = next(event for event in result.events if event.cause == "goalpost")
    assert math.hypot(*post_hit.velocity_before) > 0.0
    assert all(event.kind != "ball_kicked" for event in result.events)


def test_exhausted_flight_budget_does_not_also_report_attachment_saturation():
    state = reset()
    state = replace(state, player_a=replace(state.player_a, position=(773.0, 198.0)))
    state = assign_possession(state, PlayerId.A).state
    config = SimulationConfig(ball=BallConfig(max_impacts_per_tick=1))
    result = step_tick(state, MovementAction(Direction.NORTH, 0.0), config=config)
    limits = [event.cause for event in result.events if event.kind == "physics_limit"]
    assert limits == ["impact_budget"]
    assert result.state.ball.possession_owner is None


def test_pickup_attachment_teleport_does_not_invent_launch_energy_at_post():
    state = reset()
    state = replace(
        state,
        player_b=replace(state.player_b, position=(778.0, 186.0), facing=0.0),
        ball=BallState((778.0, 164.0)),
    )
    result = step_tick(state)
    assert [event.cause for event in result.events[:2]] == ["recovery", "post_collision"]
    assert result.state.ball.possession_owner is None
    assert result.state.ball.velocity == (0.0, 0.0)
    assert result.state.possession_protection_ticks == 0
    assert result.state.player_b.pickup_lockout_ticks == 11


def test_steal_attachment_obstructed_by_post_has_no_teleport_launch_energy():
    state = reset()
    state = replace(
        state,
        player_a=replace(state.player_a, position=(743.0, 186.0)),
        player_b=replace(state.player_b, position=(778.0, 186.0), facing=0.0),
    )
    state = assign_possession(state, PlayerId.A).state
    state = replace(state, steal_progress_ticks=17)
    result = step_tick(state)
    assert [event.cause for event in result.events[:2]] == ["steal", "post_collision"]
    assert result.state.ball.possession_owner is None
    assert result.state.ball.velocity == (0.0, 0.0)
    assert result.state.possession_protection_ticks == 0


def test_post_bounce_can_turn_off_target_shot_into_a_goal_in_one_tick():
    config = SimulationConfig(ball=BallConfig(deceleration=0.0))
    state = replace(
        reset(), ball=BallState((780.0, 304.0), (6000.0, 0.0), last_touch_player=PlayerId.A)
    )
    result = step_tick(state, config=config)
    assert [event.kind for event in result.events] == ["ball_collision", "goal"]
    assert result.events[0].cause == "goalpost"
    assert result.state.score_a == 1
    assert result.state.ball.last_touch_player == PlayerId.A
    assert result.events[0].tick_fraction < result.events[1].tick_fraction


def test_locked_player_deflection_changes_boundary_responsibility():
    config = SimulationConfig(ball=BallConfig(deceleration=0.0))
    state = reset()
    state = replace(
        state,
        player_a=replace(state.player_a, position=(400.0, 15.0), pickup_lockout_ticks=12),
        ball=BallState((380.0, 40.0), (0.0, -6000.0), last_touch_player=PlayerId.B),
    )
    result = step_tick(state, config=config)
    assert [event.kind for event in result.events] == ["ball_collision", "out_of_bounds"]
    assert result.events[0].contact_player == PlayerId.A
    assert result.events[-1].last_touch_player == PlayerId.A
    assert result.state.restart_plan.owner == PlayerId.B


def test_multiple_impacts_advance_clock_and_lockouts_only_once():
    state = reset()
    state = replace(
        state,
        player_a=replace(state.player_a, position=(200.0, 250.0), pickup_lockout_ticks=12),
        player_b=replace(state.player_b, position=(250.0, 250.0), pickup_lockout_ticks=12),
        ball=BallState((225.0, 250.0), (6000.0, 0.0)),
    )
    result = step_tick(state, config=SimulationConfig(ball=BallConfig(deceleration=0.0)))
    assert result.physics_ticks == result.state.simulation_tick == 1
    assert (
        result.state.player_a.pickup_lockout_ticks
        == result.state.player_b.pickup_lockout_ticks
        == 11
    )
    assert [event.collision_index for event in result.events] == [0, 1, 2, 3]
    assert [event.tick_fraction for event in result.events] == sorted(
        event.tick_fraction for event in result.events
    )


def test_player_cannot_clip_through_post_and_has_no_rebound():
    state = reset()
    state = replace(
        state, player_a=replace(state.player_a, position=(770.0, 186.0), velocity=(1200.0, 0.0))
    )
    result = step_tick(state)
    assert result.state.player_a.position == pytest.approx((781.0, 186.0))
    assert result.state.player_a.velocity == (0.0, 0.0)


def test_coupled_player_post_overlap_resolves_without_leaving_pitch():
    state = reset()
    state = replace(
        state, player_a=PlayerState((781.0, 186.0)), player_b=PlayerState((781.0, 186.0))
    )
    result = step_tick(state)
    assert math.dist(result.state.player_a.position, result.state.player_b.position) >= 30 - 1e-9
    for player in (result.state.player_a, result.state.player_b):
        assert 15 <= player.position[0] <= 785 and 15 <= player.position[1] <= 485
        assert all(
            math.dist(player.position, post) >= 19 - 1e-9
            for post in goalpost_centres(SimulationConfig())
        )


def test_collision_and_whistle_both_survive_last_tick_then_freeze():
    state = reset()
    state = replace(
        state,
        simulation_tick=MATCH_TICKS - 1,
        ball=BallState((780.0, 186.0), (1200.0, 0.0), last_touch_player=PlayerId.B),
    )
    result = step_tick(state)
    assert [event.kind for event in result.events] == ["ball_collision", "match_finished"]
    assert result.state.phase == MatchPhase.FINISHED
    assert step_tick(result.state).state is result.state


def test_zero_lockout_obstructed_attachment_is_bounded_not_an_infinite_loop():
    config = SimulationConfig(possession=PossessionConfig(kicker_pickup_lockout_ticks=0))
    state = reset()
    state = replace(
        state,
        player_b=replace(state.player_b, position=(778.0, 186.0), facing=0.0),
        ball=BallState((778.0, 164.0)),
    )
    result = step_tick(state, config=config)
    assert result.state.simulation_tick == 1
    assert all(
        math.isfinite(value) for value in (*result.state.ball.position, *result.state.ball.velocity)
    )


def test_seeded_contact_replay_remains_identical_and_bounded():
    def trajectory():
        rng = random.Random(114)
        state = reset()
        states, events = [], []
        for _ in range(80):
            # Deterministic stress fixtures, not a match runner or baseline agent.
            state = replace(
                state,
                ball=BallState(
                    (rng.uniform(20, 780), rng.uniform(20, 480)),
                    (rng.uniform(-12000, 12000), rng.uniform(-12000, 12000)),
                ),
                phase=MatchPhase.PLAYING,
                restart_ticks_remaining=0,
                restart_plan=None,
                steal_progress_ticks=0,
                possession_protection_ticks=0,
            )
            result = step_decision(state)
            state = result.state
            states.append(state)
            events.extend(result.events)
            assert all(math.isfinite(v) for v in (*state.ball.position, *state.ball.velocity))
        return states, events

    assert trajectory() == trajectory()


def test_seeded_post_adjoining_stress_has_solid_bodies_and_finite_ball():
    rng = random.Random(117)
    config = SimulationConfig()
    posts = goalpost_centres(config)
    for _ in range(250):
        post = posts[rng.randrange(4)]
        x = rng.uniform(15, 65) if post[0] == 0 else rng.uniform(735, 785)
        y = post[1] + rng.uniform(-35, 35)
        a = PlayerState(
            (x, y),
            velocity=(rng.uniform(-1200, 1200), rng.uniform(-1200, 1200)),
            pickup_lockout_ticks=12,
        )
        b = PlayerState((rng.uniform(15, 785), rng.uniform(15, 485)), pickup_lockout_ticks=12)
        ball = BallState(
            (rng.uniform(0, 800), rng.uniform(0, 500)),
            (rng.uniform(-12000, 12000), rng.uniform(-12000, 12000)),
        )
        result = step_tick(replace(reset(), player_a=a, player_b=b, ball=ball))
        state = result.state
        for player in (state.player_a, state.player_b):
            assert 15 <= player.position[0] <= 785 and 15 <= player.position[1] <= 485
            assert all(math.dist(player.position, p) >= 19 - 1e-8 for p in posts)
        assert math.dist(state.player_a.position, state.player_b.position) >= 30 - 1e-8
        assert all(math.isfinite(v) for v in (*state.ball.position, *state.ball.velocity))
        if state.phase == MatchPhase.PLAYING and state.ball.possession_owner is None:
            assert all(math.dist(state.ball.position, p) >= 11 - 1e-8 for p in posts)
            assert all(
                math.dist(state.ball.position, p.position) >= 22 - 1e-8
                for p in (state.player_a, state.player_b)
            )
