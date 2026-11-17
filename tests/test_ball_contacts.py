import math
from dataclasses import replace

import pytest

from game.ball import BallState
from game.ball_contacts import advance_free_ball
from game.config import BallConfig, SimulationConfig
from game.player import PlayerId, PlayerState
from game.simulation import reset

NO_FRICTION = SimulationConfig(ball=BallConfig(deceleration=0.0))


def flight(ball, a=None, b=None, *, previous_a=None, config=NO_FRICTION):
    state = reset()
    a, b = a or state.player_a, b or state.player_b
    return advance_free_ball(ball, previous_a or a, b, a, b, config=config)


def test_locked_player_bounces_half_speed_and_updates_last_touch():
    a = replace(reset().player_a, pickup_lockout_ticks=12)
    result = flight(BallState((177.0, 250.0), (300.0, 0.0), last_touch_player=PlayerId.B), a)
    assert result.ball.position == pytest.approx((176.0, 250.0))
    assert result.ball.velocity == (-150.0, 0.0)
    assert result.ball.last_touch_player == PlayerId.A
    assert result.pickup is result.crossing is None
    assert len(result.impacts) == 1
    assert result.impacts[0].player == PlayerId.A


def test_eligible_player_still_traps_instead_of_bouncing():
    result = flight(BallState((177.0, 250.0), (300.0, 0.0)))
    assert result.pickup.owner == PlayerId.A
    assert result.impacts == ()


def test_post_bounce_occurs_before_exit_and_preserves_last_touch():
    result = flight(BallState((780.0, 186.0), (1200.0, 0.0), last_touch_player=PlayerId.B))
    assert result.ball.velocity == pytest.approx((-840.0, 0.0))
    assert result.ball.position == pytest.approx((781.3, 186.0))
    assert result.ball.last_touch_player == PlayerId.B
    assert result.crossing is result.pickup is None
    assert [impact.post_index for impact in result.impacts] == [2]


def test_multiple_player_impacts_use_remaining_tick_time():
    a = PlayerState((200.0, 250.0), pickup_lockout_ticks=12)
    b = PlayerState((250.0, 250.0), pickup_lockout_ticks=12)
    result = flight(BallState((225.0, 250.0), (6000.0, 0.0)), a, b)
    assert len(result.impacts) == 4
    assert result.ball.velocity == pytest.approx((375.0, 0.0))
    assert result.ball.position == pytest.approx((222.8125, 250.0))
    assert result.ball.last_touch_player == PlayerId.A
    assert result.limited is False


def test_moving_locked_player_transfers_relative_normal_velocity_without_recoil():
    before = PlayerState((200.0, 250.0), pickup_lockout_ticks=12)
    after = replace(before, position=(203.0, 250.0), velocity=(180.0, 0.0))
    result = flight(BallState((222.0, 250.0)), after, previous_a=before)
    assert result.ball.velocity == pytest.approx((270.0, 0.0))
    assert result.ball.position == pytest.approx((226.5, 250.0))
    assert after.velocity == (180.0, 0.0)


def test_ball_can_be_recovered_after_a_post_bounce_in_the_same_tick():
    b = PlayerState((775.0, 154.0))
    result = flight(BallState((780.0, 180.0), (6000.0, 0.0)), b=b)
    assert len(result.impacts) == 1
    assert result.impacts[0].post_index == 2
    assert result.pickup.owner == PlayerId.B
    assert result.pickup.fraction > result.impacts[0].fraction


def test_equal_contested_overlap_stays_free_and_depenetrates_without_identity_bias():
    a, b = PlayerState((385.0, 250.0)), PlayerState((415.0, 250.0))
    ball = BallState((400.0, 250.0))
    result = flight(ball, a, b)
    swapped = flight(ball, b, a)
    assert result.pickup is None
    assert result.ball == swapped.ball
    assert result.ball.last_touch_player is None
    assert math.dist(result.ball.position, a.position) >= 22.0 - 1e-9
    assert math.dist(result.ball.position, b.position) >= 22.0 - 1e-9


def test_budget_saturation_is_bounded_explicit_and_nonpenetrating():
    a = PlayerState((200.0, 250.0), pickup_lockout_ticks=12)
    b = PlayerState((250.0, 250.0), pickup_lockout_ticks=12)
    config = SimulationConfig(ball=BallConfig(deceleration=0.0, max_impacts_per_tick=1))
    result = flight(BallState((225.0, 250.0), (6000.0, 0.0)), a, b, config=config)
    assert result.limited
    assert len(result.impacts) == 1
    assert result.ball.velocity == (0.0, 0.0)
    assert math.dist(result.ball.position, a.position) >= 22.0 - 1e-9
    assert math.dist(result.ball.position, b.position) >= 22.0 - 1e-9


def test_separating_surface_touch_does_not_repeat_impacts_or_change_last_touch():
    a = replace(reset().player_a, pickup_lockout_ticks=12)
    result = flight(BallState((178.0, 250.0), (-300.0, 0.0), last_touch_player=PlayerId.B), a)
    assert result.impacts == ()
    assert result.ball.last_touch_player == PlayerId.B


def test_depenetration_of_initial_post_overlap_does_not_add_energy():
    result = flight(BallState((790.0, 186.0), (-300.0, 0.0)))
    assert len(result.impacts) == 1
    assert result.ball.velocity == (-300.0, 0.0)
    assert result.ball.position == pytest.approx((784.0, 186.0))


def test_static_opposed_contacts_settle_normal_motion_but_keep_tangent():
    a = PlayerState((200.0, 250.0), pickup_lockout_ticks=12)
    b = PlayerState((244.0, 250.0), pickup_lockout_ticks=12)
    result = flight(BallState((222.0, 250.0), (6000.0, 40.0)), a, b)
    assert result.constraint is not None
    assert not result.limited
    assert result.ball.velocity == pytest.approx((0.0, 40.0))
    assert result.ball.last_touch_player is None
    assert math.dist(result.ball.position, a.position) >= 22 - 1e-9
    assert math.dist(result.ball.position, b.position) >= 22 - 1e-9
