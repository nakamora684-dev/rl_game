import math

import pytest

from game.ball import BallState, integrate_free_ball
from game.config import BallConfig, RulesConfig, SimulationConfig
from game.rebounds import project_outside_circles, reflect_velocity
from game.rules import goalpost_centres


def test_stationary_player_returns_half_normal_speed_and_preserves_tangent():
    assert reflect_velocity((300.0, 40.0), (0.0, 0.0), (-1.0, 0.0), 0.5) == (-150.0, 40.0)
    assert reflect_velocity((-150.0, 40.0), (0.0, 0.0), (-1.0, 0.0), 0.5) == (-150.0, 40.0)


def test_moving_surface_uses_relative_velocity_without_recoil():
    assert reflect_velocity((300.0, 40.0), (100.0, 0.0), (-1.0, 0.0), 0.5) == (0.0, 40.0)


def test_post_returns_seventy_percent_normal_speed():
    assert reflect_velocity((300.0, 0.0), (0.0, 0.0), (-1.0, 0.0), 0.7) == (-210.0, 0.0)


def test_post_inner_edges_preserve_clear_opening():
    posts = goalpost_centres(SimulationConfig())
    assert posts == ((0.0, 186.0), (0.0, 314.0), (800.0, 186.0), (800.0, 314.0))
    assert posts[1][1] - 4 - (posts[0][1] + 4) == 120


def test_intersection_projection_resolves_equal_overlaps_without_identity_order():
    circles = (((385.0, 250.0), 22.0), ((415.0, 250.0), 22.0))
    point = project_outside_circles((400.0, 250.0), circles)
    assert point == project_outside_circles((400.0, 250.0), tuple(reversed(circles)))
    assert all(math.dist(point, centre) >= radius - 1e-9 for centre, radius in circles)


def test_projection_respects_pitch_edge_and_solid_post():
    point = project_outside_circles(
        (785.0, 15.0), (((800.0, 20.0), 19.0),), bounds=(15, 785, 15, 485)
    )
    assert 15 <= point[0] <= 785 and 15 <= point[1] <= 485
    assert math.dist(point, (800.0, 20.0)) >= 19 - 1e-9


def test_empty_obstacle_projection_clamps_to_nearest_edge_not_corner():
    assert project_outside_circles((0.0, 250.0), (), bounds=(15, 785, 15, 485)) == (15.0, 250.0)


def test_arbitrary_duration_roll_preserves_analytic_distance():
    ball = BallState((0.0, 0.0), (300.0, 0.0))
    whole = integrate_free_ball(ball, 3.0)
    split = integrate_free_ball(integrate_free_ball(ball, 0.37), 2.63)
    assert whole.position == split.position == pytest.approx((450.0, 0.0))
    assert whole.velocity == split.velocity == (0.0, 0.0)


@pytest.mark.parametrize("value", [-0.1, 1.1, math.inf, math.nan])
def test_invalid_restitution(value):
    with pytest.raises(ValueError):
        BallConfig(player_restitution=value)
    with pytest.raises(ValueError):
        BallConfig(post_restitution=value)


@pytest.mark.parametrize("value", [0, -1, 0.5, True])
def test_invalid_impact_budget(value):
    with pytest.raises(ValueError):
        BallConfig(max_impacts_per_tick=value)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_invalid_post_radius(value):
    with pytest.raises(ValueError):
        RulesConfig(post_radius=value)
