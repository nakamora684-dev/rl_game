import math
import random
from dataclasses import replace

import pytest

from game.ball import BallState
from game.collisions import resolve_player_contact
from game.config import PitchConfig
from game.player import PlayerState
from game.simulation import reset, step_tick


def contact_scene(a, b):
    return replace(reset(), player_a=a, player_b=b, ball=BallState((50.0, 50.0)))


def test_head_on_players_stop_without_overlap_or_bounce():
    a = PlayerState((384.0, 250.0), velocity=(120.0, 0.0))
    b = PlayerState((416.0, 250.0), velocity=(-120.0, 0.0))
    result = step_tick(contact_scene(a, b))
    assert result.state.player_a.position == pytest.approx((385.0, 250.0))
    assert result.state.player_b.position == pytest.approx((415.0, 250.0))
    assert result.state.player_a.velocity == result.state.player_b.velocity == (0.0, 0.0)
    assert result.events == ()


def test_tangential_sliding_and_other_player_fields_are_preserved():
    a = PlayerState((384.0, 250.0), facing=0.7, stamina=37.0, kick_cooldown_ticks=8)
    b = PlayerState((416.0, 250.0), facing=2.0, stamina=42.0, pickup_lockout_ticks=6)
    next_a = replace(a, position=(386.0, 251.0), velocity=(120.0, 60.0))
    next_b = replace(b, position=(414.0, 251.0), velocity=(-120.0, 60.0))
    result_a, result_b = resolve_player_contact(a, b, next_a, next_b, pitch=PitchConfig())
    assert result_a.velocity == result_b.velocity == (0.0, 60.0)
    assert result_a.position == (385.0, 251.0)
    assert result_b.position == (415.0, 251.0)
    assert replace(result_a, position=next_a.position, velocity=next_a.velocity) == next_a
    assert replace(result_b, position=next_b.position, velocity=next_b.velocity) == next_b


def test_equal_mass_response_does_not_prioritize_player_identity():
    a = PlayerState((384.0, 250.0), velocity=(120.0, 0.0))
    b = PlayerState((416.0, 250.0))
    forward = step_tick(contact_scene(a, b)).state
    swapped = step_tick(contact_scene(b, a)).state
    assert forward.player_a == swapped.player_b
    assert forward.player_b == swapped.player_a
    assert math.dist(forward.player_a.position, forward.player_b.position) >= 30.0 - 1e-10


def test_high_speed_cross_through_is_prevented():
    a = PlayerState((200.0, 250.0), velocity=(18000.0, 0.0))
    b = PlayerState((600.0, 250.0), velocity=(-18000.0, 0.0))
    state = step_tick(contact_scene(a, b)).state
    assert state.player_a.position[0] < state.player_b.position[0]
    assert math.dist(state.player_a.position, state.player_b.position) == pytest.approx(30)
    assert state.player_a.velocity == state.player_b.velocity == (0.0, 0.0)


def test_overlap_at_wall_separates_without_pushing_pinned_player_outside():
    a = PlayerState((15.0, 250.0))
    b = PlayerState((44.0, 250.0), velocity=(-120.0, 0.0))
    state = step_tick(contact_scene(a, b)).state
    assert state.player_a.position == (15.0, 250.0)
    assert state.player_b.position == pytest.approx((45.0, 250.0))
    assert state.player_a.velocity == state.player_b.velocity == (0.0, 0.0)


def test_initial_coincidence_uses_equal_deterministic_separation():
    a = PlayerState((400.0, 250.0))
    result = step_tick(contact_scene(a, a)).state
    assert result.player_a.position == (385.0, 250.0)
    assert result.player_b.position == (415.0, 250.0)
    assert step_tick(contact_scene(a, a)).state == result


def test_touching_players_moving_apart_are_not_slowed():
    a = PlayerState((385.0, 250.0), velocity=(-100.0, 0.0))
    b = PlayerState((415.0, 250.0), velocity=(100.0, 0.0))
    next_a = replace(a, position=(383.0, 250.0))
    next_b = replace(b, position=(417.0, 250.0))
    assert resolve_player_contact(a, b, next_a, next_b, pitch=PitchConfig()) == (next_a, next_b)


def test_no_contact_is_a_noop():
    a, b = reset().player_a, reset().player_b
    assert resolve_player_contact(a, b, a, b, pitch=PitchConfig()) == (a, b)


def test_seeded_extreme_contact_cases_stay_finite_separated_bounded_and_symmetric():
    rng = random.Random(481)
    for _ in range(300):
        a, b = (
            PlayerState(
                (rng.uniform(15, 785), rng.uniform(15, 485)),
                velocity=(rng.uniform(-18000, 18000), rng.uniform(-18000, 18000)),
            )
            for _ in range(2)
        )
        state = step_tick(contact_scene(a, b)).state
        reverse = step_tick(contact_scene(b, a)).state
        assert math.dist(state.player_a.position, state.player_b.position) >= 30.0 - 1e-8
        assert state.player_a.position == pytest.approx(reverse.player_b.position, abs=1e-8)
        assert state.player_b.position == pytest.approx(reverse.player_a.position, abs=1e-8)
        for player in (state.player_a, state.player_b):
            assert all(math.isfinite(value) for value in (*player.position, *player.velocity))
            assert 15 <= player.position[0] <= 785
            assert 15 <= player.position[1] <= 485


def test_different_wall_constraints_still_allow_separation_along_free_axes():
    a = PlayerState(
        (24.544186989030948, 356.5953810580439), velocity=(-7545.709063050343, 16720.229610760238)
    )
    b = PlayerState(
        (77.65874724483038, 454.73796975878116), velocity=(-13652.104866060487, 6636.592165980524)
    )
    state = step_tick(contact_scene(a, b)).state
    assert math.dist(state.player_a.position, state.player_b.position) >= 30.0 - 1e-8
    for player in (state.player_a, state.player_b):
        assert 15 <= player.position[0] <= 785
        assert 15 <= player.position[1] <= 485
