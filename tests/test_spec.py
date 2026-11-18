"""Exact hybrid controls; missing optional dependencies are visible skips."""

import math

import pytest

np = pytest.importorskip("numpy", reason="Install the locked environment dependency group.")
pytest.importorskip("gymnasium", reason="Install the locked environment dependency group.")

from environment import spec
from game.actions import Direction, PlayerAction
from game.config import MATCH_TICKS, PHYSICS_HZ, TICKS_PER_DECISION

pytestmark = pytest.mark.environment


def action(direction=0, effort=0.0, kick=0, power=0.0):
    return {
        "direction": int(direction),
        "effort": np.array([effort], dtype=np.float32),
        "kick_trigger": kick,
        "kick_power": np.array([power], dtype=np.float32),
    }


def test_clock_and_dimensions():
    assert spec.PHYSICS_HZ == PHYSICS_HZ == 60
    assert spec.FRAMES_PER_DECISION == TICKS_PER_DECISION == 6
    assert spec.MAX_PHYSICS_TICKS == MATCH_TICKS == 3600
    assert spec.MAX_DECISIONS == 600
    assert spec.ACTION_DIM == 4 and spec.OBS_DIM == 50
    low, high = spec.obs_bounds()
    assert low.shape == high.shape == (50,)
    assert np.all(low < high)


def test_spaces_and_sampled_actions_decode_without_discretizing_effort_or_power():
    space = spec.make_action_space()
    space.seed(12)
    assert space["direction"].n == 9 and space["kick_trigger"].n == 2
    for _ in range(100):
        raw = space.sample()
        result = spec.decode_action(raw)
        assert isinstance(result, PlayerAction)
        assert result.direction == Direction(raw["direction"])
        assert result.effort == float(raw["effort"][0])
        assert result.kick_power == float(raw["kick_power"][0])
        assert result.kick_trigger == raw["kick_trigger"]


@pytest.mark.parametrize("direction", list(Direction))
def test_horizontal_mirror_matches_direction_vectors_and_is_involutive(direction):
    original = spec.decode_action(action(direction, 0.25, 1, 0.75))
    mirrored = spec.policy_to_world(original, -1)
    x, y = direction.vector
    assert mirrored.direction.vector == (-x, y)
    assert (mirrored.effort, mirrored.kick_trigger, mirrored.kick_power) == (0.25, 1, 0.75)
    assert spec.policy_to_world(mirrored, -1) == original
    assert spec.policy_to_world(original, 1) == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("direction", 2.5),
        ("direction", -1),
        ("direction", 9),
        ("direction", True),
        ("kick_trigger", 0.6),
        ("kick_trigger", 2),
        ("effort", [float("nan")]),
        ("effort", [float("inf")]),
        ("effort", [-0.01]),
        ("kick_power", [1.01]),
        ("effort", 0.5),
        ("effort", [0.2, 0.4]),
    ],
)
def test_invalid_actions_are_rejected_not_clipped_or_rounded(field, value):
    raw = action()
    raw[field] = value
    with pytest.raises((TypeError, ValueError)):
        spec.decode_action(raw)


def test_old_five_continuous_values_and_wrong_keys_are_rejected():
    with pytest.raises(ValueError):
        spec.decode_action(np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError):
        spec.decode_action({"direction": 0})


def test_idle_zero_power_kick_and_symmetry_helpers():
    assert spec.decode_action(action()) == PlayerAction()
    assert spec.decode_action(action(kick=1)).kick_trigger == 1
    assert spec.decode_action(action(kick=1)).kick_power == 0
    assert spec.to_policy_x(200, -1) == 600
    assert spec.to_policy_angle(0, -1) == math.pi
