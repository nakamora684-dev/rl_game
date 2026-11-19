"""Pure parts of the controls: bindings file, gamepad dead zone (no pygame)."""

import json
import math

import pytest

from rendering.controls import DEFAULT_BINDINGS, axes_to_move, load_bindings


def test_default_bindings_cover_every_action():
    b = load_bindings()
    assert set(b) == {"left", "right", "up", "down", "sprint", "kick"}
    assert b is not DEFAULT_BINDINGS and b == DEFAULT_BINDINGS
    b["kick"].append("x")  # editing a copy must not leak
    assert "x" not in DEFAULT_BINDINGS["kick"]


def test_bindings_file_overrides_only_the_given_actions(tmp_path):
    p = tmp_path / "keys.json"
    p.write_text(json.dumps({"sprint": ["left ctrl"], "kick": ["space", "return"]}))
    b = load_bindings(p)
    assert b["sprint"] == ["left ctrl"] and b["kick"] == ["space", "return"]
    assert b["left"] == DEFAULT_BINDINGS["left"]


def test_bindings_file_is_validated(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"dance": ["x"]}))
    with pytest.raises(ValueError):
        load_bindings(p)
    p.write_text(json.dumps({"kick": []}))
    with pytest.raises(ValueError):
        load_bindings(p)


def test_axes_dead_zone_and_scaling():
    assert axes_to_move(0.1, 0.1) == (0.0, 0.0)  # inside the dead zone
    x, y = axes_to_move(1.0, 0.0)
    assert math.isclose(x, 1.0) and y == 0.0
    x, y = axes_to_move(1.0, 1.0)  # diagonal: clamped to unit length
    assert math.isclose(math.hypot(x, y), 1.0)
    x, _ = axes_to_move(0.6, 0.0)
    assert 0.0 < x < 0.6  # rescaled below raw value
    assert axes_to_move(-1.0, 0.0)[0] < 0
