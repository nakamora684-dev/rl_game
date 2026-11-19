"""Renderer/HUD are read-only. Uses a FAKE pygame so no display or install is needed."""

import copy
import dataclasses
import sys
from unittest.mock import MagicMock

import pytest

from agents.engine_adapter import EngineAdapter
from agents.interface import EV_GOAL, EV_STEAL, PHASE_FINISHED, PHASE_RESTART, Event
from rendering.hud import FlashTracker, result_text


class _Img:
    def get_width(self):
        return 40

    def get_height(self):
        return 12


def _fake_pygame():
    pg = MagicMock()
    font = MagicMock()
    font.render.return_value = _Img()
    pg.font.SysFont.return_value = font
    screen = MagicMock()
    screen.get_height.return_value = 600
    pg.display.set_mode.return_value = screen
    return pg


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygame", _fake_pygame())
    from rendering.renderer import Renderer

    return Renderer()


def _states():
    s = EngineAdapter().reset()
    s.events = [Event(EV_GOAL, 0, 1), Event(EV_STEAL, 1, 1)]
    s.steal_progress_ticks, s.possession = 5, 0
    restart = copy.deepcopy(s)
    restart.phase, restart.restart_owner, restart.restart_ticks_remaining = PHASE_RESTART, 1, 90
    done = copy.deepcopy(s)
    done.phase, done.score = PHASE_FINISHED, (2, 1)
    return [s, restart, done]


@pytest.mark.parametrize("debug", [False, True])
def test_render_does_not_mutate_state(renderer, debug):
    renderer.debug = debug
    for st in _states():
        before = dataclasses.asdict(copy.deepcopy(st))
        renderer.render(st, charge=0.5)
        assert dataclasses.asdict(st) == before


def test_flash_tracker_events_show_once_then_expire():
    f = FlashTracker()
    assert f.update([Event(EV_GOAL, 0, 1)], 1 / 60) == [("GOAL!", 0)]
    assert len(f.update([], 1 / 60)) == 1  # still visible, NOT duplicated
    for _ in range(100):
        out = f.update([], 1 / 60)
    assert out == []  # expired after ~1.4 s


def test_result_text():
    s = EngineAdapter().reset()
    s.score = (1, 1)
    assert result_text(s) == "DRAW"
    s.score = (2, 1)
    assert result_text(s) == "BLUE WINS"
    s.score = (0, 3)
    assert result_text(s) == "RED WINS"
