"""P3 renderer + screens with a FAKE pygame (no display, no install)."""

import copy
import dataclasses
import sys

import pytest

from agents.engine_adapter import EngineAdapter
from agents.interface import EV_GOAL, PHASE_FINISHED, Event
from rendering.flow import Flow
from tests.test_render_readonly import _fake_pygame


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygame", _fake_pygame())
    from rendering.renderer import Renderer

    return Renderer()


def _state_with_goal():
    s = EngineAdapter().reset()
    s.events = [Event(EV_GOAL, 0, 1)]
    return s


def test_background_is_built_once_not_per_frame(renderer):
    assert renderer.pg.Surface.call_count >= 1
    calls = renderer.pg.Surface.call_count
    for _ in range(3):
        renderer.render(EngineAdapter().reset())
    assert renderer.pg.Surface.call_count == calls  # no new surfaces while playing


def test_goal_event_creates_effects_and_flash_once(renderer):
    st = _state_with_goal()
    renderer.render(st)
    assert len(renderer.fx.particles.items) > 0
    assert [t for t, _ in renderer.flash.update([], 0.0)] == ["GOAL!"]


def test_freeze_does_not_consume_events_or_change_state(renderer):
    st = _state_with_goal()
    before = dataclasses.asdict(copy.deepcopy(st))
    for _ in range(5):
        renderer.render(st, freeze=True)  # paused: same state drawn again
    assert dataclasses.asdict(st) == before
    assert renderer.fx.particles.items == [] and renderer.flash.items == []


def test_reset_effects_clears_everything(renderer):
    renderer.render(_state_with_goal())
    renderer.reset_effects()
    assert (
        renderer.fx.particles.items == [] and renderer.flash.items == [] and renderer.off == (0, 0)
    )


def test_overlay_is_called_before_flip(renderer):
    order = []
    renderer.pg.display.flip.side_effect = lambda: order.append("flip")
    renderer.render(EngineAdapter().reset(), overlay=lambda r: order.append("overlay"))
    assert order == ["overlay", "flip"]


def test_screens_draw_without_error_and_do_not_touch_state(renderer):
    from rendering.flow import MatchOptions
    from rendering.screens import Screens
    from rendering.session import Session

    s = Session(EngineAdapter(), MatchOptions())
    s.state.phase = PHASE_FINISHED
    row = s.summary_row()
    before = dataclasses.asdict(copy.deepcopy(s.state))
    sc, flow = Screens(renderer), Flow()
    sc.menu(flow)
    sc.pause(flow)
    sc.fulltime(s.state, row, replay_path="x.json")
    assert dataclasses.asdict(s.state) == before


def test_screenshot_saves_through_pygame_image(renderer, tmp_path):
    path = renderer.screenshot(tmp_path / "a" / "shot.png")
    renderer.pg.image.save.assert_called_once()
    assert path.parent.exists()


def test_debug_overlay_toggle_still_works(renderer):
    renderer.toggle_debug()
    assert renderer.debug is True
    renderer.render(EngineAdapter().reset())
