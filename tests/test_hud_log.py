"""EventLog and PossessionMeter are pure Python (no display)."""

from agents.interface import EV_GOAL, EV_KICK, EV_STEAL, PHASE_PLAYING, PHASE_RESTART, Event
from rendering.hud import EventLog, PossessionMeter
from tests._helpers import make_state


def test_log_keeps_the_last_lines_with_clock_and_expires():
    log = EventLog(keep=3, ttl=1.0)
    lines = log.update([Event(EV_GOAL, 0, 10), Event(EV_STEAL, 1, 11)], 47.0, 0.0)
    assert lines == [("0:47  GOAL BLUE", 0), ("0:47  Steal RED", 1)]
    lines = log.update([Event(EV_KICK, 0, 12), Event(EV_KICK, 1, 12)], 46.0, 0.1)
    assert len(lines) == 3 and lines[-1][0].endswith("Kick RED")  # oldest line dropped
    assert log.update([], 40.0, 5.0) == []  # everything expired
    log.update([Event("unknown", 0, 1)], 30.0, 0.0)
    assert log.items == []


def test_possession_meter_counts_only_playing_ticks_with_an_owner():
    m = PossessionMeter()
    assert m.share() == 0.5
    for _ in range(30):
        m.update(make_state(possession=0))
    for _ in range(10):
        m.update(make_state(possession=1))
    m.update(make_state(possession=None))
    m.update(make_state(possession=0, phase=PHASE_RESTART))
    assert m.ticks == [30, 10] and m.share() == 0.75
    m.clear()
    assert m.share() == 0.5 and make_state().phase == PHASE_PLAYING
