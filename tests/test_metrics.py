"""Unit tests of agents/metrics.py and the CSV schema (no engine needed)."""

import csv
import math

from agents.baseline_stats import summarize, write_csv
from agents.interface import (
    EV_GOAL,
    EV_KICK,
    EV_OUT,
    EV_RECOVER,
    EV_STEAL,
    MATCH_TICKS,
    PHASE_FINISHED,
    PHASE_RESTART,
    Event,
)
from agents.metrics import ROW_FIELDS, MatchRecorder, is_shot
from tests._helpers import make_state


def test_is_shot_geometry_both_directions():
    assert is_shot(make_state(p0=(600, 250), f0=0.0), 0)  # straight at the mouth
    assert not is_shot(make_state(p0=(600, 250), f0=math.radians(40)), 0)  # wide
    assert not is_shot(make_state(p0=(600, 250), f0=math.pi), 0)  # backwards
    left = make_state(p0=(200, 250), f0=math.pi, attack_dirs=(-1, 1))  # attacks toward -x
    assert is_shot(left, 0)
    assert not is_shot(make_state(p0=(200, 250), f0=0.0, attack_dirs=(-1, 1)), 0)


def test_recorder_counts_events_once_and_flags_duplicates():
    rec = MatchRecorder()
    s = make_state(p0=(600, 250), f0=0.0)
    s.events = [
        Event(EV_KICK, 0, 10),
        Event(EV_RECOVER, 1, 10),
        Event(EV_STEAL, 0, 11),
        Event(EV_OUT, 1, 12),
        Event(EV_GOAL, 0, 13),
    ]
    rec.on_step(s)
    assert (rec.kicks[0], rec.shots[0], rec.recoveries[1], rec.steals[0]) == (1, 1, 1, 1)
    assert (rec.oob[1], rec.goals[0]) == (1, 1)
    rec.on_step(s)  # SAME events again = engine bug
    assert rec.duplicates == 5 and rec.goals[0] == 1


def test_possession_attack_half_and_restart_exclusion():
    rec = MatchRecorder()
    for _ in range(3):  # ball in p0's attacking half, p0 owns it
        rec.on_step(make_state(ball=(700, 250), possession=0))
    rec.on_step(make_state(ball=(100, 250), possession=None))
    rec.on_step(make_state(ball=(100, 250), possession=1, phase=PHASE_RESTART))  # ignored
    assert rec.playing == 4
    assert rec.poss == [3, 0] and rec.attack == [
        3,
        1,
    ]  # p1 attacks toward -x: ball at 100 counts for p1 too


def test_idle_fraction_uses_displacement():
    rec = MatchRecorder()
    s = make_state()
    rec.on_step(s)
    rec.on_step(make_state(p0=(210, 250)))  # p0 moved, p1 did not
    assert rec.idle == [0, 1]


def test_summary_matches_schema_and_csv_roundtrip(tmp_path):
    rec = MatchRecorder()
    s = make_state(score=(2, 1), tick=MATCH_TICKS, phase=PHASE_FINISHED)
    rec.on_step(s)
    row = rec.summary(s, seed=3, swapped=True, names=("normal", "hard"))
    assert tuple(row) == ROW_FIELDS  # stable column order = stable CSV schema
    assert row["winner"] == "p0" and row["goal_diff_p0"] == 1 and row["swapped"] == 1
    path = write_csv([row, row], tmp_path / "x.csv")
    with open(path) as f:
        got = list(csv.DictReader(f))
    assert len(got) == 2 and list(got[0]) == list(ROW_FIELDS)
    assert "W/D/L = 2/0/0" in summarize([row, row])
