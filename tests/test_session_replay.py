"""Session (play.py without pygame), replay record/playback, full-time rows."""

import copy
import dataclasses
import json
import random

import pytest

from agents.engine_adapter import EngineAdapter
from agents.interface import MATCH_TICKS, Action
from agents.match import run_match
from rendering.flow import MatchOptions
from rendering.replay import REPLAY_VERSION, ReplayAgent, load_replay
from rendering.screens import fulltime_rows
from rendering.session import Session


def _scripted_human(seed=0):
    """Deterministic 'human': random but reproducible inputs."""
    rng = random.Random(seed)

    def act(t):
        if t % 90 == 0:
            act.cur = Action(
                rng.choice([-1, 0, 1]),
                rng.choice([-1, 0, 1]),
                rng.choice([0, 1]),
                1.0 if rng.random() < 0.3 else 0.0,
                rng.random(),
            )
        return act.cur

    act.cur = Action()
    return act


def _play_session(options, human):
    s = Session(EngineAdapter(), options)
    traj = []
    t = 0
    while not s.finished:
        s.tick(human(t))
        traj.append(dataclasses.asdict(copy.deepcopy(s.state)))
        t += 1
    return s, traj


@pytest.mark.parametrize("swap", [False, True])
def test_session_plays_a_full_match_and_reports_stats(swap):
    s, traj = _play_session(MatchOptions("normal", swap, 3), _scripted_human())
    assert len(traj) == MATCH_TICKS and s.finished
    row = s.summary_row()
    assert row["ticks"] == MATCH_TICKS and row["duplicate_events"] == 0
    assert row["p0_agent"] == "human" and row["p1_agent"] == "bot-normal"
    assert row["swapped"] == int(swap)
    assert (row["p0_goals"], row["p1_goals"]) == (row["score_p0"], row["score_p1"])
    assert row["p0_sprint_frac"] > 0  # human sprint input was counted


def test_tick_after_the_final_whistle_does_nothing():
    s, _ = _play_session(MatchOptions(), _scripted_human())
    t = s.state.tick
    s.tick(Action(1, 0, 0, 0, 0))
    assert s.state.tick == t


def test_replay_reproduces_the_exact_match(tmp_path):
    opts = MatchOptions("hard", True, 9)
    s, traj = _play_session(opts, _scripted_human(5))
    path = s.save_replay(tmp_path)
    data = load_replay(path)
    assert data["seed"] == 9 and data["swap_sides"] is True and len(data["actions"]) == MATCH_TICKS
    again = []
    final = run_match(
        EngineAdapter(),
        [ReplayAgent(data, 0), ReplayAgent(data, 1)],
        seed=data["seed"],
        swap_sides=data["swap_sides"],
        on_step=lambda st: again.append(dataclasses.asdict(copy.deepcopy(st))),
    )
    assert again == traj  # bit-exact
    assert final.score == s.state.score


def test_replay_version_is_checked(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"version": REPLAY_VERSION + 1}))
    with pytest.raises(ValueError):
        load_replay(p)


def test_replay_agent_pads_with_idle_when_actions_run_out():
    a = ReplayAgent({"actions": [[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]]}, 0)
    assert a.act(None).move_x == 1.0
    assert a.act(None).to_list() == [0.0] * 5
    a.reset()
    assert a.act(None).move_x == 1.0


def test_fulltime_rows_format():
    s, _ = _play_session(MatchOptions(), _scripted_human())
    rows = fulltime_rows(s.summary_row())
    assert [r[0] for r in rows] == [
        "Shots",
        "Steals",
        "Recoveries",
        "Possession",
        "Time sprinting",
        "Stamina left",
    ]
    assert all(isinstance(a, str) and isinstance(b, str) for _, a, b in rows)
    poss = rows[3]
    assert poss[1].endswith("%") and poss[2].endswith("%")
