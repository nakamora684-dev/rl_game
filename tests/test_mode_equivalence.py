"""Graphical vs headless = same simulation. The render hook must not change anything."""

import copy
import dataclasses

import pytest

from agents.base import HeldAgent
from agents.engine_adapter import EngineAdapter
from agents.interface import DECISION_EVERY, MATCH_TICKS, PHASE_FINISHED
from agents.match import run_match
from agents.rule_based import RuleBasedAgent


def _agents():
    return [HeldAgent(RuleBasedAgent(i), DECISION_EVERY) for i in (0, 1)]


def _trajectory(hook=None, swap=False, seed=7):
    traj = []

    def rec(st):
        traj.append(dataclasses.asdict(copy.deepcopy(st)))
        if hook:
            hook(st)

    final = run_match(EngineAdapter(), _agents(), seed=seed, swap_sides=swap, on_step=rec)
    return traj, final


@pytest.mark.parametrize("swap", [False, True])
def test_headless_equals_render_hook(swap):
    headless, _ = _trajectory(None, swap)
    drawn = []
    graphical, _ = _trajectory(lambda st: drawn.append(st.tick), swap)
    assert headless == graphical
    assert len(drawn) == len(graphical) == MATCH_TICKS


def test_same_seed_same_trajectory():
    a, _ = _trajectory(seed=5)
    b, _ = _trajectory(seed=5)
    assert a == b


def test_full_match_finishes_both_directions():
    for swap in (False, True):
        _, final = _trajectory(swap=swap)
        assert final.phase == PHASE_FINISHED and final.tick == MATCH_TICKS


def test_held_agent_decides_every_six_ticks():
    calls = []

    class Spy:
        def act(self, st):
            calls.append(st.tick)
            return "a"

    h = HeldAgent(Spy(), every=DECISION_EVERY)
    eng = EngineAdapter()
    s = eng.reset()
    for _ in range(18):
        h.act(s)
        s = eng.step([RuleBasedAgent(0).act(s), RuleBasedAgent(1).act(s)])
    assert calls == [0, 6, 12]
