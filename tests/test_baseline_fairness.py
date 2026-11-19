"""B-P2: baseline fairness, symmetry, restart behaviour, contract defaults."""

import dataclasses

from agents.interface import (
    EV_GOAL,
    INTERFACE_VERSION,
    PHASE_RESTART,
    BallState,
    Event,
    GameState,
    PlayerState,
)
from agents.observation_view import FREE, OPP, SELF, make_view
from agents.rule_based import RuleBasedAgent


def _state(**kw):
    base = dict(
        players=[PlayerState(200, 250), PlayerState(600, 250)],
        ball=BallState(400, 250),
        possession=None,
        attack_dirs=(1, -1),
    )
    base.update(kw)
    return GameState(**base)


def test_view_is_attack_normalized():
    s = _state()
    v0, v1 = make_view(s, 0), make_view(s, 1)
    assert not v0.mirrored and v1.mirrored
    assert v0.me.x == v1.me.x == 200  # both see themselves on the left
    assert v0.opp.x == v1.opp.x == 600
    assert v0.possession == v1.possession == FREE


def test_view_possession_categories():
    assert make_view(_state(possession=0), 0).possession == SELF
    assert make_view(_state(possession=0), 1).possession == OPP


def test_mirror_symmetry_of_decisions():
    """Swapping sides (mirror x, swap players) mirrors the action in x only."""
    s = _state(possession=None, ball=BallState(350, 120))
    a0 = RuleBasedAgent(0).act(s)
    mirrored = GameState(
        players=[
            PlayerState(800 - 600, 250, facing=3.14159),
            PlayerState(800 - 200, 250, facing=3.14159),
        ],
        ball=BallState(800 - 350, 120),
        possession=None,
        attack_dirs=(-1, 1),
    )
    a1 = RuleBasedAgent(1).act(mirrored)  # index 1 now plays the left-side role
    assert abs(a0.move_x + a1.move_x) < 1e-6 and a0.move_x > 0
    assert abs(a0.move_y - a1.move_y) < 1e-6


def test_idle_during_restart():
    a = RuleBasedAgent(0).act(_state(phase=PHASE_RESTART))
    assert a.to_list() == [0.0] * 5


def test_stamina_hysteresis():
    bot = RuleBasedAgent(0)
    s = _state()
    s.players[0].stamina = 20
    assert bot.act(s).effort == 0.0  # tired: no sprint
    s.players[0].stamina = 40
    assert bot.act(s).effort == 0.0  # still resting (below resume)
    s.players[0].stamina = 60
    assert bot.act(s).effort == 1.0  # recovered


def test_agent_reads_no_hidden_fields():
    """The baseline must work from the View alone (no events, no timers)."""
    s = _state(events=[Event(EV_GOAL, 1, 5)], restart_owner=1)
    s2 = dataclasses.replace(s, events=[], restart_owner=None)
    assert RuleBasedAgent(0).act(s).to_list() == RuleBasedAgent(0).act(s2).to_list()


def test_contract_defaults_keep_old_code_working():
    s = GameState()
    assert s.events == [] and s.restart_ticks_remaining == 0 and s.restart_owner is None
    assert INTERFACE_VERSION == 2


# ---------------------------------------------------------------- P3: fuzz + presets
import math
import random

import pytest

from agents.base import HeldAgent
from agents.config import BaselineConfig
from agents.difficulty import PRESETS, make_baseline
from agents.metrics import is_shot
from tests._helpers import mirror_state, random_state


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_symmetry_fuzz(preset):
    """300 random states: the mirrored world gives the x-mirrored action (y, effort, kick same)."""
    rng = random.Random(1234)
    for _ in range(300):
        s = random_state(rng)
        for idx in (0, 1):
            a = RuleBasedAgent(idx, PRESETS[preset]).act(s)
            b = RuleBasedAgent(idx, PRESETS[preset]).act(mirror_state(s))
            assert abs(a.move_x + b.move_x) < 1e-6, (preset, idx)
            assert abs(a.move_y - b.move_y) < 1e-6
            assert (a.effort, a.kick, a.power) == (b.effort, b.kick, b.power)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_actions_always_legal_and_finite(preset):
    rng = random.Random(99)
    for _ in range(300):
        s = random_state(rng)
        for idx in (0, 1):
            v = RuleBasedAgent(idx, PRESETS[preset]).act(s).to_list()
            assert all(math.isfinite(x) for x in v) and len(v) == 5
            assert -1 <= v[0] <= 1 and -1 <= v[1] <= 1 and all(0 <= x <= 1 for x in v[2:])


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_kick_only_with_ball_and_on_target(preset):
    """The baseline never kicks without possession, and never kicks wide of the mouth."""
    rng = random.Random(7)
    kicks = 0
    for _ in range(600):
        s = random_state(rng)
        for idx in (0, 1):
            a = RuleBasedAgent(idx, PRESETS[preset]).act(s)
            if a.kick >= 0.5:
                kicks += 1
                assert s.possession == idx
                assert is_shot(s, idx)
    assert kicks > 0  # the check above is not vacuous


def test_presets_and_factory():
    assert set(PRESETS) == {"easy", "normal", "hard"}
    assert PRESETS["normal"] == BaselineConfig()
    assert PRESETS["easy"] != PRESETS["hard"]
    assert isinstance(make_baseline(0, "hard"), HeldAgent)
    assert isinstance(make_baseline(0, "hard", held=False), RuleBasedAgent)
    with pytest.raises(ValueError):
        make_baseline(0, "impossible")
