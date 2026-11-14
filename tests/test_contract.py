"""
tests/test_contract.py -- Vérifie le contrat moteur <-> wrapper avec le faux moteur.
Lancer :  python tests/test_contract.py     (ou : python -m pytest tests/test_contract.py)

Quand le Member A aura fini son moteur, il suffira de remplacer FakeEngine par le vrai moteur
dans `make_engine()` : les mêmes tests s'appliqueront.
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # racine du dépôt
sys.path.insert(0, os.path.dirname(__file__))                      # dossier tests/ (fake_engine)

from environment import spec
from environment.contract import (
    Event,
    EventType,
    GameEngine,
    Phase,
    StepResult,
    validate_state,
    validate_step_result,
)
from fake_engine import FakeEngine


def make_engine():
    return FakeEngine()   # <- à remplacer par le vrai moteur du Member A


IDLE = spec.decode_action([0, 0, -1, -1, -1])   # action neutre
ACTIONS = (IDLE, IDLE)


def test_engine_respects_protocol():
    assert isinstance(make_engine(), GameEngine)


def test_reset_is_valid_and_deterministic():
    e = make_engine()
    s1 = e.reset(seed=0)
    s2 = e.reset(seed=0)
    validate_state(s1)
    assert s1 == s2                              # même seed => même état
    assert s1.phase is Phase.PLAYING and s1.tick == 0 and s1.score == (0, 0)


def test_attack_dirs_are_kept():
    e = make_engine()
    assert e.reset(attack_dirs=(-1, 1)).attack_dirs == (-1, 1)


def test_full_match_runs_600_decisions():
    e = make_engine()
    state = e.reset(seed=0)
    steps = 0
    all_events = []
    while not state.is_over:
        result = e.step(ACTIONS)                 # 6 ticks par décision
        validate_step_result(state, result)      # horloge, fenêtre d'événements, score/GOAL
        all_events.extend(result.events)
        state = result.state
        steps += 1
    assert steps == spec.MAX_DECISIONS           # 600
    assert state.tick == spec.MAX_PHYSICS_TICKS  # 3600
    assert state.phase is Phase.ENDED


def test_each_scripted_event_occurs_exactly_once():
    e = make_engine()
    state = e.reset()
    events = []
    while not state.is_over:
        r = e.step(ACTIONS)
        events.extend(r.events)
        state = r.state
    counts = {t: sum(1 for ev in events if ev.type is t) for t in EventType}
    assert counts[EventType.GOAL] == 1
    assert counts[EventType.STEAL] == 1
    assert counts[EventType.LOOSE_BALL_RECOVERY] == 1
    assert state.score == (0, 1)


def test_event_in_middle_of_decision_window_is_not_lost():
    # L'événement du tick 10 tombe au milieu de la 2e décision (ticks 7 à 12).
    e = make_engine()
    e.reset()
    e.step(ACTIONS)                      # ticks 1-6
    r = e.step(ACTIONS)                  # ticks 7-12
    assert [ev.tick for ev in r.events] == [10]


def test_validate_step_result_catches_bad_clock():
    e = make_engine()
    s0 = e.reset()
    r = e.step(ACTIONS)
    bad = StepResult(state=r.state, events=r.events, ticks_advanced=r.ticks_advanced + 1)
    try:
        validate_step_result(s0, bad)
    except ValueError:
        return
    raise AssertionError("une horloge incohérente aurait dû être détectée")


def test_validate_step_result_catches_goal_mismatch():
    # Un événement GOAL sans variation de score doit être rejeté (protège contre les doubles récompenses).
    e = make_engine()
    s0 = e.reset()
    r = e.step(ACTIONS)
    fake_goal = Event(EventType.GOAL, tick=3, player=0)
    bad = StepResult(state=r.state, events=r.events + (fake_goal,), ticks_advanced=r.ticks_advanced)
    try:
        validate_step_result(s0, bad)
    except ValueError:
        return
    raise AssertionError("un GOAL sans changement de score aurait dû être détecté")


def test_validate_state_catches_invalid_states():
    s = make_engine().reset()
    for bad in (
        replace(s, possession=2),
        replace(s, attack_dirs=(1, 1)),
        replace(s, score=(-1, 0)),
        replace(s, players=(replace(s.players[0], stamina=150.0), s.players[1])),
        replace(s, players=(replace(s.players[0], x=float("nan")), s.players[1])),
    ):
        try:
            validate_state(bad)
        except ValueError:
            continue
        raise AssertionError(f"état invalide non détecté : {bad}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
