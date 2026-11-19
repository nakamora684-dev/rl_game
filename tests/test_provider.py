"""B's baseline as a native ActionProvider, checked against A's runner and C's SimulationEngine."""

import pytest

from agents.difficulty import make_baseline
from agents.engine_adapter import EngineAdapter
from agents.interface import EV_GOAL, EV_KICK, EV_OUT, EV_RECOVER, EV_STEAL
from agents.provider import RuleBasedProvider, rule_based_provider
from environment.contract import EventType, validate_state, validate_step_result
from environment.engine import SimulationEngine
from game.actions import PlayerAction
from game.player import PlayerId
from game.runner import run_match
from game.simulation import MatchPhase, reset


def test_provider_has_runner_signature_and_returns_player_actions():
    prov = rule_based_provider()
    state = reset()
    for pid in (PlayerId.A, PlayerId.B):
        assert isinstance(prov(state, pid), PlayerAction)
    with pytest.raises(ValueError):
        RuleBasedProvider("impossible")


def test_runner_plays_a_full_match_with_two_providers():
    result = run_match(rule_based_provider(), rule_based_provider())
    assert result.final_state.phase == MatchPhase.FINISHED
    assert result.physics_ticks == 3600 and result.decision_count == 600


def test_fresh_providers_are_deterministic_and_reset_restores_state():
    a = run_match(rule_based_provider(), rule_based_provider()).final_state
    b = run_match(rule_based_provider(), rule_based_provider()).final_state
    assert a == b
    prov = rule_based_provider()
    first = run_match(prov, prov).final_state
    prov.reset()
    assert run_match(prov, prov).final_state == first


def test_provider_matches_the_60hz_adapter_run_by_the_renderer():
    """What FootballEnv(opponent=...) sees == what play.py / observe.py show."""
    expected = run_match(rule_based_provider(), rule_based_provider()).final_state
    eng = EngineAdapter()
    s = eng.reset()
    bots = [make_baseline(0), make_baseline(1)]
    while s.tick < 3600:
        s = eng.step([b.act(s) for b in bots])
    assert eng.raw == expected


_KIND = {
    EV_GOAL: EventType.GOAL,
    EV_OUT: EventType.OUT_OF_BOUNDS,
    EV_KICK: EventType.KICK,
    EV_STEAL: EventType.STEAL,
    EV_RECOVER: EventType.LOOSE_BALL_RECOVERY,
}


def test_event_classification_equals_cs_simulation_engine():
    """B's tick-level adapter and C's decision-level SimulationEngine tell the same story."""
    for swap in (False, True):
        sim = SimulationEngine()
        mstate = sim.reset(swap_sides=swap)
        prov = [rule_based_provider(), rule_based_provider()]
        c_events = []
        while not mstate.is_over:
            acts = (prov[0](sim.native_state, PlayerId.A), prov[1](sim.native_state, PlayerId.B))
            res = sim.step(acts)
            validate_step_result(mstate, res)
            mstate = res.state
            c_events += [(e.type, e.player) for e in res.events]
        validate_state(mstate)

        eng = EngineAdapter()
        s = eng.reset(swap_sides=swap)
        bots = [make_baseline(0), make_baseline(1)]
        b_events = []
        while s.tick < 3600:
            s = eng.step([b.act(s) for b in bots])
            b_events += [(_KIND[e.kind], e.player) for e in s.events]
        # same events, same players; order inside one decision window can differ only by stamp
        assert sorted(map(str, b_events)) == sorted(map(str, c_events))
        assert s.score == mstate.score
        assert eng.raw == sim.native_state


def test_view_uses_the_names_of_cs_match_state():
    eng = EngineAdapter()
    s = eng.reset(swap_sides=True)
    from environment.contract import snapshot

    m = snapshot(eng.raw, eng.config)
    assert s.attack_dirs == m.attack_dirs and s.possession == m.possession
    assert s.steal_progress_ticks == m.steal_progress_ticks and s.last_toucher == m.last_toucher
    assert s.restart_ticks_remaining == m.restart_ticks_remaining
    assert (s.players[0].x, s.players[1].stamina) == (m.players[0].x, m.players[1].stamina)


@pytest.mark.environment
def test_provider_plugs_into_football_env_as_the_opponent():
    """Needs the `environment` dependency group (gymnasium + numpy); visible skip otherwise."""
    pytest.importorskip("gymnasium")
    np = pytest.importorskip("numpy")
    from environment.football_env import FootballEnv

    env = FootballEnv(opponent=rule_based_provider())
    env.reset(seed=1, options={"agent_idx": 0, "swap_sides": False})
    idle = {
        "direction": 0,
        "effort": np.zeros(1, np.float32),
        "kick_trigger": 0,
        "kick_power": np.zeros(1, np.float32),
    }
    steps, done = 0, False
    while not done:
        *_, terminated, truncated, _info = env.step(idle)
        steps, done = steps + 1, terminated or truncated
    assert steps == 600  # 3600 ticks / 6, restarts included
