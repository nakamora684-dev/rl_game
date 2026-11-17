from dataclasses import replace
from itertools import pairwise

import pytest

from game.actions import PlayerAction
from game.ball import BallState
from game.config import MATCH_TICKS, PossessionConfig, SimulationConfig
from game.player import PlayerId
from game.simulation import MatchPhase, assign_possession, reset, step_decision, step_tick


def contest(owner=PlayerId.A, distance=35.0):
    state = reset()
    state = replace(state, player_b=replace(state.player_b, position=(200.0 + distance, 250.0)))
    return assign_possession(state, owner).state


@pytest.mark.parametrize("owner", list(PlayerId))
def test_steal_requires_eighteen_consecutive_ticks_and_emits_one_distinct_transition(owner):
    state = contest(owner)
    new_owner = PlayerId.B if owner == PlayerId.A else PlayerId.A
    for tick in range(1, 18):
        result = step_tick(state)
        state = result.state
        assert state.steal_progress_ticks == tick
        assert state.ball.possession_owner == owner
        assert result.events == ()
    result = step_tick(state)
    state = result.state
    assert state.ball.possession_owner == state.ball.last_touch_player == new_owner
    assert state.steal_progress_ticks == 0
    assert state.possession_protection_ticks == 30
    (event,) = result.events
    assert (event.kind, event.cause, event.previous_owner, event.new_owner, event.tick) == (
        "possession_changed",
        "steal",
        owner,
        new_owner,
        18,
    )
    assert state.ball.velocity == (0.0, 0.0)
    assert step_decision(state).events == ()


def test_protection_blocks_progress_for_thirty_complete_following_ticks():
    state = contest()
    for _ in range(18):
        state = step_tick(state).state
    for remaining in range(29, -1, -1):
        result = step_tick(state)
        state = result.state
        assert state.possession_protection_ticks == remaining
        assert state.steal_progress_ticks == 0
        assert result.events == ()
    assert step_tick(state).state.steal_progress_ticks == 1


def test_recovery_starts_protection_without_a_steal():
    state = contest()
    state = replace(state, ball=BallState((200.0, 250.0)))
    result = step_tick(state)
    assert result.state.ball.possession_owner == PlayerId.A
    assert result.state.possession_protection_ticks == 30
    assert result.state.steal_progress_ticks == 0
    assert [event.cause for event in result.events] == ["recovery"]


def test_leaving_range_resets_not_pauses_contest():
    state = contest()
    state = replace(state, steal_progress_ticks=17)
    state = replace(state, player_b=replace(state.player_b, position=(235.001, 250.0)))
    result = step_tick(state)
    assert result.state.steal_progress_ticks == 0
    assert result.events == ()
    state = replace(result.state, player_b=replace(result.state.player_b, position=(235.0, 250.0)))
    assert step_tick(state).state.steal_progress_ticks == 1


def test_kick_interrupts_contest_and_avoids_same_tick_steal():
    state = replace(contest(), steal_progress_ticks=17)
    # Kick north to avoid an intentional loose-ball interception by the defender.
    state = replace(state, player_a=replace(state.player_a, facing=-1.5707963267948966))
    result = step_tick(state, PlayerAction(kick_trigger=1, kick_power=1.0))
    assert [event.kind for event in result.events] == ["ball_kicked"]
    assert result.state.steal_progress_ticks == result.state.possession_protection_ticks == 0


def test_setup_owner_change_clears_contest_but_same_owner_assignment_is_idempotent():
    state = replace(contest(), steal_progress_ticks=10, possession_protection_ticks=3)
    assert assign_possession(state, PlayerId.A).state is state
    changed = assign_possession(state, PlayerId.B).state
    assert changed.steal_progress_ticks == changed.possession_protection_ticks == 0


def test_restart_interrupts_contest_and_advances_protection_only():
    state = replace(
        contest(),
        steal_progress_ticks=17,
        possession_protection_ticks=20,
        phase=MatchPhase.GOAL_RESTART,
        restart_ticks_remaining=2,
    )
    result = step_tick(state)
    assert result.state.steal_progress_ticks == 0
    assert result.state.possession_protection_ticks == 19
    assert result.state.ball == state.ball
    assert result.events == ()


def test_steal_and_whistle_are_both_emitted():
    state = replace(contest(), simulation_tick=MATCH_TICKS - 1, steal_progress_ticks=17)
    result = step_tick(state)
    assert [event.cause for event in result.events] == ["steal", None]
    assert result.state.phase == MatchPhase.FINISHED
    assert step_tick(result.state).state is result.state


def test_custom_contest_and_protection_settings():
    config = SimulationConfig(
        possession=PossessionConfig(steal_contest_ticks=1, protection_ticks=0)
    )
    result = step_tick(contest(), config=config)
    assert result.events[0].cause == "steal"
    assert result.state.possession_protection_ticks == 0


@pytest.mark.parametrize(
    "fields",
    [
        {"steal_distance": 0},
        {"steal_distance": float("nan")},
        {"steal_contest_ticks": 0},
        {"steal_contest_ticks": True},
        {"protection_ticks": -1},
        {"protection_ticks": 0.5},
    ],
)
def test_invalid_possession_config(fields):
    with pytest.raises(ValueError):
        PossessionConfig(**fields)


def test_free_ball_cannot_have_contest_or_protection_and_reset_clears_them():
    for field in ("steal_progress_ticks", "possession_protection_ticks"):
        with pytest.raises(ValueError):
            replace(reset(), **{field: 1})
        with pytest.raises(ValueError):
            replace(contest(), **{field: -1})
    assert reset().steal_progress_ticks == reset().possession_protection_ticks == 0


def test_full_contact_match_replays_with_one_event_per_spaced_steal():
    def trajectory():
        state = contest()
        snapshots, events = [], []
        for _ in range(600):
            result = step_decision(state)
            state = result.state
            snapshots.append(state)
            events.extend(result.events)
            assert state.ball.possession_owner is not None
            assert state.steal_progress_ticks >= 0
            assert state.possession_protection_ticks >= 0
        steals = [event for event in events if event.cause == "steal"]
        assert len(steals) == 75
        assert [event.tick for event in steals[:2]] == [18, 66]
        assert all(second.tick - first.tick == 48 for first, second in pairwise(steals))
        assert events[-1].kind == "match_finished"
        return snapshots, events

    assert trajectory() == trajectory()
