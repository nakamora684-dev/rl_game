import math
from dataclasses import replace

import pytest

from game.actions import PlayerAction
from game.ball import BallState
from game.config import MATCH_TICKS, BallConfig, RulesConfig, SimulationConfig
from game.player import PlayerId
from game.rules import BoundarySide, detect_ball_exit
from game.simulation import MatchPhase, assign_possession, reset, step_decision, step_tick


def shot(position=(804.0, 250.0), velocity=(300.0, 0.0), toucher=PlayerId.A, swap=False):
    state = reset(swap_sides=swap)
    return replace(state, ball=BallState(position, velocity, last_touch_player=toucher))


@pytest.mark.parametrize(
    "y,goal", [(196.999, False), (197.0, True), (250.0, True), (303.0, True), (303.001, False)]
)
def test_entire_ball_must_fit_centred_goal_opening(y, goal):
    crossing = detect_ball_exit((804.0, y), (810.0, y), config=SimulationConfig())
    assert crossing.is_goal is goal
    assert crossing.side == BoundarySide.RIGHT
    assert crossing.fraction == pytest.approx(0.5)


def test_strict_whole_ball_clearance_not_touching_threshold():
    config = SimulationConfig()
    assert detect_ball_exit((800.0, 250.0), (807.0, 250.0), config=config) is None
    assert detect_ball_exit((807.0, 250.0), (807.0, 250.0), config=config) is None
    assert detect_ball_exit((807.0, 250.0), (808.0, 250.0), config=config).is_goal


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("side", [BoundarySide.LEFT, BoundarySide.RIGHT])
def test_goals_score_by_attacking_side_not_last_toucher_and_emit_once(swap, side):
    left = side == BoundarySide.LEFT
    state = shot(
        (-4.0, 250.0) if left else (804.0, 250.0),
        (-300.0, 0.0) if left else (300.0, 0.0),
        toucher=PlayerId.B,
        swap=swap,
    )
    scorer = PlayerId.A if state.attack_direction_a == (-1 if left else 1) else PlayerId.B
    receiver = PlayerId.B if scorer == PlayerId.A else PlayerId.A
    result = step_tick(state)
    state = result.state
    assert state.phase == MatchPhase.GOAL_RESTART
    assert state.restart_ticks_remaining == 60
    assert (state.score_a, state.score_b) == ((1, 0) if scorer == PlayerId.A else (0, 1))
    (event,) = result.events
    assert (event.kind, event.scoring_player, event.last_touch_player, event.restart_owner) == (
        "goal",
        scorer,
        PlayerId.B,
        receiver,
    )
    assert state.ball.possession_owner is None
    assert state.ball.velocity == (0.0, 0.0)
    assert step_decision(state).events == ()


def test_goal_restart_lasts_sixty_frozen_ticks_then_conceder_controls_centre_ball():
    initial = shot()
    initial = replace(
        initial,
        player_a=replace(initial.player_a, stamina=37.0),
        player_b=replace(initial.player_b, stamina=42.0),
    )
    state = step_tick(initial).state
    frozen_stamina = (state.player_a.stamina, state.player_b.stamina)
    frozen_a, frozen_b = state.player_a.position, state.player_b.position
    action = PlayerAction(kick_trigger=1, kick_power=1.0)
    for _ in range(59):
        result = step_tick(state, action, action)
        state = result.state
        assert state.phase == MatchPhase.GOAL_RESTART
        assert state.player_a.position == frozen_a
        assert state.player_b.position == frozen_b
        assert result.events == ()
    result = step_tick(state, action, action)
    state = result.state
    assert state.simulation_tick == 61
    assert state.phase == MatchPhase.PLAYING
    assert state.restart_plan is None
    assert state.ball.possession_owner == PlayerId.B
    assert state.ball.position == pytest.approx((400.0, 250.0))
    assert state.player_a.position == (200.0, 250.0)
    assert state.player_b.position == pytest.approx((422.0, 250.0))
    assert (state.player_a.stamina, state.player_b.stamina) == frozen_stamina
    assert state.player_a.velocity == state.player_b.velocity == (0.0, 0.0)
    assert state.possession_protection_ticks == 30
    assert [event.cause for event in result.events] == ["kickoff"]
    assert (state.score_a, state.score_b) == (1, 0)


@pytest.mark.parametrize(
    "position,velocity,side",
    [
        ((400.0, -4.0), (0.0, -300.0), BoundarySide.TOP),
        ((400.0, 504.0), (0.0, 300.0), BoundarySide.BOTTOM),
        ((-4.0, 100.0), (-300.0, 0.0), BoundarySide.LEFT),
        ((804.0, 100.0), (300.0, 0.0), BoundarySide.RIGHT),
    ],
)
def test_out_of_bounds_blames_last_touch_and_gives_opponent_legal_inward_restart(
    position, velocity, side
):
    state = step_tick(shot(position, velocity, toucher=PlayerId.A)).state
    assert state.phase == MatchPhase.OUT_OF_BOUNDS_RESTART
    assert state.restart_plan.owner == PlayerId.B
    assert state.restart_plan.exit.side == side
    assert (state.score_a, state.score_b) == (0, 0)
    for _ in range(60):
        result = step_tick(state)
        state = result.state
    assert state.phase == MatchPhase.PLAYING
    assert state.ball.possession_owner == PlayerId.B
    assert state.player_a.position == (200.0, 250.0)
    assert 7 <= state.ball.position[0] <= 793
    assert 7 <= state.ball.position[1] <= 493
    assert [event.cause for event in result.events] == ["boundary_restart"]
    inward = {
        BoundarySide.TOP: (0, 1),
        BoundarySide.BOTTOM: (0, -1),
        BoundarySide.LEFT: (1, 0),
        BoundarySide.RIGHT: (-1, 0),
    }[side]
    forward = (math.cos(state.player_b.facing), math.sin(state.player_b.facing))
    assert forward == pytest.approx(inward)


def test_unknown_last_touch_restarts_neutrally_without_assigning_blame():
    state = step_tick(shot((804.0, 100.0), toucher=None)).state
    assert state.restart_plan.owner is None
    for _ in range(60):
        result = step_tick(state)
        state = result.state
    assert state.ball == reset().ball
    assert state.player_a == reset().player_a
    assert state.player_b == reset().player_b
    assert [event.kind for event in result.events] == ["restart_completed"]


def test_fast_goal_cannot_tunnel_and_earliest_boundary_exit_wins():
    result = step_tick(shot((790.0, 250.0), (120000.0, 0.0)))
    assert result.events[0].kind == "goal"
    crossing = detect_ball_exit((790.0, 490.0), (1000.0, 1000.0), config=SimulationConfig())
    assert crossing.side == BoundarySide.BOTTOM
    assert not crossing.is_goal


def test_pickup_before_exit_blocks_goal_but_contact_after_exit_cannot_rescue_ball():
    initial = shot()
    initial = replace(initial, player_b=replace(initial.player_b, position=(785.0, 250.0)))
    first = step_tick(initial)
    assert [event.cause for event in first.events] == ["recovery"]
    assert first.state.score_a == 0
    assert first.state.ball.possession_owner == PlayerId.B
    # Sweep a defender from far away so contact would occur only AFTER full clearance.
    initial = shot((804.0, 250.0), (12000.0, 0.0))
    initial = replace(
        initial,
        player_b=replace(initial.player_b, position=(785.0, 400.0), velocity=(0.0, -9000.0)),
    )
    result = step_tick(initial)
    assert [event.kind for event in result.events] == ["goal"]


def test_exact_pickup_at_clearance_threshold_prevents_strict_crossing():
    initial = shot((807.0, 250.0))
    initial = replace(initial, player_b=replace(initial.player_b, position=(785.0, 250.0)))
    result = step_tick(initial)
    assert result.state.score_a == 0
    assert result.state.ball.possession_owner == PlayerId.B


def test_goal_and_whistle_are_both_kept_and_match_can_end_during_restart():
    result = step_tick(replace(shot(), simulation_tick=MATCH_TICKS - 1))
    assert [event.kind for event in result.events] == ["goal", "match_finished"]
    assert result.state.score_a == 1
    assert result.state.phase == MatchPhase.FINISHED
    assert result.state.restart_plan is None
    assert step_tick(result.state).events == ()
    state = replace(shot(), simulation_tick=MATCH_TICKS - 5)
    result = step_decision(state)
    assert result.physics_ticks == 5
    assert [event.kind for event in result.events] == ["goal", "match_finished"]
    assert result.state.phase == MatchPhase.FINISHED


def test_custom_restart_duration_and_reset_clear_scores_plans_timers():
    config = SimulationConfig(rules=RulesConfig(restart_ticks=2))
    state = step_tick(shot(), config=config).state
    state = step_decision(state, config=config).state
    assert state.phase == MatchPhase.PLAYING
    assert state.score_a == 1
    fresh = reset()
    assert fresh.score_a == fresh.score_b == 0
    assert fresh.restart_plan is None
    assert fresh.steal_progress_ticks == fresh.possession_protection_ticks == 0


@pytest.mark.parametrize(
    "fields",
    [{"goal_width": 0}, {"goal_width": math.nan}, {"restart_ticks": 0}, {"restart_ticks": 0.5}],
)
def test_invalid_rule_config(fields):
    with pytest.raises(ValueError):
        RulesConfig(**fields)


def test_goal_must_fit_pitch_and_be_wider_than_ball():
    for width in (14.0, 501.0):
        with pytest.raises(ValueError):
            SimulationConfig(rules=RulesConfig(goal_width=width))


def test_restart_expiry_does_not_retry_kick_requested_during_restart():
    state = step_tick(shot()).state
    for _ in range(59):
        state = step_tick(state).state
    result = step_decision(state, PlayerAction(), PlayerAction(kick_trigger=1, kick_power=1.0))
    assert [event.cause for event in result.events] == ["kickoff"]
    assert result.state.ball.possession_owner == PlayerId.B
    assert result.state.phase == MatchPhase.PLAYING


def test_controlled_boundary_exit_clears_owner_and_does_not_duplicate_goal():
    config = SimulationConfig(ball=BallConfig(dribble_offset=30.0))
    state = reset(config=config)
    state = replace(state, player_a=replace(state.player_a, position=(785.0, 250.0)))
    state = assign_possession(state, PlayerId.A, config=config).state
    result = step_tick(state, config=config)
    assert [event.kind for event in result.events] == ["goal"]
    assert result.events[0].previous_owner == PlayerId.A
    assert result.state.ball.possession_owner is None
    assert result.state.score_a == 1
    assert step_decision(result.state, config=config).events == ()


def test_default_dribble_can_touch_but_not_strictly_clear_goal_threshold():
    state = reset()
    state = replace(state, player_a=replace(state.player_a, position=(785.0, 250.0)))
    state = assign_possession(state, PlayerId.A).state
    result = step_tick(state)
    assert result.state.ball.position == (807.0, 250.0)
    assert result.state.score_a == 0
    assert result.state.phase == MatchPhase.PLAYING


def test_corner_tie_is_out_of_bounds_not_goal():
    crossing = detect_ball_exit((790.0, 490.0), (824.0, 524.0), config=SimulationConfig())
    assert crossing.side == BoundarySide.BOTTOM
    assert not crossing.is_goal


def test_multiple_goals_and_restarts_replay_without_erasing_score_or_clock():
    def trajectory():
        state = reset()
        snapshots, events = [], []
        for goal in range(3):
            state = replace(
                state, ball=shot().ball, steal_progress_ticks=0, possession_protection_ticks=0
            )
            result = step_tick(state)
            state = result.state
            events.extend(result.events)
            for _ in range(60):
                result = step_tick(state)
                state = result.state
                events.extend(result.events)
                snapshots.append(state)
            assert state.score_a == goal + 1
            assert state.score_b == 0
            assert state.simulation_tick == (goal + 1) * 61
        assert [event.cause for event in events] == ["goal", "kickoff"] * 3
        return snapshots, events

    assert trajectory() == trajectory()


@pytest.mark.parametrize("field,value", [("score_a", -1), ("score_b", 1.5), ("score_a", True)])
def test_invalid_score_is_rejected(field, value):
    with pytest.raises(ValueError):
        replace(reset(), **{field: value})
