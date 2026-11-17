import json
from dataclasses import replace

import pytest

import game.scenarios as audit
from game.actions import Direction, MovementAction, PlayerAction
from game.player import PlayerId
from game.simulation import SimulationEvent, StepResult, assign_possession, reset


def test_catalog_has_50_distinct_declared_probes_not_counting_replays():
    cases = audit.build_scenarios()
    assert len(cases) == len({case.name for case in cases}) == 50
    assert all(case.expectations and case.purpose for case in cases)
    assert len({case.category for case in cases}) >= 10
    assert min(case.ticks for case in cases) == 1
    assert max(case.ticks for case in cases) == 3600
    assert sum(case.plan_a.seed is not None for case in cases) == 4


def test_agent_schedule_is_relative_to_supplied_initial_tick():
    action = PlayerAction(Direction.NORTH, 0.5, 1, 0.7)
    agent = audit.ScenarioAgent(audit.AgentPlan(schedule=((6, action),)), start_tick=10)
    assert agent(replace(reset(), simulation_tick=10), PlayerId.A) == action
    assert agent(replace(reset(), simulation_tick=16), PlayerId.A) == PlayerAction()


def test_seeded_agents_have_independent_replayable_rng_streams():
    plan = audit.AgentPlan(seed=0)
    a, b, replay = (audit.ScenarioAgent(plan, 0) for _ in range(3))
    state = reset()
    first = a(state, PlayerId.A)
    for _ in range(10):
        a(state, PlayerId.A)
    assert b(state, PlayerId.B) == replay(state, PlayerId.B) == first
    assert all(b(state, PlayerId.B) == replay(state, PlayerId.B) for _ in range(20))


@pytest.mark.parametrize(
    "fields,error",
    [
        ({"schedule": ((0, PlayerAction()),)}, ValueError),
        ({"schedule": ((6, PlayerAction()), (3, PlayerAction()))}, ValueError),
        ({"schedule": ((6, MovementAction()),)}, TypeError),
        ({"fallback": None}, TypeError),
        ({"seed": "17"}, TypeError),
        ({"schedule": ((6, PlayerAction()),), "seed": 0}, ValueError),
    ],
)
def test_invalid_agent_plans_are_not_silently_reinterpreted(fields, error):
    with pytest.raises(error):
        audit.AgentPlan(**fields)


@pytest.mark.parametrize("ticks", [-1, 1.5, True])
def test_invalid_scenario_horizon_is_rejected(ticks):
    with pytest.raises(ValueError):
        audit.Scenario("bad", "fixture", "invalid horizon", reset(), ticks)


def test_probe_strips_held_kicks_and_observes_each_tick(monkeypatch):
    original = audit.step_tick
    calls = []

    def spy(state, action_a, action_b, **kwargs):
        calls.append((state.simulation_tick, action_a))
        return original(state, action_a, action_b, **kwargs)

    monkeypatch.setattr(audit, "step_tick", spy)
    initial = assign_possession(reset(), PlayerId.A).state
    case = audit.Scenario(
        "kick",
        "fixture",
        "single decision",
        initial,
        6,
        plan_a=audit.AgentPlan(fallback=PlayerAction(kick_trigger=1)),
    )
    probe = audit._probe(case)
    assert not probe.errors
    assert len(probe.states) == 7
    assert [tick for tick, _ in calls] == list(range(6))
    assert [getattr(action, "kick_trigger", 0) for _, action in calls] == [1, 0, 0, 0, 0, 0]
    assert sum(event.kind == "ball_kicked" for event in probe.events) == 1


def test_cadence_disagreement_is_reported(monkeypatch):
    monkeypatch.setattr(audit, "step_decision", lambda state, *args, **kwargs: StepResult(state))
    case = audit.Scenario("cadence", "fixture", "bad oracle", reset(), 6)
    result = audit.run_scenario(case)
    assert result["status"] == "FAIL"
    assert any("tick/decision API disagreement" in error for error in result["errors"])


def test_engine_exception_is_recorded_with_context_not_misreported_as_success(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("fixture separation impossible")

    monkeypatch.setattr(audit, "step_tick", broken)
    case = audit.Scenario("broken", "fixture", "crash reporting", reset(), 1)
    result = audit.run_scenario(case)
    assert result["status"] == "FAIL"
    assert result["ticks_advanced"] == 0
    assert result["replay_identical"] is True  # Deterministic failure is still failure.
    assert "ValueError: fixture separation impossible" in result["errors"][0]
    assert result["first_failure"]["kind"] == "engine_exception"
    assert not any("No violated checks" in text for text in result["feedback"])


def test_false_expectation_and_oracle_exception_are_both_failures():
    case = audit.Scenario(
        "bad_expectations",
        "fixture",
        "oracle reporting",
        reset(),
        1,
        expectations=(
            audit.Expectation("always false", lambda p: False),
            audit.Expectation("missing snapshot", lambda p: p.states[100]),
        ),
    )
    result = audit.run_scenario(case)
    assert result["status"] == "FAIL"
    assert all(not check["passed"] for check in result["expectations"])
    assert any("oracle error" in error for error in result["errors"])


def test_acquisition_event_mismatch_is_a_telemetry_failure_not_corrected():
    before = reset()
    after = replace(assign_possession(before, PlayerId.B).state, simulation_tick=1)
    event = SimulationEvent(
        "possession_changed", 1, new_owner=PlayerId.B, last_touch_player=PlayerId.A, cause="kickoff"
    )
    errors = audit._invariant_errors(before, after, (event,), audit.SimulationConfig())
    assert any("telemetry:" in error for error in errors)
    assert event.last_touch_player == PlayerId.A
    assert after.ball.last_touch_player == PlayerId.B


def test_free_ball_overlap_is_checked_but_controlled_defender_overlap_is_allowed():
    from game.ball import BallState

    before = reset()
    free = replace(before, simulation_tick=1, ball=BallState(before.player_a.position))
    errors = audit._invariant_errors(before, free, (), audit.SimulationConfig())
    assert any("free ball: unresolved overlap with player A" in error for error in errors)
    contested = replace(
        before,
        player_a=replace(before.player_a, position=(400.0, 250.0)),
        player_b=replace(before.player_b, position=(430.0, 250.0)),
    )
    controlled = replace(assign_possession(contested, PlayerId.A).state, simulation_tick=1)
    assert not audit._invariant_errors(contested, controlled, (), audit.SimulationConfig())


def test_stationary_correction_energy_triggers_warning_and_regression_failure(monkeypatch):
    from game.ball import BallState

    case = next(case for case in audit.build_scenarios() if case.name == "14_coincident_bodies")
    after = replace(
        case.initial,
        simulation_tick=1,
        player_a=replace(case.initial.player_a, position=(385.0, 250.0)),
        player_b=replace(case.initial.player_b, position=(415.0, 250.0)),
        ball=BallState((400.0, 80.0), (1348.0, 0.0)),
    )
    probe = audit.Probe((case.initial, after), (), (), 1, "synthetic")
    monkeypatch.setattr(audit, "_probe", lambda scenario: probe)
    result = audit.run_scenario(case)
    assert result["status"] == "FAIL"
    assert "expectation failed: stationary correction adds no ball speed" in result["errors"]
    assert result["model_warnings"]
    assert result["max_ball_speed"] > 1000
    assert "without requested movement or kick" in result["model_warnings"][0]


@pytest.mark.parametrize(
    "name",
    [
        "17_partial_tick_stop",
        "19_turn_then_kick",
        "24_extreme_swept_pickup",
        "30_recovery_post_loss_no_energy",
        "31_locked_player_rebound",
        "33_four_rebounds_one_tick",
        "34_post_then_pickup",
        "35_post_then_goal",
        "42_final_tick_goal",
        "43_off_grid_terminal_resume",
        "44_opposed_contact_settling",
    ],
)
def test_short_edge_probes_check_declared_outcomes(name):
    case = next(case for case in audit.build_scenarios() if case.name == name)
    result = audit.run_scenario(case)
    assert result["status"] == "PASS", result["errors"]
    assert result["replay_identical"]
    assert all(check["passed"] for check in result["expectations"])


def test_deliberate_saturation_is_visible_warning_not_silently_passed():
    case = next(case for case in audit.build_scenarios() if case.expected_limit)
    result = audit.run_scenario(case)
    assert result["status"] == "WARN"
    assert result["physics_limits"] == {"impact_budget": 1}
    assert result["errors"] == []
    assert "physics_limit" in result["timeline"]


def test_cli_generates_reproducible_evidence_and_human_report(tmp_path, capsys):
    output = tmp_path / "evidence"
    report = tmp_path / "docs" / "report.md"
    assert (
        audit.main(
            ["--scenario", "17_partial_tick_stop", "--output", str(output), "--report", str(report)]
        )
        == 0
    )
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert payload["scenario_count"] == 1
    assert payload["passes_per_scenario"] == 2
    result = payload["results"][0]
    assert result["initial_state"] and result["config"] and result["agent_a"]
    assert result["status"] == "PASS"
    assert result["expectations"][0]["actual"] == pytest.approx(100.00125)
    assert report.read_text(encoding="utf-8") == (output / "report.md").read_text(encoding="utf-8")
    assert "1 distinct scenarios: 1 PASS" in report.read_text(encoding="utf-8")
    assert (output / "17_partial_tick_stop.timeline.txt").exists()
    assert "Summary: 1 scenarios" in capsys.readouterr().out


def test_cli_writes_report_even_when_a_case_fails(tmp_path, monkeypatch):
    case = audit.Scenario(
        "bad",
        "fixture",
        "forced failure",
        reset(),
        1,
        expectations=(audit.Expectation("false", lambda p: False),),
    )
    monkeypatch.setattr(audit, "build_scenarios", lambda: (case,))
    assert audit.main(["--output", str(tmp_path)]) == 1
    assert (tmp_path / "results.json").exists()
    assert "FAIL expectation: false" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_cli_rejects_unknown_scenario_and_list_does_not_execute_probes(monkeypatch, capsys):
    with pytest.raises(SystemExit) as error:
        audit.main(["--scenario", "not_a_case"])
    assert error.value.code == 2

    def forbidden(*args, **kwargs):
        pytest.fail("--list should not execute scenarios")

    monkeypatch.setattr(audit, "run_scenario", forbidden)
    assert audit.main(["--list"]) == 0
    assert "01_idle_full_match" in capsys.readouterr().out
