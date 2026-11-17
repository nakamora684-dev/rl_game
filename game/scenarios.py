"""Reproducible physics probes, not an RL agent or the evaluation baseline.

Run ``python -m game.scenarios --output results/physics_scenarios``. Scenarios
include short isolated edge probes and sustained matches; they are not 50 matches.
"""

import argparse
import hashlib
import json
import platform
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from math import cos, dist, hypot, isclose, isfinite, pi, sin
from pathlib import Path

from game.actions import Direction, MovementAction, PlayerAction
from game.ball import BallState
from game.config import (
    MATCH_TICKS,
    TICKS_PER_DECISION,
    BallConfig,
    PitchConfig,
    RulesConfig,
    SimulationConfig,
)
from game.player import PlayerId
from game.rules import goalpost_centres
from game.simulation import (
    MatchPhase,
    SimulationEvent,
    SimulationState,
    assign_possession,
    reset,
    step_decision,
    step_tick,
)
from game.timeline import format_timeline

_DEFAULT_CONFIG = SimulationConfig()


@dataclass(frozen=True)
class AgentPlan:
    """Fixed action schedule or private seeded random actions, used only as probes.

    Schedule thresholds are relative physics ticks, not wall time. The action at
    each decision is held for six ticks with the kick stripped after the first.
    """

    schedule: tuple[tuple[int, PlayerAction], ...] = ()
    fallback: PlayerAction = field(default_factory=PlayerAction)
    seed: int | None = None

    def __post_init__(self) -> None:
        previous = 0
        for until_tick, action in self.schedule:
            if type(until_tick) is not int or until_tick <= previous:
                raise ValueError("Schedule thresholds must be strictly increasing positive ticks.")
            if not isinstance(action, PlayerAction):
                raise TypeError("Scheduled actions must be PlayerAction instances.")
            previous = until_tick
        if not isinstance(self.fallback, PlayerAction):
            raise TypeError("Fallback must be a PlayerAction.")
        if self.seed is not None and type(self.seed) is not int:
            raise TypeError("Seed must be an integer or None.")
        if self.seed is not None and self.schedule:
            raise ValueError("Choose a fixed schedule or random actions, not both.")


class ScenarioAgent:
    """A diagnostic action provider; no training, tactics, or human experience."""

    def __init__(self, plan: AgentPlan, start_tick: int):
        self.plan = plan
        self.start_tick = start_tick
        self.rng = random.Random(0 if plan.seed is None else plan.seed)

    def __call__(self, state: SimulationState, player: PlayerId) -> PlayerAction:
        if self.plan.seed is not None:
            return PlayerAction(
                Direction(self.rng.randrange(9)),
                self.rng.random(),
                self.rng.randrange(2),
                self.rng.random(),
            )
        relative_tick = state.simulation_tick - self.start_tick
        for until_tick, action in self.plan.schedule:
            if relative_tick < until_tick:
                return action
        return self.plan.fallback


_IDLE_PLAN = AgentPlan()


@dataclass(frozen=True)
class Probe:
    states: tuple[SimulationState, ...]
    events: tuple[SimulationEvent, ...]
    errors: tuple[str, ...]
    decisions: int
    digest: str
    first_failure: dict | None = None

    @property
    def final(self) -> SimulationState:
        return self.states[-1]


@dataclass(frozen=True)
class Expectation:
    description: str
    check: Callable[[Probe], bool]
    measure: Callable[[Probe], object] | None = None
    expected: object = True


@dataclass(frozen=True)
class Scenario:
    name: str
    category: str
    purpose: str
    initial: SimulationState
    ticks: int
    config: SimulationConfig = field(default_factory=SimulationConfig)
    plan_a: AgentPlan = field(default_factory=AgentPlan)
    plan_b: AgentPlan = field(default_factory=AgentPlan)
    expectations: tuple[Expectation, ...] = ()
    tuning_note: str = ""
    expected_limit: bool = False

    def __post_init__(self) -> None:
        if type(self.ticks) is not int or self.ticks < 0:
            raise ValueError("Scenario horizon must be a nonnegative integer number of ticks.")


def _invariant_errors(
    before: SimulationState,
    after: SimulationState,
    events: tuple[SimulationEvent, ...],
    config: SimulationConfig,
) -> list[str]:
    """Snapshot invariants, not an oracle for the exact within-tick trajectory."""
    errors = []
    tolerance = 1e-7
    if after.simulation_tick != before.simulation_tick + 1:
        errors.append("tick did not advance exactly once")
    radius = config.pitch.player_radius
    for label, player in (("A", after.player_a), ("B", after.player_b)):
        if not all(
            isfinite(v) for v in (*player.position, *player.velocity, player.facing, player.stamina)
        ):
            errors.append(f"{label}: nonfinite state")
        x, y = player.position
        if not (
            radius - tolerance <= x <= config.pitch.width - radius + tolerance
            and radius - tolerance <= y <= config.pitch.height - radius + tolerance
        ):
            errors.append(f"{label}: player outside legal rectangle")
        if not 0 <= player.stamina <= 100:
            errors.append(f"{label}: stamina outside [0,100]")
        if any(
            type(timer) is not int or timer < 0
            for timer in (
                player.kick_cooldown_ticks,
                player.pickup_lockout_ticks,
            )
        ):
            errors.append(f"{label}: invalid interaction timer")
        if after.phase == MatchPhase.PLAYING and any(
            dist(player.position, post) < radius + config.rules.post_radius - tolerance
            for post in goalpost_centres(config)
        ):
            errors.append(f"{label}: unresolved player/post overlap")
    if dist(after.player_a.position, after.player_b.position) < 2 * radius - tolerance:
        errors.append("unresolved player/player overlap")
    ball = after.ball
    if not all(isfinite(v) for v in (*ball.position, *ball.velocity)):
        errors.append("nonfinite ball state")
    if ball.possession_owner is not None:
        carrier = after.player_a if ball.possession_owner == PlayerId.A else after.player_b
        attached = (
            carrier.position[0] + config.ball.dribble_offset * cos(carrier.facing),
            carrier.position[1] + config.ball.dribble_offset * sin(carrier.facing),
        )
        if dist(ball.position, attached) > tolerance:
            errors.append("controlled ball detached from carrier pose")
        if ball.velocity != (0.0, 0.0) or ball.last_touch_player != ball.possession_owner:
            errors.append("contradictory controlled ball velocity/last toucher")
    elif after.steal_progress_ticks or after.possession_protection_ticks:
        errors.append("free ball retained possession timers")
    if after.phase == MatchPhase.PLAYING and ball.possession_owner is None:
        for label, player in (("A", after.player_a), ("B", after.player_b)):
            if dist(ball.position, player.position) < radius + config.ball.radius - tolerance:
                errors.append(f"free ball: unresolved overlap with player {label}")
        if any(
            dist(ball.position, post) < config.ball.radius + config.rules.post_radius - tolerance
            for post in goalpost_centres(config)
        ):
            errors.append("free ball: unresolved post overlap")
    if after.attack_direction_a != before.attack_direction_a:
        errors.append("attacking orientation changed during match")
    for player, score_delta in (
        (PlayerId.A, after.score_a - before.score_a),
        (PlayerId.B, after.score_b - before.score_b),
    ):
        goals = sum(event.kind == "goal" and event.scoring_player == player for event in events)
        if score_delta != goals:
            errors.append(f"{player}: score delta disagrees with goal events")
    collision_indices = [
        event.collision_index for event in events if event.kind == "ball_collision"
    ]
    if collision_indices != list(range(len(collision_indices))):
        errors.append("collision indices not consecutive in emission order")
    for event in events:
        if event.tick not in (before.simulation_tick, after.simulation_tick):
            errors.append(f"{event.kind}: timestamp outside advanced tick")
        if event.tick_fraction is not None and not 0 <= event.tick_fraction <= 1:
            errors.append(f"{event.kind}: invalid contact fraction")
        if (
            event.kind == "possession_changed"
            and event.new_owner is not None
            and event.last_touch_player != event.new_owner
        ):
            errors.append(
                f"telemetry: {event.cause} acquisition new_owner={event.new_owner} "
                f"but event.last_touch_player={event.last_touch_player}"
            )
    if (
        before.phase in (MatchPhase.GOAL_RESTART, MatchPhase.OUT_OF_BOUNDS_RESTART)
        and before.restart_ticks_remaining > 1
    ):
        for label in ("player_a", "player_b"):
            old, new = getattr(before, label), getattr(after, label)
            if (old.position, old.velocity, old.facing, old.stamina) != (
                new.position,
                new.velocity,
                new.facing,
                new.stamina,
            ):
                errors.append("restart changed frozen player motion/stamina")
        if before.ball != after.ball:
            errors.append("restart changed frozen ball")
    return errors


def _probe(scenario: Scenario) -> Probe:
    state = scenario.initial
    agents = (
        ScenarioAgent(scenario.plan_a, state.simulation_tick),
        ScenarioAgent(scenario.plan_b, state.simulation_tick),
    )
    states, events, errors = [state], [], []
    decisions = 0
    first_failure = None
    digest = hashlib.sha256()

    def stamp(value):
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))
        digest.update(b"\n")

    stamp(asdict(state))
    try:
        while len(states) - 1 < scenario.ticks and state.phase != MatchPhase.FINISHED:
            start = state
            actions = (agents[0](state, PlayerId.A), agents[1](state, PlayerId.B))
            stamp([asdict(action) for action in actions])
            decisions += 1
            length = min(
                TICKS_PER_DECISION,
                scenario.ticks - (len(states) - 1),
                MATCH_TICKS - state.simulation_tick,
            )
            block_events = []
            for sub_tick in range(length):
                held = (
                    actions
                    if sub_tick == 0
                    else tuple(
                        MovementAction(action.direction, action.effort) for action in actions
                    )
                )
                result = step_tick(state, *held, config=scenario.config)
                tick_errors = _invariant_errors(state, result.state, result.events, scenario.config)
                if tick_errors and first_failure is None:
                    first_failure = {
                        "kind": "snapshot_or_event_invariant",
                        "errors": tick_errors,
                        "before": asdict(state),
                        "after": asdict(result.state),
                        "events": [asdict(event) for event in result.events],
                    }
                for error in tick_errors:
                    errors.append(f"tick {result.state.simulation_tick}: {error}")
                state = result.state
                states.append(state)
                events.extend(result.events)
                block_events.extend(result.events)
                stamp({"state": asdict(state), "events": [asdict(e) for e in result.events]})
            # Cross-check the instrumented loop with the public decision API.
            if length == TICKS_PER_DECISION or state.phase == MatchPhase.FINISHED:
                expected = step_decision(start, *actions, config=scenario.config)
                if (
                    expected.state != state
                    or expected.events != tuple(block_events)
                    or expected.physics_ticks != length
                ):
                    errors.append(f"tick {state.simulation_tick}: tick/decision API disagreement")
        if state.phase == MatchPhase.FINISHED:
            frozen = step_tick(state, PlayerAction(kick_trigger=1), config=scenario.config)
            if frozen.state is not state or frozen.events or frozen.physics_ticks:
                errors.append("finished stepping is not idempotent")
    except Exception as exc:  # noqa: BLE001 -- an audit must record engine failures and continue other probes
        errors.append(f"tick {state.simulation_tick}: {type(exc).__name__}: {exc}")
        if first_failure is None:
            first_failure = {
                "kind": "engine_exception",
                "state": asdict(state),
                "exception": f"{type(exc).__name__}: {exc}",
            }
    return Probe(
        tuple(states),
        tuple(events),
        tuple(dict.fromkeys(errors)),
        decisions,
        digest.hexdigest(),
        first_failure,
    )


def run_scenario(scenario: Scenario) -> dict:
    """Run twice with fresh agents and report facts separately from tuning notes."""
    first, replay = _probe(scenario), _probe(scenario)
    errors = list(first.errors)
    if first.digest != replay.digest or first.errors != replay.errors:
        errors.append("deterministic replay mismatch")
    checks = []
    for expectation in scenario.expectations:
        actual = None
        try:
            passed = bool(expectation.check(first))
            actual = expectation.measure(first) if expectation.measure is not None else passed
        except Exception as exc:  # noqa: BLE001 -- distinguish oracle exceptions from engine failures in evidence
            passed = False
            errors.append(f"oracle error: {expectation.description}: {type(exc).__name__}: {exc}")
        checks.append(
            {
                "description": expectation.description,
                "passed": passed,
                "expected": expectation.expected,
                "actual": actual,
            }
        )
        if not passed:
            errors.append(f"expectation failed: {expectation.description}")
    final = first.states[-1]
    advanced = len(first.states) - 1
    expected_ticks = min(scenario.ticks, MATCH_TICKS - scenario.initial.simulation_tick)
    if advanced != expected_ticks:
        errors.append(f"advanced {advanced} ticks, expected {expected_ticks}")
    counts = Counter(event.kind for event in first.events)
    limits = Counter(event.cause for event in first.events if event.kind == "physics_limit")
    max_ball_speed = max(hypot(*state.ball.velocity) for state in first.states)
    model_warnings = []
    if (
        scenario.initial.player_a.velocity == scenario.initial.player_b.velocity == (0.0, 0.0)
        and scenario.initial.ball.velocity == (0.0, 0.0)
        and scenario.plan_a == scenario.plan_b == _IDLE_PLAN
        and dist(scenario.initial.player_a.position, scenario.initial.player_b.position)
        < 2 * scenario.config.pitch.player_radius
        and max_ball_speed > 1e-7
    ):
        model_warnings.append(
            f"Stationary overlapping-body correction generated snapshot-observed ball speed "
            f"{max_ball_speed:.6g} units/s without requested movement or kick. "
            "Inspect correction-derived collision energy before using overlapping spawn fixtures."
        )
    feedback = []
    if limits:
        feedback.append(
            "Explicit physics-limit saturation occurred; inspect trace, not a silent success."
        )
    if any(event.kind == "ball_constraint" for event in first.events):
        feedback.append(
            "Opposed-contact settling used a bounded approximation; visual inspection is still needed."
        )
    if counts["goal"] and not any("score delta disagrees" in error for error in first.errors):
        feedback.append("Goal events agree with score changes at every observed physics tick.")
    if scenario.tuning_note:
        feedback.append(scenario.tuning_note)
    if not feedback:
        feedback.append(
            "Recorded failures need investigation; see failure entries."
            if errors
            else "Declared assertions passed; a model-hardening warning remains."
            if model_warnings
            else "No violated checks in this probe; this does not establish realistic tuning."
        )
    feedback.extend(model_warnings)
    if limits and not scenario.expected_limit:
        feedback.append(
            "Unexpected limit telemetry: investigate before accepting this scenario as safe."
        )
    players = [player for state in first.states for player in (state.player_a, state.player_b)]
    return {
        "name": scenario.name,
        "category": scenario.category,
        "purpose": scenario.purpose,
        "status": "FAIL" if errors else "WARN" if limits or model_warnings else "PASS",
        "errors": errors,
        "first_failure": first.first_failure,
        "model_warnings": model_warnings,
        "expectations": checks,
        "feedback": feedback,
        "ticks_requested": scenario.ticks,
        "ticks_advanced": advanced,
        "decision_count": first.decisions,
        "initial_tick": scenario.initial.simulation_tick,
        "final_tick": final.simulation_tick,
        "final_phase": final.phase.value,
        "initial_score": [scenario.initial.score_a, scenario.initial.score_b],
        "final_score": [final.score_a, final.score_b],
        "final_owner": final.ball.possession_owner,
        "min_stamina": min(player.stamina for player in players),
        "max_player_speed": max(hypot(*player.velocity) for player in players),
        "max_ball_speed": max_ball_speed,
        "min_body_clearance": min(
            dist(state.player_a.position, state.player_b.position)
            - 2 * scenario.config.pitch.player_radius
            for state in first.states[1:]
        )
        if advanced
        else None,
        "event_counts": dict(counts),
        "physics_limits": dict(limits),
        "expected_limit": scenario.expected_limit,
        "replay_identical": first.digest == replay.digest and first.errors == replay.errors,
        "trajectory_sha256": first.digest,
        "initial_state": asdict(scenario.initial),
        "config": asdict(scenario.config),
        "agent_a": asdict(scenario.plan_a),
        "agent_b": asdict(scenario.plan_b),
        "final_state": asdict(final),
        "events": [asdict(event) for event in first.events],
        "timeline": format_timeline(first.events),
    }


def _fixture(
    *, a=None, b=None, ball=None, owner=None, swap=False, config=_DEFAULT_CONFIG
) -> SimulationState:
    state = reset(swap_sides=swap, config=config)
    state = replace(
        state,
        player_a=replace(state.player_a, **(a or {})),
        player_b=replace(state.player_b, **(b or {})),
        ball=state.ball if ball is None else ball,
    )
    return assign_possession(state, owner, config=config).state if owner is not None else state


def _count(probe: Probe, kind: str, cause: str | None = None) -> int:
    return sum(e.kind == kind and (cause is None or e.cause == cause) for e in probe.events)


def _events(kind: str, count: int, cause: str | None = None) -> Expectation:
    return Expectation(
        f"{kind}/{cause or '*'} count = {count}",
        lambda p: _count(p, kind, cause) == count,
        lambda p: _count(p, kind, cause),
        count,
    )


def _value(description: str, getter: Callable, expected: float, tolerance=1e-7) -> Expectation:
    return Expectation(
        description,
        lambda p: isclose(getter(p), expected, abs_tol=tolerance, rel_tol=1e-9),
        getter,
        expected,
    )


def build_scenarios() -> tuple[Scenario, ...]:
    """Exactly 50 named cases; replays are verification, not extra scenarios.

    Test fixtures may contain explicit velocities, ownership, timers and compressed
    restarts. They are declared initial conditions, never controller cheating.
    """
    cases = []
    idle = PlayerAction()
    east = PlayerAction(Direction.EAST, 1.0)
    west = PlayerAction(Direction.WEST, 1.0)
    kick = PlayerAction(kick_trigger=1, kick_power=1.0)
    frictionless = SimulationConfig(ball=BallConfig(deceleration=0.0))

    def add(
        name,
        category,
        purpose,
        *,
        initial=None,
        ticks=60,
        config=_DEFAULT_CONFIG,
        a=_IDLE_PLAN,
        b=_IDLE_PLAN,
        checks=(),
        note="",
        expected_limit=False,
    ):
        cases.append(
            Scenario(
                name,
                category,
                purpose,
                reset(config=config) if initial is None else initial,
                ticks,
                config,
                a,
                b,
                tuple(checks),
                note,
                expected_limit,
            )
        )

    # 01-05: movement, facing, and a complete idle match.
    add(
        "01_idle_full_match",
        "movement",
        "Exact 60-second match and one whistle",
        ticks=3600,
        checks=(
            _events("match_finished", 1),
            _value("ends at tick 3600", lambda p: p.final.simulation_tick, 3600),
        ),
    )
    add(
        "02_normalized_diagonal",
        "movement",
        "Diagonal target does not multiply speed by sqrt(2)",
        initial=_fixture(a={"position": (100.0, 100.0)}),
        ticks=60,
        a=AgentPlan(fallback=PlayerAction(Direction.SOUTHEAST, 0.5)),
        checks=(_value("diagonal speed = 80", lambda p: hypot(*p.final.player_a.velocity), 80.0),),
    )
    add(
        "03_velocity_reversal",
        "movement",
        "Eastward run then bounded westward reversal",
        initial=_fixture(a={"position": (300.0, 80.0)}),
        ticks=240,
        a=AgentPlan(schedule=((120, east),), fallback=west),
        checks=(
            Expectation("final x velocity is negative", lambda p: p.final.player_a.velocity[0] < 0),
        ),
    )
    add(
        "04_idle_braking",
        "movement",
        "Open-space coast from 160 at braking 40",
        initial=_fixture(a={"position": (100.0, 80.0), "velocity": (160.0, 0.0)}),
        ticks=240,
        checks=(
            _value("stops at 4 seconds", lambda p: hypot(*p.final.player_a.velocity), 0),
            _value(
                "semi-implicit travel = 318 2/3",
                lambda p: p.final.player_a.position[0] - 100,
                318 + 2 / 3,
            ),
        ),
        note="Measured stop uses player semi-implicit integration, not the ball's analytic integration. Four-second idle stopping is an approved tuning concern.",
    )
    add(
        "05_zero_effort_turn",
        "movement",
        "Facing turns north without driving acceleration",
        ticks=30,
        a=AgentPlan(fallback=PlayerAction(Direction.NORTH, 0.0)),
        checks=(
            _value("faces north", lambda p: p.final.player_a.facing, -pi / 2),
            _value("no speed", lambda p: hypot(*p.final.player_a.velocity), 0),
        ),
    )

    # 06-10: stamina and the approved zero-effort braking discontinuity.
    add(
        "06_sprint_against_wall",
        "stamina",
        "Requested effort drains stamina despite blocked motion",
        initial=_fixture(a={"position": (785.0, 400.0)}),
        ticks=1200,
        a=AgentPlan(fallback=east),
        checks=(
            _value("stamina depleted after 20 seconds", lambda p: p.final.player_a.stamina, 0),
            _value("wall x unchanged", lambda p: p.final.player_a.position[0], 785),
        ),
    )
    add(
        "07_empty_stamina_speed",
        "stamina",
        "Fatigue halves full-effort target speed",
        initial=_fixture(a={"position": (100.0, 80.0), "stamina": 0.0}),
        ticks=120,
        a=AgentPlan(fallback=east),
        checks=(_value("empty target = 80", lambda p: hypot(*p.final.player_a.velocity), 80),),
    )
    add(
        "08_sprint_then_recover",
        "stamina",
        "20-second sprint followed by 15 seconds idle",
        initial=_fixture(a={"position": (100.0, 80.0)}),
        ticks=2100,
        a=AgentPlan(schedule=((1200, east),)),
        checks=(
            _value("sprint reaches empty", lambda p: p.states[1200].player_a.stamina, 0),
            _value("rest restores full stamina", lambda p: p.final.player_a.stamina, 100),
        ),
    )
    add(
        "09_neutral_effort",
        "stamina",
        "Effort 0.5 neither consumes nor regenerates",
        initial=_fixture(a={"position": (100.0, 80.0), "stamina": 50.0}),
        ticks=600,
        a=AgentPlan(fallback=PlayerAction(Direction.EAST, 0.5)),
        checks=(_value("neutral stamina stays 50", lambda p: p.final.player_a.stamina, 50),),
    )
    add(
        "10_tiny_effort_brakes_harder",
        "stamina",
        "Compare coasting with tiny active target",
        initial=_fixture(
            a={"position": (100.0, 80.0), "velocity": (160.0, 0.0)},
            b={"position": (100.0, 400.0), "velocity": (160.0, 0.0)},
        ),
        ticks=6,
        b=AgentPlan(fallback=PlayerAction(Direction.EAST, 1e-6)),
        checks=(
            Expectation(
                "tiny active effort brakes more than idle",
                lambda p: hypot(*p.final.player_b.velocity) < hypot(*p.final.player_a.velocity),
            ),
        ),
        note="Known tuning discontinuity: idle uses braking 40; tiny positive active effort uses acceleration up to 320. Do not mistake it for smooth effort-dependent braking.",
    )

    # 11-15: bodies and coupled wall/post geometry.
    add(
        "11_head_on_bodies",
        "body_contact",
        "Symmetric approach must not pass through",
        initial=_fixture(
            a={"position": (300.0, 250.0)},
            b={"position": (500.0, 250.0)},
            ball=BallState((400.0, 80.0)),
        ),
        ticks=240,
        a=AgentPlan(fallback=east),
        b=AgentPlan(fallback=west),
        checks=(
            Expectation(
                "A remains left of B",
                lambda p: all(s.player_a.position[0] <= s.player_b.position[0] for s in p.states),
            ),
        ),
    )
    add(
        "12_tangential_slide",
        "body_contact",
        "Touching bodies retain lateral sliding",
        initial=_fixture(a={"position": (400.0, 100.0)}, b={"position": (430.0, 100.0)}),
        ticks=60,
        a=AgentPlan(fallback=PlayerAction(Direction.SOUTHEAST, 0.5)),
        checks=(
            Expectation(
                "A slides down rather than freezing", lambda p: p.final.player_a.position[1] > 120
            ),
        ),
    )
    add(
        "13_high_speed_body_crossing",
        "body_contact",
        "Artificial extreme velocities must not tunnel",
        initial=_fixture(
            a={"position": (300.0, 100.0), "velocity": (30000.0, 0.0)},
            b={"position": (500.0, 100.0), "velocity": (-30000.0, 0.0)},
        ),
        ticks=1,
        checks=(
            Expectation(
                "body order preserved",
                lambda p: p.final.player_a.position[0] < p.final.player_b.position[0],
            ),
        ),
    )
    add(
        "14_coincident_bodies",
        "body_contact",
        "Deterministic split of exactly coincident fixture",
        initial=_fixture(a={"position": (400.0, 250.0)}, b={"position": (400.0, 250.0)}),
        ticks=1,
        checks=(
            Expectation(
                "coincidence resolved along pitch long axis",
                lambda p: p.final.player_a.position[0] < p.final.player_b.position[0],
            ),
            _value(
                "stationary correction adds no ball speed",
                lambda p: hypot(*p.final.ball.velocity),
                0,
            ),
        ),
    )
    add(
        "15_body_post_wall_pressure",
        "body_contact",
        "A second body pushes near a solid goalpost and wall",
        initial=_fixture(a={"position": (780.0, 186.0)}, b={"position": (750.0, 186.0)}),
        ticks=120,
        a=AgentPlan(fallback=east),
        b=AgentPlan(fallback=east),
        checks=(
            Expectation(
                "A stays left of post clearance",
                lambda p: p.final.player_a.position[0] <= 781.000001,
            ),
        ),
    )

    # 16-20: analytic rolling and facing-based action release.
    off_path = {"a": {"position": (100.0, 400.0)}, "b": {"position": (600.0, 400.0)}}
    add(
        "16_reference_roll",
        "ball_motion",
        "300-unit/s roll stops after 3 seconds and 450 units",
        initial=_fixture(**off_path, ball=BallState((100.0, 80.0), (300.0, 0.0))),
        ticks=180,
        checks=(
            _value(
                "final x = 550 after 450-unit reference roll",
                lambda p: p.final.ball.position[0],
                550,
            ),
            _value("reference ball stopped", lambda p: hypot(*p.final.ball.velocity), 0),
        ),
    )
    add(
        "17_partial_tick_stop",
        "ball_motion",
        "Very slow ball stops partway through one tick",
        initial=_fixture(**off_path, ball=BallState((100.0, 80.0), (0.5, 0.0))),
        ticks=1,
        checks=(
            _value(
                "final x = 100.00125 after partial-tick stop",
                lambda p: p.final.ball.position[0],
                100.00125,
            ),
        ),
    )
    add(
        "18_frictionless_roll",
        "ball_motion",
        "Zero friction preserves velocity and linear travel",
        config=frictionless,
        initial=_fixture(**off_path, ball=BallState((100.0, 80.0), (300.0, 0.0))),
        ticks=60,
        checks=(
            _value("one-second x = 400", lambda p: p.final.ball.position[0], 400),
            _value("speed remains 300", lambda p: hypot(*p.final.ball.velocity), 300),
        ),
    )
    add(
        "19_turn_then_kick",
        "ball_motion",
        "First-tick facing update changes release direction",
        initial=_fixture(b={"position": (600.0, 400.0)}, owner=PlayerId.A),
        ticks=1,
        a=AgentPlan(fallback=PlayerAction(Direction.NORTH, 0.0, 1, 1.0)),
        checks=(
            _events("ball_kicked", 1),
            Expectation(
                "northward component after east-to-north turn",
                lambda p: p.final.ball.velocity[1] < 0,
            ),
        ),
        note="A kick during a turn follows that tick's partially turned facing, not the requested direction immediately.",
    )
    add(
        "20_midpower_stop_distance",
        "ball_motion",
        "Power 0.5 maps to speed 175, not half the maximum distance",
        initial=_fixture(b={"position": (600.0, 400.0)}, owner=PlayerId.A),
        ticks=105,
        a=AgentPlan(schedule=((6, PlayerAction(kick_trigger=1, kick_power=0.5)),)),
        checks=(
            _events("ball_kicked", 1),
            _value(
                "final x = 375.125 after 153.125-unit roll",
                lambda p: p.final.ball.position[0],
                375.125,
            ),
        ),
        note="Kick power is linear in speed, but stopping distance is quadratic in speed.",
    )

    # 21-25: requests, cooldown, restart expiry, swept pickup and an exact tie.
    add(
        "21_pickup_does_not_retry_kick",
        "request_pickup",
        "No possession at request, recovery later in the decision",
        initial=_fixture(ball=BallState((220.0, 250.0))),
        ticks=6,
        a=AgentPlan(fallback=kick),
        checks=(_events("ball_kicked", 0), _events("possession_changed", 1, "recovery")),
    )
    cooldown = _fixture(owner=PlayerId.A)
    cooldown = replace(cooldown, player_a=replace(cooldown.player_a, kick_cooldown_ticks=1))
    add(
        "22_cooldown_expiry_no_retry",
        "request_pickup",
        "Cooldown expires inside a held decision",
        initial=cooldown,
        ticks=6,
        a=AgentPlan(fallback=kick),
        checks=(
            _events("ball_kicked", 0),
            _value("cooldown reaches zero", lambda p: p.final.player_a.kick_cooldown_ticks, 0),
        ),
    )
    restart = step_tick(
        _fixture(ball=BallState((804.0, 250.0), (300.0, 0.0), last_touch_player=PlayerId.A))
    ).state
    restart = replace(restart, restart_ticks_remaining=2)
    add(
        "23_restart_expiry_no_retry",
        "request_pickup",
        "Compressed real restart ends mid-decision; kick stays discarded",
        initial=restart,
        ticks=6,
        a=AgentPlan(fallback=PlayerAction(Direction.NORTH, 0.5, 1, 1.0)),
        b=AgentPlan(fallback=PlayerAction(Direction.NORTH, 0.5, 1, 1.0)),
        checks=(
            _events("ball_kicked", 0),
            _events("possession_changed", 1, "kickoff"),
            Expectation("held movement resumes", lambda p: p.final.player_b.velocity[1] < 0),
        ),
    )
    add(
        "24_extreme_swept_pickup",
        "request_pickup",
        "Ball crosses receiver and boundary within one tick; pickup wins first",
        initial=_fixture(
            a={"position": (400.0, 250.0)},
            b={"position": (600.0, 400.0)},
            ball=BallState((100.0, 250.0), (60000.0, 0.0)),
        ),
        ticks=1,
        checks=(
            _events("possession_changed", 1, "recovery"),
            _events("goal", 0),
            Expectation(
                "A traps the swept ball", lambda p: p.final.ball.possession_owner == PlayerId.A
            ),
        ),
    )
    add(
        "25_exact_pickup_tie",
        "request_pickup",
        "Exactly simultaneous equal-distance pickups stay free",
        initial=_fixture(a={"position": (378.0, 250.0)}, b={"position": (422.0, 250.0)}),
        ticks=6,
        checks=(
            _events("possession_changed", 0),
            Expectation("no A-first owner", lambda p: p.final.ball.possession_owner is None),
        ),
    )

    # 26-30: contests/protection and attachment-loss energy.
    contest = _fixture(
        a={"position": (400.0, 250.0)}, b={"position": (430.0, 250.0)}, owner=PlayerId.A
    )
    add(
        "26_steal_then_protected_return",
        "stealing",
        "Consecutive 18-tick steals separated by 30 protected ticks",
        initial=contest,
        ticks=66,
        checks=(
            _events("possession_changed", 2, "steal"),
            Expectation(
                "steals at ticks 18 and 66",
                lambda p: [e.tick for e in p.events if e.cause == "steal"] == [18, 66],
                lambda p: [e.tick for e in p.events if e.cause == "steal"],
                [18, 66],
            ),
        ),
        note="Controlled attachment can geometrically overlap the defender; the hybrid model resolves control via the steal timer, not a free-ball bounce.",
    )
    add(
        "27_contest_interrupted",
        "stealing",
        "Leaving range clears, rather than pauses, steal progress",
        initial=_fixture(
            a={"position": (400.0, 250.0)}, b={"position": (435.0, 250.0)}, owner=PlayerId.A
        ),
        ticks=60,
        b=AgentPlan(schedule=((12, idle), (24, east))),
        checks=(
            _events("possession_changed", 0, "steal"),
            _value("12 initial contest ticks", lambda p: p.states[12].steal_progress_ticks, 12),
            _value("contest cleared", lambda p: p.final.steal_progress_ticks, 0),
        ),
    )
    add(
        "28_protection_exact_boundary",
        "stealing",
        "30 following ticks block progress before 18 eligible ticks",
        initial=replace(contest, possession_protection_ticks=30),
        ticks=48,
        checks=(
            Expectation(
                "first steal exactly tick 48",
                lambda p: [e.tick for e in p.events if e.cause == "steal"] == [48],
            ),
            _value("no progress while protected", lambda p: p.states[30].steal_progress_ticks, 0),
        ),
    )
    add(
        "29_kick_clears_contest",
        "stealing",
        "A kick interrupts a nearly completed steal",
        initial=replace(
            _fixture(
                a={"position": (400.0, 250.0)}, b={"position": (365.0, 250.0)}, owner=PlayerId.A
            ),
            steal_progress_ticks=17,
        ),
        ticks=1,
        a=AgentPlan(fallback=kick),
        checks=(
            _events("ball_kicked", 1),
            _events("possession_changed", 0, "steal"),
            _value("free-ball contest cleared", lambda p: p.final.steal_progress_ticks, 0),
        ),
    )
    add(
        "30_recovery_post_loss_no_energy",
        "stealing",
        "Recovery attachment into a post loses control without teleport-derived energy",
        initial=_fixture(
            b={"position": (778.0, 186.0), "facing": 0.0}, ball=BallState((778.0, 164.0))
        ),
        ticks=1,
        checks=(
            _events("possession_changed", 1, "recovery"),
            _events("possession_changed", 1, "post_collision"),
            _events("ball_kicked", 0),
            _value("no invented speed", lambda p: hypot(*p.final.ball.velocity), 0),
        ),
    )

    # 31-35: locked-body/post rebounds and ordered remaining flights.
    add(
        "31_locked_player_rebound",
        "rebounds",
        "Locked player is solid and becomes last toucher",
        config=frictionless,
        initial=_fixture(
            a={"pickup_lockout_ticks": 12},
            ball=BallState((177.0, 250.0), (300.0, 0.0), last_touch_player=PlayerId.B),
        ),
        ticks=1,
        checks=(
            _events("ball_collision", 1, "player_contact"),
            _value(
                "normal speed reflected at restitution 0.5",
                lambda p: p.final.ball.velocity[0],
                -150,
            ),
            Expectation(
                "last touch changes to A", lambda p: p.final.ball.last_touch_player == PlayerId.A
            ),
        ),
    )
    add(
        "32_post_restitution",
        "rebounds",
        "Post bounce preserves last touch and uses restitution 0.7",
        config=frictionless,
        initial=_fixture(
            ball=BallState((780.0, 186.0), (1200.0, 0.0), last_touch_player=PlayerId.B)
        ),
        ticks=1,
        checks=(
            _events("ball_collision", 1, "goalpost"),
            _value("post vx = -840", lambda p: p.final.ball.velocity[0], -840),
            Expectation(
                "post preserves B last touch",
                lambda p: p.final.ball.last_touch_player == PlayerId.B,
            ),
        ),
    )
    trapped = _fixture(
        a={"position": (200.0, 250.0), "pickup_lockout_ticks": 12},
        b={"position": (250.0, 250.0), "pickup_lockout_ticks": 12},
        ball=BallState((225.0, 250.0), (6000.0, 0.0)),
    )
    add(
        "33_four_rebounds_one_tick",
        "rebounds",
        "Multiple locked-player impacts consume the remaining tick",
        config=frictionless,
        initial=trapped,
        ticks=1,
        checks=(
            _events("ball_collision", 4),
            _value("four-bounce vx = 375", lambda p: p.final.ball.velocity[0], 375),
        ),
    )
    add(
        "34_post_then_pickup",
        "rebounds",
        "Post bounce can precede a recovery within one tick",
        config=frictionless,
        initial=_fixture(
            b={"position": (775.0, 154.0)}, ball=BallState((780.0, 180.0), (6000.0, 0.0))
        ),
        ticks=1,
        checks=(
            Expectation(
                "post collision before recovery",
                lambda p: (
                    [e.kind for e in p.events[:2]] == ["ball_collision", "possession_changed"]
                ),
            ),
            Expectation("B recovers", lambda p: p.final.ball.possession_owner == PlayerId.B),
        ),
    )
    add(
        "35_post_then_goal",
        "rebounds",
        "Post turns off-target high-speed shot into a goal",
        config=frictionless,
        initial=_fixture(
            ball=BallState((780.0, 304.0), (6000.0, 0.0), last_touch_player=PlayerId.A)
        ),
        ticks=1,
        checks=(
            Expectation(
                "post then goal event order",
                lambda p: [e.kind for e in p.events] == ["ball_collision", "goal"],
            ),
        ),
    )

    # 36-40: strict exits, mouth clearance, own goal, and responsibility.
    add(
        "36_strict_threshold_touch",
        "rules",
        "Touching x=807 alone is not a strict whole-ball exit",
        initial=_fixture(ball=BallState((807.0, 250.0))),
        ticks=6,
        checks=(_events("goal", 0), _events("out_of_bounds", 0)),
    )
    add(
        "37_inclusive_mouth_edge",
        "rules",
        "Whole ball fits exactly at lower legal centre y=197",
        initial=_fixture(
            ball=BallState((804.0, 197.0), (300.0, 0.0), last_touch_player=PlayerId.A)
        ),
        ticks=1,
        checks=(_events("goal", 1), _events("out_of_bounds", 0)),
    )
    add(
        "38_swapped_own_goal",
        "rules",
        "Credit attack direction, not the last toucher",
        initial=_fixture(
            swap=True, ball=BallState((804.0, 250.0), (300.0, 0.0), last_touch_player=PlayerId.A)
        ),
        ticks=1,
        checks=(_value("B scores despite A last touch", lambda p: p.final.score_b, 1),),
    )
    add(
        "39_known_touch_boundary_restart",
        "rules",
        "A's top exit awards an inset restart to B",
        initial=_fixture(
            ball=BallState((400.0, -4.0), (0.0, -300.0), last_touch_player=PlayerId.A)
        ),
        ticks=66,
        checks=(
            _events("out_of_bounds", 1),
            _events("possession_changed", 1, "boundary_restart"),
            Expectation("B owns restart", lambda p: p.final.ball.possession_owner == PlayerId.B),
        ),
    )
    add(
        "40_unknown_touch_neutral_restart",
        "rules",
        "Unknown responsibility creates free centre restart without blame",
        initial=_fixture(ball=BallState((400.0, -4.0), (0.0, -300.0))),
        ticks=66,
        checks=(
            _events("restart_completed", 1, "neutral_restart"),
            Expectation(
                "neutral centre ball",
                lambda p: (
                    p.final.ball.possession_owner is None
                    and p.final.ball.position == (400.0, 250.0)
                ),
            ),
        ),
    )

    # 41-45: complete restarts, terminal edges and explicitly bounded settling.
    goal_shot = _fixture(ball=BallState((804.0, 250.0), (300.0, 0.0), last_touch_player=PlayerId.A))
    add(
        "41_full_restart_freeze",
        "terminal_diagnostics",
        "Goal followed by all 60 frozen restart ticks and play resumption",
        initial=goal_shot,
        ticks=67,
        checks=(
            _events("goal", 1),
            _events("possession_changed", 1, "kickoff"),
            Expectation(
                "restart completes at tick 61",
                lambda p: any(e.cause == "kickoff" and e.tick == 61 for e in p.events),
            ),
        ),
    )
    add(
        "42_final_tick_goal",
        "terminal_diagnostics",
        "Final-tick goal and whistle both survive; no future kickoff",
        initial=replace(goal_shot, simulation_tick=3599),
        ticks=1,
        checks=(
            Expectation(
                "goal then whistle",
                lambda p: [e.kind for e in p.events] == ["goal", "match_finished"],
            ),
            Expectation("pending restart cleared", lambda p: p.final.restart_plan is None),
        ),
    )
    add(
        "43_off_grid_terminal_resume",
        "terminal_diagnostics",
        "Seven remaining ticks span one full and one partial decision",
        initial=replace(reset(), simulation_tick=3593),
        ticks=7,
        checks=(
            _events("match_finished", 1),
            _value("two decision queries", lambda p: p.decisions, 2),
        ),
    )
    opposed = _fixture(
        a={"position": (200.0, 250.0), "pickup_lockout_ticks": 12},
        b={"position": (244.0, 250.0), "pickup_lockout_ticks": 12},
        ball=BallState((222.0, 250.0), (6000.0, 40.0)),
    )
    add(
        "44_opposed_contact_settling",
        "terminal_diagnostics",
        "Normal squeezing settles while tangential roll survives",
        initial=opposed,
        config=frictionless,
        ticks=1,
        checks=(
            _events("ball_constraint", 1, "opposed_contacts"),
            _events("physics_limit", 0),
            _value("tangential speed retained", lambda p: p.final.ball.velocity[1], 40),
        ),
    )
    add(
        "45_deliberate_budget_saturation",
        "terminal_diagnostics",
        "Artificial one-impact budget reports saturation explicitly",
        initial=trapped,
        config=SimulationConfig(ball=BallConfig(deceleration=0.0, max_impacts_per_tick=1)),
        ticks=1,
        checks=(
            _events("physics_limit", 1, "impact_budget"),
            _value("saturated ball stops", lambda p: hypot(*p.final.ball.velocity), 0),
        ),
        expected_limit=True,
        note="Deliberately reduced budget, not a failure at the default 32-impact budget.",
    )

    # 46-50: sustained stochastic probes plus the historical narrow-reset regression.
    for number, name, seed, initial in (
        (46, "random_default", 17, reset()),
        (47, "random_swapped", 17, reset(swap_sides=True)),
        (
            48,
            "random_near_posts",
            23,
            _fixture(
                a={"position": (773.0, 198.0)},
                b={"position": (730.0, 240.0)},
                ball=BallState((773.0, 176.0)),
            ),
        ),
        (49, "random_low_stamina", 41, _fixture(a={"stamina": 0.0}, b={"stamina": 5.0})),
    ):
        add(
            f"{number}_{name}",
            "sustained_stress",
            "Both agents issue seeded random actions for a full match",
            initial=initial,
            ticks=3600,
            a=AgentPlan(seed=seed),
            b=AgentPlan(seed=seed + 1000),
            checks=(_events("match_finished", 1),),
            note="Random probes test combinations, not tactical quality; a low score does not itself imply bad physics.",
        )
    narrow = SimulationConfig(
        pitch=PitchConfig(width=50.0, height=100.0), rules=RulesConfig(goal_width=40.0)
    )
    add(
        "50_unsupported_narrow_reset",
        "custom_geometry",
        "Historical 50x100 reset failure: fallback spawns must be separated and advance normally",
        initial=reset(config=narrow),
        config=narrow,
        ticks=6,
        checks=(
            _value(
                "accepted configuration produces 7 snapshots including initial",
                lambda p: len(p.states),
                7,
            ),
            Expectation(
                "reset bodies are already separated",
                lambda p: dist(p.states[0].player_a.position, p.states[0].player_b.position) >= 30,
            ),
        ),
        note="Historical scenario ID retained. Narrow reset now uses vertical fallback spawns; this does not prove support for every custom pitch/post layout.",
    )
    if len(cases) != 50 or len({case.name for case in cases}) != 50:
        raise RuntimeError("The scenario catalog must contain exactly 50 unique cases.")
    return tuple(cases)


def render_report(results: list[dict]) -> str:
    """Evidence summary; feedback is numerical/model-based, not subjective play."""
    statuses = Counter(result["status"] for result in results)
    failures = [result for result in results if result["status"] == "FAIL"]
    telemetry = [
        result["name"] for result in results if any("telemetry:" in e for e in result["errors"])
    ]
    lines = [
        "# Physics scenario audit",
        "",
        (
            f"**{len(results)} distinct scenarios: {statuses['PASS']} PASS, "
            f"{statuses['WARN']} WARN, {statuses['FAIL']} FAIL.**"
        ),
        "",
        (
            "Each scenario runs twice with fresh deterministic controllers. Every advanced physics "
            "tick is inspected; complete six-tick blocks and terminal partial blocks are also "
            "compared with `step_decision`. Replay compares state/action/event trajectory hashes. "
            "Replays and cadence checks are not counted as additional scenarios."
        ),
        "",
        (
            "PASS means only that the declared expectations and snapshot invariants held. WARN "
            "means explicit limit telemetry or a marked model-hardening concern occurred. FAIL "
            "can mean a state/event inconsistency, a failed expectation or an exception. Identical "
            "repeated failures are still failures, even if replay is deterministic."
        ),
        "",
        (
            f"Python: {platform.python_version()}. Primary-pass inspected ticks: "
            f"{sum(r['ticks_advanced'] for r in results)}. "
            f"Identical replays: {sum(r['replay_identical'] for r in results)}/{len(results)}."
        ),
        "",
        "## Findings",
        "",
    ]
    if telemetry:
        lines.append(
            f"- **Restart acquisition telemetry inconsistency in {len(telemetry)} scenarios:** "
            "`possession_changed` reports an acquired owner but a different event last toucher. "
            "Compare the event with the resulting state's correct ball owner/last toucher in "
            "the raw JSON. The timeline preserves the original discrepancy. This is an "
            "instrumentation defect, not evidence that restart possession itself is wrong."
        )
    for result in failures:
        other = [error for error in result["errors"] if "telemetry:" not in error]
        if other:
            lines.append(f"- **{result['name']}:** " + "; ".join(other[:3]) + ".")
    if not failures:
        lines.append(
            "- No violated checks in this catalog; this is not exhaustive correctness evidence."
        )
    for result in results:
        if result["physics_limits"]:
            designation = "deliberate" if result["expected_limit"] else "unexpected"
            lines.append(
                f"- **{designation.capitalize()} saturation in {result['name']}:** `{result['physics_limits']}`."
            )
        for warning in result["model_warnings"]:
            lines.append(f"- **Model-hardening concern in {result['name']}:** {warning}")
    lines += [
        "",
        "Baseline audit and engine repair history: `docs/physics_fixes.md`. This report records the current executed probes, not the pre-fix results.",
        "",
        "## Physics/tuning feedback",
        "",
        (
            "- The reference roll and stopping probes check analytic distance, while player coast "
            "uses semi-implicit integration. Their distances should not be compared as if they "
            "used the same integrator; inspect the measured values below."
        ),
        (
            "- Four-second idle braking at speed 160 and much stronger braking at tiny positive "
            "effort are existing tuning choices. These need human visual play-testing, not a "
            "claim of physical realism from passing assertions."
        ),
        (
            "- Controlled possession may overlap the defender's ball-contact circle. This is a "
            "hybrid attachment/steal model; it is not an independent free-ball collision failure."
        ),
        (
            "- Explicit high-speed, reduced-budget and narrow-pitch fixtures are adversarial "
            "tests, not normal gameplay settings. Do not interpret their velocities as action kick powers."
        ),
        "",
        "## Scenario results",
        "",
        "| Scenario | Status | Ticks | Decisions | Goals | Collisions | Constraints | Limits | Replay |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        counts = result["event_counts"]
        lines.append(
            f"| {result['name']} | {result['status']} | {result['ticks_advanced']} | "
            f"{result['decision_count']} | {counts.get('goal', 0)} | "
            f"{counts.get('ball_collision', 0)} | {counts.get('ball_constraint', 0)} | "
            f"{counts.get('physics_limit', 0)} | {'same' if result['replay_identical'] else 'DIFFERENT'} |"
        )
    lines += ["", "## Per-scenario evidence and feedback", ""]
    for result in results:
        lines += [
            f"### {result['name']} — {result['status']}",
            "",
            result["purpose"],
            "",
            (
                f"- Advanced {result['ticks_advanced']}/{result['ticks_requested']} requested ticks; "
                f"final phase `{result['final_phase']}`, score {result['final_score']}, "
                f"owner `{result['final_owner'] or 'free'}`."
            ),
            (
                f"- Minimum stamina {result['min_stamina']:.6g}; maximum snapshot-observed player/ball "
                f"speeds {result['max_player_speed']:.6g}/{result['max_ball_speed']:.6g} units/s "
                "(includes injected initial velocities)."
            ),
        ]
        if result["min_body_clearance"] is not None:
            lines.append(
                f"- Minimum completed-snapshot body clearance: {result['min_body_clearance']:.6g} units."
            )
        for check in result["expectations"]:
            lines.append(
                f"- {'PASS' if check['passed'] else 'FAIL'} expectation: {check['description']}; "
                f"expected `{check['expected']}`, observed `{check['actual']}`."
            )
        for error in result["errors"]:
            lines.append(f"- **Failure:** {error}.")
        lines.extend(f"- {feedback}" for feedback in result["feedback"])
        lines.append("")
    lines += [
        "## Reproduction and artifacts",
        "",
        "```powershell",
        "uv run --locked --only-group engine python -m game.scenarios --output results/physics_scenarios",
        "# Single case:",
        "uv run --locked --only-group engine python -m game.scenarios --scenario 45_deliberate_budget_saturation --output results/physics_budget_probe",
        "```",
        "",
        (
            "The output directory contains `results.json` (initial/final state, first-failure context, config, agent plans, "
            "expectations, errors, all events and replay hashes), `report.md`, and one event "
            "timeline per scenario. `--report PATH` additionally writes this report at PATH. "
            "Audit exit status is 1 if any scenario FAILs, otherwise 0; reports are written before exit. "
            "Generated `results/` artifacts are ignored by the existing repository policy."
        ),
        "",
        "## Limitations",
        "",
        (
            "The catalog contains 50 named probes, not 50 full matches: horizons range from one tick to a "
            "full 3,600-tick match. Setup fixtures may inject positions, velocities, ownership, "
            "timers and compressed restarts. Seeded agents randomize actions only; resets are "
            "not randomized. This diagnostic controller is not the team's evaluation opponent "
            "and is not an RL policy. There is no subjective experience, visual observation, "
            "wall-clock throughput benchmark or reward evaluation. Linear snapshot sweeps, "
            "turning chords and bounded settling are not an exact accelerated rigid-body solver. "
            "The cadence check shares engine code and cannot independently prove physics correctness. "
            "State snapshots are inspected after ticks; assertions do not reconstruct every instant "
            "within a tick. Event timeline seconds use stored tick stamps; fractions remain explicit. "
            "No physics constants or mechanics were changed by this audit."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="list the 50 named probes without running"
    )
    parser.add_argument(
        "--scenario", action="append", help="run only this named probe (repeatable)"
    )
    parser.add_argument("--output", type=Path, default=Path("results/physics_scenarios"))
    parser.add_argument("--report", type=Path, help="also save the Markdown report at this path")
    args = parser.parse_args(argv)
    scenarios = build_scenarios()
    if args.scenario:
        unknown = set(args.scenario) - {scenario.name for scenario in scenarios}
        if unknown:
            parser.error(f"unknown scenarios: {', '.join(sorted(unknown))}")
        scenarios = tuple(scenario for scenario in scenarios if scenario.name in args.scenario)
    if args.list:
        for scenario in scenarios:
            print(f"{scenario.name}: {scenario.purpose}")
        return 0
    results = []
    for scenario in scenarios:
        result = run_scenario(scenario)
        results.append(result)
        print(
            f"{result['status']:4} {scenario.name}: {result['ticks_advanced']} ticks, "
            f"{len(result['errors'])} failed checks, replay={result['replay_identical']}",
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    status_counts = dict(Counter(result["status"] for result in results))
    payload = {
        "schema_version": 1,
        "scenario_count": len(results),
        "status_counts": status_counts,
        "python_version": platform.python_version(),
        "passes_per_scenario": 2,
        "reset_randomness": False,
        "results": results,
    }
    (args.output / "results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = render_report(results)
    (args.output / "report.md").write_text(report, encoding="utf-8")
    for result in results:
        (args.output / f"{result['name']}.timeline.txt").write_text(
            result["timeline"] + "\n", encoding="utf-8"
        )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(f"Summary: {len(results)} scenarios, {status_counts}. Evidence: {args.output}")
    return int(any(result["status"] == "FAIL" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
