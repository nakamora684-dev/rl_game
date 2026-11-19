"""Full-match behaviour of the baseline (P3).

Runs on Member A's real engine (agents.engine_adapter). These tests check SANITY
(finishes, moves, events once); the strength ordering of the presets is measured in
P5 with agents.baseline_stats.
"""

import pytest

from agents.baseline_stats import play_paired
from agents.difficulty import PRESETS, make_baseline
from agents.engine_adapter import EngineAdapter
from agents.interface import MATCH_TICKS, PHASE_FINISHED, PITCH_H, PITCH_W
from agents.match import run_match
from agents.metrics import MatchRecorder


def _engine():
    return EngineAdapter()


def _play(preset_a="normal", preset_b="normal", seeds=(1, 2, 3)):
    return play_paired(
        _engine(),
        lambda i: make_baseline(i, preset_a),
        lambda i: make_baseline(i, preset_b),
        seeds,
        names=(preset_a, preset_b),
    )


@pytest.mark.parametrize(
    "pa,pb", [("easy", "easy"), ("normal", "normal"), ("hard", "hard"), ("easy", "hard")]
)
def test_matches_finish_in_both_directions_with_clean_events(pa, pb):
    rows = _play(pa, pb, seeds=(1, 2))
    assert len(rows) == 4
    for r in rows:
        assert r["ticks"] == MATCH_TICKS and r["end_reason"] == "time"
        assert r["duplicate_events"] == 0
        # goal events counted once == the final score (events are the truth)
        assert (r["p0_goals"], r["p1_goals"]) == (r["score_p0"], r["score_p1"])


def test_baseline_can_score():
    rows = _play("normal", "normal", seeds=(1, 2, 3))
    assert sum(r["score_p0"] + r["score_p1"] for r in rows) >= 1


def test_baseline_is_not_passive_and_has_the_ball_sometimes():
    for r in _play("normal", "normal", seeds=(1, 2)):
        for i in (0, 1):
            assert r[f"p{i}_idle_frac"] < 0.5
        assert r["p0_possession_share"] + r["p1_possession_share"] > 0.4
        assert r["p0_kicks"] + r["p1_kicks"] >= 1


def test_no_out_of_bounds_loops():
    for r in _play("normal", "normal", seeds=(1, 2, 3)):
        assert r["p0_out_of_bounds"] + r["p1_out_of_bounds"] <= 10


def test_stamina_stays_in_range_and_is_actually_used():
    for r in _play("normal", "normal", seeds=(1,)):
        for i in (0, 1):
            assert 0.0 <= r[f"p{i}_stamina_end"] <= 1.0
            assert 0.0 < r[f"p{i}_sprint_frac"] < 1.0  # sprints, but not always


def test_players_stay_on_the_pitch():
    bad = []

    def hook(st):
        for p in st.players:
            if not (-1 <= p.x <= PITCH_W + 1 and -1 <= p.y <= PITCH_H + 1):
                bad.append((st.tick, p.x, p.y))

    agents = [make_baseline(i) for i in (0, 1)]
    run_match(_engine(), agents, seed=1, on_step=hook)
    assert not bad


def test_every_main_rule_fires_in_a_match():
    seen = set()
    agents = [make_baseline(i) for i in (0, 1)]

    def hook(st):
        for a in agents:
            seen.add(a.agent.last_rule)

    final = run_match(_engine(), agents, seed=1, on_step=hook)
    assert final.phase == PHASE_FINISHED
    assert {"chase", "carry", "shoot"} <= seen
    assert seen & {"contest", "block"}


def test_same_seed_gives_identical_rows():
    assert _play(seeds=(5,)) == _play(seeds=(5,))


def test_presets_play_without_error_against_each_other():
    for a in PRESETS:
        for b in PRESETS:
            rec = MatchRecorder()
            ag = [rec.wrap(make_baseline(0, a), 0), rec.wrap(make_baseline(1, b), 1)]
            final = run_match(_engine(), ag, seed=3, on_step=rec.on_step, max_ticks=600)
            assert final.tick == 600
