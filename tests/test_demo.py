import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_headless_module_demo():
    result = subprocess.run(
        [sys.executable, "-m", "game"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    summary = json.loads(result.stdout)
    assert summary["final_tick"] == 3600
    assert summary["finished"] is True
    assert summary["terminal_events"] == 1
    assert summary["checkpoints"][0]["tick"] == 30
    assert summary["checkpoints"][0]["player_a_speed"] == pytest.approx(158.0)
    assert summary["checkpoints"][0]["player_a_stamina"] == pytest.approx(97.5)
    assert summary["checkpoints"][1]["tick"] == 270
    assert summary["checkpoints"][1]["player_a_speed"] == 0.0
    assert summary["checkpoints"][1]["player_a_stamina"] == 100.0
    assert 200 < summary["checkpoints"][0]["player_a_position"][0]
    assert (
        summary["checkpoints"][0]["player_a_position"][0]
        < summary["checkpoints"][1]["player_a_position"][0]
        < 785
    )


def test_stamina_inspection_csv():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--stamina"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert len(rows) == 36
    assert float(rows[1]["stamina"]) == 95.0
    assert float(rows[20]["stamina"]) == 0.0
    assert float(rows[20]["speed_limit"]) == 80.0
    assert float(rows[20]["acceleration_limit"]) == 160.0
    assert float(rows[35]["stamina"]) == 100.0
    assert float(rows[35]["speed"]) == 0.0


def test_ball_inspection_csv():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--ball"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert len(rows) == 5
    assert float(rows[1]["x"]) == 250.0
    assert float(rows[1]["speed"]) == 200.0
    assert float(rows[2]["x"]) == 400.0
    assert float(rows[3]["x"]) == 450.0
    assert float(rows[3]["speed"]) == 0.0
    assert float(rows[4]["x"]) == 450.0


def test_dribble_inspection_csv_follows_facing_while_coasting_sideways():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--dribble"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert [int(row["tick"]) for row in rows] == [0, 6, 24, 84]
    assert all(row["owner"] == row["last_toucher"] == "A" for row in rows)
    assert float(rows[0]["ball_x"]) == float(rows[0]["player_x"]) + 22.0
    assert float(rows[2]["player_vx"]) > 0.0
    assert float(rows[2]["player_vy"]) == 0.0
    assert float(rows[2]["facing"]) == pytest.approx(-1.570796, abs=1e-6)
    assert float(rows[2]["ball_x"]) == float(rows[2]["player_x"])
    assert float(rows[2]["ball_y"]) == float(rows[2]["player_y"]) - 22.0
    assert float(rows[3]["player_vx"]) == float(rows[3]["player_vy"]) == 0.0
    assert all(float(row["ball_vx"]) == float(row["ball_vy"]) == 0.0 for row in rows)


def test_kick_inspection_csv_has_one_release_then_free_motion():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--kick"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert [int(row["tick"]) for row in rows] == [0, 60, 120, 180, 240]
    assert all(row["owner"] == "none" and row["last_toucher"] == "A" for row in rows)
    assert [row["events"] for row in rows] == ["ball_kicked", "", "", "", ""]
    assert [float(row["ball_x"]) for row in rows] == [222.0, 472.0, 622.0, 672.0, 672.0]
    assert [float(row["speed"]) for row in rows] == [300.0, 200.0, 100.0, 0.0, 0.0]


def test_kick_action_inspection_csv_compares_powers():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--kick-actions"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert [float(row["power"]) for row in rows] == [0.0, 0.5, 1.0]
    assert [float(row["ball_x"]) for row in rows] == [226.5, 239.0, 251.5]
    assert [float(row["speed"]) for row in rows] == [40.0, 165.0, 290.0]
    assert all(row["tick"] == row["cooldown_ticks"] == "6" for row in rows)
    assert all(row["kick_events"] == "1" for row in rows)


def test_pickup_inspection_shows_one_free_ball_recovery_not_a_steal():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--pickup"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert [int(row["tick"]) for row in rows] == [0, 1, 12, 98, 104]
    assert [row["owner"] for row in rows] == ["A", "none", "none", "B", "B"]
    assert [row["last_toucher"] for row in rows] == ["A", "A", "A", "B", "B"]
    assert [row["events"] for row in rows] == ["setup", "kick", "", "recovery", ""]
    assert [int(row["a_pickup_lockout"]) for row in rows] == [0, 11, 0, 0, 0]
    assert float(rows[3]["ball_x"]) == 578.0


def test_steal_inspection_shows_contest_then_protection_without_immediate_toggle():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--steal"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert [int(row["tick"]) for row in rows] == [0, 17, 18, 48, 65, 66]
    assert [row["owner"] for row in rows] == ["A", "A", "B", "B", "B", "A"]
    assert [int(row["contest_ticks"]) for row in rows] == [0, 17, 0, 0, 17, 0]
    assert [int(row["protection_ticks"]) for row in rows] == [0, 0, 30, 0, 0, 30]
    assert [row["events"] for row in rows] == ["setup", "", "steal", "", "", "steal"]


def test_rules_inspection_distinguishes_goals_boundaries_and_restart_causes():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--rules"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert [int(row["tick"]) for row in rows] == [1, 31, 61, 67] * 2
    assert [row["score_a"] for row in rows] == ["1"] * 4 + ["0"] * 4
    assert [row["events"] for row in rows] == [
        "goal:goal",
        "",
        "possession_changed:kickoff",
        "",
        "out_of_bounds:boundary",
        "",
        "possession_changed:boundary_restart",
        "",
    ]
    assert rows[2]["owner"] == rows[6]["owner"] == "B"
    assert float(rows[2]["ball_x"]) == 400.0
    assert float(rows[6]["ball_x"]) == 763.0


def test_contacts_inspection_shows_rebounds_control_loss_and_post_goal():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--contacts"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert [row["scenario"] for row in rows] == [
        "locked_player",
        "post",
        "controlled_post",
        "post_then_goal",
    ]
    assert all(row["tick"] == "1" for row in rows)
    assert float(rows[0]["vx"]) == -150.0
    assert float(rows[1]["vx"]) == -840.0
    assert rows[0]["last_toucher"] == "A" and rows[1]["last_toucher"] == "B"
    assert rows[2]["events"].startswith("possession_changed:post_collision|")
    assert "ball_constraint:opposed_contacts" in rows[2]["events"]
    assert "physics_limit" not in rows[2]["events"]
    assert rows[3]["phase"] == "goal_restart"
    assert rows[3]["events"] == "ball_collision:goalpost|goal:goal"


@pytest.mark.parametrize("script", ["idle", "scripted"])
@pytest.mark.parametrize("swap", [False, True])
def test_full_match_cli(script, swap):
    arguments = [sys.executable, "-m", "game", "--match", script]
    if swap:
        arguments.append("--swap-sides")
    result = subprocess.run(
        arguments,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    summary = json.loads(result.stdout)
    assert summary["controllers"] == script
    assert summary["swap_sides"] is swap
    assert summary["final_tick"] == summary["physics_ticks"] == 3600
    assert summary["decision_count"] == 600
    assert summary["finished"] is True
    assert summary["event_counts"]["match_finished"] == 1
    assert set(summary["diagnostics"]) == {"ball_collision", "ball_constraint", "physics_limit"}
    for kind, diagnostic in summary["diagnostics"].items():
        assert diagnostic["count"] == summary["event_counts"].get(kind, 0)
        assert sum(diagnostic["causes"].values()) == diagnostic["count"]
    if script == "idle":
        assert (summary["score_a"], summary["score_b"]) == (0, 0)
    else:
        assert summary["event_counts"].get("ball_kicked", 0) == 1


def test_match_cli_defaults_to_idle(capsys):
    from game.__main__ import main

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", ["game", "--match"])
        main()
    assert json.loads(capsys.readouterr().out)["controllers"] == "idle"


def test_swap_sides_cli_requires_match():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--swap-sides"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 2
    assert "--swap-sides requires --match" in result.stderr


def test_match_timeline_uses_stderr_and_preserves_json_stdout():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--match", "scripted", "--timeline"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    summary = json.loads(result.stdout)
    assert summary["event_counts"]["goal"] == 1
    assert "owner=free->A" in result.stderr
    assert "cause=kickoff" in result.stderr
    assert "match_finished" in result.stderr
    assert "60.000" in result.stderr


def test_timeline_requires_a_match():
    result = subprocess.run(
        [sys.executable, "-m", "game", "--timeline"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 2
    assert "--timeline requires --match" in result.stderr
