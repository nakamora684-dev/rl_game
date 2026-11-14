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
