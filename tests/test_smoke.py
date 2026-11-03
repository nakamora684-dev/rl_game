import pathlib

def test_imports():
    import numpy, gymnasium, torch, stable_baselines3  # noqa: F401

def test_game_does_not_import_pygame():
    """Plan rule: the shared simulation must not depend on Pygame."""
    for f in pathlib.Path("game").rglob("*.py"):
        text = f.read_text()
        assert "import pygame" not in text and "from pygame" not in text, f