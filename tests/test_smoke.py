import pathlib

import pytest


@pytest.mark.full_stack
def test_imports():
    import gymnasium  # noqa: F401
    import numpy  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch  # noqa: F401


def test_game_does_not_import_pygame():
    """Plan rule: the shared simulation must not depend on Pygame."""
    for f in pathlib.Path("game").rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert "import pygame" not in text and "from pygame" not in text, f
