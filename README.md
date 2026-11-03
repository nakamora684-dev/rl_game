# 1v1 Football DRL
PPO agent vs rule-based baseline in a 2D top-down football game.
RLOC project 2026/2027, deadline 31 Dec 2026.

## Setup
```bash
git clone <repo-url> && cd <repo>
uv sync
uv run pytest
```

## Run
- `uv run python play.py`      – human-playable game
- `uv run python observe.py`   – watch an agent/baseline
- `uv run python train.py`     – train PPO (headless)
- `uv run python test.py`      – evaluate RL vs baseline → results/*.csv
- `uv run jupyter lab`         – open evaluate.ipynb

## Structure
See project plan, section 10.

## Machine specs
(fill in: CPU, GPU, RAM, OS, Python version)