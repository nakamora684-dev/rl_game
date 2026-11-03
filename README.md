## Prerequisites

- **Git**: https://git-scm.com/downloads
- **uv** (Python package manager): https://docs.astral.sh/uv/getting-started/installation/

Install uv on Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install uv on macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Check it worked (reopen the terminal first):

```bash
uv --version
```

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