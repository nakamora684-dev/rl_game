# 1v1 Football DRL
PPO agent vs rule-based baseline in a 2D top-down football game.
RLOC project 2026/2027, deadline 31 Dec 2026.

## Setup
Use uv (validated with 0.12.15). `.python-version` pins Python **3.12.14**.
Python packages and their transitive versions are recorded in `pyproject.toml` and `uv.lock`.
Do not commit `.venv/`; teammates recreate it from these files.

### Member A: lightweight engine environment
Only pytest, Ruff, and their locked dependencies are installed. No graphics or ML packages are needed.

```powershell
uv sync --locked --only-group engine
uv run --locked --only-group engine pytest -m "not full_stack"
uv run --locked --only-group engine ruff check game tests
uv run --locked --only-group engine ruff format --check game tests
uv run --locked --only-group engine python -m game
uv run --locked --only-group engine python -m game --stamina
uv run --locked --only-group engine python -m game --ball
```

The inspection commands print a movement/clock demonstration, a 20-second sprint /
15-second rest CSV trace, and a reference free-ball roll CSV. None is a playable game.
The `full_stack` dependency-import test is intentionally excluded from engine-only runs.
Use `--only-group engine` on `uv run` too: omitting it can install the full project stack.

### Full team environment
This also installs rendering, training, notebook, and evaluation dependencies (including PyTorch).

```powershell
uv sync --locked
uv run --locked pytest
```

For future dependency changes, use `uv add <package>` (runtime) or
`uv add --group engine <package>` (engine tooling), then share both `pyproject.toml` and `uv.lock`.
Keep the Python pin updated if the team deliberately changes interpreter versions.

### Optional Graphify tooling
Graphify is installed separately with uv tools, not imported by the game or added to
the lightweight engine environment. Its version is recorded in the optional `graphify`
dependency group; `uv.lock` records the corresponding dependency versions.

```powershell
uv tool install --python 3.12.14 graphifyy==0.9.103
graphify install --platform opencode
graphify --version
```

This registers a user-level OpenCode skill and reminder plugin under
`~/.config/opencode/`. Each teammate must register their own installation; global
tool environments and registrations are not included in a Git push. In this release,
the bare `graphify install` defaults to Claude Code on Windows, so select OpenCode explicitly.
Installing does not build a graph or configure a hosted service. Graph building is
a separate action; document/media semantic extraction may use an LLM provider.

### Review the project knowledge graph
Start with [docs/project_review.md](docs/project_review.md) for implementation status
and approved-versus-planned decisions. Graphify outputs live in `graphify-out/`:
open `graph.html` for the interactive map, `GRAPH_TREE.html` for a file-oriented tree,
or `GRAPH_REPORT.md` for the text summary; `graph.json` is the queryable graph.
The graph covers code, tests, current documentation, and the original Word plan's
text/tables/figures. Current decisions supersede older plan proposals where noted.

```powershell
graphify query "free ball stopping distance" --budget 1200
graphify query "implemented features and planned possession" --budget 1200
```

The graph is a snapshot: rebuild with `/graphify .` after substantive code/design
changes (use `/graphify . --update` for an incremental skill refresh). A bare CLI
`graphify update .` only refreshes code and is not a replacement for revisiting changed
design documents. No cloud upload, paid API key, or Git hooks are needed for this setup.
Only share reviewable outputs you want in Git; local caches/previews are ignored.

## Planned entry points (currently scaffold files)
- `uv run python play.py`      – human-playable game
- `uv run python observe.py`   – watch an agent/baseline
- `uv run python train.py`     – train PPO (headless)
- `uv run python test.py`      – evaluate RL vs baseline → results/*.csv
- `uv run jupyter lab`         – open evaluate.ipynb

## Structure
See project plan, section 10. Current engine API, tuning, and limitations: [docs/engine.md](docs/engine.md).

## Machine specs
(fill in: CPU, GPU, RAM, OS, Python version)
