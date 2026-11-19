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
uv run --locked --only-group engine python -m game --dribble
uv run --locked --only-group engine python -m game --kick
uv run --locked --only-group engine python -m game --kick-actions
uv run --locked --only-group engine python -m game --pickup
uv run --locked --only-group engine python -m game --steal
uv run --locked --only-group engine python -m game --rules
uv run --locked --only-group engine python -m game --contacts
```

The inspection commands print a movement/clock demonstration, a 20-second sprint /
15-second rest CSV trace, a reference free-ball roll, controlled dribbling, and direct kick release CSVs.
None is a playable game; possession assignment in the dribble demo is explicit setup,
not an implemented pickup/steal rule.
`--kick` isolates direct release at the reference speed. `--kick-actions` compares full
decision actions at powers 0/0.5/1: linear 50–300 units/s mapping, first-tick-only kick
attempts, and a 12-tick per-player cooldown. See `docs/engine.md` for exact semantics.
`--pickup` shows automatic interception: contact radius 22, any-speed control, swept
first-contact selection, and a separate 12-tick kicker pickup lockout. Exact ties stay
free. Recovery events are distinct from setup and stealing. `--kick` keeps the opponent
off the shot path to isolate roll physics. Player bodies now use sliding, non-bouncy
contact. `--steal` shows the 18-tick contest and 30-tick protection; `--rules` compares
goals, last-touch boundaries and one-second restarts. Stamina freezes during restarts
while match time continues. `--contacts` isolates locked-player/post rebounds, controlled
post loss and post→goal in one tick; it is not a full-match runner. Ball normal restitution
is 0.5 off pickup-locked players (no recoil) and 0.7 off radius-4 posts; eligible players
still trap. Bounded multi-contact processing emits collision/constraint/limit telemetry.
The shared core has a headless full-match runner, and `environment/` now wraps that
same simulation for Gymnasium; no renderer or PPO action integration exists yet.
Default strict-clearance goals intentionally require kicks. Snapshot sweeps are not an exact rigid-body solver;
visual validation and tuning remain. See [the next-session handoff](docs/next_session.md).
The `full_stack` dependency-import test is intentionally excluded from engine-only runs.
Use `--only-group engine` on `uv run` too: omitting it can install the full project stack.

### Members A/C: lightweight real-engine environment tests

```powershell
uv sync --locked --only-group environment
uv run --locked --only-group environment pytest -m "not full_stack" -q
uv run --locked --only-group environment ruff check game environment tests
uv run --locked --only-group environment ruff format --check game environment tests
```

This adds only NumPy/Gymnasium to the engine tools, not Torch/Pygame/SB3. Engine-only
runs visibly skip optional-dependency tests. See [the environment handoff](docs/environment.md)
for usage, exact hybrid actions, the new 50-feature observation and restart/event rules.
**Standard SB3 PPO cannot directly use this mixed Dict action space**; its C/D
integration remains separate. Experimental reward settings were preserved, not finalized.

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
open `index.html` for the review landing page, `graph.html` for the interactive map, `GRAPH_TREE.html` for a file-oriented tree,
or `GRAPH_REPORT.md` for the text summary; `graph.json` is the queryable graph.
The graph covers code, tests, current documentation, and the original Word plan's
text/tables/figures. Current decisions supersede older plan proposals where noted.

```powershell
graphify query "free ball stopping distance" --budget 1200
graphify query "controlled possession setup versus automatic pickup" --budget 1200
```

The graph is a snapshot: rebuild with `/graphify .` after substantive code/design
changes (use `/graphify . --update` for an incremental skill refresh). A bare CLI
`graphify update .` only refreshes code and is not a replacement for revisiting changed
design documents. No cloud upload, paid API key, or Git hooks are needed for this setup.
Only share reviewable outputs you want in Git; local caches/previews are ignored.

## Planned entry points (currently scaffold files)
- `uv run python play.py`      – human-playable game vs the rule-based baseline (Member B, implemented)
- `uv run python observe.py`   – watch baseline vs baseline, or `--headless --episodes N --swap` (Member B, implemented)
- `uv run python train.py`     – train PPO (headless)
- `uv run python test.py`      – evaluate RL vs baseline → results/*.csv
- `uv run jupyter lab`         – open evaluate.ipynb

## Member B: graphical mode and rule-based baseline
See [docs/B_integration.md](docs/B_integration.md): `agents/engine_adapter.py` is the 60 Hz bridge used by the
renderer, and `agents/provider.py` exposes the baseline as a native `ActionProvider` for `FootballEnv(opponent=...)`.

## Structure
See project plan, section 10. Current engine API, tuning, and limitations: [docs/engine.md](docs/engine.md).

## Machine specs
(fill in: CPU, GPU, RAM, OS, Python version)
