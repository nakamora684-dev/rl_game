"""Run baseline-vs-baseline matches headless and write a CSV + a short summary.

    uv run python -m agents.baseline_stats --seeds 25                  # normal vs normal
    uv run python -m agents.baseline_stats --a hard --b easy --seeds 25
    uv run python -m agents.baseline_stats --seeds 5 --out results/tmp.csv

Every seed is played TWICE (attack direction normal, then swapped), as required
by the evaluation protocol (plan, section 7). Agent A is always player 0, agent B
player 1; `swapped` only flips the attack direction. The CSV schema is
agents.metrics.ROW_FIELDS (same schema Member D can use for PPO-vs-baseline).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from agents.difficulty import PRESETS, make_baseline
from agents.match import run_match
from agents.metrics import ROW_FIELDS, MatchRecorder


def play_paired(
    engine,
    make_a: Callable,
    make_b: Callable,
    seeds: Iterable[int],
    names: tuple[str, str] = ("A", "B"),
) -> list[dict]:
    """Play each seed in both attack directions. `make_x(index)` builds an agent."""
    rows = []
    for seed in seeds:
        for swapped in (False, True):
            rec = MatchRecorder()
            agents = [rec.wrap(make_a(0), 0), rec.wrap(make_b(1), 1)]
            final = run_match(engine, agents, seed=seed, swap_sides=swapped, on_step=rec.on_step)
            rows.append(rec.summary(final, seed=seed, swapped=swapped, names=names))
    return rows


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def summarize(rows: list[dict]) -> str:
    """W/D/L and mean goal difference (with a 95% normal-approximation CI) for A."""
    n = len(rows)
    if n == 0:
        return "no matches"
    gds = [r["goal_diff_p0"] for r in rows]
    mean = sum(gds) / n
    sd = math.sqrt(sum((g - mean) ** 2 for g in gds) / (n - 1)) if n > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(n)
    w = sum(g > 0 for g in gds)
    lost = sum(g < 0 for g in gds)
    d = n - w - lost
    out = [
        f"{rows[0]['p0_agent']} (p0) vs {rows[0]['p1_agent']} (p1): n={n} matches",
        f"  W/D/L = {w}/{d}/{lost}   goal diff = {mean:+.2f} +/- {ci:.2f} (95% CI)",
    ]
    for swapped in (0, 1):
        sub = [r["goal_diff_p0"] for r in rows if r["swapped"] == swapped]
        if sub:
            out.append(
                f"  attack direction {'swapped' if swapped else 'normal '}: "
                f"mean goal diff {sum(sub) / len(sub):+.2f} (n={len(sub)})"
            )
    dup = sum(r["duplicate_events"] for r in rows)
    out.append(f"  duplicate events: {dup} (must be 0)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Baseline-vs-baseline statistics")
    ap.add_argument("--a", default="normal", choices=sorted(PRESETS))
    ap.add_argument("--b", default="normal", choices=sorted(PRESETS))
    ap.add_argument("--seeds", type=int, default=10, help="seeds (2 matches each)")
    ap.add_argument("--first-seed", type=int, default=1000, help="held-out seed range start")
    ap.add_argument("--out", default="results/baseline_stats.csv")
    args = ap.parse_args(argv)

    from play import load_engine  # lazy: play.py imports no pygame at top

    engine, label = load_engine()
    print(f"Using {label}")
    rows = play_paired(
        engine,
        lambda i: make_baseline(i, args.a),
        lambda i: make_baseline(i, args.b),
        range(args.first_seed, args.first_seed + args.seeds),
        names=(args.a, args.b),
    )
    print(f"wrote {write_csv(rows, args.out)}")
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
