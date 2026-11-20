"""TensorBoard + CSV logging for training (no torch needed to import this file)."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path


class _CsvSink:
    def __init__(self, path: Path) -> None:
        self.path, self._writer, self._file = path, None, None

    def write(self, row: dict) -> None:
        if self._writer is None:
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()


class TrainingLogger:
    """Logs the undiscounted per-episode reward (raw + moving average) as required.

    Writes ``episodes.csv`` and ``updates.csv`` next to the TensorBoard events so that
    ``evaluate.ipynb`` can plot learning curves without parsing event files.
    """

    def __init__(self, log_dir: str | Path, moving_average: int = 100, tensorboard: bool = True):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(self.dir))
            except ImportError:
                print("[logger] tensorboard not installed: CSV logging only.")
        self._episodes = _CsvSink(self.dir / "episodes.csv")
        self._updates = _CsvSink(self.dir / "updates.csv")
        self._evals = _CsvSink(self.dir / "evals.csv")
        self._returns: deque[float] = deque(maxlen=moving_average)
        self.episode_count = 0

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), step)

    def episode(self, ep: dict, step: int) -> None:
        self.episode_count += 1
        self._returns.append(ep["ep_return"])
        self.scalar("rollout/ep_return", ep["ep_return"], step)  # undiscounted total
        self.scalar("rollout/ep_return_ma", sum(self._returns) / len(self._returns), step)
        self.scalar("episode/goal_diff", ep["goal_diff"], step)
        self.scalar("episode/win", float(ep["result"] == "win"), step)
        self.scalar("episode/length", ep["steps"], step)
        for key, value in ep.items():
            if key.startswith("r_"):
                self.scalar(f"reward_components/{key[2:]}", value, step)
        self._episodes.write(
            {"global_step": step, "episode": self.episode_count, **ep}
        )

    def update(self, stats: dict, step: int) -> None:
        for key, value in stats.items():
            self.scalar(key, value, step)
        self._updates.write({"global_step": step, **{k.replace("/", "_"): v for k, v in stats.items()}})

    def evaluation(self, stats: dict, step: int) -> None:
        for key, value in stats.items():
            self.scalar(f"eval/{key}", value, step)
        self._evals.write({"global_step": step, **stats})

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        for sink in (self._episodes, self._updates, self._evals):
            sink.close()
