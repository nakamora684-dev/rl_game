"""One human-vs-baseline match, WITHOUT pygame (so it is unit-tested headless).

play.py only adds windows, keys and drawing around this class. The same engine,
the same baseline (agents.difficulty) and the same metrics (agents.metrics) as
headless evaluation are used: graphical mode is not a second implementation.
"""

from __future__ import annotations

from pathlib import Path

from agents.difficulty import make_baseline
from agents.interface import PHASE_FINISHED, PHASE_PLAYING, Action, GameState
from agents.metrics import SPRINT_EFFORT, MatchRecorder
from rendering.flow import MatchOptions
from rendering.replay import ReplayRecorder

HUMAN, BOT = 0, 1


class Session:
    def __init__(self, engine, options: MatchOptions) -> None:
        self.engine, self.options = engine, options
        self.bot = make_baseline(BOT, options.difficulty)
        self.rec = MatchRecorder()
        self.replay = ReplayRecorder(
            options.seed, options.swap_sides, difficulty=options.difficulty, human=HUMAN
        )
        self.state: GameState = (
            engine.reset(seed=options.seed, swap_sides=options.swap_sides)
            if options.swap_sides
            else engine.reset(seed=options.seed)
        )
        self.bot.reset()

    @property
    def finished(self) -> bool:
        return self.state.phase == PHASE_FINISHED

    def tick(self, human: Action) -> GameState:
        """Advance ONE physics update with the human action and the baseline decision."""
        if self.finished:
            return self.state
        bot = self.bot.act(self.state)
        if self.state.phase == PHASE_PLAYING:
            self.rec.sprint[HUMAN] += 1 if human.effort >= SPRINT_EFFORT else 0
            self.rec.sprint[BOT] += 1 if bot.effort >= SPRINT_EFFORT else 0
        self.replay.record([human, bot])
        self.state = self.engine.step([human, bot])
        self.rec.on_step(self.state)
        return self.state

    def summary_row(self) -> dict:
        return self.rec.summary(
            self.state,
            seed=self.options.seed,
            swapped=self.options.swap_sides,
            names=("human", f"bot-{self.options.difficulty}"),
        )

    def save_replay(self, folder: str | Path = "results/replays") -> Path:
        o = self.options
        return self.replay.save(
            Path(folder) / f"replay_seed{o.seed}_{o.difficulty}"
            f"{'_swap' if o.swap_sides else ''}.json"
        )
