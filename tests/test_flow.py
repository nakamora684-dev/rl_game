"""Menu / pause / full-time state machine (pure Python)."""

from rendering.flow import (
    DIFFICULTIES,
    FULLTIME,
    MENU,
    PAUSED,
    PLAYING,
    Flow,
    MatchOptions,
)


def test_menu_changes_options_and_starts():
    f = Flow()
    assert f.state == MENU and f.options.difficulty == "normal"
    f.handle("right")
    assert f.options.difficulty == "hard"
    f.handle("right")  # wraps around
    assert f.options.difficulty == "easy"
    f.handle("left")
    assert f.options.difficulty == "hard"
    f.handle("down")
    f.handle("right")
    assert f.options.swap_sides is True
    f.handle("up")
    f.handle("up")  # wraps to the last row ("start")
    assert f.menu_row == 2
    assert f.handle("enter") == "start" and f.state == PLAYING


def test_menu_escape_quits_and_unknown_keys_do_nothing():
    f = Flow()
    assert f.handle("x") is None and f.handle(None) is None
    assert f.handle("escape") == "quit"


def test_pause_resume_restart_menu_quit():
    f = Flow(start_in_menu=False)
    assert f.state == PLAYING
    assert f.handle("p") is None and f.state == PAUSED
    assert f.handle("escape") == "resume" and f.state == PLAYING
    f.handle("escape")
    f.handle("down")
    assert f.handle("enter") == "restart" and f.state == PLAYING
    f.handle("p")
    f.handle("down")
    f.handle("down")
    assert f.handle("enter") == "menu" and f.state == MENU
    f.state = PAUSED
    f.pause_row = 3
    assert f.handle("enter") == "quit"


def test_finish_and_rematch_next_seed_and_alternating_sides():
    f = Flow(MatchOptions(seed=4, alternate=True), start_in_menu=False)
    f.finish()
    assert f.state == FULLTIME
    assert f.handle("r") == "rematch" and f.state == PLAYING
    assert f.options.seed == 5 and f.options.swap_sides is True
    f.finish()
    f.handle("enter")
    assert f.options.seed == 6 and f.options.swap_sides is False


def test_rematch_keeps_side_without_alternate_and_m_returns_to_menu():
    f = Flow(MatchOptions(swap_sides=True), start_in_menu=False)
    f.finish()
    f.handle("r")
    assert f.options.swap_sides is True
    f.finish()
    assert f.handle("m") == "menu" and f.state == MENU


def test_finish_is_ignored_outside_playing():
    f = Flow()
    f.finish()
    assert f.state == MENU
    assert set(DIFFICULTIES) == {"easy", "normal", "hard"}
