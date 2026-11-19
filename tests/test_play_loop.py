"""End-to-end run of play.main() on a FAKE pygame: menu-less start, pause/resume,
debug + screenshot keys, a full 60 s match, replay saved, rematch, quit.
No display and no pygame install needed. (A real window still has to be checked by eye.)"""

import os
import sys
import time
from types import SimpleNamespace

from tests.test_render_readonly import _fake_pygame


class _NoKeys(dict):
    def __missing__(self, key):
        return False


def _scripted_pygame(script):
    pg = _fake_pygame()
    pg.key.get_pressed.return_value = _NoKeys()
    pg.joystick.get_count.return_value = 0
    calls = {"n": 0}

    def get():
        calls["n"] += 1
        out = []
        for kind in script.get(calls["n"], []):
            if kind == "quit":
                out.append(SimpleNamespace(type=pg.QUIT))
            else:
                out.append(SimpleNamespace(type=pg.KEYDOWN, key=getattr(pg, kind)))
        return out

    pg.event.get.side_effect = get
    return pg, calls


def test_play_runs_a_whole_session_on_fake_pygame(tmp_path):
    script = {10: ["K_p"], 15: ["K_p"], 20: ["K_F1"], 25: ["K_F12"], 3800: ["K_r"], 3830: ["quit"]}
    pg, calls = _scripted_pygame(script)
    old_cwd, old_pg = os.getcwd(), sys.modules.get("pygame")
    sys.modules["pygame"] = pg
    os.chdir(tmp_path)
    try:
        import play

        t0 = time.time()
        rc = play.main(["--no-menu", "--seed", "2", "--swap"])
        assert rc == 0
        replays = list((tmp_path / "results" / "replays").glob("*.json"))
        assert len(replays) == 1 and "seed2" in replays[0].name  # saved once, at full time
        assert pg.image.save.call_count == 1  # F12 screenshot
        assert pg.display.flip.call_count > 3600
        assert pg.quit.called
        assert calls["n"] >= 3830 and time.time() - t0 < 120
    finally:
        os.chdir(old_cwd)
        if old_pg is None:
            sys.modules.pop("pygame", None)
        else:
            sys.modules["pygame"] = old_pg


def test_play_menu_flow_starts_and_quits(tmp_path):
    script = {
        2: ["K_RIGHT"],
        3: ["K_DOWN"],
        4: ["K_RIGHT"],
        5: ["K_RETURN"],
        40: ["K_ESCAPE"],
        41: ["K_DOWN"],
        42: ["K_DOWN"],
        43: ["K_DOWN"],
        44: ["K_RETURN"],
    }  # pause -> Quit
    pg, calls = _scripted_pygame(script)
    old_cwd, old_pg = os.getcwd(), sys.modules.get("pygame")
    sys.modules["pygame"] = pg
    os.chdir(tmp_path)
    try:
        import play

        assert play.main([]) == 0
        assert calls["n"] == 44  # loop ended on the Quit entry
    finally:
        os.chdir(old_cwd)
        if old_pg is None:
            sys.modules.pop("pygame", None)
        else:
            sys.modules["pygame"] = old_pg
