"""Pure helpers: colours, clock, blink, pitch geometry (no pygame)."""

import math

from agents.interface import GOAL_W, PITCH_H, PITCH_W
from rendering import pitch, theme
from rendering.hud import blink, format_clock


def test_stamina_color_thresholds():
    assert theme.stamina_color(1.0) == theme.STAMINA_OK
    assert theme.stamina_color(0.5) == theme.STAMINA_MID  # 0.5 is not "> 0.5"
    assert theme.stamina_color(0.26) == theme.STAMINA_MID
    assert theme.stamina_color(0.25) == theme.STAMINA_LOW
    assert theme.stamina_color(0.0) == theme.STAMINA_LOW


def test_shade_and_mix_stay_in_range():
    assert theme.shade((200, 100, 50), 0.5) == (100, 50, 25)
    assert theme.shade((200, 200, 200), 5.0) == (255, 255, 255)
    assert theme.mix((0, 0, 0), (200, 100, 50), 0.5) == (100, 50, 25)
    assert theme.mix((0, 0, 0), (10, 10, 10), 9.0) == (10, 10, 10)


def test_format_clock():
    assert format_clock(60) == "1:00"
    assert format_clock(59.2) == "1:00"  # rounds up
    assert format_clock(9.0) == "0:09"
    assert format_clock(0.0) == "0:00" and format_clock(-3) == "0:00"


def test_blink_alternates_twice_per_second():
    states = [blink(t, 2.0) for t in range(60)]
    assert states[0] is True and states[15] is False and states[30] is True
    assert sum(states) == 30


def test_boxes_are_centred_on_each_goal():
    for side in (0, 1):
        for depth, height in ((pitch.PEN_DEPTH, pitch.PEN_H), (pitch.SIX_DEPTH, pitch.SIX_H)):
            x, y, w, h = pitch.box(side, depth, height)
            assert math.isclose(y + h / 2, PITCH_H / 2)
            assert (x == 0.0) if side == 0 else math.isclose(x + w, PITCH_W)
    assert pitch.PEN_H > GOAL_W and pitch.SIX_H > GOAL_W


def test_nets_sit_behind_the_goal_line_and_match_the_mouth():
    x0, y0, w0, h0 = pitch.net_rect(0)
    x1, y1, w1, h1 = pitch.net_rect(1)
    assert x0 + w0 == 0.0 and x1 == PITCH_W  # outside the pitch
    assert math.isclose(h0, GOAL_W) and math.isclose(h1, GOAL_W)
    assert theme.NET_DEPTH * 1.0 < theme.MARGIN  # fits in the margin at scale 1


def test_penalty_spots_symmetric():
    (xa, ya), (xb, yb) = pitch.penalty_spot(0), pitch.penalty_spot(1)
    assert math.isclose(xa, PITCH_W - xb) and ya == yb == PITCH_H / 2
