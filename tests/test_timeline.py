from game.player import PlayerId
from game.simulation import SimulationEvent
from game.timeline import format_timeline


def test_timeline_preserves_emission_order_and_raw_tick_stamps():
    events = (
        SimulationEvent("possession_changed", 1, new_owner=PlayerId.A, cause="recovery"),
        SimulationEvent("ball_kicked", 0, previous_owner=PlayerId.A, cause="kick"),
        SimulationEvent("match_finished", 3600),
    )
    text = format_timeline(iter(events))
    assert text.index("cause=recovery") < text.index("cause=kick")
    assert "0.000" in text and "0.017" in text and "60.000" in text
    assert "owner=free->A" in text
    assert "owner=A->free" in text
    assert "stamp_s = stored tick / 60" in text
    assert "Contact fractions are shown separately" in text
    assert len(events) == 3


def test_timeline_includes_contact_metadata_and_zero_indices():
    event = SimulationEvent(
        "ball_collision",
        12,
        cause="goalpost",
        post_index=0,
        tick_fraction=0.25,
        collision_index=0,
        ball_position=(789.0, 186.0),
        velocity_before=(300.0, 0.0),
        velocity_after=(-210.0, 0.0),
        last_touch_player=PlayerId.B,
    )
    text = format_timeline([event])
    for detail in (
        "post=0",
        "fraction=0.25",
        "collision=0",
        "last_touch=B",
        "ball=(789.0, 186.0)",
        "velocity_after=(-210.0, 0.0)",
    ):
        assert detail in text


def test_timeline_names_scorer_restart_receiver_and_unknown_last_touch():
    text = format_timeline(
        [
            SimulationEvent("goal", 60, scoring_player=PlayerId.B, restart_owner=PlayerId.A),
            SimulationEvent("out_of_bounds", 120),
        ]
    )
    assert "scorer=B receiver=A" in text
    assert "last_touch=unknown receiver=neutral" in text


def test_empty_timeline_is_explicit():
    assert "(no events)" in format_timeline(())
