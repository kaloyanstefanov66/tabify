import pytest

from tabify import TabifyError
from tabify.fretting import assign_frets
from tabify.notes import Note, name_to_midi
from tabify.synth import RENDER_HELP, render_audio, resolve_soundfont, schedule
from tabify.tuning import parse_tuning

STANDARD = parse_tuning("standard")


def test_schedule_produces_matching_on_off_pairs_at_the_right_time():
    events = assign_frets([Note(0, 2, name_to_midi("E2"))], STANDARD).events
    sched = schedule(events, STANDARD, capo=0, bpm=120)
    assert sched == [(0.0, True, name_to_midi("E2")), (1.0, False, name_to_midi("E2"))]


def test_schedule_applies_capo_to_sounding_pitch():
    events = assign_frets([Note(0, 1, name_to_midi("E2"))], STANDARD).events
    sched = schedule(events, STANDARD, capo=2, bpm=120)
    assert sched[0][2] == name_to_midi("E2") + 2


def test_schedule_orders_note_off_before_note_on_at_a_tie():
    # Two adjacent notes on the same string: the first note's "off" and the second's "on" land
    # at the same instant, and the off must come first so the note actually retriggers.
    events = assign_frets([Note(0, 1, name_to_midi("E2")), Note(1, 1, name_to_midi("F2"))], STANDARD).events
    sched = schedule(events, STANDARD, capo=0, bpm=60)
    at_the_tie = [ev for ev in sched if ev[0] == 1.0]
    assert [ev[1] for ev in at_the_tie] == [False, True]


def test_chord_produces_simultaneous_events():
    events = assign_frets([Note(0, 1, name_to_midi("E2")), Note(0, 1, name_to_midi("B2"))], STANDARD).events
    sched = schedule(events, STANDARD, capo=0, bpm=60)
    on_times = {t for t, is_on, _ in sched if is_on}
    assert on_times == {0.0}


def test_unknown_instrument_is_rejected_before_touching_fluidsynth():
    with pytest.raises(TabifyError, match="unknown instrument"):
        render_audio([], STANDARD, "out.wav", instrument="kazoo")


def test_missing_fluidsynth_gives_actionable_error(monkeypatch):
    # Simulate FluidSynth not being installed, regardless of whether this
    # particular machine happens to have it.
    monkeypatch.setattr("tabify.synth.importlib.util.find_spec", lambda name: None)
    with pytest.raises(TabifyError, match="FluidSynth"):
        render_audio([], STANDARD, "out.wav", instrument="steel")
    assert "pip install" in RENDER_HELP


def test_resolve_soundfont_missing_explicit_path():
    with pytest.raises(TabifyError, match="not found"):
        resolve_soundfont("does-not-exist.sf2")


def test_resolve_soundfont_uses_explicit_path(tmp_path):
    sf2 = tmp_path / "my.sf2"
    sf2.write_bytes(b"fake soundfont data")
    assert resolve_soundfont(str(sf2)) == sf2
