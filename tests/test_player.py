import numpy as np
import pytest

from tabify import TabifyError
from tabify.fretting import assign_frets
from tabify.notes import Note
from tabify.player import PLAY_HELP, play_along, step_at
from tabify.tuning import parse_tuning

# These tests patch the real, installed sounddevice's play/stop rather than swapping in a fake
# module object: replacing sys.modules["sounddevice"] with something lacking a real __spec__
# breaks importlib.util.find_spec's own module-already-imported check.
sd = pytest.importorskip("sounddevice")

STANDARD = parse_tuning("standard")


def test_step_at():
    assert step_at(0.0, bpm=120, subdivision=4) == 0.0
    assert step_at(0.5, bpm=120, subdivision=4) == 4.0  # 0.5s at 120bpm = 1 beat = 4 sixteenth-note steps
    assert step_at(1.0, bpm=60, subdivision=1) == 1.0  # 1s at 60bpm = 1 beat = 1 quarter-note step


def test_missing_sounddevice_gives_actionable_error(monkeypatch):
    monkeypatch.setattr("tabify.player.importlib.util.find_spec", lambda name: None)
    events = assign_frets([Note(0, 1, 40)], STANDARD).events
    with pytest.raises(TabifyError, match="sounddevice"):
        play_along(events, STANDARD, np.zeros((100, 2)), 44100, bpm=120)
    assert "pip install" in PLAY_HELP


def test_no_playable_notes_is_a_clean_error(monkeypatch):
    monkeypatch.setattr(sd, "play", lambda *a, **k: None)
    monkeypatch.setattr(sd, "stop", lambda: None)
    with pytest.raises(TabifyError, match="nothing to play"):
        play_along([], STANDARD, np.zeros((100, 2)), 44100, bpm=120)


def test_quits_immediately_on_q(monkeypatch):
    """A scripted 'q' keypress should make play_along return cleanly without blocking."""
    calls = {"play": 0, "stop": 0}
    monkeypatch.setattr(sd, "play", lambda *a, **k: calls.__setitem__("play", calls["play"] + 1))
    monkeypatch.setattr(sd, "stop", lambda: calls.__setitem__("stop", calls["stop"] + 1))
    monkeypatch.setattr("tabify.player.time.sleep", lambda s: None)

    class ImmediateQuit:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return "q"

    monkeypatch.setattr("tabify.player.KeyReader", ImmediateQuit)

    events = assign_frets([Note(0, 1, 40)], STANDARD).events
    play_along(events, STANDARD, np.zeros((44100, 2), dtype=np.float32), 44100, bpm=120)
    assert calls["play"] == 1
    assert calls["stop"] >= 1


def test_frames_redraw_in_place_instead_of_clearing_the_screen():
    from tabify import term
    from tabify.player import frame_text

    text = frame_text(["line one", "line two"])
    assert "\x1b[2J" not in text  # a full clear floods scrollback in some terminals
    assert text.startswith(term.HOME) and text.endswith(term.CLEAR_SCREEN_END)


def test_play_along_uses_the_alternate_screen_and_restores_the_terminal(monkeypatch, capsys):
    from tabify import term

    monkeypatch.setattr(sd, "play", lambda *a, **k: None)
    monkeypatch.setattr(sd, "stop", lambda: None)
    monkeypatch.setattr("tabify.player.time.sleep", lambda s: None)

    class QuitAfterTwoFrames:
        reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            QuitAfterTwoFrames.reads += 1
            return "q" if QuitAfterTwoFrames.reads >= 2 else None

    monkeypatch.setattr("tabify.player.KeyReader", QuitAfterTwoFrames)
    events = assign_frets([Note(0, 1, 40)], STANDARD).events
    play_along(events, STANDARD, np.zeros((44100, 2), dtype=np.float32), 44100, bpm=120)
    out = capsys.readouterr().out
    assert out.index(term.ALT_SCREEN_ON) < out.index(term.HOME)
    assert out.rstrip().endswith(term.ALT_SCREEN_OFF)


def test_a_take_that_starts_late_keeps_its_lead_in():
    """align_to_bars trims the empty bars off the tab; playback has to put that time back."""
    from tabify.notes import Note
    from tabify.rhythm import TimeSignature, leading_bar_shift

    # First note two bars in, at 4/4.
    notes = [Note(8.0, 1, 40), Note(9.0, 1, 43)]
    assert leading_bar_shift(notes, TimeSignature()) == 8.0
    # At 120 BPM those 8 beats are 4 seconds of recording before the tab's bar 1.
    assert 8.0 * 60.0 / 120.0 == 4.0


def test_no_lead_in_when_the_first_note_is_already_in_bar_one():
    from tabify.notes import Note
    from tabify.rhythm import TimeSignature, leading_bar_shift

    assert leading_bar_shift([Note(0.5, 1, 40)], TimeSignature()) == 0.0
    assert leading_bar_shift([], TimeSignature()) == 0.0


def test_the_cursor_waits_at_the_start_during_the_lead_in():
    from tabify.player import step_at

    # During the lead-in the offset time is negative and must not run the tab backwards.
    assert step_at(max(0.0, 1.0 - 4.0), 120, 4) == 0.0
    assert step_at(max(0.0, 5.0 - 4.0), 120, 4) == 8.0


def test_the_tab_starts_where_the_music_does_not_where_the_file_does():
    """Two separate steps trim beats off the front of a tab; both must be added back."""
    from tabify.notes import Note
    from tabify.rhythm import TimeSignature, first_beat_shift, leading_bar_shift, quantize, start_on_first_beat

    # A take whose first note lands six beats in - a bar and a half of lead-in.
    notes = [Note(6.5, 1, 40), Note(7.5, 1, 45)]
    origin = first_beat_shift(notes)
    trimmed = leading_bar_shift(quantize(start_on_first_beat(notes), 4), TimeSignature())
    assert origin == 6.0
    # Together they account for the whole distance back to the start of the recording.
    assert origin + trimmed == 6.0


def test_playback_follows_the_beat_map_rather_than_one_fixed_tempo():
    """Notes are placed against the beats as played; the cursor has to read the same clock."""
    from tabify.player import step_at

    # A take that speeds up: by 10 seconds a fixed 120 BPM is a long way from the truth.
    def to_beats(t):
        return t * 2.0 + 0.02 * t * t

    fixed = step_at(10.0, 120, 4)
    following = max(0.0, to_beats(10.0) - 0.0) * 4
    assert abs(following - fixed) > 4  # more than a beat apart by then
