from tabify.fretting import assign_frets
from tabify.notes import Note, name_to_midi
from tabify.render import render_tab
from tabify.rhythm import TimeSignature, align_to_bars, parse_time_signature, quantize
from tabify.tuning import parse_tuning

STANDARD = parse_tuning("standard")


def test_render_simple_bar():
    notes = [Note(0, 1, name_to_midi("E2")), Note(1, 1, name_to_midi("G2")), Note(2, 1, name_to_midi("A2"))]
    events = assign_frets(notes, STANDARD).events
    text = render_tab(events, STANDARD, subdivision=1, width=80)
    assert text.splitlines()[-6:] == [
        "e|---------|",
        "B|---------|",
        "G|---------|",
        "D|---------|",
        "A|-----0---|",
        "E|-0-3-----|",
    ]


def test_two_digit_frets_keep_strings_aligned():
    notes = [Note(0, 1, name_to_midi("E5")), Note(1, 1, name_to_midi("E4"))]
    events = assign_frets(notes, STANDARD).events
    lines = render_tab(events, STANDARD, subdivision=1).splitlines()[-6:]
    assert len({len(line) for line in lines}) == 1


def test_bars_wrap_to_width():
    notes = [Note(bar * 4.0, 1, 40) for bar in range(8)]
    events = assign_frets(notes, STANDARD).events
    text = render_tab(events, STANDARD, subdivision=4, width=80)
    assert all(len(line) <= 80 for line in text.splitlines())
    assert text.count("\ne|") > 1


def test_color_adds_escapes_only_when_asked():
    events = assign_frets([Note(0, 1, 45)], STANDARD).events
    assert "\x1b[" not in render_tab(events, STANDARD)
    assert "\x1b[" in render_tab(events, STANDARD, color=True)


def test_no_events_message():
    assert "no playable notes" in render_tab([], STANDARD)


def test_quantize_and_align():
    notes = [Note(8.13, 0.2, 40), Note(9.49, 0.6, 42)]
    q = align_to_bars(quantize(notes, 4), TimeSignature())
    assert [(n.start, n.duration) for n in q] == [(0.25, 0.25), (1.5, 0.5)]


def test_parse_time_signature():
    ts = parse_time_signature("6/8")
    assert ts.bar_length == 3
