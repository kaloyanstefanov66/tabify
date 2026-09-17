from tabify.fretting import assign_frets
from tabify.notes import Note, name_to_midi
from tabify.render import build_layout, format_system, render_tab, system_for_step
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


def test_highlight_lands_on_the_right_column_for_every_step():
    # Every string's row gets a highlighted column at the current step, even strings that are
    # silent there (which just highlight a dash) - so check the set of marks across all rows,
    # rather than assuming the first highlighted line is the one with the actual note.
    notes = [Note(0, 1, name_to_midi("E2")), Note(1, 1, name_to_midi("G2")), Note(2, 1, name_to_midi("A2"))]
    events = assign_frets(notes, STANDARD).events
    layout = build_layout(events, STANDARD, subdivision=1)
    expected_fret_at_step = {0: "0", 1: "3", 2: "0"}  # low-E fret 0, fret 3, then A-string fret 0
    for step, fret in expected_fret_at_step.items():
        lines = format_system(layout, 0, highlight_step=step)
        marks = {line.split("\x1b[7m")[1].split("\x1b[0m")[0] for line in lines if "\x1b[7m" in line}
        assert marks == {fret, "-"}


def test_system_for_step_picks_the_right_line_when_wrapped():
    notes = [Note(bar * 4.0, 1, 40) for bar in range(8)]
    events = assign_frets(notes, STANDARD).events
    layout = build_layout(events, STANDARD, subdivision=4, width=40)
    assert len(layout.systems) > 1  # sanity check that this actually wraps into multiple lines
    assert system_for_step(layout, 0) == 0
    last_bar_step = (len(layout.bars) - 1) * layout.bar_steps
    assert system_for_step(layout, last_bar_step) == len(layout.systems) - 1


def test_system_for_step_clamps_out_of_range():
    events = assign_frets([Note(0, 1, 40)], STANDARD).events
    layout = build_layout(events, STANDARD, subdivision=4)
    assert system_for_step(layout, 10_000) == len(layout.systems) - 1
    assert system_for_step(layout, -5) == 0


def test_palm_mute_markers_sit_over_the_muted_run_only():
    from tabify.fretting import Position, TabEvent

    def stroke(step, palm_mute):
        return TabEvent(step / 4, [Note(step / 4, 0.25, 40)], [Position(0, 0)], palm_mute=palm_mute)

    events = [stroke(0, True), stroke(1, True), stroke(2, True), stroke(8, False)]
    lines = render_tab(events, STANDARD, subdivision=4).splitlines()
    pm_line, low_e = lines[-7], lines[-1]
    assert low_e.startswith("E|-0-0-0-----------0")
    # "PM" starts right above the first muted fret and the dashes end at the last muted one
    assert pm_line.index("PM") == low_e.index("0")
    assert pm_line.rstrip().endswith("-") and len(pm_line.rstrip()) == low_e.index("0-0-0") + len("0-0-0")


def test_no_palm_mute_line_when_nothing_is_muted():
    events = assign_frets([Note(0, 1, 40), Note(1, 1, 43)], STANDARD).events
    assert "PM" not in render_tab(events, STANDARD)
