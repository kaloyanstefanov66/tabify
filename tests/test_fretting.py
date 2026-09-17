from tabify.fretting import FretOptions, Position, assign_frets
from tabify.notes import Note, name_to_midi
from tabify.tuning import parse_tuning

STANDARD = parse_tuning("standard")


def seq(names, step=0.5):
    return [Note(i * step, step, name_to_midi(n)) for i, n in enumerate(names.split())]


def chord(names, start=0.0):
    return [Note(start, 1.0, name_to_midi(n)) for n in names.split()]


def test_open_low_e():
    result = assign_frets(seq("E2"), STANDARD)
    assert result.events[0].positions == [Position(0, 0)]


def test_open_a_minor_chord():
    result = assign_frets(chord("A2 E3 A3 C4 E4"), STANDARD)
    frets = {p.string: p.fret for p in result.events[0].positions}
    assert frets == {1: 0, 2: 2, 3: 2, 4: 1, 5: 0}


def test_scale_stays_in_one_position():
    # C major scale: should be played in one hand position, not up one string.
    result = assign_frets(seq("C3 D3 E3 F3 G3 A3 B3 C4"), STANDARD)
    fretted = [p.fret for e in result.events for p in e.positions if p.fret > 0]
    assert max(fretted) - min(fretted) <= 4
    assert len({p.string for e in result.events for p in e.positions}) >= 3


def test_high_melody_stays_in_a_box():
    # A-minor pentatonic an octave up: should sit in one box, not jump between positions.
    result = assign_frets(seq("A4 C5 D5 E5 G5 A5 G5 E5 D5 C5 A4"), STANDARD)
    fretted = [p.fret for e in result.events for p in e.positions if p.fret > 0]
    assert max(fretted) - min(fretted) <= 5, fretted


def test_capo_makes_notes_relative_and_drops_notes_below_it():
    opts = FretOptions(capo=2)
    result = assign_frets(seq("E2 F#2"), STANDARD, opts)
    assert [n.pitch for n in result.dropped] == [name_to_midi("E2")]
    assert result.events[0].positions == [Position(0, 0)]


def test_too_many_notes_are_dropped_not_crashing():
    result = assign_frets(chord("E2 A2 D3 G3 B3 E4 G4"), STANDARD)
    assert len(result.events[0].positions) == 6
    assert len(result.dropped) == 1


def test_unison_duplicates_merged():
    notes = [Note(0, 1, 52), Note(0, 2, 52)]
    result = assign_frets(notes, STANDARD)
    assert len(result.events[0].notes) == 1
    assert result.events[0].notes[0].duration == 2


def test_empty_input():
    assert assign_frets([], STANDARD).events == []
