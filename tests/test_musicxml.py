import re
import xml.etree.ElementTree as ET

from tabify.fretting import FretOptions, assign_frets
from tabify.musicxml import to_musicxml
from tabify.notes import Note, name_to_midi
from tabify.rhythm import TimeSignature
from tabify.tuning import parse_tuning

STANDARD = parse_tuning("standard")


def parse(xml_text: str) -> ET.Element:
    # Strip the doctype so the test doesn't depend on network/DTD resolution.
    body = re.sub(r"<!DOCTYPE[^>]*>\n", "", xml_text)
    return ET.fromstring(body)


def test_valid_structure_and_metadata():
    events = assign_frets([Note(0, 1, name_to_midi("E2"))], STANDARD).events
    root = parse(to_musicxml(events, STANDARD, bpm=120, title="My Riff"))
    assert root.tag == "score-partwise"
    assert root.find("work/work-title").text == "My Riff"
    assert root.find("part-list/score-part").get("id") == "P1"
    assert root.find(".//staff-details/staff-lines").text == "6"
    assert root.find(".//metronome/per-minute").text == "120"


def test_open_low_e_string_and_fret():
    events = assign_frets([Note(0, 1, name_to_midi("E2"))], STANDARD).events
    root = parse(to_musicxml(events, STANDARD))
    note = root.find(".//note")
    assert note.find("pitch/step").text == "E"
    assert note.find("pitch/octave").text == "2"
    assert note.find("pitch/alter") is None
    assert note.find("notations/technical/string").text == "6"  # low E is string 6, tab convention
    assert note.find("notations/technical/fret").text == "0"


def test_gap_becomes_a_rest():
    events = assign_frets([Note(0, 0.5, name_to_midi("E2")), Note(2, 0.5, name_to_midi("G2"))], STANDARD).events
    notes = parse(to_musicxml(events, STANDARD)).findall(".//note")
    assert notes[1].find("rest") is not None
    assert notes[1].find("duration").text == "6"  # 1.5 beats of rest at 16th-note (subdivision 4) resolution


def test_chord_notes_use_chord_element():
    events = assign_frets([Note(0, 1, name_to_midi("E2")), Note(0, 1, name_to_midi("B2"))], STANDARD).events
    notes = [n for n in parse(to_musicxml(events, STANDARD)).findall(".//note") if n.find("pitch") is not None]
    assert len(notes) == 2
    assert notes[0].find("chord") is None
    assert notes[1].find("chord") is not None


def test_note_crossing_a_barline_is_tied():
    # A whole note starting on beat 3 of a 4/4 bar spans into the next measure.
    events = assign_frets([Note(2, 4, name_to_midi("A2"))], STANDARD).events
    measures = parse(to_musicxml(events, STANDARD, time_sig=TimeSignature(4, 4))).findall("part/measure")
    assert len(measures) == 2
    n1 = measures[0].find("note[pitch]")
    n2 = measures[1].find("note[pitch]")
    assert n1.find("tie[@type='start']") is not None
    assert n2.find("tie[@type='stop']") is not None
    assert int(n1.find("duration").text) + int(n2.find("duration").text) == 4 * 4  # subdivision 4


def test_empty_events_still_produces_valid_xml():
    root = parse(to_musicxml([], STANDARD))
    assert root.find("part/measure") is not None


def test_capo_is_reflected_in_the_sounding_pitch():
    # pos.fret is capo-relative (tab convention); the <pitch> must be the true concert pitch.
    # G2, fretted at fret 1 behind a capo on fret 2, still sounds as G2 - not as A2
    # (what fret 1 would mean with no capo).
    events = assign_frets([Note(0, 1, name_to_midi("G2"))], STANDARD, FretOptions(capo=2)).events
    note = parse(to_musicxml(events, STANDARD, capo=2)).find(".//note[pitch]")
    assert note.find("notations/technical/fret").text == "1"
    assert note.find("pitch/step").text == "G"
    assert note.find("pitch/alter") is None
    assert note.find("pitch/octave").text == "2"
