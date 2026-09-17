"""MusicXML export.

Songsterr's own uploader wants a Guitar Pro file, but Guitar Pro, TuxGuitar and
MuseScore all import MusicXML directly - so this is the path to get a tabify
transcription into any of them (and from there, re-saved as .gp5 if a site
insists on that exact format). MusicXML is plain, documented XML, which makes
it far more reliable to generate correctly than reverse-engineering Guitar
Pro's binary format.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace

from tabify.fretting import Position, TabEvent
from tabify.rhythm import TimeSignature
from tabify.tuning import Tuning

# (step, semitones sharp of the step) for each of the 12 pitch classes, C..B.
_STEP_ALTER = [
    ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
    ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
]
# Standard note values in beats, largest first, used only to pick a display
# <type> for engraving - the <duration> element carries the real timing.
_NOTE_TYPES = [(4.0, "whole"), (2.0, "half"), (1.0, "quarter"), (0.5, "eighth"), (0.25, "16th"), (0.125, "32nd")]

DOCTYPE = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">\n'
)


def _pitch_xml(pitch: int) -> tuple[str, int, int]:
    step, alter = _STEP_ALTER[pitch % 12]
    return step, alter, pitch // 12 - 1


def _note_type(beats: float) -> tuple[str, bool]:
    """Best-effort (type, dotted) for a note lasting `beats` quarter notes."""
    for base, name in _NOTE_TYPES:
        if abs(beats - base * 1.5) < 1e-6:
            return name, True
        if abs(beats - base) < 1e-6:
            return name, False
    for base, name in _NOTE_TYPES:  # odd length (e.g. a triplet grid) - closest fit, no dot
        if base <= beats + 1e-9:
            return name, False
    return "64th", False


@dataclass
class _Segment:
    start: int  # in grid steps
    duration: int
    positions: tuple[Position, ...]  # empty means a rest
    continued: bool = False  # True once this segment is what's left after a bar split


def _segments(events: list[TabEvent], subdivision: int, total_steps: int) -> list[_Segment]:
    segments = []
    cursor = 0
    for e in sorted(events, key=lambda ev: ev.start):
        start = round(e.start * subdivision)
        duration = max(1, round(max(n.duration for n in e.notes) * subdivision))
        if start > cursor:
            segments.append(_Segment(cursor, start - cursor, ()))
        segments.append(_Segment(start, duration, tuple(e.positions)))
        cursor = start + duration
    if cursor < total_steps:
        segments.append(_Segment(cursor, total_steps - cursor, ()))
    return segments


def to_musicxml(
    events: list[TabEvent],
    tuning: Tuning,
    *,
    time_sig: TimeSignature = TimeSignature(),
    subdivision: int = 4,
    bpm: float | None = None,
    title: str | None = None,
    capo: int = 0,
) -> str:
    bar_steps = max(1, round(time_sig.bar_length * subdivision))
    last = max(
        (round(e.start * subdivision) + max(1, round(max(n.duration for n in e.notes) * subdivision)) for e in events),
        default=0,
    )
    total_steps = max(bar_steps, -(-last // bar_steps) * bar_steps)  # round up to a whole number of bars
    segments = _segments(events, subdivision, total_steps)
    n_strings = len(tuning.strings)

    root = ET.Element("score-partwise", version="3.1")
    ET.SubElement(ET.SubElement(root, "work"), "work-title").text = title or "Untitled"
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Guitar"
    part = ET.SubElement(root, "part", id="P1")

    measure_no, step_i, seg_i = 1, 0, 0
    while step_i < total_steps:
        measure = ET.SubElement(part, "measure", number=str(measure_no))
        if measure_no == 1:
            attrs = ET.SubElement(measure, "attributes")
            ET.SubElement(attrs, "divisions").text = str(subdivision)
            ET.SubElement(ET.SubElement(attrs, "key"), "fifths").text = "0"
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(time_sig.beats)
            ET.SubElement(time_el, "beat-type").text = str(time_sig.unit)
            clef = ET.SubElement(attrs, "clef")
            ET.SubElement(clef, "sign").text = "TAB"
            ET.SubElement(clef, "line").text = "5"
            staff_details = ET.SubElement(attrs, "staff-details")
            ET.SubElement(staff_details, "staff-lines").text = str(n_strings)
            for i, pitch in enumerate(tuning.strings):  # line 1 = lowest string, matching tab convention
                step, _, octave = _pitch_xml(pitch)
                tuning_el = ET.SubElement(staff_details, "staff-tuning", line=str(i + 1))
                ET.SubElement(tuning_el, "tuning-step").text = step
                ET.SubElement(tuning_el, "tuning-octave").text = str(octave)
            if bpm:
                direction = ET.SubElement(measure, "direction", placement="above")
                metronome = ET.SubElement(ET.SubElement(direction, "direction-type"), "metronome")
                ET.SubElement(metronome, "beat-unit").text = "quarter"
                ET.SubElement(metronome, "per-minute").text = str(round(bpm))
                ET.SubElement(direction, "sound", tempo=str(round(bpm)))

        bar_end = step_i + bar_steps
        while seg_i < len(segments) and segments[seg_i].start < bar_end:
            seg = segments[seg_i]
            seg_end = seg.start + seg.duration
            piece_end = min(seg_end, bar_end)
            piece_dur = piece_end - seg.start
            is_first, is_last = not seg.continued, piece_end == seg_end
            note_type, dotted = _note_type(piece_dur / subdivision)

            if not seg.positions:
                rest = ET.SubElement(measure, "note")
                ET.SubElement(rest, "rest")
                ET.SubElement(rest, "duration").text = str(piece_dur)
                ET.SubElement(rest, "voice").text = "1"
                ET.SubElement(rest, "type").text = note_type
                if dotted:
                    ET.SubElement(rest, "dot")
            for idx, pos in enumerate(seg.positions):
                note_el = ET.SubElement(measure, "note")
                if idx > 0:
                    ET.SubElement(note_el, "chord")
                # pos.fret is relative to the capo (standard tab convention); the actual
                # sounding pitch - what a <pitch> element must represent - includes it.
                step, alter, octave = _pitch_xml(tuning.strings[pos.string] + capo + pos.fret)
                pitch_el = ET.SubElement(note_el, "pitch")
                ET.SubElement(pitch_el, "step").text = step
                if alter:
                    ET.SubElement(pitch_el, "alter").text = str(alter)
                ET.SubElement(pitch_el, "octave").text = str(octave)
                ET.SubElement(note_el, "duration").text = str(piece_dur)
                if not is_first:
                    ET.SubElement(note_el, "tie", type="stop")
                if not is_last:
                    ET.SubElement(note_el, "tie", type="start")
                ET.SubElement(note_el, "voice").text = "1"
                ET.SubElement(note_el, "type").text = note_type
                if dotted:
                    ET.SubElement(note_el, "dot")
                notations = ET.SubElement(note_el, "notations")
                if not is_first or not is_last:
                    ET.SubElement(notations, "tied", type="stop" if not is_first else "start")
                technical = ET.SubElement(notations, "technical")
                ET.SubElement(technical, "string").text = str(n_strings - pos.string)  # string 1 = highest, tab convention
                ET.SubElement(technical, "fret").text = str(pos.fret)

            if piece_end >= seg_end:
                seg_i += 1
            else:
                segments[seg_i] = replace(seg, start=piece_end, duration=seg_end - piece_end, continued=True)
        step_i = bar_end
        measure_no += 1

    ET.indent(root, space="  ")
    return DOCTYPE + ET.tostring(root, encoding="unicode") + "\n"
