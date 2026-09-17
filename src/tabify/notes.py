"""Note model and pitch-name helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tabify import TabifyError

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_LETTER_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")


@dataclass(frozen=True)
class Note:
    """A single note. Times are measured in quarter-note beats."""

    start: float
    duration: float
    pitch: int  # MIDI note number
    velocity: int = 90


def midi_to_name(pitch: int, octave: bool = True) -> str:
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{pitch // 12 - 1}" if octave else name


def name_to_midi(name: str) -> int:
    """Parse a scientific pitch name such as ``E2``, ``F#3`` or ``Bb1``."""
    m = _NOTE_RE.match(name.strip())
    if not m:
        raise TabifyError(f"invalid note name {name!r} (expected something like E2, F#3, Bb1)")
    letter, accidental, octave = m.groups()
    pc = _LETTER_TO_PC[letter.upper()] + {"#": 1, "b": -1, "": 0}[accidental]
    return (int(octave) + 1) * 12 + pc
