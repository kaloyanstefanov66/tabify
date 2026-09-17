"""Instrument tunings."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tabify import TabifyError
from tabify.notes import midi_to_name, name_to_midi

# Strings are listed from lowest to highest.
TUNINGS: dict[str, str] = {
    # Standard and whole-guitar downtunings
    "standard": "E2 A2 D3 G3 B3 E4",
    "half-step-down": "D#2 G#2 C#3 F#3 A#3 D#4",
    "d-standard": "D2 G2 C3 F3 A3 D4",
    "c#-standard": "C#2 F#2 B2 E3 G#3 C#4",
    "c-standard": "C2 F2 A#2 D#3 G3 C4",
    "b-standard": "B1 E2 A2 D3 F#3 B3",
    "a#-standard": "A#1 D#2 G#2 C#3 F3 A#3",
    "a-standard": "A1 D2 G2 C3 E3 A3",
    # Drop tunings, every semitone from drop D down to drop F#
    "drop-d": "D2 A2 D3 G3 B3 E4",
    "drop-c#": "C#2 G#2 C#3 F#3 A#3 D#4",
    "drop-c": "C2 G2 C3 F3 A3 D4",
    "drop-b": "B1 F#2 B2 E3 G#3 C#4",
    "drop-a#": "A#1 F2 A#2 D#3 G3 C4",
    "drop-a": "A1 E2 A2 D3 F#3 B3",
    "drop-g#": "G#1 D#2 G#2 C#3 F3 A#3",
    "drop-g": "G1 D2 G2 C3 E3 A3",
    "drop-f#": "F#1 C#2 F#2 B2 D#3 G#3",
    # Open and modal
    "open-g": "D2 G2 D3 G3 B3 D4",
    "open-d": "D2 A2 D3 F#3 A3 D4",
    "open-e": "E2 B2 E3 G#3 B3 E4",
    "open-c": "C2 G2 C3 G3 C4 E4",
    "dadgad": "D2 A2 D3 G3 A3 D4",
    # Extended range
    "7-string": "B1 E2 A2 D3 G3 B3 E4",
    "7-string-drop-a": "A1 E2 A2 D3 G3 B3 E4",
    "7-string-a#-standard": "A#1 D#2 G#2 C#3 F#3 A#3 D#4",
    "8-string": "F#1 B1 E2 A2 D3 G3 B3 E4",
    "8-string-drop-e": "E1 B1 E2 A2 D3 G3 B3 E4",
    "bass": "E1 A1 D2 G2",
    "bass-5": "B0 E1 A1 D2 G2",
    "bass-drop-d": "D1 A1 D2 G2",
}

# Flat spellings and common nicknames for the presets above. Sharps and flats name the
# same pitch, and which one a band uses is arbitrary, so both should work.
ALIASES: dict[str, str] = {
    "e-standard": "standard",
    "eb-standard": "half-step-down",
    "d#-standard": "half-step-down",
    "half-step": "half-step-down",
    "e-flat": "half-step-down",
    "whole-step-down": "d-standard",
    "full-step-down": "d-standard",
    "db-standard": "c#-standard",
    "bb-standard": "a#-standard",
    "drop-db": "drop-c#",
    "drop-bb": "drop-a#",
    "drop-ab": "drop-g#",
    "drop-gb": "drop-f#",
    "7-string-standard": "7-string",
    "7-string-bb-standard": "7-string-a#-standard",
    "8-string-standard": "8-string",
}


@dataclass(frozen=True)
class Tuning:
    name: str
    strings: tuple[int, ...]  # open-string MIDI pitches, lowest string first

    def labels(self) -> list[str]:
        """String labels, lowest string first (e.g. E A D G B e)."""
        names = [midi_to_name(p, octave=False) for p in self.strings]
        return [n.lower() if i > 0 and n == names[0] else n for i, n in enumerate(names)]

    def describe(self) -> str:
        return " ".join(midi_to_name(p) for p in self.strings)


def preset_key(spec: str) -> str | None:
    """The preset this name refers to, allowing spaces, underscores and flat spellings."""
    key = re.sub(r"[\s_]+", "-", spec.strip().lower())
    key = ALIASES.get(key, key)
    return key if key in TUNINGS else None


def parse_tuning(spec: str) -> Tuning:
    """Accept a preset name (``drop-d``, ``drop-db``) or explicit notes (``D2 A2 D3 G3 B3 E4``)."""
    key = preset_key(spec)
    if key:
        return Tuning(key, tuple(name_to_midi(n) for n in TUNINGS[key].split()))

    parts = [p for p in re.split(r"[\s,]+", spec.strip()) if p]
    if len(parts) < 2:
        raise TabifyError(
            f"unknown tuning {spec!r}. Use a preset (see --list-tunings) "
            "or notes low-to-high like 'D2 A2 D3 G3 B3 E4'"
        )
    strings = tuple(name_to_midi(p) for p in parts)
    if list(strings) != sorted(strings):
        raise TabifyError("tuning notes must be listed from the lowest string to the highest")
    return Tuning("custom", strings)
