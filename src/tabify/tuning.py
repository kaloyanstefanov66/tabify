"""Instrument tunings."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tabify import TabifyError
from tabify.notes import midi_to_name, name_to_midi

# Strings are listed from lowest to highest.
TUNINGS: dict[str, str] = {
    "standard": "E2 A2 D3 G3 B3 E4",
    "drop-d": "D2 A2 D3 G3 B3 E4",
    "half-step-down": "D#2 G#2 C#3 F#3 A#3 D#4",
    "d-standard": "D2 G2 C3 F3 A3 D4",
    "drop-c": "C2 G2 C3 F3 A3 D4",
    "drop-b": "B1 F#2 B2 E3 G#3 C#4",
    "open-g": "D2 G2 D3 G3 B3 D4",
    "open-d": "D2 A2 D3 F#3 A3 D4",
    "open-e": "E2 B2 E3 G#3 B3 E4",
    "dadgad": "D2 A2 D3 G3 A3 D4",
    "7-string": "B1 E2 A2 D3 G3 B3 E4",
    "bass": "E1 A1 D2 G2",
    "bass-5": "B0 E1 A1 D2 G2",
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


def parse_tuning(spec: str) -> Tuning:
    """Accept a preset name (``drop-d``) or explicit notes (``D2 A2 D3 G3 B3 E4``)."""
    key = re.sub(r"[\s_]+", "-", spec.strip().lower())
    if key in TUNINGS:
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
