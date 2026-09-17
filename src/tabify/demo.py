"""A short built-in piece so `tabify --demo` works with nothing installed."""

from __future__ import annotations

from tabify.notes import Note, name_to_midi

# Eighth-note arpeggios over Am - F - C - G, ending on a ringing Am chord.
_ARPEGGIOS = [
    "A2 E3 A3 C4 E4 C4 A3 E3",
    "F2 C3 F3 A3 C4 A3 F3 C3",
    "C3 G3 C4 E4 G4 E4 C4 G3",
    "G2 D3 G3 B3 D4 B3 G3 D3",
]
_FINAL_CHORD = "A2 E3 A3 C4 E4"


def demo_notes() -> list[Note]:
    notes = []
    for bar, pattern in enumerate(_ARPEGGIOS):
        for i, name in enumerate(pattern.split()):
            notes.append(Note(bar * 4 + i * 0.5, 0.5, name_to_midi(name)))
    notes += [Note(16.0, 4.0, name_to_midi(n)) for n in _FINAL_CHORD.split()]
    return notes
