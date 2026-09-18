"""What a single guitar stroke can be.

The stroke model answers two questions about each pick attack: what note is it rooted on,
and what shape was played on top of that root. Keeping the vocabulary here means the data
generator and the model agree on what a label means.

The shapes are the ones that carry rhythm guitar - a bare note, a power chord with or
without its octave, an octave pair, and the two triads - rather than every chord in music,
because that's what the strokes we get wrong actually are.
"""

from __future__ import annotations

from tabify.tuning import Tuning

# Semitones above the root, per shape.
SHAPE_INTERVALS: dict[str, tuple[int, ...]] = {
    "single": (0,),
    "power2": (0, 7),  # root + fifth
    "power3": (0, 7, 12),  # root + fifth + octave, the drop-tuning barre
    "octave": (0, 12),
    "triad-minor": (0, 3, 7),
    "triad-major": (0, 4, 7),
}
SHAPES: tuple[str, ...] = tuple(SHAPE_INTERVALS)

# Roots the model predicts: E1 (below any standard guitar) up to G4 (high on the neck).
ROOT_LOW, ROOT_HIGH = 28, 67
ROOTS: tuple[int, ...] = tuple(range(ROOT_LOW, ROOT_HIGH + 1))


def shape_index(shape: str) -> int:
    return SHAPES.index(shape)


def root_index(pitch: int) -> int:
    return pitch - ROOT_LOW


def positions_for(
    tuning: Tuning, root: int, shape: str, *, max_fret: int = 22, max_span: int = 4
) -> list[tuple[int, int]] | None:
    """Where a guitarist would put this shape: (string, fret) per note, or None if it won't fit.

    Each note goes on a higher string than the one below it, at the fret closest to the root's -
    which is what makes an octave a string skip rather than a twelve-fret stretch - and the whole
    shape has to stay within a hand span.
    """
    strings = tuning.strings
    lowest_string = next((s for s in range(len(strings)) if 0 <= root - strings[s] <= max_fret), None)
    if lowest_string is None:
        return None

    root_fret = root - strings[lowest_string]
    positions = [(lowest_string, root_fret)]
    string = lowest_string
    for interval in SHAPE_INTERVALS[shape][1:]:
        pitch = root + interval
        options = [
            (s, pitch - strings[s])
            for s in range(string + 1, len(strings))
            if 0 <= pitch - strings[s] <= max_fret and abs(pitch - strings[s] - root_fret) <= max_span
        ]
        if not options:
            return None
        string, fret = min(options, key=lambda p: (abs(p[1] - root_fret), p[0]))
        positions.append((string, fret))

    frets = [fret for _, fret in positions if fret > 0]
    if frets and max(frets) - min(frets) > max_span:
        return None
    return positions
