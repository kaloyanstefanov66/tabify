"""Reading tab the way people actually write it.

ASCII tab is what guitarists already have and already type, so that's the format tabify
reads: paste it from a forum, a text file or your own notes. It's parsed forgivingly,
because handwritten tab is never tidy - columns drift, bar lines wander, techniques get
scribbled between the notes - and none of that changes which frets were played.

What's recovered is the *order of the strokes and the frets in each*, not their timing.
Timing comes from the audio, which knows better than a column of dashes does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tabify import TabifyError
from tabify.notes import name_to_midi
from tabify.tuning import Tuning, parse_tuning

# A tab line is mostly dashes and fret numbers, optionally behind a string label ("e|", "E |").
_LINE = re.compile(r"^\s*(?:([A-Ga-g][#b]?\d?)\s*)?\|?([-\d|hpbrx/\\~^()* .]+)$")
_LABEL_ONLY = re.compile(r"^[A-Ga-g][#b]?\d?$")
_METADATA = re.compile(r"^\s*(tuning|tone|capo|notes|tempo|bpm)\s*:\s*(.+?)\s*$", re.IGNORECASE)
# Techniques are noted between frets; they say how a note was played, not which fret it was.
_TECHNIQUES = "hpbr/\\~^()*"


@dataclass
class TabStrokes:
    """Strokes in the order they're played: each a list of (string, fret), lowest string first."""

    strokes: list[list[tuple[int, int]]]
    tuning: Tuning | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.strokes)


def _is_tab_line(text: str) -> bool:
    match = _LINE.match(text.rstrip())
    if not match:
        return False
    body = match.group(2)
    dashes = body.count("-")
    return dashes >= 4 and dashes >= len(body.replace(" ", "")) * 0.4


def _blocks(lines: list[str]) -> list[list[str]]:
    """Group the tab lines into systems - one block per set of consecutive string lines."""
    blocks, current = [], []
    for line in lines:
        if _is_tab_line(line):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return [b for b in blocks if len(b) >= 4]


def _frets_in(line: str) -> list[tuple[int, int]]:
    """Every fret number in one string's line, as (column where it starts, fret).

    Columns are counted inside the tab itself, after the string label, so lines whose labels
    are different widths ("e|" against "G |") still line up with each other.
    """
    match = _LINE.match(line.rstrip())
    body = match.group(2) if match else line
    return [(m.start(), int(m.group())) for m in re.finditer(r"\d+", body)]


def _label_of(line: str) -> str | None:
    match = _LINE.match(line.rstrip())
    return match.group(1) if match and match.group(1) else None


def parse_tab(text: str, *, tuning: Tuning | None = None, column_tolerance: int = 1) -> TabStrokes:
    """Read ASCII tab into ordered strokes.

    Notes whose columns line up (within `column_tolerance`, since handwritten tab drifts) are
    one stroke. Techniques like hammer-ons and bends are ignored - they don't change the fret.
    """
    lines = text.splitlines()
    metadata = {}
    for line in lines:
        found = _METADATA.match(line)
        if found:
            metadata[found.group(1).lower()] = found.group(2)
    if tuning is None and "tuning" in metadata:
        tuning = parse_tuning(metadata["tuning"])

    blocks = _blocks(lines)
    if not blocks:
        raise TabifyError("no tab found in that text - expected lines of dashes and fret numbers")

    strokes: list[list[tuple[int, int]]] = []
    for block in blocks:
        # Tab is written with the highest string on top, so the file's order is reversed.
        rows = list(reversed(block))
        if tuning is None:
            tuning = _tuning_from_labels(rows)
        columns: dict[int, list[tuple[int, int]]] = {}
        for string, line in enumerate(rows):
            for column, fret in _frets_in(line):
                anchor = next((c for c in columns if abs(c - column) <= column_tolerance), column)
                columns.setdefault(anchor, []).append((string, fret))
        for column in sorted(columns):
            strokes.append(sorted(columns[column]))

    return TabStrokes(strokes, tuning, metadata)


def _tuning_from_labels(rows: list[str]) -> Tuning | None:
    """Work out the tuning from the string labels down the left edge, if they're there."""
    labels = [_label_of(line) for line in rows]
    if not all(labels):
        return None
    pitches = []
    for label, reference in zip(labels, parse_tuning("standard").strings):
        name = label if label[-1].isdigit() else None
        if name:
            pitches.append(name_to_midi(name))
        else:  # a bare letter: take the octave from where that string sits on a guitar
            base = name_to_midi(f"{label.upper()}{reference // 12 - 1}")
            pitches.append(base if abs(base - reference) <= 6 else base + 12 * (1 if base < reference else -1))
    if len(pitches) >= 4 and pitches == sorted(pitches):
        return Tuning("custom", tuple(pitches))
    return None


def read_tab_file(path) -> TabStrokes:
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_tab(text)
