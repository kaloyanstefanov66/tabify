"""Quantization and bar alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from tabify import TabifyError
from tabify.notes import Note


@dataclass(frozen=True)
class TimeSignature:
    beats: int = 4
    unit: int = 4

    @property
    def bar_length(self) -> float:
        """Bar length in quarter-note beats."""
        return self.beats * 4 / self.unit

    def __str__(self) -> str:
        return f"{self.beats}/{self.unit}"


def parse_time_signature(spec: str) -> TimeSignature:
    try:
        beats, unit = (int(x) for x in spec.split("/"))
    except ValueError:
        raise TabifyError(f"invalid time signature {spec!r} (expected e.g. 4/4 or 6/8)") from None
    if beats < 1 or unit not in (1, 2, 4, 8, 16, 32):
        raise TabifyError(f"invalid time signature {spec!r}")
    return TimeSignature(beats, unit)


def quantize(notes: list[Note], subdivision: int) -> list[Note]:
    """Snap note starts and durations to a grid of ``1/subdivision`` beats."""
    step = 1 / subdivision
    out = []
    for n in notes:
        start = round(n.start / step) * step
        duration = max(step, round(n.duration / step) * step)
        out.append(replace(n, start=start, duration=duration))
    return out


def start_on_first_beat(notes: list[Note]) -> list[Note]:
    """Shift notes so the beat containing the first note becomes beat 0."""
    if not notes:
        return notes
    shift = math.floor(min(n.start for n in notes) + 0.25)  # tolerate notes played slightly early
    return [replace(n, start=n.start - shift) for n in notes]


def leading_bar_shift(notes: list[Note], time_sig: TimeSignature) -> float:
    """How many beats of empty bars sit before the first note.

    Worth keeping rather than discarding: it is the distance between the tab's first bar and
    the start of the recording, which is exactly what playback needs to stay lined up when a
    take begins with a few seconds of silence or a count-in.
    """
    if not notes:
        return 0.0
    first = min(n.start for n in notes)
    return math.floor(first / time_sig.bar_length + 1e-9) * time_sig.bar_length


def align_to_bars(notes: list[Note], time_sig: TimeSignature) -> list[Note]:
    """Drop leading empty bars so the first note lands in bar 1, keeping its position in the bar."""
    if not notes:
        return notes
    shift = leading_bar_shift(notes, time_sig)
    return [replace(n, start=n.start - shift) for n in notes]
