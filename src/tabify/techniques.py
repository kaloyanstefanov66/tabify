"""Playing techniques that can't be read straight off the pitches.

Palm mutes are inferred, not heard. On a real isolated distorted guitar track, muted and
let-ring strokes measured the same decay rate and brightness - heavy distortion
compresses both - so no audio measurement tried could tell them apart. What does
separate them in practice is how they're played: palm mutes are fast, repeated hits on
the low strings, while let-ring hits are held. That's the rule here, and it's why the
marks are a convention-based guess you can switch off, not a measurement.
"""

from __future__ import annotations

from dataclasses import replace

from tabify.fretting import TabEvent


def infer_palm_mutes(
    events: list[TabEvent], *, max_gap_beats: float = 0.5, low_strings: int = 2, min_run: int = 2,
) -> list[TabEvent]:
    """Mark runs of fast strokes on the lowest strings as palm-muted.

    A stroke qualifies when it uses one of the `low_strings` lowest strings and the next
    stroke follows within `max_gap_beats` (an 8th note by default). Only runs of at least
    `min_run` qualifying strokes are marked, so a lone quick note isn't flagged.
    """
    events = sorted(events, key=lambda e: e.start)
    qualifies = []
    for i, e in enumerate(events):
        if i + 1 < len(events):
            gap = events[i + 1].start - e.start
        else:
            gap = e.start - events[i - 1].start if i > 0 else float("inf")
        low = any(p.string < low_strings for p in e.positions)
        qualifies.append(low and gap <= max_gap_beats + 1e-9)

    muted = [False] * len(events)
    i = 0
    while i < len(events):
        if not qualifies[i]:
            i += 1
            continue
        j = i
        while j < len(events) and qualifies[j]:
            j += 1
        if j - i >= min_run:
            for k in range(i, j):
                muted[k] = True
        i = j
    return [replace(e, palm_mute=m) for e, m in zip(events, muted)]
