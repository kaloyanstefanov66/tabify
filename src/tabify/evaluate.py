"""Scoring a transcription against a known tab.

Without this, "does this change help?" is answered by squinting at a tab, which is how a
change that restores 212 missing roots can still leave the output worse. Two scores are
reported, because they fail for different reasons:

* **note F1** - did we hear the right pitch at the right time? Blames the pitch model.
* **tab F1** - did we also put it on the string and fret it was played on? Blames the
  fingering search (a note can be right while its position is wrong, since the same pitch
  sits in several places on the neck).
"""

from __future__ import annotations

from dataclasses import dataclass

from tabify.fretting import TabEvent
from tabify.tuning import Tuning


@dataclass(frozen=True)
class Stroke:
    """One note of a tab: when it's struck, and where it's fretted."""

    start: float  # in beats
    string: int  # 0 = lowest string
    fret: int

    def pitch(self, tuning: Tuning, capo: int = 0) -> int:
        return tuning.strings[self.string] + capo + self.fret


@dataclass(frozen=True)
class Scorecard:
    matched: int
    predicted: int
    reference: int

    @property
    def precision(self) -> float:
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.reference if self.reference else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def __str__(self) -> str:
        return f"F1 {self.f1:.0%} (precision {self.precision:.0%}, recall {self.recall:.0%}, {self.matched}/{self.reference})"


def _match(reference: list[tuple[float, object]], predicted: list[tuple[float, object]], tolerance: float) -> int:
    """Count reference items that have a predicted item with the same key, within `tolerance` beats.

    Each prediction can only be used once, so duplicates don't inflate the score.
    """
    used: set[int] = set()
    matched = 0
    for ref_start, ref_key in reference:
        best = None
        for i, (start, key) in enumerate(predicted):
            if i in used or key != ref_key:
                continue
            distance = abs(start - ref_start)
            if distance <= tolerance and (best is None or distance < best[0]):
                best = (distance, i)
        if best is not None:
            used.add(best[1])
            matched += 1
    return matched


def strokes_of(events: list[TabEvent]) -> list[Stroke]:
    return [Stroke(e.start, p.string, p.fret) for e in events for p in e.positions]


def score(
    reference: list[Stroke], predicted: list[Stroke], tuning: Tuning, *, tolerance: float = 0.25, capo: int = 0,
) -> tuple[Scorecard, Scorecard]:
    """Compare a predicted tab against the real one. Returns (note score, tab score)."""
    note_ref = [(s.start, s.pitch(tuning, capo)) for s in reference]
    note_pred = [(s.start, s.pitch(tuning, capo)) for s in predicted]
    tab_ref = [(s.start, (s.string, s.fret)) for s in reference]
    tab_pred = [(s.start, (s.string, s.fret)) for s in predicted]
    return (
        Scorecard(_match(note_ref, note_pred, tolerance), len(note_pred), len(note_ref)),
        Scorecard(_match(tab_ref, tab_pred, tolerance), len(tab_pred), len(tab_ref)),
    )
