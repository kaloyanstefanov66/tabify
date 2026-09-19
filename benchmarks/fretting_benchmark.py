"""How close tabify's fingering is to the one a guitarist actually wrote.

This benchmark has no audio in it. Each stroke's *pitches* are taken from a real Guitar Pro
tab and handed straight to the fretting stage, which then has to pick strings and frets
without any transcription error to hide behind. The tab says what a human chose, so any
disagreement is the fretting stage's own.

That makes it the fastest loop in the project - it runs in seconds, it is deterministic,
and there are thousands of labelled strokes - and the one place where tabify can be
compared against published work directly: TART reports 71.8% on this task ("string-fret Tab
F1 in oracle settings", arXiv 2609.11904), meaning string and fret assignment given correct
notes.

The scores are strict: `exact` counts a stroke only when every string and fret matches,
`per-note` counts individual positions. Published tab F1 is closer to the second.

Guitar Pro files live in `recordings/`, which is gitignored, so this is skipped rather than
failed when they are not there.

One caveat that matters more than it looks: every tab here is drop-tuned metal. Fitting the
cost weights against these alone produced weights that scored far better on them and broke
ordinary guitar playing - an open A minor stopped being the open shape, and scales crawled
along one string instead of staying in position. The unit tests in `tests/test_fretting.py`
are the only thing in the project representing standard-tuning chords and scales, so they
are part of the objective, not a formality. Treat any large gain here that is not matched on
held-out tabs as a sign the weights have learnt this genre rather than the guitar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabify.fretting import FretOptions, assign_frets  # noqa: E402
from tabify.gpfile import read_gp  # noqa: E402
from tabify.notes import Note  # noqa: E402

RECORDINGS = Path(__file__).resolve().parents[1] / "recordings"

# Files are split so that changes can be judged on tabs they were not tuned against. Keep
# anything added to HOLDOUT out of the fitting, or the numbers stop meaning anything.
TRAIN = [
    ("Comfortable Liar (Drop C#).gp", None),
    ("isolated guitart.gp", None),
    ("everytime-i-die.gp", "Rhythm"),
]
HOLDOUT = [
    ("Are You Dead Yet_.gp", "Rhythm"),
    ("everytime-i-die.gp", "Lead"),
]


def load(name: str, track: str | None):
    """One tab as (notes to finger, the fingering the human chose, tuning)."""
    tab = read_gp(RECORDINGS / name, track=track)
    if tab.tuning is None or not tab.strokes:
        return None
    notes, want = [], []
    for i, stroke in enumerate(tab.strokes):
        for string, fret in stroke:
            notes.append(Note(float(i), 1.0, tab.tuning.strings[string] + fret, 90))
        want.append(set(stroke))
    return notes, want, tab.tuning


def score(case, opts: FretOptions) -> tuple[int, int, int, int]:
    notes, want, tuning = case
    chosen: dict[float, set[tuple[int, int]]] = {}
    for event in assign_frets(notes, tuning, opts).events:
        chosen.setdefault(round(event.start, 3), set()).update(
            (p.string, p.fret) for p in event.positions
        )
    exact = notes_right = notes_total = 0
    for i, expected in enumerate(want):
        got = chosen.get(float(i), set())
        exact += got == expected
        notes_right += len(got & expected)
        notes_total += len(expected)
    return exact, len(want), notes_right, notes_total


def report(label: str, cases, opts: FretOptions) -> tuple[float, float]:
    exact = strokes = right = total = 0
    for name, case in cases:
        e, s, r, t = score(case, opts)
        print(f"  {name:<34} {s:5d} strokes   exact {e/s:6.1%}   per-note {r/max(t,1):6.1%}")
        exact, strokes, right, total = exact + e, strokes + s, right + r, total + t
    print(f"  {label:<34} {strokes:5d} strokes   exact {exact/strokes:6.1%}   per-note {right/total:6.1%}")
    return exact / strokes, right / total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-only", action="store_true")
    args = ap.parse_args()

    groups = {"TRAIN": TRAIN, "HOLDOUT": HOLDOUT}
    if args.holdout_only:
        groups.pop("TRAIN")

    opts = FretOptions()
    missing = False
    for label, files in groups.items():
        cases = []
        for name, track in files:
            if not (RECORDINGS / name).exists():
                missing = True
                continue
            loaded = load(name, track)
            if loaded:
                cases.append((f"{name[:24]}{' / ' + track if track else ''}", loaded))
        if not cases:
            continue
        print(f"\n{label}")
        report(f"-- {label.lower()} overall", cases, opts)

    if missing:
        print("\n(some tabs were not found in recordings/ - they are gitignored)")
    print("\nfor reference: TART reports 71.8% string-fret Tab F1 given correct notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
