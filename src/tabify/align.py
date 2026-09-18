"""Lining up two versions of the same riff.

A transcription and a written tab never match stroke for stroke: a note gets missed, a ghost
appears, a repeat gets played once more than it was written. Comparing them position by
position would then mark everything after the first difference as wrong, which says nothing
useful about either.

So they're aligned the way two versions of a text are: find the pairing that matches as much
as possible, and account for the rest as notes that were added or missed. What comes out is
which strokes correspond, and where each side has something the other doesn't.

Strokes are compared by the pitches they sound rather than by string and fret, because the
same chord fingered in two places is the same chord - where it sits on the neck is judged
separately, once the strokes are paired up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GAP_PENALTY = -0.6  # skipping a stroke costs less than forcing a bad pairing


@dataclass
class Alignment:
    """How two sequences of strokes correspond."""

    pairs: list[tuple[int, int]] = field(default_factory=list)  # (index in a, index in b)
    only_in_a: list[int] = field(default_factory=list)
    only_in_b: list[int] = field(default_factory=list)

    @property
    def matched(self) -> int:
        return len(self.pairs)


def similarity(a: set[int], b: set[int]) -> float:
    """How alike two strokes are, from 0 (nothing shared) to 1 (the same notes)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def align(first: list[set[int]], second: list[set[int]], *, gap: float = GAP_PENALTY) -> Alignment:
    """Pair up two sequences of strokes, allowing for extra and missing ones.

    Each stroke is the set of pitches it sounds. Order is respected - a riff is a sequence,
    so a pairing that jumps backwards isn't a pairing.
    """
    rows, columns = len(first), len(second)
    score = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    back = [[""] * (columns + 1) for _ in range(rows + 1)]

    for i in range(1, rows + 1):
        score[i][0] = score[i - 1][0] + gap
        back[i][0] = "up"
    for j in range(1, columns + 1):
        score[0][j] = score[0][j - 1] + gap
        back[0][j] = "left"

    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            options = (
                (score[i - 1][j - 1] + similarity(first[i - 1], second[j - 1]), "diag"),
                (score[i - 1][j] + gap, "up"),
                (score[i][j - 1] + gap, "left"),
            )
            score[i][j], back[i][j] = max(options)

    alignment = Alignment()
    i, j = rows, columns
    while i > 0 or j > 0:
        move = back[i][j] if (i and j) else ("up" if i else "left")
        if move == "diag":
            i, j = i - 1, j - 1
            if similarity(first[i], second[j]) > 0:
                alignment.pairs.append((i, j))
            else:  # paired only because they sit in the same place in the riff, not because they match
                alignment.only_in_a.append(i)
                alignment.only_in_b.append(j)
        elif move == "up":
            i -= 1
            alignment.only_in_a.append(i)
        else:
            j -= 1
            alignment.only_in_b.append(j)

    alignment.pairs.reverse()
    alignment.only_in_a.reverse()
    alignment.only_in_b.reverse()
    return alignment


def agreement(first: list[set[int]], second: list[set[int]], alignment: Alignment) -> dict:
    """Summarize an alignment: how much of each side the other one got."""
    exact = sum(1 for i, j in alignment.pairs if first[i] == second[j])
    partial = sum(1 for i, j in alignment.pairs if first[i] != second[j] and first[i] & second[j])
    return {
        "strokes_a": len(first),
        "strokes_b": len(second),
        "exact": exact,
        "partial": partial,
        "missed": len(alignment.only_in_a),
        "extra": len(alignment.only_in_b),
        "recall": exact / len(first) if first else 0.0,
        "precision": exact / len(second) if second else 0.0,
    }
