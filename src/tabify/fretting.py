"""Choose a playable string/fret for every note.

Any pitch can usually be played in several places on the neck. We enumerate
the possible fingerings for each chord (or single note), score each one on
how comfortable it is by itself, and then run a Viterbi search over the whole
piece to minimise total hand movement. The result reads like a tab a guitarist
would actually write, instead of jumping around the neck.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tabify.notes import Note
from tabify.tuning import Tuning

MAX_CANDIDATES = 48


@dataclass(frozen=True)
class Position:
    string: int  # 0 = lowest string
    fret: int  # relative to the capo; 0 = open (or capo)


@dataclass(frozen=True)
class FretOptions:
    capo: int = 0
    max_fret: int = 22  # highest fret on the neck (absolute, ignoring capo)
    max_span: int = 4  # max distance between lowest and highest fretted note in a chord
    # Cost weights
    shift_weight: float = 1.0  # per fret of hand movement between events
    big_shift_penalty: float = 2.0  # extra per fret beyond `max_span` in one move
    span_weight: float = 0.5  # per fret of stretch inside a chord
    height_weight: float = 0.08  # prefer lower positions a little
    string_jump_weight: float = 0.15  # prefer staying on nearby strings


@dataclass
class TabEvent:
    """Notes that start together, with the chosen fingering."""

    start: float
    notes: list[Note]
    positions: list[Position]


@dataclass
class FretResult:
    events: list[TabEvent]
    dropped: list[Note] = field(default_factory=list)


@dataclass(frozen=True)
class _Candidate:
    positions: tuple[Position, ...]
    cost: float
    hand: float | None  # centre of the fretted notes, None if all open strings
    mean_string: float


def _options(pitch: int, tuning: Tuning, opts: FretOptions) -> list[Position]:
    top = opts.max_fret - opts.capo
    out = []
    for s, open_pitch in enumerate(tuning.strings):
        fret = pitch - open_pitch - opts.capo
        if 0 <= fret <= top:
            out.append(Position(s, fret))
    return out


def _candidates(pitches: list[int], tuning: Tuning, opts: FretOptions) -> list[_Candidate]:
    per_note = [_options(p, tuning, opts) for p in pitches]
    found: list[tuple[Position, ...]] = []

    def backtrack(i: int, chosen: list[Position], used: set[int], lo: int, hi: int) -> None:
        if i == len(per_note):
            found.append(tuple(chosen))
            return
        for pos in per_note[i]:
            if pos.string in used:
                continue
            nlo, nhi = lo, hi
            if pos.fret > 0:
                nlo, nhi = min(lo, pos.fret), max(hi, pos.fret)
                if nhi - nlo > opts.max_span:
                    continue
            chosen.append(pos)
            used.add(pos.string)
            backtrack(i + 1, chosen, used, nlo, nhi)
            used.discard(pos.string)
            chosen.pop()

    backtrack(0, [], set(), 10**6, -1)

    cands = []
    for positions in found:
        fretted = [p.fret for p in positions if p.fret > 0]
        if fretted:
            lo, hi = min(fretted), max(fretted)
            hand = (lo + hi) / 2
            cost = opts.span_weight * (hi - lo) + opts.height_weight * lo
        else:
            hand, cost = None, 0.0
        mean_string = sum(p.string for p in positions) / len(positions)
        cands.append(_Candidate(positions, cost, hand, mean_string))
    cands.sort(key=lambda c: c.cost)
    return cands[:MAX_CANDIDATES]


def _playable(pitches: list[int], tuning: Tuning, opts: FretOptions) -> tuple[list[int], list[int], list[_Candidate]]:
    """Return (kept, dropped, candidates), dropping as few notes as needed to make the chord playable."""
    kept = [p for p in pitches if _options(p, tuning, opts)]
    dropped = [p for p in pitches if p not in kept]
    kept = kept[-len(tuning.strings):] if len(kept) > len(tuning.strings) else kept
    dropped += [p for p in pitches if p not in kept and p not in dropped]

    cands = _candidates(kept, tuning, opts) if kept else []
    while kept and not cands:
        best: tuple[int, list[_Candidate]] | None = None
        for i in range(len(kept)):
            trial = _candidates(kept[:i] + kept[i + 1:], tuning, opts)
            if best is None or len(trial) > len(best[1]):
                best = (i, trial)
        assert best is not None
        dropped.append(kept.pop(best[0]))
        cands = best[1]
    return kept, dropped, cands


def _transition(a: _Candidate, b: _Candidate, opts: FretOptions) -> float:
    cost = opts.string_jump_weight * abs(a.mean_string - b.mean_string)
    if a.hand is None or b.hand is None:
        return cost
    d = abs(a.hand - b.hand)
    cost += opts.shift_weight * d
    if d > opts.max_span:
        cost += opts.big_shift_penalty * (d - opts.max_span)
    return cost


def assign_frets(notes: list[Note], tuning: Tuning, opts: FretOptions | None = None) -> FretResult:
    opts = opts or FretOptions()

    # Group notes that start together; keep the longest copy of duplicate pitches.
    groups: dict[float, dict[int, Note]] = {}
    for n in notes:
        g = groups.setdefault(round(n.start, 6), {})
        if n.pitch not in g or n.duration > g[n.pitch].duration:
            g[n.pitch] = n

    dropped: list[Note] = []
    steps: list[tuple[float, list[Note], list[_Candidate]]] = []
    for start in sorted(groups):
        by_pitch = groups[start]
        kept, lost, cands = _playable(sorted(by_pitch), tuning, opts)
        dropped.extend(by_pitch[p] for p in lost)
        if kept:
            steps.append((start, [by_pitch[p] for p in kept], cands))

    if not steps:
        return FretResult([], dropped)

    # Viterbi over the candidate fingerings.
    scores = [c.cost for c in steps[0][2]]
    back: list[list[int]] = []
    for i in range(1, len(steps)):
        prev, cur = steps[i - 1][2], steps[i][2]
        new_scores, pointers = [], []
        for c in cur:
            best_j = min(range(len(prev)), key=lambda j: scores[j] + _transition(prev[j], c, opts))
            new_scores.append(scores[best_j] + _transition(prev[best_j], c, opts) + c.cost)
            pointers.append(best_j)
        scores = new_scores
        back.append(pointers)

    k = min(range(len(scores)), key=scores.__getitem__)
    chosen = [k]
    for pointers in reversed(back):
        k = pointers[k]
        chosen.append(k)
    chosen.reverse()

    events = [
        TabEvent(start, step_notes, list(cands[k].positions))
        for (start, step_notes, cands), k in zip(steps, chosen)
    ]
    return FretResult(events, dropped)
