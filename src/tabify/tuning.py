"""Instrument tunings."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from tabify import TabifyError
from tabify.notes import midi_to_name, name_to_midi

# Strings are listed from lowest to highest.
TUNINGS: dict[str, str] = {
    # Standard and whole-guitar downtunings
    "standard": "E2 A2 D3 G3 B3 E4",
    "half-step-down": "D#2 G#2 C#3 F#3 A#3 D#4",
    "d-standard": "D2 G2 C3 F3 A3 D4",
    "c#-standard": "C#2 F#2 B2 E3 G#3 C#4",
    "c-standard": "C2 F2 A#2 D#3 G3 C4",
    "b-standard": "B1 E2 A2 D3 F#3 B3",
    "a#-standard": "A#1 D#2 G#2 C#3 F3 A#3",
    "a-standard": "A1 D2 G2 C3 E3 A3",
    # Drop tunings, every semitone from drop D down to drop F#
    "drop-d": "D2 A2 D3 G3 B3 E4",
    "drop-c#": "C#2 G#2 C#3 F#3 A#3 D#4",
    "drop-c": "C2 G2 C3 F3 A3 D4",
    "drop-b": "B1 F#2 B2 E3 G#3 C#4",
    "drop-a#": "A#1 F2 A#2 D#3 G3 C4",
    "drop-a": "A1 E2 A2 D3 F#3 B3",
    "drop-g#": "G#1 D#2 G#2 C#3 F3 A#3",
    "drop-g": "G1 D2 G2 C3 E3 A3",
    "drop-f#": "F#1 C#2 F#2 B2 D#3 G#3",
    # Open and modal
    "open-g": "D2 G2 D3 G3 B3 D4",
    "open-d": "D2 A2 D3 F#3 A3 D4",
    "open-e": "E2 B2 E3 G#3 B3 E4",
    "open-c": "C2 G2 C3 G3 C4 E4",
    "dadgad": "D2 A2 D3 G3 A3 D4",
    # Extended range
    "7-string": "B1 E2 A2 D3 G3 B3 E4",
    "7-string-drop-a": "A1 E2 A2 D3 G3 B3 E4",
    "7-string-a#-standard": "A#1 D#2 G#2 C#3 F#3 A#3 D#4",
    "8-string": "F#1 B1 E2 A2 D3 G3 B3 E4",
    "8-string-drop-e": "E1 B1 E2 A2 D3 G3 B3 E4",
    "bass": "E1 A1 D2 G2",
    "bass-5": "B0 E1 A1 D2 G2",
    "bass-drop-d": "D1 A1 D2 G2",
}

# Flat spellings and common nicknames for the presets above. Sharps and flats name the
# same pitch, and which one a band uses is arbitrary, so both should work.
ALIASES: dict[str, str] = {
    "e-standard": "standard",
    "eb-standard": "half-step-down",
    "d#-standard": "half-step-down",
    "half-step": "half-step-down",
    "e-flat": "half-step-down",
    "whole-step-down": "d-standard",
    "full-step-down": "d-standard",
    "db-standard": "c#-standard",
    "bb-standard": "a#-standard",
    "drop-db": "drop-c#",
    "drop-bb": "drop-a#",
    "drop-ab": "drop-g#",
    "drop-gb": "drop-f#",
    "7-string-standard": "7-string",
    "7-string-bb-standard": "7-string-a#-standard",
    "8-string-standard": "8-string",
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


def preset_key(spec: str) -> str | None:
    """The preset this name refers to, allowing spaces, underscores and flat spellings."""
    key = re.sub(r"[\s_]+", "-", spec.strip().lower())
    key = ALIASES.get(key, key)
    return key if key in TUNINGS else None


def parse_tuning(spec: str) -> Tuning:
    """Accept a preset name (``drop-d``, ``drop-db``) or explicit notes (``D2 A2 D3 G3 B3 E4``)."""
    key = preset_key(spec)
    if key:
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


def score_tuning(tuning: Tuning, pitches: list[int], *, max_fret: int = 22) -> float:
    """How well a set of played pitches fits a tuning. Higher is better; 0 means unplayable.

    Pitch content alone can't pin a tuning down - a lower tuning can reach every note a
    higher one can, just at higher frets. What distinguishes them is how guitarists use
    them: riffs lean on the open low string, and sit low on the neck. So this rewards
    notes landing exactly on open strings, the lowest note being the lowest string, and
    low fret positions, while anything unplayable is disqualifying.
    """
    if not pitches:
        return 0.0

    lowest, highest = tuning.strings[0], tuning.strings[-1] + max_fret
    playable = [p for p in pitches if lowest <= p <= highest]
    if not playable:
        return 0.0
    # A note heard once is a misdetection; a note heard again and again is the instrument.
    # Judging unplayable notes by that, rather than by how many there are, is what keeps a
    # handful of low ghosts from ruling out the right tuning - while a bass line's low E,
    # which recurs constantly, still rules out every guitar tuning above it.
    counts = Counter(pitches)
    unplayable = [p for p in pitches if not lowest <= p <= highest]
    recurring = sum(1 for p in unplayable if counts[p] >= 3)
    unplayable_share = (recurring + 0.2 * (len(unplayable) - recurring)) / len(pitches)
    open_share = sum(p in tuning.strings for p in playable) / len(playable)
    low_string_share = sum(p == lowest for p in playable) / len(playable)
    # Where the playing bottoms out: the lowest note heard repeatedly, rather than the single
    # lowest note (most likely a ghost) or a percentile (which sits above the lowest string
    # whenever someone favours their second string, as bass lines do).
    recurring_pitches = [p for p, seen in counts.items() if seen >= 3]
    floor_pitch = min(recurring_pitches) if recurring_pitches else min(pitches)
    starts_on_low_string = abs(floor_pitch - lowest) < 1.0
    # Nobody tunes to a string they never touch. Without this, a 7-string tuning wins on any
    # 6-string song, since it can play everything standard can plus a low string it never uses.
    unused_low_string = not any(lowest <= p < lowest + 5 for p in playable)
    # Lowest fret each note could be played at, as a fraction of the neck.
    mean_fret = sum(min(p - s for s in tuning.strings if s <= p) for p in playable) / len(playable) / max_fret

    # Weights picked by grid search over a small set of labelled riffs (12 of 12 correct);
    # with that few cases they're a sensible starting point, not a tuned optimum.
    return max(
        0.0,
        1.0
        - 2.0 * unplayable_share
        + 0.6 * open_share
        + 0.8 * low_string_share  # leaning on the open low string is the signature of a drop tuning
        + (0.8 if starts_on_low_string else 0.0)
        - (0.7 if unused_low_string else 0.0)
        - 0.5 * mean_fret,
    )


# Roughly how often each tuning turns up in real music, most common first. Used only to break
# near-ties: a riff that never touches an open string fits many tunings equally well, and
# standard is a far better guess than, say, dadgad when nothing else separates them.
POPULARITY = (
    "standard", "drop-d", "half-step-down", "d-standard", "drop-c", "drop-c#", "drop-b", "open-g",
    "open-d", "dadgad", "7-string", "drop-a", "drop-a#", "c-standard", "open-e", "c#-standard",
    "b-standard", "drop-g", "drop-g#", "open-c", "7-string-drop-a", "8-string", "a#-standard",
    "a-standard", "drop-f#", "8-string-drop-e", "7-string-a#-standard", "bass", "bass-5", "bass-drop-d",
)
PRIOR_WEIGHT = 0.7


def _prior(name: str) -> float:
    if name not in POPULARITY:
        return 0.0
    return PRIOR_WEIGHT * (1 - POPULARITY.index(name) / len(POPULARITY))


def rank_tunings(pitches: list[int], *, candidates=None, max_fret: int = 22) -> list[tuple[float, Tuning]]:
    """Score every preset tuning against the pitches heard, best first."""
    names = candidates if candidates is not None else list(TUNINGS)
    scored = []
    for name in names:
        tuning = parse_tuning(name)
        fit = score_tuning(tuning, pitches, max_fret=max_fret)
        scored.append((fit + _prior(tuning.name) if fit else 0.0, tuning))
    return sorted(scored, key=lambda s: (-s[0], s[1].strings[0]))
