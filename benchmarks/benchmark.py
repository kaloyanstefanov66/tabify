"""Score tabify against riffs whose tab we already know.

Each piece below is written as a tab, rendered to audio with a real guitar tone, then fed
back through the whole pipeline and scored. That makes "did this change help?" a number.

Caveat worth keeping in mind: this is synthesized audio, so it is kinder than a real
recording (no room, no amp, no player noise). It is useful for measuring *changes*, not
for claiming real-world accuracy - which is why one piece deliberately high-passes away
the low end, the way the isolated stem that prompted all this did.

    python benchmarks/benchmark.py            # score everything
    python benchmarks/benchmark.py --keep-audio out/   # also write the rendered wavs
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from tabify.evaluate import Stroke, score, strokes_of  # noqa: E402
from tabify.fretting import Position, TabEvent, assign_frets  # noqa: E402
from tabify.notes import Note  # noqa: E402
from tabify.rhythm import TimeSignature, align_to_bars, quantize  # noqa: E402
from tabify.synth import synthesize  # noqa: E402
from tabify.transcribe import transcribe_audio  # noqa: E402
from tabify.tuning import parse_tuning  # noqa: E402

SAMPLE_RATE = 44100


@dataclass
class Piece:
    name: str
    tuning: str
    bpm: float
    instrument: str
    # (beat, [(string, fret), ...], duration in beats)
    strokes: list[tuple[float, list[tuple[int, int]], float]]
    high_pass_hz: float = 0.0  # simulate a recording whose low end was rolled off
    noise_db: float = -60.0
    palm_muted: bool = False
    grid: int = 4

    def events(self) -> list[TabEvent]:
        out = []
        for start, positions, duration in self.strokes:
            notes = [Note(start, duration, 0) for _ in positions]  # pitch filled in by the caller
            out.append(TabEvent(start, notes, [Position(s, f) for s, f in positions], palm_mute=self.palm_muted))
        return out

    def reference(self) -> list[Stroke]:
        return [Stroke(start, s, f) for start, positions, _ in self.strokes for s, f in positions]


def _chug_riff() -> Piece:
    """Drop-C style: open power chords and single-note chugs, 8th notes - the shape that failed."""
    chord = [(0, 0), (1, 0), (2, 0)]
    pattern = [chord, [(0, 0)], [(0, 0)], chord, [(0, 0)], [(1, 1), (2, 1), (0, 1)], chord, [(0, 0)]]
    strokes = []
    for bar in range(4):
        for i, positions in enumerate(pattern):
            strokes.append((bar * 4 + i * 0.5, positions, 0.45))
    return Piece("drop-C chug riff", "drop-c", 92, "distortion", strokes, palm_muted=True)


def _missing_low_end() -> Piece:
    """The same riff, with everything under 90 Hz taken away - like an isolated stem."""
    piece = _chug_riff()
    piece.name = "drop-C chug riff, low end cut"
    piece.high_pass_hz = 90.0
    return piece


def _power_chord_moves() -> Piece:
    """Held power chords moving around the neck: tests pitch and position, not rhythm."""
    shapes = [0, 3, 5, 1, 8, 6, 0, 10]
    strokes = [(i * 2.0, [(0, f), (1, f), (2, f)], 1.9) for i, f in enumerate(shapes)]
    return Piece("drop-C power chords", "drop-c", 100, "overdrive", strokes)


def _clean_arpeggio() -> Piece:
    """Clean picked arpeggios in standard tuning - the easy case, as a control."""
    shapes = [[(1, 0), (2, 2), (3, 2), (4, 1), (5, 0)], [(0, 1), (1, 3), (2, 3), (3, 2), (4, 1)]]
    strokes = []
    for bar in range(4):
        shape = shapes[bar % 2]
        for i, position in enumerate(shape + shape[::-1][1:4]):
            strokes.append((bar * 4 + i * 0.5, [position], 0.5))
    return Piece("standard clean arpeggio", "standard", 90, "steel", strokes)


def _single_note_line() -> Piece:
    """A single-note lead line in one pentatonic box, 16th notes.

    Written inside one box on purpose: the same pitch exists in several places on the neck,
    so a reference that jumps around would mark a perfectly good fingering wrong.
    """
    box = [(0, 5), (0, 8), (1, 5), (1, 7), (2, 5), (2, 7), (2, 5), (1, 7), (1, 5), (0, 8), (0, 5), (0, 8)]
    strokes = [(i * 0.25, [position], 0.25) for i, position in enumerate(box)]
    return Piece("standard single-note line", "standard", 110, "clean", strokes)


PIECES = [_chug_riff(), _missing_low_end(), _power_chord_moves(), _clean_arpeggio(), _single_note_line()]


def render(piece: Piece, path: Path) -> None:
    tuning = parse_tuning(piece.tuning)
    events = []
    for start, positions, duration in piece.strokes:
        notes = [Note(start, duration, tuning.strings[s] + f) for s, f in positions]
        events.append(TabEvent(start, notes, [Position(s, f) for s, f in positions], palm_mute=piece.palm_muted))
    audio = synthesize(events, tuning, instrument=piece.instrument, bpm=piece.bpm, sample_rate=SAMPLE_RATE)
    if piece.high_pass_hz:
        spectrum = np.fft.rfft(audio, axis=0)
        freqs = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)
        spectrum[freqs < piece.high_pass_hz] *= 0.03
        audio = np.fft.irfft(spectrum, len(audio), axis=0).astype(np.float32)
    if piece.noise_db > -120:
        rms = float(np.sqrt((audio**2).mean())) or 1e-6
        audio = audio + np.random.default_rng(0).normal(0, rms * 10 ** (piece.noise_db / 20), audio.shape)
    sf.write(str(path), audio, SAMPLE_RATE)


def transcribe(piece: Piece, path: Path, **kwargs) -> list[Stroke]:
    tuning = parse_tuning(piece.tuning)
    result = transcribe_audio(
        path, lowest=tuning.strings[0], highest=tuning.strings[-1] + 22, engine="basic-pitch", bpm=piece.bpm, **kwargs
    )
    notes = align_to_bars(quantize(result.notes, piece.grid), TimeSignature())
    return strokes_of(assign_frets(notes, tuning).events)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep-audio", metavar="DIR", help="write the rendered audio here for listening")
    parser.add_argument("--no-cleanup", action="store_true", help="score the raw model output, without tabify's cleanup")
    parser.add_argument("--only", help="only run pieces whose name contains this")
    args = parser.parse_args()

    out_dir = Path(args.keep_audio) if args.keep_audio else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    pieces = [p for p in PIECES if not args.only or args.only.lower() in p.name.lower()]
    print(f"{'piece':32s} {'notes':>28s}   {'tab':>28s}")
    print("-" * 94)
    note_f1s, tab_f1s = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for piece in pieces:
            path = (out_dir or Path(tmp)) / (piece.name.replace(" ", "_").replace(",", "") + ".wav")
            render(piece, path)
            predicted = transcribe(piece, path, refine=not args.no_cleanup)
            notes, tab = score(piece.reference(), predicted, parse_tuning(piece.tuning))
            note_f1s.append(notes.f1)
            tab_f1s.append(tab.f1)
            print(f"{piece.name:32s} {str(notes):>28s}   {str(tab):>28s}")
    print("-" * 94)
    print(f"{'mean':32s} {'F1 ' + format(np.mean(note_f1s), '.0%'):>28s}   {'F1 ' + format(np.mean(tab_f1s), '.0%'):>28s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
