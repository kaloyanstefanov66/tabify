"""Audio -> notes.

Two engines, both optional installs:

* ``basic-pitch`` - Spotify's polyphonic model, handles chords (``pip install tabify-cli[ml]``)
* ``pyin``        - librosa's monophonic pitch tracker, for single-note lines (``pip install tabify-cli[audio]``)

Note times come back in seconds, get cleaned up against the audio's actual pick
attacks and low end (see `tabify.refine`), then are converted to beats using
librosa's beat tracker (or a user-supplied tempo) so the tab lines up with bars.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from tabify import TabifyError
from tabify.notes import Note
from tabify.refine import RefineReport, TimedNote as _TimedNote

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
ENGINES = ("auto", "basic-pitch", "pyin")
SAMPLE_RATE = 22050
HOP = 256


@dataclass
class AudioTranscription:
    notes: list[Note]  # in beats
    bpm: float
    engine: str
    refine: RefineReport | None = None
    tuning: object | None = None  # the Tuning picked when detection was asked for
    tuning_ranking: list | None = None  # (score, Tuning) best first, when detection ran


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def resolve_engine(engine: str) -> str:
    if engine == "auto":
        if _has("basic_pitch"):
            return "basic-pitch"
        if _has("librosa"):
            return "pyin"
        raise TabifyError(
            "audio support is not installed. Install one of:\n"
            "  pipx install 'tabify-cli[audio]'   # single-note lines, any Python\n"
            "  pipx install 'tabify-cli[ml]' --python 3.11   # chords too (basic-pitch)"
        )
    needed = {"basic-pitch": ("basic_pitch", "ml"), "pyin": ("librosa", "audio")}[engine]
    if not _has(needed[0]):
        raise TabifyError(f"engine {engine!r} needs extra dependencies: pip install 'tabify-cli[{needed[1]}]'")
    return engine


def _midi_to_hz(pitch: float) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def _basic_pitch(path: Path, lo: int, hi: int, onset_threshold: float, min_note_ms: float) -> list[_TimedNote]:
    from basic_pitch.inference import predict

    _, _, events = predict(
        str(path),
        onset_threshold=onset_threshold,
        minimum_note_length=min_note_ms,
        minimum_frequency=_midi_to_hz(lo - 0.5),
        maximum_frequency=_midi_to_hz(hi + 0.5),
    )
    return [
        _TimedNote(float(start), float(end), int(pitch), max(1, min(127, round(float(amp) * 127))))
        for start, end, pitch, amp, *_ in events
    ]


def _pyin(y, sr: int, lo: int, hi: int, min_note_ms: float) -> list[_TimedNote]:
    import librosa
    import numpy as np

    f0, voiced, _ = librosa.pyin(
        y, fmin=_midi_to_hz(lo - 1), fmax=_midi_to_hz(hi + 1), sr=sr, frame_length=2048, hop_length=HOP
    )
    pitches = np.where(voiced, np.round(librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))), -1).astype(int)
    onsets = set(librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP).tolist())
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    loudest = float(rms.max()) or 1.0

    frame_time = HOP / sr
    min_frames = max(1, round(min_note_ms / 1000 / frame_time))
    notes: list[_TimedNote] = []
    current, start = -1, 0

    def close(end: int) -> None:
        if current >= 0 and end - start >= min_frames:
            level = float(rms[start:end].mean()) / loudest
            notes.append(_TimedNote(start * frame_time, end * frame_time, current, max(1, round(level * 127))))

    for i, p in enumerate(pitches):
        if p != current or (i in onsets and i > start):
            close(i)
            current, start = int(p), i
    close(len(pitches))
    return notes


def _beat_mapper(y, sr: int, bpm: float | None):
    """Return (seconds -> beats function, tempo)."""
    import librosa
    import numpy as np

    if bpm:
        return (lambda t: t * bpm / 60.0), bpm

    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0]) or 120.0
    if len(beat_times) < 2:
        return (lambda t: t * tempo / 60.0), tempo

    beats = np.asarray(beat_times, dtype=float)
    period = float(np.median(np.diff(beats)))

    def to_beats(t: float) -> float:
        if t < beats[0]:
            return (t - beats[0]) / period
        if t > beats[-1]:
            return len(beats) - 1 + (t - beats[-1]) / period
        return float(np.interp(t, beats, np.arange(len(beats))))

    return to_beats, 60.0 / period


def transcribe_audio(
    path: str | Path,
    *,
    lowest: int,
    highest: int,
    engine: str = "auto",
    bpm: float | None = None,
    onset_threshold: float = 0.5,
    min_note_ms: float = 80.0,
    refine: bool = True,
    detect_tuning: bool = False,
) -> AudioTranscription:
    path = Path(path)
    engine = resolve_engine(engine)

    import librosa

    try:
        y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:  # librosa raises a zoo of backend errors
        raise TabifyError(f"could not load audio {path}: {exc}") from exc

    # Detecting the tuning means listening first and deciding after, so the pitch model gets
    # the whole guitar range (C1 to E6) rather than one tuning's range.
    if detect_tuning:
        lowest, highest = 24, 88

    if engine == "basic-pitch":
        timed = _basic_pitch(path, lowest, highest, onset_threshold, min_note_ms)
    else:
        timed = _pyin(y, sr, lowest, highest, min_note_ms)

    tuning = ranking = None
    if detect_tuning:
        from tabify.tuning import rank_tunings

        ranking = rank_tunings([n.pitch for n in timed])
        tuning = ranking[0][1]
        lowest = tuning.strings[0]

    report = None
    if refine:
        from tabify.refine import refine as refine_notes

        timed, report = refine_notes(timed, y, sr, lowest=lowest)

    to_beats, tempo = _beat_mapper(y, sr, bpm)
    notes = []
    for n in timed:
        start = to_beats(n.start)
        notes.append(Note(start, max(to_beats(n.end) - start, 0.0), n.pitch, n.velocity))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return AudioTranscription(notes, tempo, engine, report, tuning, ranking)
