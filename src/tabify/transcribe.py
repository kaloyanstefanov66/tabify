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
    beat_map: object | None = None  # the BeatMap used, for reporting how much the tempo moved


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def chords_available() -> bool:
    """Whether the polyphonic model can run: onnxruntime, and the model file beside it."""
    from tabify.pitchmodel import MODEL

    return _has("onnxruntime") and MODEL.exists()


def resolve_engine(engine: str) -> str:
    if engine == "auto":
        if chords_available():
            return "basic-pitch"
        if _has("librosa"):
            return "pyin"
        raise TabifyError("audio support is not installed: pip install tabify-cli")
    if engine == "basic-pitch" and not chords_available():
        raise TabifyError("the chord engine needs onnxruntime: pip install onnxruntime")
    if engine == "pyin" and not _has("librosa"):
        raise TabifyError("pyin needs librosa: pip install librosa")
    return engine


# How loud a frame must stay for a note to count as still sounding. The pitch model's own
# default, kept so predictions match what tabify produced before it ran the model directly.
FRAME_THRESHOLD = 0.3


# A note shorter than this is taken to be noise rather than something played. It cannot be a
# fixed number of milliseconds: at 195 BPM a sixteenth note lasts 77 ms, so the 80 ms floor
# tabify used to apply everywhere threw away every sixteenth in a fast riff before anything
# else ran. It is derived from how fast the playing actually is, measured from the attacks
# themselves, and bounded so neither a slow ballad nor a blast beat pushes it somewhere silly.
FLOOR_SHARE = 0.5  # of the shortest gaps between attacks
FLOOR_QUANTILE = 25  # not the median: see below
FLOOR_MIN_MS, FLOOR_MAX_MS = 20.0, 80.0


def note_floor_ms(y, sr: int) -> float:
    """How short a note may be before it is not worth believing, judged from the playing.

    Measured from the *shortest* gaps between attacks rather than the typical one, because
    the detector misses fast notes too, and the notes it misses are exactly the ones this
    floor decides the fate of. On a riff whose strokes were 77 ms apart it found two thirds
    of them, which pushed the median gap to 151 ms and would have left the floor where it
    started - discarding the fast notes for being fast. The gaps it did catch still show how
    quick the playing is, so a low quantile survives the missed ones.
    """
    import numpy as np

    from tabify.refine import detect_onsets

    onsets = np.asarray(detect_onsets(y, sr))
    if len(onsets) < 3:
        return FLOOR_MAX_MS
    quickest = float(np.percentile(np.diff(onsets), FLOOR_QUANTILE)) * 1000.0
    return float(np.clip(quickest * FLOOR_SHARE, FLOOR_MIN_MS, FLOOR_MAX_MS))


# Fast playing is hard for a pitch model in two different ways, and only one of them is
# physics. Identifying a low pitch needs several cycles of it, and a 65 Hz string gives a
# cycle every 15 ms, so a 77 ms sixteenth barely contains enough waveform to measure - that
# part cannot be argued with. But the model also runs at a fixed 86 frames a second, so that
# same note is only six frames wide, and *that* is just arithmetic. Stretching time before
# the model runs buys frames back: on two real recordings it took strokes transcribed exactly
# from 60% to 66% and from 14% to 20%.
#
# It has to be a time stretch that keeps the pitch. Slowing the audio like a tape instead
# drops every note out of the range the model handles best, and precision collapsed to 9%
# when that was measured. Only the frames are worth buying, not the transposition.
STRETCH_FACTOR = 2.0
# Slower playing gains nothing from this and would pay for it in time, so it is only done
# where the notes are short enough to be losing frames. Measured: real drop-tuned riffing
# sits at 81-104 ms between the quickest attacks, while a moderate single-note line sits at
# 136 ms and got worse when it was stretched. The line is drawn between them.
STRETCH_WHEN_GAPS_BELOW_MS = 120.0


def quickest_gap_ms(y, sr: int) -> float:
    """How close together the quickest attacks are - how fast this is really being played."""
    import numpy as np

    from tabify.refine import detect_onsets

    onsets = np.asarray(detect_onsets(y, sr))
    if len(onsets) < 3:
        return float("inf")
    return float(np.percentile(np.diff(onsets), FLOOR_QUANTILE)) * 1000.0


def _midi_to_hz(pitch: float) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


# How much of a recording's energy has to sit above BRIGHT_HZ before it counts as a
# distorted or otherwise dense tone. Measured across real riffs and synthesized test pieces:
# distorted guitar sat at 55%, synthesized chugs at 22%, a clean arpeggio at 16%.
BRIGHT_SHARE = 0.20
BRIGHT_HZ = 2000.0
# basic-pitch's own default is 0.5, which is tuned for clean, sparse playing. On distorted
# guitar it misses most of every chord - dropping it to 0.3 roughly doubled how many strokes
# came out exactly right on two real recordings. On clean material the same change fills the
# tab with ghost notes instead, so the tone decides.
SENSITIVE, CAUTIOUS = 0.3, 0.5


def sensitivity_for(y, sr: int) -> float:
    """How hard to listen, judged from the tone of the recording.

    A pitch model has one threshold for "is this a note", and the right setting is not the
    same for a distorted wall of harmonics as for a clean arpeggio: what finds the notes in
    one invents them in the other. Distortion shows up as energy high above the fundamentals,
    which is measurable before the model runs.
    """
    import librosa
    import numpy as np

    spectrum = np.abs(librosa.stft(y, n_fft=4096, hop_length=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    total = spectrum.sum()
    if not total:
        return CAUTIOUS
    return SENSITIVE if spectrum[freqs > BRIGHT_HZ].sum() / total >= BRIGHT_SHARE else CAUTIOUS


def _basic_pitch(
    path: Path, lo: int, hi: int, onset_threshold: float, min_note_ms: float, audio=None
) -> list[_TimedNote]:
    """Hear the notes, running the pitch model straight from its ONNX file.

    This used to go through the basic-pitch package, which on Python 3.11 and newer drags
    TensorFlow along - 1.3 GB that was downloaded and never loaded, since the ONNX copy of
    the same model was doing the work either way. `tabify.pitchmodel` does what the package
    did for us, and a test checks the notes still come out identical.
    """
    import librosa

    from tabify import pitchmodel

    if audio is None:
        audio, _ = librosa.load(str(path), sr=pitchmodel.SAMPLE_RATE, mono=True)
    output = pitchmodel.predict(audio)
    n_frames = output["note"].shape[0]
    events = pitchmodel.notes_from_output(
        output["note"],
        output["onset"],
        onset_threshold=onset_threshold,
        frame_threshold=FRAME_THRESHOLD,
        min_note_frames=int(round(min_note_ms / 1000 * pitchmodel.FRAMES_PER_SECOND)),
        lowest_hz=_midi_to_hz(lo - 0.5),
        highest_hz=_midi_to_hz(hi + 0.5),
    )
    times = pitchmodel.frame_times(n_frames)
    notes = []
    for first, last, pitch, amplitude in events:
        start = float(times[min(first, n_frames - 1)])
        finish = float(times[min(last, n_frames - 1)])
        notes.append(_TimedNote(start, finish, int(pitch), max(1, min(127, round(float(amplitude) * 127)))))
    return notes


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


@dataclass
class BeatMap:
    """Maps recording time to musical time, following the tempo as it actually moves."""

    to_beats: object  # callable: seconds -> beats
    bpm: float  # the typical tempo, for display and for exports that want one number
    bpm_low: float = 0.0
    bpm_high: float = 0.0
    beats: object = None  # detected beat times, if any

    @property
    def varies(self) -> bool:
        """True when the tempo moves enough that a single BPM would misplace notes."""
        return bool(self.bpm_high and self.bpm_high - self.bpm_low > 0.08 * self.bpm)


def _beat_mapper(y, sr: int, bpm: float | None) -> BeatMap:
    """Work out where the beats are, following a tempo that drifts rather than assuming one.

    Nobody plays to a perfect grid, so the beat positions are kept and interpolated between:
    a note lands where it falls *between the beats around it*, not where a fixed tempo says
    it should be. The beats come from tabify's own attack detection, which finds far more of
    a dense distorted riff than a general-purpose beat tracker does.
    """
    import librosa
    import numpy as np

    if bpm:  # the user told us the tempo, so take them at their word
        return BeatMap(lambda t: t * bpm / 60.0, bpm)

    from tabify.refine import onset_envelope

    envelope = onset_envelope(y, sr)
    tempo, beat_times = librosa.beat.beat_track(
        onset_envelope=envelope, sr=sr, hop_length=HOP, units="time", trim=False
    )
    tempo = float(np.atleast_1d(tempo)[0]) or 120.0
    if len(beat_times) < 2:
        return BeatMap(lambda t: t * tempo / 60.0, tempo)

    beats = np.asarray(beat_times, dtype=float)
    intervals = np.diff(beats)
    period = float(np.median(intervals))
    local_bpm = 60.0 / intervals[intervals > 0]

    def to_beats(t: float) -> float:
        if t < beats[0]:  # before the first beat / after the last, carry the tempo on
            return (t - beats[0]) / period
        if t > beats[-1]:
            return len(beats) - 1 + (t - beats[-1]) / period
        return float(np.interp(t, beats, np.arange(len(beats))))

    return BeatMap(
        to_beats,
        60.0 / period,
        float(np.percentile(local_bpm, 10)),
        float(np.percentile(local_bpm, 90)),
        beats,
    )


def transcribe_audio(
    path: str | Path,
    *,
    lowest: int,
    highest: int,
    engine: str = "auto",
    bpm: float | None = None,
    onset_threshold: float | None = None,
    min_note_ms: float | None = None,
    stretch: float | None = None,
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

    gap_ms = quickest_gap_ms(y, sr)
    if min_note_ms is None:
        min_note_ms = float(min(max(gap_ms * FLOOR_SHARE, FLOOR_MIN_MS), FLOOR_MAX_MS))

    if engine == "basic-pitch":
        if onset_threshold is None:
            onset_threshold = sensitivity_for(y, sr)
        if stretch is None:
            stretch = STRETCH_FACTOR if gap_ms < STRETCH_WHEN_GAPS_BELOW_MS else 1.0
        if stretch > 1.0:
            slowed = librosa.effects.time_stretch(y, rate=1.0 / stretch)
            timed = _basic_pitch(path, lowest, highest, onset_threshold, min_note_ms * stretch, audio=slowed)
            # Back onto the recording's own clock, before anything else looks at the audio.
            timed = [
                _TimedNote(n.start / stretch, n.end / stretch, n.pitch, n.velocity) for n in timed
            ]
        else:
            timed = _basic_pitch(path, lowest, highest, onset_threshold, min_note_ms)
    else:
        timed = _pyin(y, sr, lowest, highest, min_note_ms)

    tuning = ranking = None
    report = None
    if refine:
        from tabify.refine import refine as refine_notes

    if detect_tuning:
        from tabify.tuning import lowest_possible_string, rank_tunings

        # What identifies a drop tuning is its low notes, and those are exactly the notes a
        # pitch model misses: a distorted low chug reaches it as overtones with the
        # fundamental gone. Asked about the raw output, detection sees a riff with no low
        # end and answers "standard", which then makes the real notes unplayable.
        #
        # So the roots are restored first, with the floor set as low as any known tuning
        # goes rather than one tuning's, and the tuning is judged on what was really played.
        # The cleanup is then redone against the tuning that won, since it uses the lowest
        # string to decide which roots are reachable at all.
        judged = timed
        if refine:
            judged, _ = refine_notes(timed, y, sr, lowest=lowest_possible_string())
        ranking = rank_tunings([n.pitch for n in judged])
        tuning = ranking[0][1]
        lowest = tuning.strings[0]

    if refine:
        timed, report = refine_notes(timed, y, sr, lowest=lowest)

    beat_map = _beat_mapper(y, sr, bpm)
    notes = []
    for n in timed:
        start = beat_map.to_beats(n.start)
        notes.append(Note(start, max(beat_map.to_beats(n.end) - start, 0.0), n.pitch, n.velocity))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return AudioTranscription(notes, beat_map.bpm, engine, report, tuning, ranking, beat_map)
