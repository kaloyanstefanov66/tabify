"""Guitar-aware cleanup of a pitch model's raw notes, using the audio itself.

Pitch models report notes; a tab needs pick strokes. On a real isolated distorted
guitar track, three things went wrong between the two - each measured, not assumed:

* The notes of one strummed chord got onsets up to ~80 ms apart, so they quantized
  into separate steps ("6 - 6 - 6" instead of a stacked chord). Fix: find the actual
  pick attacks in the audio and snap notes to them.
* Fast repeated chugs merged into one long note, since the pitch never changes
  between them. Fix: split sounding notes at attacks where their chord is re-struck.
* Low roots whose fundamental isn't in the recording at all (isolated stems and small
  speakers roll off the low end) were never reported, so a drop-C power chord
  (C2 G2 C3) came out as just G2 C3, on the wrong strings. Fix: measure where the
  recording's low end actually stops, and restore roots below that point - from the
  chord's shape for dyads, from the root's own overtones for single notes.

Palm mutes are not detected here: on that same track, muted and let-ring strokes
measured the same decay and brightness, because heavy distortion compresses both.
See `tabify.techniques` for how they're inferred instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

HOP = 256
_EPS = 1e-9


@dataclass(frozen=True)
class TimedNote:
    start: float  # seconds
    end: float
    pitch: int  # MIDI
    velocity: int


@dataclass
class RefineReport:
    onsets: int = 0
    snapped: int = 0
    splits: int = 0
    roots_added: int = 0
    low_end_hz: float = 0.0


def _hz(pitch: float) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


# --- attacks -----------------------------------------------------------------


def detect_onsets(y: np.ndarray, sr: int, *, delta: float = 0.05, min_gap: float = 0.07, gate_db: float = -45.0):
    """Pick attacks, tuned for dense distorted guitar. Returns onset times in seconds."""
    import librosa
    from scipy.ndimage import maximum_filter1d

    spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP))
    log_spec = np.log1p(100 * spec)
    flux = np.maximum(0.0, np.diff(log_spec, axis=1, prepend=log_spec[:, :1])).sum(axis=0)
    # Normalize against the local maximum (1.5 s), not the whole track's: a handful of big
    # accents otherwise push every ordinary chug below the peak picker's threshold.
    envelope = flux / (maximum_filter1d(flux, max(3, int(1.5 * sr / HOP))) + _EPS)
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope, sr=sr, hop_length=HOP, delta=delta, wait=max(1, int(min_gap * sr / HOP)),
        pre_max=3, post_max=3, pre_avg=10, post_avg=10, normalize=False,
    )
    # Local normalization also amplifies noise in silence, so drop "attacks" nobody could hear.
    rms_db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0] + _EPS)
    loudest = rms_db.max()
    keep = [f for f in frames if rms_db[f : f + 5].max(initial=-np.inf) > loudest + gate_db]
    return librosa.frames_to_time(np.asarray(keep, dtype=int), sr=sr, hop_length=HOP)


def snap_to_onsets(notes: list[TimedNote], onsets, *, before: float = 0.12, after: float = 0.04):
    """Move each note's start to the nearest attack in [start - before, start + after].

    The window is lopsided because pitch models report low notes late: the pitch only
    becomes unambiguous a few cycles after the pick hits the string.
    Returns (notes, how many moved).
    """
    onsets = np.asarray(onsets)
    out, moved = [], 0
    for n in notes:
        lo = np.searchsorted(onsets, n.start - before, "left")
        hi = np.searchsorted(onsets, n.start + after, "right")
        if hi > lo:
            candidates = onsets[lo:hi]
            t = float(candidates[np.argmin(np.abs(candidates - n.start))])
            if t < n.end - 0.01 and t != n.start:
                n = replace(n, start=t)
                moved += 1
        out.append(n)
    return cluster_starts(out), moved


def cluster_starts(notes: list[TimedNote], window: float = 0.06) -> list[TimedNote]:
    """Give notes that start within `window` of each other one shared start (the earliest)."""
    out: list[TimedNote] = []
    anchor = None
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if anchor is None or n.start - anchor > window:
            anchor = n.start
        out.append(n if n.start == anchor else replace(n, start=anchor))
    return out


def split_restrikes(notes: list[TimedNote], onsets, *, min_piece: float = 0.05):
    """Split notes that keep sounding through an attack where they were actually re-struck.

    A sounding note is split at an attack if nothing else starts there (the attack must
    belong to something already ringing), or if a note it was struck together with
    starts again there (its chord was re-struck, but the model merged this string).
    Returns (notes, number of splits).
    """
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    splits = 0
    for t in np.asarray(onsets, dtype=float):
        starting = {n.pitch for n in notes if abs(n.start - t) < 1e-6}
        updated = []
        for n in notes:
            if n.start < t - min_piece and n.end > t + min_piece:
                mates = {m.pitch for m in notes if m is not n and abs(m.start - n.start) < 1e-6}
                if not starting or mates & starting:
                    updated.append(replace(n, end=float(t)))
                    updated.append(replace(n, start=float(t)))
                    splits += 1
                    continue
            updated.append(n)
        notes = sorted(updated, key=lambda n: (n.start, n.pitch))
    return notes, splits


# --- missing low roots ----------------------------------------------------------


_THIRD_OCTAVES = (25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000)


def low_end_floor(y: np.ndarray, sr: int, *, within_db: float = 12.0) -> float:
    """The lowest frequency (Hz) the recording carries real energy at.

    Measured in third-octave bands: the lower edge of the lowest band that comes within
    `within_db` of the loudest band between 100 Hz and 1 kHz. On a real isolated drop-C
    track this read ~89 Hz (the 63 Hz band sat 23 dB down), even though the guitar's
    lowest string is 65 Hz - which is exactly why its roots went unreported.
    """
    import librosa

    n_fft = 8192
    power = (np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 2)) ** 2).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    edge = 2 ** (1 / 6)
    levels = {}
    for center in _THIRD_OCTAVES:
        band = (freqs >= center / edge) & (freqs < center * edge)
        if band.any():
            levels[center] = 10 * np.log10(power[band].sum() + 1e-20)
    loudest = max(level for center, level in levels.items() if 100 <= center <= 1000)
    for center, level in levels.items():
        if level >= loudest - within_db:
            return float(center / edge)
    return 1000.0


class _Spectra:
    """Short-time magnitude spectra of a recording, for checking overtones at specific times."""

    def __init__(self, y: np.ndarray, sr: int):
        self.y, self.sr = y, sr

    def peak_levels(self, t0: float, t1: float, freqs_hz):
        seg = self.y[int(t0 * self.sr) : int(t1 * self.sr)]
        if len(seg) < int(0.04 * self.sr):
            return None
        n = 1 << int(np.ceil(np.log2(max(len(seg), 8192))))
        mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n))
        bins = np.fft.rfftfreq(n, 1 / self.sr)
        levels = []
        for f in freqs_hz:
            band = (bins >= f * 0.975) & (bins <= f * 1.025)
            levels.append(float(mag[band].max()) if band.any() else 0.0)
        return np.asarray(levels)


def overtone_evidence(spectra: _Spectra, t0: float, t1: float, candidate: int, present: list[int]) -> float:
    """How strongly `candidate`'s own overtones show up, relative to the notes already heard.

    Only overtones that none of the `present` notes could have produced count as
    evidence - e.g. for a C2 under a heard G2, the 5th harmonic (E4) is C2's alone.
    Valid for single notes; for dyads, distortion's intermodulation products produce the
    missing root's overtone series on their own, so this can't tell those apart.
    """
    f0 = _hz(candidate)
    heard = [_hz(p) * m for p in present for m in range(1, 24)]
    own = [k * f0 for k in range(2, 11) if k * f0 < 2500 and all(abs(k * f0 - h) / (k * f0) > 0.03 for h in heard)]
    reference = [_hz(p) * m for p in present for m in range(1, 7) if _hz(p) * m < 2500]
    if len(own) < 2 or not reference:
        return 0.0
    own_levels = spectra.peak_levels(t0, t1, own)
    ref_levels = spectra.peak_levels(t0, t1, reference)
    if own_levels is None:
        return 0.0
    return float(own_levels.mean() / (ref_levels.mean() + _EPS))


def restore_missing_roots(
    notes: list[TimedNote], y: np.ndarray, sr: int, *, lowest: int, floor_hz: float, evidence: float = 0.35,
):
    """Add power-chord roots that sit below the recording's low-end floor. Returns (notes, added)."""
    spectra = _Spectra(y, sr)
    groups: dict[float, list[TimedNote]] = {}
    for n in notes:
        groups.setdefault(n.start, []).append(n)

    added: list[TimedNote] = []
    for start, group in groups.items():
        pitches = sorted({n.pitch for n in group})
        low = pitches[0]
        root = None
        if low + 5 in pitches:
            # A bare fourth as the two lowest notes is almost always the fifth and octave of a
            # power chord whose root got lost - but only trust that if the root really is inaudible.
            if low - 7 >= lowest and _hz(low - 7) < floor_hz:
                root = low - 7
        elif len(pitches) == 1:
            end = min(n.end for n in group)
            t0, t1 = start + 0.015, min(end, start + 0.25)
            candidates = [c for c in (low - 7, low - 12) if c >= lowest and _hz(c) < floor_hz]
            scored = [(overtone_evidence(spectra, t0, t1, c, pitches), c) for c in candidates]
            scored = [s for s in scored if s[0] >= evidence]
            if scored:
                root = max(scored)[1]
        if root is not None:
            ref = max(group, key=lambda n: n.velocity)
            added.append(TimedNote(start, min(n.end for n in group), root, ref.velocity))
    return sorted(notes + added, key=lambda n: (n.start, n.pitch)), len(added)


# --- all together ----------------------------------------------------------------------


def refine(notes: list[TimedNote], y: np.ndarray, sr: int, *, lowest: int) -> tuple[list[TimedNote], RefineReport]:
    report = RefineReport()
    if not notes:
        return notes, report
    onsets = detect_onsets(y, sr)
    report.onsets = len(onsets)
    notes, report.snapped = snap_to_onsets(notes, onsets)
    notes, report.splits = split_restrikes(notes, onsets)
    report.low_end_hz = low_end_floor(y, sr)
    notes, report.roots_added = restore_missing_roots(notes, y, sr, lowest=lowest, floor_hz=report.low_end_hz)
    return notes, report
