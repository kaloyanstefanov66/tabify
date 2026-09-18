"""Turning a stroke of guitar audio into something a model can read.

One shared definition, used both to build training data and to run the trained model, so
the two can't drift apart - a model fed features computed even slightly differently at
inference is a model quietly performing worse than its test score says.

A stroke becomes a constant-Q patch: a short window starting just before the pick lands,
log-scaled, with pitch spaced evenly (a semitone is the same distance everywhere, unlike a
plain spectrogram) so the same chord shape looks the same wherever it's played.
"""

from __future__ import annotations

import numpy as np

HOP = 256
SAMPLE_RATE = 22050
BINS_PER_OCTAVE = 24  # quarter-tone resolution: enough to separate semitones, cheap enough to train on
OCTAVES = 6  # C1 (32.7 Hz) to C7, which covers every guitar string and the overtones that matter
N_BINS = BINS_PER_OCTAVE * OCTAVES
FMIN_HZ = 32.70  # C1
BEFORE = 0.03  # seconds of lead-in, so the attack itself is inside the patch
AFTER = 0.20  # seconds after it: long enough to hear the chord, short enough to stay one stroke
FRAMES = round((BEFORE + AFTER) * SAMPLE_RATE / HOP)


def cqt(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Log-scaled constant-Q spectrogram of a whole recording, shaped (bins, frames)."""
    import librosa

    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    spectrum = np.abs(
        librosa.cqt(y, sr=SAMPLE_RATE, hop_length=HOP, fmin=FMIN_HZ, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
    )
    return librosa.amplitude_to_db(spectrum, ref=np.max).astype(np.float32)


DB_FLOOR = -80.0


def patch_at(spectrogram: np.ndarray, time_s: float) -> np.ndarray:
    """One stroke's patch, shaped (N_BINS, FRAMES), scaled into 0-1.

    Loudness is already handled per recording (`cqt` references the loudest moment), so the
    patch keeps its level rather than being rescaled against its own maximum - doing that
    made a silent patch come out as bright as a struck chord.
    """
    start = int(round((time_s - BEFORE) * SAMPLE_RATE / HOP))
    window = np.full((spectrogram.shape[0], FRAMES), DB_FLOOR, dtype=np.float32)
    source_from, source_to = max(start, 0), min(start + FRAMES, spectrogram.shape[1])
    if source_to > source_from:
        window[:, source_from - start : source_to - start] = spectrogram[:, source_from:source_to]
    return np.clip((window - DB_FLOOR) / -DB_FLOOR, 0.0, 1.0)


def patches_at(y: np.ndarray, times, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Patches for every stroke time in one recording, shaped (strokes, N_BINS, FRAMES)."""
    spectrogram = cqt(y, sr)
    if len(times) == 0:
        return np.zeros((0, N_BINS, FRAMES), dtype=np.float32)
    return np.stack([patch_at(spectrogram, float(t)) for t in times])
