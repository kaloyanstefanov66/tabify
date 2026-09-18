"""Is this one instrument, or a whole band?

Handed a full mix, a transcriber will faithfully tab *everything it hears* into one guitar
part - the bass on the low string, keys, vocals - and the result looks like nonsense rather
than like a failure, which is worse. Reported from a real run on a band track.

The giveaway is energy below the instrument's lowest string. A guitar can't produce it; a
kick drum and a bass guitar produce plenty. Measured across real files, isolated tracks sit
at 0.000-0.001 of their energy down there, while mixes and backing tracks sit at 0.11-0.29 -
two orders of magnitude apart, which is a comfortable place to put a threshold.

It's relative to the instrument you asked for, so a bass stem reads as clean when you tell
tabify it's a bass, and as "something else is in here" when you claim it's a guitar.
"""

from __future__ import annotations

import numpy as np

# An isolated track measured 0.000-0.001; mixes measured 0.109 and up. Anywhere in between
# works; this sits close to the clean end so a bass-heavy solo guitar isn't flagged.
FULL_MIX_THRESHOLD = 0.03
EXCERPT_SECONDS = 40.0


def _hz(pitch: float) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def low_energy_share(y: np.ndarray, sr: int, below_hz: float) -> float:
    """Fraction of the recording's energy that sits below `below_hz`."""
    import librosa

    if len(y) < sr:
        return 0.0
    spectrum = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    return float(spectrum[freqs < below_hz].sum() / (spectrum.sum() + 1e-12))


def full_mix_share(y: np.ndarray, sr: int, lowest_pitch: int) -> float:
    """How much of this recording couldn't have come from the instrument you asked for.

    Measured a little under the lowest string, so a slightly flat guitar isn't mistaken for
    a bass player.
    """
    return low_energy_share(y, sr, _hz(lowest_pitch) * 0.85)


def looks_like_full_mix(y: np.ndarray, sr: int, lowest_pitch: int, *, threshold: float = FULL_MIX_THRESHOLD):
    """Returns (is a mix, the measured share)."""
    share = full_mix_share(y, sr, lowest_pitch)
    return share > threshold, share
