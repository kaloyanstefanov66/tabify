"""Telling one instrument from a whole band."""

import numpy as np
import pytest

pytest.importorskip("librosa")

from tabify.mixcheck import FULL_MIX_THRESHOLD, _hz, full_mix_share, looks_like_full_mix  # noqa: E402

SR = 22050
LOW_E, LOW_E_BASS = 40, 28  # guitar's lowest string, and a bass guitar's


def instrument(pitches, seconds=6.0, amplitude=1.0):
    t = np.arange(int(seconds * SR)) / SR
    signal = sum(
        (1 / k) * np.sin(2 * np.pi * _hz(p) * k * t)
        for p in pitches
        for k in range(1, 12)
        if _hz(p) * k < SR / 2
    )
    return (amplitude * signal / (np.abs(signal).max() + 1e-9)).astype(np.float32)


def kick_drum(seconds=6.0, every=0.5):
    """Thumps at 55 Hz, well below any guitar."""
    y = np.zeros(int(seconds * SR), dtype=np.float32)
    hit = np.sin(2 * np.pi * 55 * np.arange(int(0.12 * SR)) / SR) * np.exp(-np.arange(int(0.12 * SR)) / (0.03 * SR))
    for start in range(0, len(y) - len(hit), int(every * SR)):
        y[start : start + len(hit)] += hit.astype(np.float32)
    return y


def test_a_guitar_on_its_own_is_not_a_mix():
    guitar = instrument([40, 47, 52])  # an E5 power chord
    is_mix, share = looks_like_full_mix(guitar, SR, LOW_E)
    assert not is_mix
    assert share < FULL_MIX_THRESHOLD


def test_a_guitar_with_a_bass_and_kick_under_it_is_a_mix():
    band = instrument([40, 47, 52]) * 0.6 + instrument([28, 35], amplitude=0.8) + kick_drum()
    is_mix, share = looks_like_full_mix(band, SR, LOW_E)
    assert is_mix
    assert share > FULL_MIX_THRESHOLD


def test_the_judgement_is_relative_to_the_instrument_you_asked_for():
    """A bass stem is clean as a bass, and 'something else is in here' as a guitar.

    Measured on real files: an isolated bass read 0.001 against a bass tuning and 0.124
    against a guitar's.
    """
    bass = instrument([28, 33, 35, 38])
    assert not looks_like_full_mix(bass, SR, LOW_E_BASS)[0]
    assert looks_like_full_mix(bass, SR, LOW_E)[0]


def test_a_recording_with_its_low_end_rolled_off_reads_as_clean():
    # The isolated stem that prompted all this has nothing below ~89 Hz at all.
    guitar = instrument([40, 47, 52])
    spectrum = np.fft.rfft(guitar)
    spectrum[np.fft.rfftfreq(len(guitar), 1 / SR) < 90] = 0
    cut = np.fft.irfft(spectrum, len(guitar)).astype(np.float32)
    assert full_mix_share(cut, SR, LOW_E) < 0.005


def test_silence_does_not_crash_or_accuse():
    assert not looks_like_full_mix(np.zeros(SR * 2, dtype=np.float32), SR, LOW_E)[0]
