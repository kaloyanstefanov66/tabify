import numpy as np
import pytest

pytest.importorskip("librosa")

from tabify.features import BEFORE, FRAMES, N_BINS, SAMPLE_RATE, cqt, patch_at, patches_at  # noqa: E402


def tone(pitch_hz: float, seconds: float = 1.0, at: float = 0.0, total: float = 1.5) -> np.ndarray:
    """A plucked note starting at `at` seconds inside a `total`-second recording."""
    y = np.zeros(int(total * SAMPLE_RATE), dtype=np.float32)
    seconds = min(seconds, total - at)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    note = sum((1 / k) * np.sin(2 * np.pi * pitch_hz * k * t) for k in range(1, 8)) * np.exp(-3 * t)
    start = int(at * SAMPLE_RATE)
    y[start : start + len(note)] += note.astype(np.float32)
    return y


def test_patch_has_the_shape_the_model_expects():
    patches = patches_at(tone(110.0, at=0.5), [0.5])
    assert patches.shape == (1, N_BINS, FRAMES)


def test_patches_are_normalized_so_loudness_does_not_matter():
    quiet = patches_at(tone(110.0, at=0.5) * 0.01, [0.5])
    loud = patches_at(tone(110.0, at=0.5) * 1.0, [0.5])
    assert np.allclose(quiet, loud, atol=0.02)
    assert quiet.min() >= 0.0 and quiet.max() <= 1.0


def test_the_patch_starts_just_before_the_attack():
    # The pick has to be inside the window, so the patch begins slightly before the stroke.
    y = tone(110.0, at=0.5)
    spectrogram = cqt(y)
    on_time = patch_at(spectrogram, 0.5)
    silence_before = patch_at(spectrogram, 0.1)
    assert on_time.mean() > silence_before.mean()


def test_a_stroke_at_the_very_start_or_end_is_padded_not_dropped():
    y = tone(110.0, at=0.0, total=0.6)
    assert patches_at(y, [0.0]).shape == (1, N_BINS, FRAMES)  # nothing before it to read
    assert patches_at(y, [0.59]).shape == (1, N_BINS, FRAMES)  # runs off the end
    assert np.isfinite(patches_at(y, [0.0, 0.59])).all()


def test_different_pitches_look_different():
    low = patches_at(tone(82.41, at=0.5), [0.5])[0]  # E2
    high = patches_at(tone(164.81, at=0.5), [0.5])[0]  # E3, an octave up
    assert np.abs(low - high).mean() > 0.02


def test_no_strokes_gives_an_empty_batch_rather_than_an_error():
    assert patches_at(tone(110.0), []).shape == (0, N_BINS, FRAMES)


def test_lead_in_is_inside_the_window():
    assert 0 < BEFORE < 0.1
