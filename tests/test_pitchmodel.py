"""Running the pitch model without the package that used to wrap it.

The package is no longer a dependency, so these mostly test the reimplementation on its own
terms. When basic-pitch does happen to be installed, one test pins the two against each
other note for note - that equivalence is the whole justification for dropping it.
"""

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("librosa")
pytest.importorskip("scipy")

from tabify import pitchmodel as P  # noqa: E402

SR = P.SAMPLE_RATE


def tone(midi, seconds=1.5, sr=SR):
    hz = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(seconds * sr)) / sr
    wave = sum((1 / k) * np.sin(2 * np.pi * hz * k * t) for k in range(1, 8))
    envelope = np.minimum(1.0, np.linspace(40, 0, len(t)).clip(0, 1) + 0.2)
    return (wave * envelope / np.abs(wave).max()).astype(np.float32)


def test_the_model_ships_with_tabify():
    """No download, no separate package - a 0.2 MB file inside the wheel."""
    assert P.MODEL.exists()
    assert P.MODEL.stat().st_size < 1_000_000
    licence = P.MODEL.parent / "basic-pitch-LICENSE"
    assert licence.exists() and "Apache" in licence.read_text(encoding="utf-8", errors="replace")


def test_windows_cover_the_audio_with_the_expected_overlap():
    audio = np.zeros(P.SAMPLE_RATE * 5, dtype=np.float32)
    windows = P._windows(audio)
    assert windows.shape[1:] == (P.WINDOW_SAMPLES, 1)
    # Front-padded by half the overlap, then advanced by the hop until the audio runs out.
    expected = len(range(0, len(audio) + P.OVERLAP_SAMPLES // 2, P.HOP_SAMPLES))
    assert windows.shape[0] == expected


def test_a_short_clip_is_padded_rather_than_dropped():
    windows = P._windows(np.zeros(1000, dtype=np.float32))
    assert windows.shape[0] == 1 and windows.shape[1] == P.WINDOW_SAMPLES


def test_it_hears_a_plain_note():
    notes = _notes(tone(45))  # A2
    assert notes, "a clean sustained note should be heard"
    assert min(abs(p - 45) for _, _, p, _ in notes) == 0


def test_a_chord_comes_back_as_several_notes():
    chord = tone(40) + tone(47) + tone(52)
    heard = {p for _, _, p, _ in _notes(chord / np.abs(chord).max())}
    assert {40, 47, 52} <= heard


def test_the_frequency_range_is_respected():
    chord = tone(40) + tone(76)
    chord = (chord / np.abs(chord).max()).astype(np.float32)
    heard = {p for _, _, p, _ in _notes(chord, lowest_hz=200.0)}
    assert 40 not in heard  # E2 is 82 Hz, well below the floor asked for


def _notes(audio, lowest_hz=None, highest_hz=None):
    out = P.predict(audio)
    return P.notes_from_output(
        out["note"], out["onset"], onset_threshold=0.5, frame_threshold=0.3,
        min_note_frames=7, lowest_hz=lowest_hz, highest_hz=highest_hz,
    )


def test_frame_times_increase_and_start_near_zero():
    times = P.frame_times(500)
    assert times[0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(times) > -1e-9)  # the per-window correction must not go backwards


def test_unknown_model_outputs_fail_loudly():
    """Naming the outputs by position made them silently swap once; a shape change must not."""
    with pytest.raises(RuntimeError, match="unexpected pitch model outputs"):
        P._label_outputs({"a": np.zeros((1, 4, 5)), "b": np.zeros((1, 4, 6))})


def test_matches_the_reference_implementation_note_for_note():
    """The reason the basic-pitch package could be dropped: the notes are identical.

    Skipped when it isn't installed, which is the normal case now.
    """
    from pathlib import Path

    import soundfile as sf

    # Not importorskip: that only skips on ImportError, and this package fails with an
    # AttributeError under NumPy 2 (it still calls np.complex_). Either way there is simply
    # nothing to compare against on this machine, which is the normal case now.
    try:
        import basic_pitch
        from basic_pitch.inference import predict as reference
    except Exception as exc:
        pytest.skip(f"reference implementation cannot run here: {type(exc).__name__}")

    chord = tone(40) + tone(47) + tone(52)
    # A held chord, then a lone re-struck root, so both the onset-driven and the leftover
    # energy paths through the note builder are exercised.
    played = np.concatenate([chord, tone(40, seconds=0.6)])
    audio = (played / np.abs(played).max()).astype(np.float32)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.wav"
        # Float, not the default 16-bit PCM: otherwise the reference reads a quantised copy
        # of what this side is given, and the two legitimately disagree about quiet partials.
        sf.write(str(path), audio, SR, subtype="FLOAT")
        model = Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"
        try:
            _, _, theirs = reference(str(path), model_or_model_path=model, onset_threshold=0.5)
        except Exception as exc:
            # The reference drags TensorFlow in, which does not import under NumPy 2 - one of
            # the reasons for no longer depending on it. Nothing to compare against here.
            pytest.skip(f"reference implementation cannot run here: {type(exc).__name__}")
        out = P.predict(audio)
        mine = P.notes_from_output(
            out["note"], out["onset"], onset_threshold=0.5, frame_threshold=0.3,
            min_note_frames=int(round(127.7 / 1000 * P.FRAMES_PER_SECOND)),
        )
        times = P.frame_times(out["note"].shape[0])
        n = out["note"].shape[0]
        ours = sorted(
            (round(float(times[min(a, n - 1)]), 6), round(float(times[min(b, n - 1)]), 6), int(p))
            for a, b, p, _ in mine
        )
        reference_notes = sorted((round(s, 6), round(e, 6), int(p)) for s, e, p, *_ in theirs)
    assert ours == reference_notes
