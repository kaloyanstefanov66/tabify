import pytest

from tabify import TabifyError
from tabify.separate import SELECTABLE, SEPARATE_HELP, parse_stems, separate_stems, separation_available


def _installed(*names):
    """Pretend exactly these modules are importable."""
    return lambda name: object() if name in names else None


def test_missing_separation_backend_gives_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr("tabify.separate.importlib.util.find_spec", _installed())
    with pytest.raises(TabifyError, match="separation"):
        separate_stems("song.wav", tmp_path)
    assert "pip install" in SEPARATE_HELP


def test_either_backend_counts_as_available(monkeypatch):
    """The ONNX build and the PyTorch build both separate; the CLI shouldn't care which.

    It did care, once: it asked whether `demucs` specifically was importable, so with only
    the ONNX backend installed it told people to install what they already had.
    """
    monkeypatch.setattr("tabify.separate.importlib.util.find_spec", _installed("demucs_onnx"))
    assert separation_available()
    monkeypatch.setattr("tabify.separate.importlib.util.find_spec", _installed("demucs"))
    assert separation_available()
    monkeypatch.setattr("tabify.separate.importlib.util.find_spec", _installed())
    assert not separation_available()


def test_the_onnx_backend_is_preferred_over_pytorch(tmp_path, monkeypatch):
    # Same model either way, but the ONNX build doesn't drag PyTorch in behind it.
    monkeypatch.setattr("tabify.separate.importlib.util.find_spec", _installed("demucs_onnx", "demucs"))
    used = []
    monkeypatch.setattr("tabify.separate._separate_onnx", lambda path, out: used.append("onnx") or {})
    monkeypatch.setattr("tabify.separate._separate_torch", lambda path, out: used.append("torch") or {})
    separate_stems("song.wav", tmp_path)
    assert used == ["onnx"]


def test_stem_choice_accepts_a_list_or_all():
    assert parse_stems("guitar") == ["guitar"]
    assert parse_stems(" Guitar , Bass ") == ["guitar", "bass"]
    assert parse_stems("all") == list(SELECTABLE)


def test_drums_are_not_on_offer():
    # There's nothing to fret on a drum kit, so it isn't one of the choices.
    assert "drums" not in SELECTABLE
    with pytest.raises(TabifyError, match="nothing to fret"):
        parse_stems("drums")


def test_an_unknown_part_lists_the_real_ones():
    with pytest.raises(TabifyError, match="guitar"):
        parse_stems("banjo")


def test_loudness_tells_an_empty_stem_from_a_played_one(tmp_path):
    import numpy as np
    import soundfile as sf

    from tabify.separate import SILENCE_RMS, loudness

    quiet, played = tmp_path / "piano.wav", tmp_path / "guitar.wav"
    sf.write(str(quiet), np.full(22050, 1e-4, dtype="float32"), 22050)
    sf.write(str(played), np.sin(np.linspace(0, 400, 22050)).astype("float32"), 22050)
    assert loudness(quiet) < SILENCE_RMS <= loudness(played)
