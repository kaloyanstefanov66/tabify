import pytest

from tabify import TabifyError
from tabify.separate import SEPARATE_HELP, separate_stems, separation_available


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
