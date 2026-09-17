import pytest

from tabify import TabifyError
from tabify.separate import SEPARATE_HELP, separate_stems


def test_missing_demucs_gives_actionable_error(tmp_path, monkeypatch):
    # Simulate demucs not being installed, regardless of whether this
    # particular machine happens to have it.
    monkeypatch.setattr("tabify.separate.importlib.util.find_spec", lambda name: None)
    with pytest.raises(TabifyError, match="Demucs"):
        separate_stems("song.wav", tmp_path)
    assert "pip install" in SEPARATE_HELP
