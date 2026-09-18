import pytest

from tabify import TabifyError
from tabify.fetch import FETCH_HELP, fetch, fetch_media, is_url


@pytest.mark.parametrize(
    "text, expected",
    [
        ("https://www.youtube.com/watch?v=abc123", True),
        ("http://example.com/riff.wav", True),
        ("C:\\Users\\me\\riff.wav", False),
        ("riff.mp3", False),
        ("/home/me/riff.mp3", False),
    ],
)
def test_is_url(text, expected):
    assert is_url(text) is expected


def test_direct_file_links_skip_yt_dlp(monkeypatch, tmp_path):
    """A plain .wav link is just a download - no need to involve yt-dlp."""
    calls = []
    monkeypatch.setattr("tabify.fetch.fetch_direct", lambda url, into=None: calls.append(("direct", url)) or (tmp_path, "x"))
    monkeypatch.setattr("tabify.fetch.fetch_media", lambda url, into=None: calls.append(("media", url)) or (tmp_path, "x"))

    fetch("https://example.com/my-riff.wav")
    fetch("https://example.com/my-song.mid")
    fetch("https://www.youtube.com/watch?v=abc123")
    assert [kind for kind, _ in calls] == ["direct", "direct", "media"]


def test_missing_yt_dlp_gives_actionable_error(monkeypatch):
    monkeypatch.setattr("tabify.fetch.importlib.util.find_spec", lambda name: None)
    with pytest.raises(TabifyError, match="yt-dlp"):
        fetch_media("https://www.youtube.com/watch?v=abc123")
    assert "pip install" in FETCH_HELP and "ffmpeg" in FETCH_HELP
