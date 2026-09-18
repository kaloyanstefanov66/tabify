"""Fetching audio to transcribe: a YouTube (or other site) link, or a direct file URL.

[yt-dlp](https://github.com/yt-dlp/yt-dlp) handles the site-specific part; tabify just asks
it for the audio and caches the result, so transcribing the same link twice only downloads
once. Whether you may download a given video is between you and whoever holds the rights -
your own uploads and openly licensed material are the safe cases.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from tabify import TabifyError
from tabify.synth import cache_dir

URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)
# Extensions we can hand straight to the audio pipeline without a conversion step.
DIRECT_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".mid", ".midi", ".m4a"}

FETCH_HELP = (
    'fetching audio from a link needs yt-dlp:\n  pip install "tabify-cli[youtube]"\n'
    "and ffmpeg on PATH to convert what it downloads (https://ffmpeg.org/download.html)."
)


def is_url(text: str) -> bool:
    return bool(URL_PATTERN.match(text.strip()))


def downloads_dir() -> Path:
    path = cache_dir() / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(text: str, limit: int = 60) -> str:
    cleaned = re.sub(r"[^\w\s.-]", "", text).strip().replace(" ", "_")
    return cleaned[:limit] or "audio"


def fetch_direct(url: str, into: Path | None = None) -> tuple[Path, str]:
    """Download a plain file URL (someone's own .wav, .mp3, .mid). Returns (path, title)."""
    name = _safe_name(Path(urllib.parse.urlparse(url).path).name) or "audio"
    target = (into or downloads_dir()) / name
    if not target.exists():
        print(f"Downloading {url} ...", file=sys.stderr)
        try:
            urllib.request.urlretrieve(url, target)
        except OSError as exc:
            raise TabifyError(f"could not download {url}: {exc}") from exc
    return target, Path(name).stem


def fetch_media(url: str, into: Path | None = None, *, attempts: int = 3) -> tuple[Path, str]:
    """Download the audio of a video/streaming link via yt-dlp. Returns (path, title)."""
    if importlib.util.find_spec("yt_dlp") is None:
        raise TabifyError(FETCH_HELP)
    import yt_dlp

    into = into or downloads_dir()
    quiet = {"quiet": True, "no_warnings": True, "noprogress": True}
    try:
        with yt_dlp.YoutubeDL(quiet) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt-dlp raises its own error types for every site problem
        raise TabifyError(f"could not read {url}: {exc}") from exc

    title = info.get("title") or "audio"
    # FLAC rather than WAV: lossless either way, about half the size, and read natively.
    cached = into / f"{info.get('id', _safe_name(title))}.flac"
    if cached.exists():
        print(f"Using previously downloaded audio: {cached.name}", file=sys.stderr)
        return cached, title

    if not shutil.which("ffmpeg"):
        raise TabifyError(
            "ffmpeg isn't on PATH, and it's needed to turn what yt-dlp downloads into audio "
            "tabify can read.\nInstall it from https://ffmpeg.org/download.html "
            "(Windows: winget install Gyan.FFmpeg)."
        )

    print(f"Downloading audio: {title}", file=sys.stderr)
    options = {
        **quiet,
        "format": "bestaudio/best",
        "outtmpl": str(cached.with_suffix("")) + ".%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "flac"}],
    }
    # YouTube hands out intermittent 403s even for a link that worked a minute earlier, so a
    # single failure isn't worth giving up on - observed and retried successfully in testing.
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                print(f"Download failed ({str(exc)[-80:].strip()}), retrying ...", file=sys.stderr)
                time.sleep(2)
    if not cached.exists():
        raise TabifyError(
            f"could not download {url}: {last_error}\n"
            "Sites change often - `pip install -U yt-dlp` fixes most of these, and rate limits pass."
        )
    return cached, title


def fetch(url: str, into: Path | None = None) -> tuple[Path, str]:
    """Get a local audio/MIDI file for `url`, downloading it once and caching it."""
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix in DIRECT_EXTENSIONS:
        return fetch_direct(url, into)
    return fetch_media(url, into)
