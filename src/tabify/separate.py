"""Split a full mix into instrument stems before transcribing each one.

Uses Meta's Demucs (https://github.com/facebookresearch/demucs) source
separation model. The 6-source model used here is the one that pulls guitar
and piano out as their own stems, rather than lumping everything non-vocal,
non-bass, non-drum into one "other" bucket.

Two guitars playing at the same time can't be reliably split into two
separate tracks by this or any current source-separation model - it only
separates *across* instrument types, not between two of the same instrument.
A mix with two interleaved guitar parts will still come out as one, more
complex, guitar stem.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tabify import TabifyError

MODEL = "htdemucs_6s"
# Stems worth running guitar/bass transcription on; drums and vocals are skipped.
TRANSCRIBABLE = ("guitar", "bass")

SEPARATE_HELP = (
    "full-mix separation needs Demucs, which pulls in PyTorch (a few hundred MB):\n"
    '  pip install "tabify-cli[separate]"\n'
    f"The first run also downloads the {MODEL} model (roughly 100 MB) from Hugging Face."
)


def separate_stems(path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Split `path` into instrument stems, written as wav files under `out_dir`."""
    if importlib.util.find_spec("demucs") is None:
        raise TabifyError(SEPARATE_HELP)
    from demucs.api import Separator, save_audio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        separator = Separator(model=MODEL)
        _, stems = separator.separate_audio_file(str(path))
    except Exception as exc:  # demucs/torch raise a variety of backend-specific errors
        raise TabifyError(f"stem separation failed: {exc}") from exc

    paths = {}
    for name, tensor in stems.items():
        out_path = out_dir / f"{name}.wav"
        save_audio(tensor, str(out_path), samplerate=separator.samplerate)
        paths[name] = out_path
    return paths
