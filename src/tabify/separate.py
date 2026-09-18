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

import numpy as np

from tabify import TabifyError

MODEL = "htdemucs_6s"
# Stems that can be tabbed at all, in the order most people want them. Drums are left out:
# there's nothing to fret. "other" is where second guitars and keys usually end up.
SELECTABLE = ("guitar", "bass", "other", "piano", "vocals")
DEFAULT_STEMS = ("guitar",)
# Stems the model found nothing for come back near-silent (measured around 0.0002 on a real
# mix with no piano in it), and tabbing silence just wastes a minute.
SILENCE_RMS = 0.002

SEPARATE_HELP = (
    'full-mix separation needs a separation model:\n  pip install "tabify-cli[separate]"\n'
    f"The first run also downloads the {MODEL} model (roughly 100 MB)."
)


def _separate_onnx(path: Path, out_dir: Path) -> dict[str, Path]:
    """Separate with the ONNX build: same model, no PyTorch, and about as fast on CPU."""
    import demucs_onnx
    import soundfile as sf

    try:
        stems = demucs_onnx.separate(path, model=MODEL, progress=False)
    except Exception as exc:  # model download and onnxruntime both have their own failure modes
        raise TabifyError(f"stem separation failed: {exc}") from exc

    paths = {}
    for name, audio in stems.items():
        out_path = out_dir / f"{name}.wav"
        sf.write(str(out_path), np.asarray(audio, dtype=np.float32).T, 44100)
        paths[name] = out_path
    return paths


def _separate_torch(path: Path, out_dir: Path) -> dict[str, Path]:
    from demucs.api import Separator, save_audio

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


def parse_stems(spec: str) -> list[str]:
    """Which stems to tab, from a comma-separated list or "all"."""
    if spec.strip().lower() == "all":
        return list(SELECTABLE)
    wanted = [name.strip().lower() for name in spec.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in SELECTABLE]
    if unknown:
        raise TabifyError(
            f"can't tab {', '.join(unknown)} - choose from {', '.join(SELECTABLE)}, or 'all'"
            + (" (drums have nothing to fret)" if "drums" in unknown else "")
        )
    return wanted or list(DEFAULT_STEMS)


def loudness(path: Path) -> float:
    """RMS level of a stem, for telling an empty one from a real part."""
    import soundfile as sf

    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return float(np.sqrt((audio**2).mean())) if len(audio) else 0.0


def separation_available() -> bool:
    """Whether either separation backend is installed - ONNX preferred, PyTorch accepted."""
    return any(importlib.util.find_spec(name) is not None for name in ("demucs_onnx", "demucs"))


def separate_stems(path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Split `path` into instrument stems, written as wav files under `out_dir`.

    Prefers the ONNX build of the same model, which needs only onnxruntime - tabify already
    ships that for the chord engine - instead of PyTorch's several hundred megabytes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("demucs_onnx") is not None:
        return _separate_onnx(Path(path), out_dir)
    if importlib.util.find_spec("demucs") is not None:
        return _separate_torch(Path(path), out_dir)
    raise TabifyError(SEPARATE_HELP)
