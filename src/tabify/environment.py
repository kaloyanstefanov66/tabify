"""What this install can actually do, and which engine each stage will really use.

Editing a dependency list changes nothing about an environment that is already installed,
so it is entirely possible to read the code, see that the light path is preferred, and still
be running the heavy one because nothing was ever reinstalled. Nothing in the normal output
gives that away - a tab looks the same either way, it just took a slow route to get there.

`tabify --doctor` asks the environment instead of the source: every check below imports or
looks for the real thing, so what it prints is what the next run will do.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    """One capability: whether it works, which backend won, and what to do if it doesn't."""

    name: str
    detail: str
    ok: bool
    fix: str = ""
    warn: bool = False  # works, but not by the route it should take

    @property
    def mark(self) -> str:
        return "!" if self.warn else ("ok" if self.ok else "--")


# basic-pitch, which hears chords, has no build for Python 3.12 or newer.
BASIC_PITCH_MAX_PYTHON = (3, 12)
# --force deliberately keeps the interpreter the venv already has, so it cannot fix a venv
# built on the wrong Python. Removing it first is the only thing that does.
REBUILD_ON_311 = chr(10).join((
    "rebuild on Python 3.11, which is the only version that gets every feature:",
    "    pipx uninstall tabify-cli",
    '    pipx install --python 3.11 "tabify-cli[all]"',
))


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # a half-removed package can raise rather than return None
        return False


def _bundled_onnx_model() -> Path | None:
    """basic-pitch's ONNX model file, found without importing the package."""
    spec = importlib.util.find_spec("basic_pitch")
    if spec is None or not spec.origin:
        return None
    model = Path(spec.origin).parent / "saved_models" / "icassp_2022" / "nmp.onnx"
    return model if model.exists() else None


def _pitch_engine() -> Check:
    """basic-pitch runs from a 0.2 MB ONNX model or a 1.2 GB TensorFlow one - same predictions."""
    if not _have("basic_pitch"):
        if not _have("librosa"):
            return Check("chords", "no transcription engine at all", ok=False, fix="pip install tabify-cli")
        if sys.version_info >= BASIC_PITCH_MAX_PYTHON:
            # The usual cause, and it is silent: basic-pitch has no build for a Python this
            # new, so installing simply leaves it out rather than failing. Telling someone to
            # install it again cannot work - the interpreter is what has to change.
            running = f"{sys.version_info[0]}.{sys.version_info[1]}"
            return Check(
                "chords", f"unavailable on Python {running} - single notes only, via librosa pYIN",
                ok=False, fix=REBUILD_ON_311,
            )
        return Check(
            "chords", "not installed - single notes only, via librosa pYIN", ok=False,
            fix="pip install basic-pitch",
        )
    if not _have("onnxruntime"):
        return Check(
            "chords", "basic-pitch via TensorFlow (1.2 GB, ~10 s to load)", ok=True, warn=True,
            fix="pip install onnxruntime   # same model, 0.2 MB, loads instantly",
        )
    # Located from the spec rather than by importing basic_pitch, which drags TensorFlow in
    # with it - a ten second wait to answer a question about a file on disk.
    if not _bundled_onnx_model():
        return Check(
            "chords", "basic-pitch via TensorFlow (this build ships no ONNX model)", ok=True, warn=True,
            fix='pip install --upgrade "basic-pitch>=0.4"',
        )
    return Check("chords", "basic-pitch via ONNX (0.2 MB model)", ok=True)


def _separation() -> Check:
    if _have("demucs_onnx"):
        return Check("full mixes", "demucs via ONNX", ok=True)
    if _have("demucs"):
        return Check(
            "full mixes", "demucs via PyTorch (550 MB)", ok=True, warn=True,
            fix="pip install demucs-onnx   # same model, no PyTorch",
        )
    return Check(
        "full mixes", "not installed - a band recording gets tabbed as one part", ok=False,
        fix='pip install "tabify-cli[separate]"',
    )


def _rendering() -> Check:
    if not _have("fluidsynth"):
        return Check("--audio-out", "not installed", ok=False, fix='pip install "tabify-cli[render]"')
    try:
        import fluidsynth  # noqa: F401
    except Exception:  # the Python binding is there, the native library it wraps is not
        return Check(
            "--audio-out", "pyfluidsynth is installed but libfluidsynth isn't", ok=False,
            fix="install FluidSynth itself - see the README's audio rendering section",
        )
    return Check("--audio-out", "FluidSynth ready", ok=True)


def _youtube() -> Check:
    if not _have("yt_dlp"):
        return Check("youtube links", "not installed", ok=False, fix='pip install "tabify-cli[youtube]"')
    if shutil.which("ffmpeg") is None:
        return Check(
            "youtube links", "yt-dlp is installed but ffmpeg isn't on PATH", ok=False,
            fix="install ffmpeg and make sure it's on PATH",
        )
    return Check("youtube links", "yt-dlp + ffmpeg ready", ok=True)


def checks() -> list[Check]:
    audio = Check("audio files", "librosa ready", ok=True) if _have("librosa") else Check(
        "audio files", "librosa missing - MIDI input only", ok=False, fix='pip install "tabify-cli[audio]"'
    )
    play = Check("--play", "sounddevice ready", ok=True) if _have("sounddevice") else Check(
        "--play", "not installed", ok=False, fix='pip install "tabify-cli[play]"'
    )
    return [audio, _pitch_engine(), _separation(), play, _rendering(), _youtube()]


# Heavy backends tabify can use but no longer needs, and what supersedes each. Having one
# installed is not a problem - tabify won't import it - but seeing it in `pip list` looks
# exactly like tabify still running on it, so the doctor says plainly that it doesn't.
_SUPERSEDED = {
    "tensorflow": ("onnxruntime", "chords", "1.2 GB"),
    "torch": ("demucs_onnx", "full mixes", "550 MB"),
}


def idle_heavyweights() -> list[str]:
    """Big backends that are installed, and unused because the light one won."""
    return [
        f"{name} ({size}) is installed but unused - {stage} run on the ONNX model instead"
        for name, (replacement, stage, size) in _SUPERSEDED.items()
        if _have(name) and _have(replacement)
    ]


def report() -> str:
    """The doctor output: one line per capability, then only the fixes that apply."""
    found = checks()
    width = max(len(c.name) for c in found)
    lines = [f"tabify environment  (python {sys.version.split()[0]}, {sys.executable})", ""]
    lines += [f"  [{c.mark:>2}] {c.name:<{width}}  {c.detail}" for c in found]

    fixes = [c for c in found if c.fix and not c.ok]
    slow = [c for c in found if c.fix and c.warn]
    if slow:
        lines += ["", "Works, but the slow way round:"]
        lines += [f"  {c.name}: {c.fix}" for c in slow]
    if fixes:
        lines += ["", "Missing:"]
        lines += [f"  {c.name}: {c.fix}" for c in fixes]
    idle = idle_heavyweights()
    if idle:
        lines += ["", "Installed but not used by tabify:"]
        lines += [f"  {note}" for note in idle]
        lines += ["  Uninstalling them frees the space; nothing here needs them."]
    if not fixes and not slow:
        lines += ["", "Everything is installed and taking the fast path."]
    else:
        lines += [
            "",
            "Installed with pipx? pipx applies a changed dependency list only on reinstall:",
            '  pipx install --force "tabify-cli[all]"',
            "  (--force keeps the interpreter the venv already has - to change that, uninstall first)",
        ]
    return "\n".join(lines)
