"""Render a transcription to real audio with an actual guitar tone.

A MIDI file only records which notes to play and which General MIDI program
number to play them with; the acoustic/distorted *sound* only exists once
something synthesizes that into audio. This uses FluidSynth (the standard
open-source softsynth) driven by a General MIDI soundfont to do that.

FluidSynth's native library isn't bundled - see `RENDER_HELP` for how to get
it. The soundfont (a set of sampled instrument sounds) is fetched once into a
local cache the first time it's needed, unless one is supplied explicitly.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import urllib.request
from pathlib import Path

from tabify import TabifyError
from tabify.fretting import TabEvent
from tabify.instruments import ALL_PROGRAMS
from tabify.tuning import Tuning

DEFAULT_SOUNDFONT_URL = "https://raw.githubusercontent.com/arbruijn/TimGM6mb/master/TimGM6mb.sf2"
DEFAULT_SOUNDFONT_NAME = "TimGM6mb.sf2"  # ~6 MB, GPLv2, Tim Brechbill - small but complete GM soundfont
DEFAULT_SOUNDFONT_SIZE = "~6 MB"

RENDER_HELP = (
    "audio rendering needs the FluidSynth library, plus its Python bindings.\n"
    "  1. Install FluidSynth itself:\n"
    "       Windows: download the release zip from "
    "https://github.com/FluidSynth/fluidsynth/releases and add its bin\\ folder to PATH\n"
    "       macOS:   brew install fluid-synth\n"
    "       Linux:   apt install fluidsynth  (or your distro's equivalent)\n"
    "  2. pip install \"tabify-cli[render]\""
)


def cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    d = Path(base) / "tabify"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_soundfont(path: str | None) -> Path:
    if path:
        p = Path(path)
        if not p.is_file():
            raise TabifyError(f"soundfont not found: {p}")
        return p

    cached = cache_dir() / DEFAULT_SOUNDFONT_NAME
    if not cached.is_file():
        print(
            f"No soundfont given - downloading a small General MIDI soundfont "
            f"({DEFAULT_SOUNDFONT_NAME}, {DEFAULT_SOUNDFONT_SIZE}, GPLv2, from {DEFAULT_SOUNDFONT_URL}) "
            f"to {cached}\nUse --soundfont to supply a bigger one for a better tone.",
            file=sys.stderr,
        )
        tmp = cached.with_suffix(".part")
        try:
            urllib.request.urlretrieve(DEFAULT_SOUNDFONT_URL, tmp)
        except OSError as exc:
            raise TabifyError(
                f"could not download the default soundfont: {exc}\nUse --soundfont to supply your own."
            ) from exc
        tmp.rename(cached)
    return cached


# Per-instrument default drive (0 = clean passthrough), used when --drive isn't given explicitly.
# Distortion is real signal processing on top of the soundfont's own patch, not just picking a
# different GM program number - a small soundfont's "Distortion Guitar" sample alone tends to
# sound thin, so this adds the actual clipping/harmonics a distortion pedal or overdriven amp adds.
DEFAULT_DRIVE = {"distortion": 0.75, "overdrive": 0.4, "muted": 0.15}


def _lowpass(audio, cutoff_hz: float, sample_rate: int):
    """A one-pole-shaped low-pass, applied in the frequency domain (fast, no scipy needed).

    Stands in for a guitar speaker cabinet, which rolls off the harsh high harmonics that
    clipping adds - without it, distortion sounds like digital fuzz rather than an amp.
    """
    import numpy as np

    n = audio.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    response = 1.0 / np.sqrt(1.0 + (freqs / cutoff_hz) ** 2)  # |H(f)| of a single-pole low-pass
    spectrum = np.fft.rfft(audio, axis=0)
    return np.fft.irfft(spectrum * response[:, None], n=n, axis=0).astype(np.float32)


def _distort(audio, sample_rate: int, drive: float, tone: float = 0.5):
    """Soft-clip waveshaping distortion, the same basic technique real distortion pedals use,
    followed by a cabinet-style low-pass so the added harmonics sound like an amp, not fuzz.

    `drive` in [0, 1]: how hard the signal clips (0 leaves audio untouched). `tone` in [0, 1]:
    how dark the post-clip low-pass is (0 = brighter/~7kHz, 1 = darker/~2.5kHz).
    """
    import numpy as np

    if drive <= 0:
        return audio
    gain = 1.0 + drive * 14.0
    shaped = np.tanh(audio * gain)
    shaped = _lowpass(shaped, 7000.0 - tone * 4500.0, sample_rate)
    # Renormalize toward the original peak so different drive amounts stay comparably loud,
    # with a little headroom so the waveshaper's own peaks don't clip on export.
    peak = float(np.abs(audio).max()) or 1e-9
    shaped_peak = float(np.abs(shaped).max()) or 1e-9
    return (shaped * (peak / shaped_peak) * 0.92).astype(np.float32)


def schedule(events: list[TabEvent], tuning: Tuning, capo: int, bpm: float) -> list[tuple[float, bool, int]]:
    """Turn tab events into (seconds, is_note_on, pitch) triples, ready to feed to a synth.

    At an exact tie, note-offs are ordered before note-ons so a repeated note retriggers cleanly.
    """
    seconds_per_beat = 60.0 / bpm
    raw: list[tuple[float, bool, int]] = []
    for e in events:
        duration = max(n.duration for n in e.notes)
        for pos in e.positions:
            # pos.fret is relative to the capo (tab convention); the sounding pitch includes it.
            pitch = tuning.strings[pos.string] + capo + pos.fret
            raw.append((e.start * seconds_per_beat, True, pitch))
            raw.append(((e.start + duration) * seconds_per_beat, False, pitch))
    raw.sort(key=lambda ev: (ev[0], ev[1]))  # False (note-off) sorts before True (note-on)
    return raw


def synthesize(
    events: list[TabEvent],
    tuning: Tuning,
    *,
    instrument: str = "steel",
    capo: int = 0,
    bpm: float = 120.0,
    soundfont: str | None = None,
    sample_rate: int = 44100,
    tail_seconds: float = 1.5,
    drive: float | None = None,
    tone: float = 0.5,
):
    """Render tab events to an in-memory (frames, 2) float32 array at `sample_rate`.

    Shared by `render_audio` (writes it to a file) and the play-along player
    (streams it straight to the speakers), so both hear exactly the same thing.
    """
    if instrument not in ALL_PROGRAMS:
        raise TabifyError(f"unknown instrument {instrument!r} (choose from: {', '.join(ALL_PROGRAMS)})")
    if importlib.util.find_spec("fluidsynth") is None:
        raise TabifyError(RENDER_HELP)
    import fluidsynth  # imported lazily: needs a native library that may not be installed
    import numpy as np

    sf2 = resolve_soundfont(soundfont)
    synth = fluidsynth.Synth(samplerate=float(sample_rate))
    try:
        sfid = synth.sfload(str(sf2))
        if sfid < 0:
            raise TabifyError(f"could not load soundfont {sf2} (it may be corrupt - delete it and try again)")
        synth.program_select(0, sfid, 0, ALL_PROGRAMS[instrument])

        chunks = []
        cursor = 0.0
        for t, is_on, pitch in schedule(events, tuning, capo, bpm):
            if t > cursor:
                chunks.append(synth.get_samples(round((t - cursor) * sample_rate)))
                cursor = t
            if 0 <= pitch <= 127:  # extreme capo/tuning combinations can push a pitch out of MIDI range
                if is_on:
                    synth.noteon(0, pitch, 100)
                else:
                    synth.noteoff(0, pitch)
        chunks.append(synth.get_samples(round(tail_seconds * sample_rate)))  # let the last notes ring out
    finally:
        synth.delete()

    audio = np.concatenate(chunks).astype(np.float32).reshape(-1, 2) / 32768.0
    effective_drive = DEFAULT_DRIVE.get(instrument, 0.0) if drive is None else drive
    return _distort(audio, sample_rate, effective_drive, tone)


def render_audio(
    events: list[TabEvent],
    tuning: Tuning,
    out_path: str | Path,
    *,
    instrument: str = "steel",
    capo: int = 0,
    bpm: float = 120.0,
    soundfont: str | None = None,
    sample_rate: int = 44100,
    tail_seconds: float = 1.5,
    drive: float | None = None,
    tone: float = 0.5,
) -> None:
    import soundfile as sf

    audio = synthesize(
        events, tuning, instrument=instrument, capo=capo, bpm=bpm, soundfont=soundfont,
        sample_rate=sample_rate, tail_seconds=tail_seconds, drive=drive, tone=tone,
    )
    sf.write(str(out_path), audio, sample_rate)
