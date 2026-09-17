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

    return np.concatenate(chunks).astype(np.float32).reshape(-1, 2) / 32768.0


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
) -> None:
    import soundfile as sf

    audio = synthesize(
        events, tuning, instrument=instrument, capo=capo, bpm=bpm,
        soundfont=soundfont, sample_rate=sample_rate, tail_seconds=tail_seconds,
    )
    sf.write(str(out_path), audio, sample_rate)
