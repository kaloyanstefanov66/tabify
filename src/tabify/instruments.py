"""General MIDI instrument programs used for MIDI and audio export.

A MIDI file only stores *which notes* to play and *which General MIDI
program number* to play them with - the acoustic/distorted guitar sound
itself only appears once something synthesizes that into audio (see
`tabify.synth`). This module is just the shared program-number table both
`midi_io` and `synth` draw from.
"""

from __future__ import annotations

from tabify.tuning import Tuning

# General MIDI program numbers (0-indexed).
GUITAR_PROGRAMS = {
    "nylon": 24, "steel": 25, "jazz": 26, "clean": 27,
    "muted": 28, "overdrive": 29, "distortion": 30, "harmonics": 31,
}
BASS_PROGRAMS = {"acoustic-bass": 32, "finger": 33, "pick": 34, "fretless": 35, "slap": 36}
ALL_PROGRAMS = {**GUITAR_PROGRAMS, **BASS_PROGRAMS}


def default_instrument(tuning: Tuning) -> str:
    """A reasonable default patch, guessed from the tuning: our bass presets get a bass sound."""
    return "finger" if "bass" in tuning.name else "steel"
