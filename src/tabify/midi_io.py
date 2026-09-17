"""Reading and writing MIDI files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from tabify import TabifyError
from tabify.notes import Note
from tabify.rhythm import TimeSignature

DRUM_CHANNEL = 9


@dataclass
class MidiContent:
    notes: list[Note]
    bpm: float | None
    time_sig: TimeSignature | None


def read_midi(path: str | Path, track: int | None = None) -> MidiContent:
    try:
        mid = mido.MidiFile(str(path))
    except (OSError, EOFError, ValueError) as exc:
        raise TabifyError(f"could not read MIDI file {path}: {exc}") from exc

    if track is not None and not 0 <= track < len(mid.tracks):
        raise TabifyError(f"track {track} does not exist (file has {len(mid.tracks)} tracks: 0-{len(mid.tracks) - 1})")

    tpb = mid.ticks_per_beat
    bpm: float | None = None
    time_sig: TimeSignature | None = None
    notes: list[Note] = []

    for index, trk in enumerate(mid.tracks):
        tick = 0
        held: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for msg in trk:
            tick += msg.time
            if msg.type == "set_tempo" and bpm is None:
                bpm = mido.tempo2bpm(msg.tempo)
            elif msg.type == "time_signature" and time_sig is None:
                time_sig = TimeSignature(msg.numerator, msg.denominator)
            if track is not None and index != track:
                continue
            if msg.type not in ("note_on", "note_off") or msg.channel == DRUM_CHANNEL:
                continue
            key = (msg.channel, msg.note)
            if msg.type == "note_on" and msg.velocity > 0:
                held.setdefault(key, []).append((tick, msg.velocity))
            elif held.get(key):
                start, velocity = held[key].pop(0)
                notes.append(Note(start / tpb, max(tick - start, 1) / tpb, msg.note, velocity))

    notes.sort(key=lambda n: (n.start, n.pitch))
    return MidiContent(notes, bpm, time_sig)


def write_midi(
    path: str | Path,
    notes: list[Note],
    bpm: float = 120.0,
    time_sig: TimeSignature = TimeSignature(),
    ticks_per_beat: int = 480,
    program: int = 25,  # General MIDI "Acoustic Guitar (steel)", zero-based
) -> None:
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    trk = mido.MidiTrack()
    mid.tracks.append(trk)

    timed: list[tuple[int, int, mido.Message | mido.MetaMessage]] = [
        (0, 0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm))),
        (0, 0, mido.MetaMessage("time_signature", numerator=time_sig.beats, denominator=time_sig.unit)),
        (0, 0, mido.Message("program_change", program=program, channel=0)),
    ]
    for n in notes:
        on = max(0, round(n.start * ticks_per_beat))
        off = max(on + 1, round((n.start + n.duration) * ticks_per_beat))
        velocity = min(127, max(1, n.velocity))
        # note_off sorts before note_on at the same tick so repeated notes retrigger cleanly
        timed.append((on, 2, mido.Message("note_on", note=n.pitch, velocity=velocity, channel=0)))
        timed.append((off, 1, mido.Message("note_off", note=n.pitch, velocity=0, channel=0)))

    timed.sort(key=lambda t: (t[0], t[1]))
    last = 0
    for tick, _, msg in timed:
        trk.append(msg.copy(time=tick - last))
        last = tick
    mid.save(str(path))
