"""Reading Guitar Pro files, because that's what people already have.

A `.gp` file (Guitar Pro 7 and up) is a zip with an XML score inside it, so the notes can be
read straight out: which string, which fret, in which order. That makes a real tab usable as
ground truth for `--compare` without anyone retyping it as ASCII first.

Only what tabify compares against is read - the ordered strokes, the tuning and the tempo.
Everything else in a score (lyrics, automation, mixing, notation layout) is skipped.

Guitar Pro numbers strings from the lowest pitch up, the same way `TabStrokes` does, so the
string indices carry over unchanged.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from tabify import TabifyError
from tabify.tabfile import TabStrokes
from tabify.tuning import Tuning

SCORE_ENTRY = "Content/score.gpif"
# Instrument names Guitar Pro gives to things with frets. A score usually holds the whole
# band, and tabbing a recording of a guitar against the drum part helps nobody.
_FRETTED = ("guitar", "bass")


def _index(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    found = root.find(tag)
    return {el.get("id"): el for el in found} if found is not None else {}


def _ids(text: str | None) -> list[str]:
    """Guitar Pro stores child lists as space-separated ids, with -1 meaning "nothing here"."""
    return [i for i in (text or "").split() if i != "-1"]


def _property(el: ET.Element, name: str) -> ET.Element | None:
    for prop in el.iter("Property"):
        if prop.get("name") == name:
            return prop
    return None


def _tuning_of(track: ET.Element) -> Tuning | None:
    prop = _property(track, "Tuning")
    pitches = prop.findtext("Pitches") if prop is not None else None
    if not pitches:
        return None
    strings = tuple(int(p) for p in pitches.split())
    return Tuning("custom", strings) if len(strings) >= 4 else None


def _is_fretted(track: ET.Element) -> bool:
    haystack = " ".join(filter(None, (track.findtext("Name"), track.findtext(".//InstrumentSet/Type")))).lower()
    return any(word in haystack for word in _FRETTED)


def _tracks(root: ET.Element) -> list[ET.Element]:
    found = root.find("Tracks")
    return list(found) if found is not None else []


def _pick_track(tracks: list[ET.Element], wanted: str | None) -> int:
    """Which track to read: the one asked for by name, else the first fretted one."""
    if wanted:
        for i, track in enumerate(tracks):
            if wanted.lower() in (track.findtext("Name") or "").lower():
                return i
        names = ", ".join((t.findtext("Name") or "?") for t in tracks)
        raise TabifyError(f"no track matching {wanted!r} - this score has: {names}")
    for i, track in enumerate(tracks):
        if _is_fretted(track):
            return i
    raise TabifyError("no guitar or bass track found in that score")


def track_names(path: str | Path) -> list[str]:
    """Every track in the score, for telling someone what they can choose from."""
    root = _score(path)
    return [(t.findtext("Name") or f"track {i}") for i, t in enumerate(_tracks(root))]


def _score(path: str | Path) -> ET.Element:
    try:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read(SCORE_ENTRY)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise TabifyError(
            f"{Path(path).name} isn't a Guitar Pro 7 file. Older .gp3/.gp4/.gp5 files are a "
            f"different, binary format - re-save it as .gp, or export the tab as text."
        ) from exc
    return ET.fromstring(raw)


def read_gp(path: str | Path, *, track: str | None = None, bars: int | None = None) -> TabStrokes:
    """Read the ordered strokes of one track out of a Guitar Pro file.

    `bars` reads only the first N bars, which is what you want when the score runs the whole
    song but the recording is just the intro.
    """
    root = _score(path)
    tracks = _tracks(root)
    if not tracks:
        raise TabifyError("that Guitar Pro file has no tracks")
    chosen = _pick_track(tracks, track)

    bars_by_id = _index(root, "Bars")
    voices_by_id = _index(root, "Voices")
    beats_by_id = _index(root, "Beats")
    notes_by_id = _index(root, "Notes")

    master_bars = root.find("MasterBars")
    strokes: list[list[tuple[int, int]]] = []
    for bar_number, master in enumerate(master_bars if master_bars is not None else []):
        if bars is not None and bar_number >= bars:
            break
        ids = _ids(master.findtext("Bars"))
        if chosen >= len(ids):
            continue
        bar = bars_by_id.get(ids[chosen])
        if bar is None:
            continue
        for voice_id in _ids(bar.findtext("Voices")):
            voice = voices_by_id.get(voice_id)
            if voice is None:
                continue
            for beat_id in _ids(voice.findtext("Beats")):
                beat = beats_by_id.get(beat_id)
                if beat is None:
                    continue
                stroke = []
                for note_id in _ids(beat.findtext("Notes")):
                    note = notes_by_id.get(note_id)
                    if note is None:
                        continue
                    string = _property(note, "String")
                    fret = _property(note, "Fret")
                    if string is None or fret is None:
                        continue  # a drum or unpitched note has neither
                    stroke.append((int(string.findtext("String")), int(fret.findtext("Fret"))))
                if stroke:  # a beat with no notes is a rest, which is not a stroke
                    strokes.append(sorted(stroke))

    metadata = {"track": tracks[chosen].findtext("Name") or ""}
    tempo = root.findtext(".//MasterTrack/Automations/Automation/Value")
    if tempo:
        metadata["tempo"] = tempo.split()[0]
    return TabStrokes(strokes, _tuning_of(tracks[chosen]), metadata)
