"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from tabify import TabifyError, __version__
from tabify.fretting import FretOptions, assign_frets
from tabify.instruments import ALL_PROGRAMS, default_instrument
from tabify.notes import Note
from tabify.render import render_tab
from tabify.rhythm import TimeSignature, align_to_bars, parse_time_signature, quantize, start_on_first_beat
from tabify.tuning import TUNINGS, parse_tuning

MIDI_EXTENSIONS = {".mid", ".midi"}
GP_EXTENSIONS = {".gp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tabify",
        description="Turn guitar recordings and MIDI files into playable guitar tabs.",
        epilog="examples:\n"
        "  tabify --demo\n"
        "  tabify riff.wav\n"
        "  tabify song.mid --tuning drop-d --capo 2 -o song.txt\n"
        "  tabify solo.mp3 --bpm 96 --midi-out solo.mid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="audio file (wav, mp3, flac, ...) or MIDI file")
    p.add_argument("-o", "--output", metavar="FILE", help="write the tab to FILE instead of the terminal")
    p.add_argument("--demo", action="store_true", help="render a built-in example, no input needed")
    p.add_argument("--version", action="version", version=f"tabify {__version__}")
    p.add_argument(
        "--doctor", action="store_true",
        help="show which engine each stage of this install will actually use, and exit",
    )

    g = p.add_argument_group("instrument")
    g.add_argument(
        "-t", "--tuning", default="standard",
        help="preset name, notes low-to-high ('D2 A2 D3 G3 B3 E4'), or 'auto' to work it out from the audio",
    )
    g.add_argument("--capo", type=int, default=0, help="capo fret (tab numbers are relative to the capo)")
    g.add_argument("--max-fret", type=int, default=22, help="highest fret on your guitar (default: 22)")
    g.add_argument("--max-span", type=int, default=4, help="max fret stretch within a chord (default: 4)")
    g.add_argument("--list-tunings", action="store_true", help="show tuning presets and exit")

    g = p.add_argument_group("full mix")
    g.add_argument(
        "--separate", action="store_true",
        help="split a full-band recording into instrument stems first (via Demucs) and tab each one separately",
    )
    g.add_argument(
        "--stems", default="guitar", metavar="NAMES",
        help="which separated parts to tab: any of guitar, bass, other, piano, vocals (comma-separated), "
        "or 'all' (default: guitar). 'other' is usually second guitars and keys",
    )
    g.add_argument("--bass-tuning", default="bass", help="tuning used for the separated bass stem (default: bass)")
    g.add_argument(
        "--no-auto-separate", action="store_true",
        help="tab a full band mix as one part instead of splitting it up (every instrument lands in the same tab)",
    )

    g = p.add_argument_group("rhythm")
    g.add_argument("--bpm", type=float, help="tempo (default: read from MIDI or detected from audio)")
    g.add_argument("--time-sig", help="time signature, e.g. 3/4 (default: from MIDI, else 4/4)")
    g.add_argument(
        "--grid", type=int, default=4, choices=(1, 2, 3, 4, 6, 8),
        help="grid steps per beat: 2=8ths, 4=16ths, 3/6=triplets (default: 4)",
    )

    g = p.add_argument_group("audio")
    g.add_argument("--engine", default="auto", choices=("auto", "basic-pitch", "pyin"), help="transcription engine")
    g.add_argument("--onset-threshold", type=float, default=0.5, help="basic-pitch note sensitivity, 0-1 (default: 0.5)")
    g.add_argument("--min-note-ms", type=float, default=80.0, help="ignore notes shorter than this (default: 80)")
    g.add_argument(
        "--no-cleanup", action="store_true",
        help="skip snapping notes to pick attacks, splitting merged re-strikes and restoring missing low roots",
    )
    g.add_argument(
        "--no-palm-mute", action="store_true",
        help="don't mark palm mutes (they're inferred from fast repeated low-string hits, not heard)",
    )

    g = p.add_argument_group("midi")
    g.add_argument("--track", type=int, help="only use this MIDI track index (default: all non-drum tracks)")
    g.add_argument("--midi-out", metavar="FILE", help="also save the transcribed notes as a MIDI file")
    g.add_argument(
        "--musicxml-out", metavar="FILE",
        help="also save as MusicXML, importable into Guitar Pro, TuxGuitar or MuseScore",
    )
    g.add_argument(
        "--instrument", choices=sorted(ALL_PROGRAMS), metavar="NAME",
        help=f"tone for --midi-out and --audio-out: {', '.join(sorted(ALL_PROGRAMS))} (default: guessed from tuning)",
    )

    g = p.add_argument_group("audio rendering")
    g.add_argument(
        "--audio-out", metavar="FILE",
        help="render actual audio (wav/flac/ogg) of the transcription with --instrument's tone, via FluidSynth",
    )
    g.add_argument("--soundfont", metavar="FILE", help="a .sf2 file for --audio-out (default: a small one, auto-downloaded once)")
    g.add_argument(
        "--drive", type=float, metavar="0-1",
        help="distortion amount for --audio-out/--play (0=clean, 1=heavy; default: a sensible amount for "
        "--instrument distortion/overdrive/muted, 0 for cleaner tones)",
    )
    g.add_argument("--tone", type=float, default=0.5, metavar="0-1", help="distortion tone: 0=brighter, 1=darker (default: 0.5)")

    g = p.add_argument_group("play-along")
    g.add_argument(
        "--play", action="store_true",
        help="interactive play-along: the tab scrolls and highlights in sync with audio playback (needs [play])",
    )
    g.add_argument(
        "--source", choices=("original", "synth"), metavar="NAME",
        help="what --play plays: 'original' (the recording, most accurate) or 'synth' (a rendered guitar tone, "
        "perfectly in sync with the tab but not the real sound) - default: original if available, else synth",
    )
    g.add_argument("--seek-seconds", type=float, default=5.0, help="seconds to jump with arrow keys in --play (default: 5)")

    g = p.add_argument_group("checking")
    g.add_argument(
        "--compare", metavar="TABFILE",
        help="compare the transcription against a tab you already have (a Guitar Pro .gp file "
        "or ASCII tab), and report what each side has that the other doesn't",
    )
    g.add_argument(
        "--compare-track", metavar="NAME",
        help="which track of a Guitar Pro score to compare against (default: the first guitar or bass)",
    )
    g.add_argument(
        "--compare-bars", type=int, metavar="N",
        help="only compare the first N bars of the score - use it when the tab runs the whole "
        "song but the recording is just a section",
    )

    g = p.add_argument_group("display")
    g.add_argument("--title", help="title shown above the tab")
    g.add_argument("--width", type=int, help="max line width (default: terminal width)")
    g.add_argument("--no-color", action="store_true", help="disable colored output")
    return p


def _use_color(args: argparse.Namespace) -> bool:
    if args.no_color or args.output or os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    from tabify.term import enable_ansi

    return enable_ansi()


def _info(msg: str) -> None:
    print(msg, file=sys.stderr)


def _suffixed(path: str, label: str | None) -> str:
    """Insert `.{label}` before the extension, e.g. ('song.txt', 'bass') -> 'song.bass.txt'."""
    if not label:
        return path
    p = Path(path)
    return str(p.with_name(f"{p.stem}.{label}{p.suffix}"))


def _transcribe_path(path: Path, tuning, args: argparse.Namespace) -> tuple[list[Note], float, str]:
    """Run the configured engine over an audio file.

    `tuning` may be None, meaning "work it out from the audio". Returns
    (notes, bpm, engine used, tuning).
    """
    from tabify.transcribe import transcribe_audio

    detect = tuning is None
    result = transcribe_audio(
        path,
        lowest=(tuning.strings[0] + args.capo) if tuning else 0,
        highest=(tuning.strings[-1] + args.max_fret) if tuning else 0,
        engine=args.engine,
        bpm=args.bpm,
        onset_threshold=args.onset_threshold,
        min_note_ms=args.min_note_ms,
        refine=not args.no_cleanup,
        detect_tuning=detect,
    )
    if detect:
        tuning = result.tuning
        ranking = result.tuning_ranking or []
        best, runner_up = (ranking + [(0.0, None)] * 2)[:2]
        others = ", ".join(f"{t.name} ({s:.2f})" for s, t in ranking[1:4] if t)
        confident = best[0] - runner_up[0] > 0.15
        _info(f"Tuning: {tuning.name} ({tuning.describe()}), fit {best[0]:.2f}; next best {others}")
        if not confident:
            _info(
                "         that was a close call - this riff barely uses open strings, which is what "
                "separates tunings. Pass --tuning if you know it."
            )
    r = result.refine
    if r:
        roots = f", restored {r.roots_added} low root(s)" if r.roots_added else ""
        _info(
            f"Cleanup: {r.onsets} pick attacks found, {r.snapped} note(s) snapped to them, "
            f"{r.splits} merged re-strike(s) split{roots} (recording's low end stops near {round(r.low_end_hz)} Hz)"
        )
    beats = result.beat_map
    if beats is not None and beats.varies:
        _info(
            f"Tempo moves between {round(beats.bpm_low)} and {round(beats.bpm_high)} BPM - notes are placed "
            "against the beats as played, not a fixed grid. Pass --bpm to force a steady tempo instead."
        )
    # Beat tracking finds beats but not bar lines, so start bar 1 on the first played beat.
    return start_on_first_beat(result.notes), result.bpm, result.engine, tuning


def _quantize_and_fret(
    notes: list[Note], time_sig: TimeSignature, tuning, args: argparse.Namespace, from_audio: bool,
):
    notes = align_to_bars(quantize(notes, args.grid), time_sig)
    opts = FretOptions(capo=args.capo, max_fret=args.max_fret, max_span=args.max_span)
    fretted = assign_frets(notes, tuning, opts)
    if fretted.dropped:
        _info(f"warning: skipped {len(fretted.dropped)} note(s) that don't fit this tuning/capo")
    # Only for audio: a MIDI file's notes carry no technique information to base a guess on.
    if from_audio and not args.no_palm_mute:
        from tabify.techniques import infer_palm_mutes

        fretted.events = infer_palm_mutes(fretted.events)
    return notes, fretted


def _process_one(
    notes: list[Note], bpm: float, time_sig: TimeSignature, title: str, tuning,
    args: argparse.Namespace, label: str | None, from_audio: bool = False,
) -> None:
    """Quantize, fret, render and export one instrument's notes."""
    notes, fretted = _quantize_and_fret(notes, time_sig, tuning, args, from_audio)

    width = args.width or shutil.get_terminal_size((100, 24)).columns
    text = render_tab(
        fretted.events, tuning, time_sig=time_sig, subdivision=args.grid, width=max(width, 20),
        title=title, capo=args.capo, bpm=bpm, color=_use_color(args),
    )

    if args.output:
        out = _suffixed(args.output, label)
        Path(out).write_text(text, encoding="utf-8")
        _info(f"Wrote tab to {out}")
    else:
        if label:
            print(f"=== {label} ===", file=sys.stdout)
        sys.stdout.write(text)

    instrument = args.instrument or default_instrument(tuning)

    if args.midi_out:
        from tabify.midi_io import write_midi

        out = _suffixed(args.midi_out, label)
        write_midi(out, notes, bpm or 120.0, time_sig, program=ALL_PROGRAMS[instrument])
        _info(f"Wrote MIDI to {out}")

    if args.musicxml_out:
        from tabify.musicxml import to_musicxml

        out = _suffixed(args.musicxml_out, label)
        xml_text = to_musicxml(
            fretted.events, tuning, time_sig=time_sig, subdivision=args.grid, bpm=bpm, title=title, capo=args.capo
        )
        Path(out).write_text(xml_text, encoding="utf-8")
        _info(f"Wrote MusicXML to {out}")

    if args.audio_out:
        from tabify.synth import render_audio

        out = _suffixed(args.audio_out, label)
        _info(f"Rendering audio ({instrument}) ...")
        render_audio(
            fretted.events, tuning, out,
            instrument=instrument, capo=args.capo, bpm=bpm or 120.0, soundfont=args.soundfont,
            drive=args.drive, tone=args.tone,
        )
        _info(f"Wrote audio to {out}")

    if args.compare:
        _compare_with_tab(fretted, tuning, args)


def _play_one(
    notes: list[Note], bpm: float, time_sig: TimeSignature, title: str, tuning,
    args: argparse.Namespace, audio_path: Path | None,
) -> None:
    from tabify.player import load_audio_file, play_along

    notes, fretted = _quantize_and_fret(notes, time_sig, tuning, args, from_audio=audio_path is not None)
    bpm = bpm or 120.0

    source = args.source or ("original" if audio_path else "synth")
    if source == "original" and not audio_path:
        raise TabifyError("--source original needs an audio file (this input has no original recording to play)")

    if source == "original":
        _info(f"Loading {audio_path.name} for playback ...")
        audio, sample_rate = load_audio_file(audio_path)
    else:
        from tabify.synth import synthesize

        instrument = args.instrument or default_instrument(tuning)
        _info(f"Synthesizing audio ({instrument}) ...")
        audio = synthesize(
            fretted.events, tuning, instrument=instrument, capo=args.capo, bpm=bpm,
            soundfont=args.soundfont, drive=args.drive, tone=args.tone,
        )
        sample_rate = 44100

    width = args.width or shutil.get_terminal_size((100, 24)).columns
    play_along(
        fretted.events, tuning, audio, sample_rate,
        bpm=bpm, time_sig=time_sig, subdivision=args.grid, capo=args.capo,
        title=title, width=max(width, 20), color=_use_color(args), seek_seconds=args.seek_seconds,
    )


def _is_full_mix(path: Path, tuning, args: argparse.Namespace) -> bool:
    """Warn when a whole band was handed in as if it were one guitar, and separate if we can.

    Without this, every instrument gets faithfully tabbed into the same part - the bass on the
    low string, keys, vocals - which reads as nonsense rather than as a failure.
    """
    if args.separate or args.no_auto_separate or tuning is None:
        return False  # with --tuning auto there's no lowest string to measure against yet

    import librosa

    from tabify.mixcheck import looks_like_full_mix
    from tabify.separate import separation_available

    try:
        y, sr = librosa.load(str(path), sr=22050, mono=True, duration=40)
    except Exception:  # if it won't load here, the real load will report it properly
        return False
    is_mix, share = looks_like_full_mix(y, sr, tuning.strings[0])
    if not is_mix:
        return False

    _info(
        f"This sounds like a full band mix: {share:.0%} of its energy is below the lowest string of "
        f"{tuning.name}, which a guitar can't make - that's bass and kick drum."
    )
    if separation_available():
        _info("Splitting the instruments apart first, so they don't all land in one tab.")
        return True
    _info(
        "Every instrument will be tabbed into the same part. Either give tabify an isolated guitar "
        'track, or install separation: pip install "tabify-cli[separate]"\n'
        "         (--no-auto-separate silences this and tabs the mix as-is.)"
    )
    return False


def _pitches_of(strokes, tuning, capo: int = 0) -> list[set[int]]:
    """Each stroke as the set of pitches it sounds, so two tabs can be compared by ear, not by fret."""
    return [{tuning.strings[s] + capo + f for s, f in stroke} for stroke in strokes]


def _read_reference(args: argparse.Namespace):
    """The tab to compare against - a Guitar Pro score or hand-written ASCII."""
    path = Path(args.compare)
    if path.suffix.lower() in GP_EXTENSIONS:
        from tabify.gpfile import read_gp

        written = read_gp(path, track=args.compare_track, bars=args.compare_bars)
        _info(f"Comparing against the {written.metadata.get('track', 'first fretted')} track of {path.name}.")
        return written

    from tabify.tabfile import read_tab_file

    return read_tab_file(path)


def _compare_with_tab(fretted, tuning, args: argparse.Namespace) -> None:
    """Say how a transcription differs from a tab someone already has."""
    from tabify.align import agreement, align

    written = _read_reference(args)
    reference_tuning = written.tuning or tuning
    if reference_tuning.strings != tuning.strings:
        _info(
            f"note: that tab is in {reference_tuning.describe()} and this was transcribed in "
            f"{tuning.describe()} - comparing the notes each one sounds, not the fret numbers."
        )
    reference = _pitches_of(written.strokes, reference_tuning)
    heard = _pitches_of([[(p.string, p.fret) for p in e.positions] for e in fretted.events], tuning, args.capo)

    result = align(reference, heard)
    summary = agreement(reference, heard, result)
    print(
        f"\nAgainst {Path(args.compare).name}: {summary['strokes_a']} strokes written, "
        f"{summary['strokes_b']} heard\n"
        f"  {summary['exact']} matched exactly ({summary['recall']:.0%} of the written tab)\n"
        f"  {summary['partial']} nearly (a chord missing or gaining a string)\n"
        f"  {summary['missed']} written but not heard, {summary['extra']} heard but not written"
    )

    from tabify.notes import midi_to_name

    disagreements = [(i, j) for i, j in result.pairs if reference[i] != heard[j]]
    if disagreements:
        print("  where they differ:")
        for i, j in disagreements[:8]:
            want = " ".join(midi_to_name(p) for p in sorted(reference[i]))
            got = " ".join(midi_to_name(p) for p in sorted(heard[j]))
            print(f"    stroke {i + 1:>4}: tab says {want:<20} tabify heard {got}")
        if len(disagreements) > 8:
            print(f"    ... and {len(disagreements) - 8} more")


def _separate_and_process(path: Path, title: str, tuning, time_sig, args: argparse.Namespace) -> None:
    import tempfile

    from tabify.separate import SELECTABLE, SILENCE_RMS, loudness, parse_stems, separate_stems

    wanted = parse_stems(args.stems)
    _info(f"Separating {path.name} into stems (this downloads a model on first use) ...")
    with tempfile.TemporaryDirectory(prefix="tabify-") as tmp:
        stems = separate_stems(path, tmp)
        levels = {name: loudness(p) for name, p in stems.items() if name in SELECTABLE}
        playing = [name for name, level in levels.items() if level >= SILENCE_RMS]
        found = [name for name in wanted if name in playing]

        empty = [name for name in wanted if name in levels and name not in playing]
        if empty:
            _info(f"Nothing playing in the {', '.join(empty)} stem, so there's nothing to tab there.")
        if not found:
            raise TabifyError(
                f"nothing to tab in {', '.join(wanted)}. This track has: {', '.join(playing) or 'nothing'}"
            )
        others = [name for name in playing if name not in found]
        if others:
            _info(f"Also in this track: {', '.join(others)} - tab those with --stems {','.join(others)}")

        # Playing along is one part at a time - there is a single screen and a single pair of
        # ears - so say which one rather than silently picking.
        if args.play and len(found) > 1:
            raise TabifyError(
                f"--play follows one part at a time, and {', '.join(found)} were found. "
                f"Pick one, e.g. --stems {found[0]}"
            )

        for name in found:
            stem_tuning = parse_tuning(args.bass_tuning) if name == "bass" else tuning
            _info(f"Transcribing {name} stem ...")
            notes, bpm, engine, stem_tuning = _transcribe_path(stems[name], stem_tuning, args)
            ts = time_sig or TimeSignature()
            _info(f"  Engine: {engine}, {len(notes)} notes, ~{round(bpm)} BPM")
            if args.play:
                # Play the separated stem, not the original mix: following a guitar tab is far
                # easier against the guitar alone, which is the whole point of separating.
                _play_one(notes, bpm, ts, f"{title} ({name})", stem_tuning, args, stems[name])
                continue
            # Only tag the output with the stem's name when there's more than one to tell apart,
            # so asking for a single part writes exactly the file you asked for.
            label = name if len(found) > 1 else None
            _process_one(notes, bpm, ts, f"{title} ({name})", stem_tuning, args, label, from_audio=True)


def run(args: argparse.Namespace) -> int:
    if args.doctor:
        from tabify.environment import report

        print(report())
        return 0

    if args.list_tunings:
        from tabify.tuning import ALIASES

        also_known_as: dict[str, list[str]] = {}
        for alias, target in ALIASES.items():
            also_known_as.setdefault(target, []).append(alias)
        name_w = max(len(n) for n in TUNINGS)
        notes_w = max(len(v) for v in TUNINGS.values())
        for name, notes in TUNINGS.items():
            aliases = also_known_as.get(name)
            extra = f"   (also: {', '.join(aliases)})" if aliases else ""
            print(f"{name:<{name_w}}  {notes:<{notes_w}}{extra}".rstrip())
        print("\nAny other tuning: pass the open strings low to high, e.g. --tuning 'D2 A2 D3 G3 B3 E4'")
        return 0

    # None means "work it out from the audio"; every path that can't (MIDI, --demo) checks below.
    tuning = None if args.tuning.strip().lower() == "auto" else parse_tuning(args.tuning)
    if not 0 <= args.capo < args.max_fret:
        raise TabifyError(f"capo must be between 0 and {args.max_fret - 1}")
    if args.max_span < 1:
        raise TabifyError("--max-span must be at least 1")

    time_sig: TimeSignature | None = parse_time_signature(args.time_sig) if args.time_sig else None
    title = args.title

    if args.demo:
        from tabify.demo import demo_notes

        time_sig = time_sig or TimeSignature()
        tuning = tuning or parse_tuning("standard")  # nothing to listen to, so nothing to detect
        notes, bpm, title = demo_notes(), args.bpm or 90, title or "tabify demo"
        if args.play:
            _play_one(notes, bpm, time_sig, title, tuning, args, None)
        else:
            _process_one(notes, bpm, time_sig, title, tuning, args, None)
        return 0

    if not args.input:
        raise TabifyError("no input file given (try: tabify --demo, or tabify --help)")

    from tabify.fetch import fetch, is_url

    if is_url(args.input):
        path, fetched_title = fetch(args.input)
        title = title or fetched_title
    else:
        path = Path(args.input)
        if not path.is_file():
            raise TabifyError(f"file not found: {path}")
    title = title or path.stem

    is_midi = path.suffix.lower() in MIDI_EXTENSIONS
    if args.separate or (not is_midi and _is_full_mix(path, tuning, args)):
        if is_midi:
            raise TabifyError("--separate needs an audio file, not MIDI (a MIDI file already has separate tracks - use --track)")
        _separate_and_process(path, title, tuning, time_sig, args)
        return 0

    if is_midi:
        from tabify.midi_io import read_midi

        if tuning is None:
            raise TabifyError("--tuning auto needs audio to listen to; a MIDI file has no tone to judge from")
        content = read_midi(path, args.track)
        notes = content.notes
        bpm = args.bpm or content.bpm
        time_sig = time_sig or content.time_sig
    else:
        _info(f"Transcribing {path.name} ...")
        notes, bpm, engine, tuning = _transcribe_path(path, tuning, args)
        _info(f"Engine: {engine}, {len(notes)} notes, ~{round(bpm)} BPM")

    time_sig = time_sig or TimeSignature()
    if args.play:
        _play_one(notes, bpm, time_sig, title, tuning, args, None if is_midi else path)
    else:
        _process_one(notes, bpm, time_sig, title, tuning, args, None, from_audio=not is_midi)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except TabifyError as exc:
        _info(f"tabify: error: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130
