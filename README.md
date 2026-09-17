# tabify 🎸

**Turn guitar recordings and MIDI files into guitar tabs you can actually play, right in your terminal.**

[![CI](https://github.com/OWNER/tabify/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/tabify/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tabify-cli)](https://pypi.org/project/tabify-cli/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

```console
$ tabify riff.wav
Transcribing riff.wav ...
Engine: pyin, 16 notes, ~99 BPM
riff
====
Tuning: E2 A2 D3 G3 B3 E4 (standard)
Time: 4/4   Tempo: 99 BPM

  1                                 2
e|---------------------------------|---------------------------------|
B|-------------0---3---0-----------|-------------1---0---------------|
G|-----0---2---------------2---0---|-----0---2-----------2---0-------|
D|-2-------------------------------|-2---------------------------2---|
A|---------------------------------|---------------------------------|
E|---------------------------------|---------------------------------|
```

Most audio-to-tab tools are paid, closed, or only work in a browser. tabify is free and open source, and it runs locally with one command.

## Why the tabs are playable

Most notes can be played in several places on the neck. Tools that pick the lowest fret, or the first string that works, give you tabs that jump all over the place.

tabify lists every way to finger each note and chord and gives each one a cost for stretch and position. Then it searches the whole piece for the fingering path with the **least hand movement**. What you get looks like a tab a guitarist would write: open chord shapes, scale runs kept in one position, and no pointless shifts.

## Install

```bash
pipx install "tabify-cli[audio]"
```

| Install | Gives you | Python |
| --- | --- | --- |
| `tabify-cli` | MIDI → tab, all the tab features | 3.10+ |
| `tabify-cli[audio]` | + audio transcription of **single-note** lines (librosa pYIN) | 3.10+ |
| `tabify-cli[ml]` | + **chords, power chords and other intervals** via Spotify's [basic-pitch](https://github.com/spotify/basic-pitch) | 3.10–3.11 |
| `tabify-cli[separate]` | + **full-mix support**: split a band recording into stems (guitar/bass/drums/vocals/piano) before tabbing, via [Demucs](https://github.com/facebookresearch/demucs) | 3.10+ |
| `tabify-cli[render]` | + **real audio rendering** of a transcription (acoustic/distorted guitar, bass, ...) via FluidSynth | 3.10+ |

```bash
# chord/power-chord engine (basic-pitch currently needs Python 3.11 or older)
pipx install "tabify-cli[ml]" --python 3.11
```

`[ml]` pulls in TensorFlow, which is a genuinely large download (~1.2 GB) - basic-pitch's model needs it on Windows/Linux for Python 3.11+. Worth knowing before you install it, not after.

Try it without any audio:

```bash
tabify --demo
```

## Usage

```bash
tabify song.wav                         # print the tab
tabify song.mp3 -o song.txt             # save it to a file
tabify song.mid --track 2               # tab one track of a MIDI file
tabify riff.wav --tuning drop-d         # alternate tunings
tabify riff.wav --capo 3                # frets shown relative to the capo
tabify riff.wav --tuning "D2 G2 D3 G3 B3 D4"   # any custom tuning
tabify solo.flac --bpm 120 --grid 2     # set the tempo, snap to 8th notes
tabify solo.wav --midi-out solo.mid     # also export the notes as MIDI
tabify solo.wav --musicxml-out solo.musicxml   # opens in Guitar Pro, TuxGuitar, MuseScore
tabify bassline.wav --tuning bass       # 4- and 5-string bass work too
tabify --list-tunings

# full-mix support - splits into stems first, then tabs guitar and bass separately
tabify full_band_song.mp3 --separate -o song.txt   # writes song.guitar.txt, song.bass.txt

# hear it back with a real guitar tone (needs FluidSynth - see below)
tabify riff.wav --audio-out riff_render.wav --instrument distortion
tabify riff.wav --audio-out heavy.wav --instrument distortion --drive 1.0 --tone 0.7   # crunchier, darker

# play along: the tab scrolls and highlights in sync with playback, Songsterr-style
tabify riff.wav --play                          # plays the original recording
tabify riff.wav --play --source synth           # plays a synthesized guitar render instead
tabify song.mid --play                          # MIDI has no "original recording", so this uses synth
```

| Option | What it does |
| --- | --- |
| `-t, --tuning` | Preset (`standard`, `drop-d`, `dadgad`, `open-g`, `bass`, `7-string`, …) or notes from low to high |
| `--capo N` | Capo fret |
| `--max-span N` | Biggest fret stretch allowed within a chord (default 4) |
| `--max-fret N` | Highest fret on your guitar (default 22) |
| `--bpm`, `--time-sig` | Override the detected tempo or time signature |
| `--grid` | Steps per beat: `2` = 8ths, `4` = 16ths, `3`/`6` = triplets |
| `--engine` | `auto`, `basic-pitch` (chords) or `pyin` (single notes) |
| `--onset-threshold` | Note sensitivity for basic-pitch (lower finds more notes) |
| `--min-note-ms` | Ignore notes shorter than this, to filter out noise |
| `--midi-out FILE` | Save the transcribed notes as a MIDI file |
| `--musicxml-out FILE` | Save as MusicXML (string/fret included), for Guitar Pro, TuxGuitar or MuseScore |
| `--separate` | Split a full-band recording into instrument stems first, and tab each one (needs `[separate]`) |
| `--bass-tuning` | Tuning used for the separated bass stem (default: `bass`) |
| `--instrument NAME` | Tone for `--midi-out`/`--audio-out`: `nylon`, `steel`, `jazz`, `clean`, `muted`, `overdrive`, `distortion`, `harmonics`, or a bass patch |
| `--audio-out FILE` | Render real audio of the transcription with `--instrument`'s tone (needs `[render]`) |
| `--soundfont FILE` | Use your own `.sf2` for `--audio-out` (default: a small one, downloaded once) |
| `--drive 0-1` | Distortion amount for `--audio-out`/`--play` (real waveshaping DSP, not just a soundfont patch - default: a sensible amount for `distortion`/`overdrive`/`muted`, 0 for cleaner tones) |
| `--tone 0-1` | Distortion tone: 0 = brighter, 1 = darker/more muffled (default: 0.5) |
| `--play` | Play along: the tab scrolls and highlights in sync with audio, Songsterr-style (needs `[play]`) |
| `--source` | What `--play` plays: `original` (the recording) or `synth` (a rendered tone) - default: original if available |
| `--seek-seconds N` | Seconds to jump with the arrow keys in `--play` (default: 5) |
| `--width`, `--title`, `--no-color` | Display options |

### Full-mix support and its real limit

`--separate` uses [Demucs](https://github.com/facebookresearch/demucs) to pull guitar and bass out of a full band recording before tabbing each one. This works well *across* instrument types.

It can't reliably split **two guitars playing at the same time** into two separate tracks - no current source-separation model can do that for two instances of the same instrument, tabify included. A recording with two interleaved guitar parts will still come out as one, more complex, guitar tab rather than two.

### Real audio rendering

`--audio-out` doesn't just export a MIDI file - it actually synthesizes audio with a chosen guitar tone via [FluidSynth](https://www.fluidsynth.org/), so you can listen to what tabify transcribed. FluidSynth's native library needs a separate install:

```bash
# Windows: download the release zip from https://github.com/FluidSynth/fluidsynth/releases
#          and add its bin\ folder to PATH
brew install fluid-synth       # macOS
apt install fluidsynth         # Linux (or your distro's equivalent)
pip install "tabify-cli[render]"
```

The first render auto-downloads a small (~6 MB) General MIDI soundfont. Pass `--soundfont your.sf2` to use a bigger one for a better tone.

`distortion`/`overdrive`/`muted` also run through real waveshaping distortion DSP on top of whatever the soundfont provides - a small soundfont's own "Distortion Guitar" sample tends to sound thin on its own, so tabify adds the actual clipping and harmonics a pedal or overdriven amp would, then rolls off the harsh top end the way a speaker cabinet does. Tune it with `--drive`/`--tone`, or turn it off entirely on any instrument with `--drive 0`.

### Play-along mode

```bash
pip install "tabify-cli[play]"
tabify riff.wav --play
```

Space to pause/resume, ←/→ to seek, `q` to quit. Two sources, picked with `--source`:

- **`original`** (the default, when available): plays the real recording. Sync quality depends on tabify's tempo detection - solid for a riff or a short clip, but can drift on a longer piece with tempo changes or rubato, since tabify currently detects one tempo rather than a full tempo map.
- **`synth`**: plays a FluidSynth render of the transcription (needs `[render]` too). Always perfectly in sync with the tab, since both are generated from the exact same quantized notes - but it's a synthesized tone, not the real recording.

### Getting a tab into Songsterr

Songsterr's own uploader wants a Guitar Pro file, not MusicXML directly. The
free route: `tabify song.wav --musicxml-out song.musicxml`, open it in
[TuxGuitar](https://tuxguitar.app/) (free) or Guitar Pro, then save/export as
`.gp5` from there and upload that.

## How it works

```
audio ──► pitch detection ──► beat tracking ──► quantize to grid ──► fingering search ──► ASCII tab
         (basic-pitch/pYIN)      (librosa)                        (Viterbi, least movement)
MIDI  ─────────────────────────────────────────►┘
```

## Limitations

Automatic music transcription is still an open research problem, so here's what to expect:

- **Works best on:** a clean recording of one guitar, such as a DI or close-mic recording, a riff, or a solo. With `--engine basic-pitch` (needs `[ml]`), chords, power chords and dyads (thirds, fourths, fifths, octaves) transcribe correctly - verified against a set of synthesized power chords and intervals, all detected with the right notes.
- **Harder:** full band mixes, heavy distortion, and dense strumming. The notes will be rough.
- It doesn't detect bends, slides, hammer-ons, or palm muting yet.
- Beat tracking finds the beats but can't be sure where bar 1 starts. Use `--bpm` if the tempo is off.

Treat the output as a strong first draft, then fix it by ear.

## Roadmap

- [ ] Direct Guitar Pro (`.gp5`) export (MusicXML export already works, see `--musicxml-out`)
- [ ] Hammer-on, pull-off, slide, and bend detection
- [ ] Rhythm notation under the tab
- [ ] `--position` to force a region of the neck

Done, and verified end-to-end (not just unit-tested) rather than assumed to work:
- **Full-mix stem separation** (`--separate`) - separating a synthetic guitar+bass+drums mix produced a guitar tab that exactly matched the original.
- **Real audio rendering** (`--audio-out`) - produced real, audible acoustic and distortion guitar renders.
- **Play-along mode** (`--play`) - confirmed correct real-time sync and scrolling against both a synthesized render and an actual recording, for runs of 5-20+ seconds. Not personally verified: the feel of the interactive controls (space/arrows/`q`) in a live terminal, since that needs an actual person at the keyboard - the pause/seek/quit logic itself is unit-tested with simulated keypresses.

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
