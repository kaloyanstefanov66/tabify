# tabify 🎸

**Turn guitar recordings and MIDI files into guitar tabs you can actually play, right in your terminal.**

[![CI](https://github.com/kaloyanstefanov66/tabify/actions/workflows/ci.yml/badge.svg)](https://github.com/kaloyanstefanov66/tabify/actions/workflows/ci.yml)
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

One command, and nothing is held back:

```bash
pipx install --python 3.11 tabify-cli
```

That gives you audio in, tabs out: chords, power chords, and full band mixes split into
instruments first. No extras to pick, no compiler, no native installers. Separation used to
mean installing PyTorch (~550 MB); it now runs on the same ONNX runtime the chord engine
uses, which is why it's simply part of tabify rather than an afterthought - it's a tab
transcriber, so hearing a band recording is the job.

Two optional extras remain, and only because each needs something pip can't install:

| Install | Adds | Extra setup |
| --- | --- | --- |
| `tabify-cli` | everything below the line: MIDI and audio in, chords, full-mix separation, `--play` | none |
| `tabify-cli[render]` | **rendering audio** of a transcription, and `--play` with a synth tone | needs [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases) |
| `tabify-cli[youtube]` | transcribing straight from a **link** | needs [ffmpeg](https://ffmpeg.org/download.html) |
| `tabify-cli[all]` | both | both |

`[audio]`, `[ml]`, `[separate]` and `[play]` still work as install names so older commands
don't break, but they're empty now - you get all four either way.

### Why `--python 3.11`

Two upstream packages disagree about Python, and 3.11 is the only version both accept:

- **basic-pitch**, which hears chords, runs on 3.11 and older.
- **demucs-onnx**, which splits a mix into instruments, runs on 3.11 and newer.

tabify installs and runs on anything from 3.10 up, just without whichever piece its Python
can't have. On 3.11, basic-pitch drags TensorFlow (1.2 GB) along as a hard dependency, but
tabify never loads it: it runs the 0.2 MB ONNX copy of the same model, which gives
**identical predictions** (checked note for note) and loads in a tenth of a second. You can
uninstall TensorFlow afterwards and everything keeps working.

### Check what your install can actually do

```bash
tabify --doctor
```

It reports which engine each stage will really use on **your** machine, because an
environment installed earlier keeps whatever it was built with - changing a dependency list
does nothing to an install that already exists. If something looks like it's taking the slow
route, this is what says so, and pipx applies a changed dependency list only on reinstall:

```bash
pipx install --force --python 3.11 tabify-cli
```

### What genuinely needs a separate install

Only two things, both optional:

- **ffmpeg**, to convert what gets downloaded from a link (`winget install Gyan.FFmpeg`).
- **FluidSynth**, to synthesize audio from a transcription. On macOS and Linux that's
  `brew install fluid-synth` / `apt install fluidsynth`; on Windows it's a zip from
  [their releases](https://github.com/FluidSynth/fluidsynth/releases) with its `bin\`
  folder added to PATH, which is the most tedious step in the whole project.

Everything else is one `pipx install`.

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

# straight from a link - the audio is downloaded once and cached
tabify "https://www.youtube.com/watch?v=..."
tabify "https://example.com/my-riff.wav"          # direct file links work too

# don't know the tuning? let tabify work it out from the audio
tabify riff.wav --tuning auto

# full-mix support - splits into stems first, then tabs guitar and bass separately
tabify full_band_song.mp3 -o song.txt              # noticed as a mix: tabs the guitar
tabify full_band_song.mp3 --stems bass             # or pick another part
tabify full_band_song.mp3 --stems guitar,bass      # or several at once

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
| `-t, --tuning` | Preset or notes from low to high. 30 presets: every drop tuning from `drop-d` down to `drop-f#` (`drop-c#`, `drop-a#`, `drop-g#`, …), whole-guitar downtunings (`half-step-down` … `a-standard`), `open-g`/`open-d`/`open-e`/`open-c`, `dadgad`, 7- and 8-string, and bass. Flat spellings work too (`drop-db` = `drop-c#`). See `tabify --list-tunings` |
| `--capo N` | Capo fret |
| `--max-span N` | Biggest fret stretch allowed within a chord (default 4) |
| `--max-fret N` | Highest fret on your guitar (default 22) |
| `--bpm`, `--time-sig` | Override the detected tempo or time signature |
| `--grid` | Steps per beat: `2` = 8ths, `4` = 16ths, `3`/`6` = triplets |
| `--engine` | `auto`, `basic-pitch` (chords) or `pyin` (single notes) |
| `--onset-threshold` | Note sensitivity for basic-pitch (lower finds more notes) |
| `--min-note-ms` | Ignore notes shorter than this, to filter out noise |
| `--no-cleanup` | Skip snapping to pick attacks, splitting merged chugs and restoring missing low roots (see [the cleanup step](#the-cleanup-step)) |
| `--no-palm-mute` | Don't mark palm mutes (they're inferred from rhythm and string, not heard) |
| `--tuning auto` | Work the tuning out from the audio, and say how confident it is (see [detecting the tuning](#detecting-the-tuning)) |
| `--midi-out FILE` | Save the transcribed notes as a MIDI file |
| `--musicxml-out FILE` | Save as MusicXML (string/fret included), for Guitar Pro, TuxGuitar or MuseScore |
| `--separate` | Force separation even if the track doesn't look like a mix |
| `--stems NAMES` | Which separated part to tab: `guitar` (default), `bass`, `other`, `piano`, `vocals`, a comma-separated list, or `all`. `other` is usually second guitars and keys |
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

### Transcribing from a link

```bash
pip install "tabify-cli[youtube]"   # plus ffmpeg on PATH
tabify "https://www.youtube.com/watch?v=..."
```

[yt-dlp](https://github.com/yt-dlp/yt-dlp) fetches the audio, tabify caches it (as FLAC,
under your local app-data folder) so transcribing the same link twice only downloads once,
and a direct file URL is downloaded without involving yt-dlp at all. Sites hand out the odd
403 even for a link that worked a minute ago, so failed downloads are retried.

Whether you may download a given video is between you and whoever holds the rights - your
own uploads and openly licensed material are the safe cases. tabify won't scrape tab sites
like Songsterr or Ultimate Guitar: those tabs are licensed content and their terms forbid it.

### Following the tempo

Nobody plays to a perfect grid, and a tab built on a fixed tempo drifts further out of sync
the longer someone rushes or drags. So tabify tracks where the beats actually fall and places
notes *between the beats around them*, rather than against one tempo for the whole take. The
beats come from tabify's own attack detection, which finds far more of a dense distorted riff
than a general-purpose beat tracker does.

On a benchmark take that rushes from roughly 83 to 101 BPM, the difference is stark:

| | note F1 |
| --- | --- |
| following the tempo | **94%** |
| assuming one fixed tempo | 60% |

When the tempo moves, tabify says so:

```console
Tempo moves between 86 and 99 BPM - notes are placed against the beats as played, not a
fixed grid. Pass --bpm to force a steady tempo instead.
```

Pass `--bpm` when you know the take is to a click and want a rigid grid.

### Detecting the tuning

```bash
tabify riff.wav --tuning auto
```

```console
Tuning: drop-c (C2 G2 C3 F3 A3 D4), fit 2.67; next best open-c (2.31), c-standard (2.31)
```

Pitch alone can't pin a tuning down, because a lower tuning can reach every note a higher
one can, just at higher frets. What separates them is how they're *used*: riffs lean on the
open low string and sit low on the neck. So each preset is scored on notes landing on open
strings, the lowest note being the lowest string, and fret economy, with anything unplayable
disqualified and ties broken toward tunings people actually use.

That works when a riff touches open strings. When it doesn't - a lick played entirely up the
neck fits nearly every tuning equally - tabify says so instead of pretending:

```console
Tuning: drop-d (D2 A2 D3 G3 B3 E4), fit 1.55; next best half-step-down (1.51), standard (1.50)
         that was a close call - this riff barely uses open strings, which is what separates
         tunings. Pass --tuning if you know it.
```

### Full mixes are noticed, not silently mangled

Handed a whole band, a transcriber will faithfully tab *everything it hears* into one guitar
part - the bass on the low string, keys, vocals - and the result reads as nonsense rather
than as a failure, which is worse. So tabify checks first:

```console
This sounds like a full band mix: 17% of its energy is below the lowest string of standard,
which a guitar can't make - that's bass and kick drum.
Splitting the instruments apart first, so they don't all land in one tab.
```

The giveaway is energy below the instrument's lowest string, which a guitar can't produce
but a kick drum and bass guitar produce plenty of. Across real files, isolated tracks measure
0.000-0.001 of their energy down there and mixes measure 0.11-0.29 - two orders of magnitude
apart. It's judged against the instrument *you asked for*, so a bass stem reads as clean when
you say `--tuning bass` and as "something else is in here" when you call it a guitar.

You get the guitar by default, and tabify says what else is in there:

```console
Also in this track: bass, vocals - tab those with --stems bass,vocals
```

Drums aren't on the list - there's nothing to fret - and a stem the model found nothing for
is reported as empty rather than tabbed into a page of silence.

tabify splits the mix and tabs each instrument separately. `--no-auto-separate` tabs the mix
as one part anyway. If separation isn't available on your Python, `tabify --doctor` says so.

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
audio ──► pitch detection ──► cleanup ──► beat tracking ──► quantize ──► fingering search ──► ASCII tab
         (basic-pitch/pYIN)       │         (librosa)                   (Viterbi, least movement)
MIDI  ────────────────────────────┼─────────────────────────────►┘
                                  └ snap chord notes to real pick attacks, split merged chugs,
                                    restore low roots the recording can't carry
```

### The cleanup step

Pitch models report *notes*; a tab needs *pick strokes*. Tested on a real isolated drop-C metal guitar track, the raw model output had three problems, and the cleanup step (`tabify.refine`) fixes each:

- **Chord notes arrived tens of milliseconds apart** (up to 81 ms), so a stacked `6/6/6` came out as `6 - 6 - 6`. tabify now finds the actual pick attacks in the audio and snaps notes onto them.
- **Fast repeated chugs merged into one long note**, since the pitch never changes between them. Notes still ringing through an attack where their chord was re-struck get split there.
- **Low roots were missing.** That track has essentially nothing below ~89 Hz (the 63 Hz band sits 23 dB down), but a drop-C string's root is at 65 Hz - so `0/0/0` came out as just the fifth and octave, on the wrong strings. tabify measures where the recording's low end actually stops, and restores a root below that point when the notes above it form a power-chord shape, or when the root's own overtones are present.

Turn it off with `--no-cleanup` to compare against the model's raw output.

### Measuring accuracy

Guesswork is how a change that restores 212 missing roots can still make a tab worse, so
tabify ships a benchmark: riffs whose tab is already known get rendered to audio with a
real guitar tone, transcribed, and scored.

```bash
python benchmarks/benchmark.py              # score every piece
python benchmarks/benchmark.py --no-cleanup # score the raw model output, for comparison
```

Two scores, because they fail for different reasons: **note F1** (right pitch, right time -
blames the pitch model) and **tab F1** (also the right string and fret - blames the fingering
search). Current mean across the benchmark, up from 65% / 46% before the cleanup step:

| | note F1 | tab F1 |
| --- | --- | --- |
| drop-C chug riff | 94% | 94% |
| drop-C chug riff, low end cut | 96% | 96% |
| drop-C chug riff, human timing | 99% | 99% |
| drop-C power chords | 79% | 68% |
| standard clean arpeggio | 82% | 82% |
| standard single-note line | 86% | 14% |
| **mean** | **89%** | **75%** |

Read those as a way to compare changes, not as real-world accuracy: the audio is
synthesized, so it's kinder than a real recording. For scale, [TART (2026)](https://arxiv.org/html/2609.11904),
the best published audio-to-tab system, reports 54% end-to-end tab F1 on real recordings -
this is an unsolved problem, and heavily distorted downtuned guitar is its hardest corner.
Recall is near 100% on riffs; the remaining gap is precision (extra notes), which is what
the cleanup step keeps chipping away at.

### Checking a transcription against a tab you have

```bash
tabify riff.wav --compare my_riff.txt
```

```console
Against my_riff.txt: 32 strokes written, 36 heard
  22 matched exactly (69% of the written tab)
  10 nearly (a chord missing or gaining a string)
  0 written but not heard, 4 heard but not written
  where they differ:
    stroke    2: tab says C2                   tabify heard C2 C3
```

The two are lined up the way two versions of a text are, so an extra chug or a missed note
doesn't mark everything after it as wrong, and strokes are compared by the notes they sound
rather than by fret, since the same chord fingered in two places is the same chord.

This is how to find out what tabify actually gets wrong on your playing, rather than guessing
from how a tab looks - and the first time it ran, all ten of its disagreements turned out to
be one fixable bug.

## Limitations

Automatic music transcription is still an open research problem, so here's what to expect:

- **Works best on:** a clean recording of one guitar, such as a DI or close-mic recording, a riff, or a solo. With `--engine basic-pitch` (the default when it's installed), chords, power chords and dyads (thirds, fourths, fifths, octaves) transcribe correctly - verified against a set of synthesized power chords and intervals, all detected with the right notes.
- **Harder:** full band mixes, heavy distortion, and dense strumming. The notes will be rough.
- **Palm mutes are inferred, not heard.** On distorted guitar, muted and let-ring strokes measured the same decay and brightness - the distortion compresses both - so tabify marks `PM` on fast (8th note or quicker) repeated hits on the two lowest strings, the way they're played in practice. Expect it to be right for chug riffs and wrong on anything unconventional; `--no-palm-mute` turns it off.
- It doesn't detect bends, slides or hammer-ons yet.
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
