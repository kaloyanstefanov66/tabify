"""Make labelled training data for the stroke model, by playing tabs we already know.

You can't get much labelled guitar audio, but you can write a tab, render it, and keep the
tab as the answer - the trick behind SynthTab (https://arxiv.org/html/2309.09085v3). Each
clip is a randomly written riff, rendered through a randomly chosen guitar tone, and every
pick attack in it becomes one labelled example: what root, what shape.

The point of the randomisation is the domain gap. A model trained on one clean synthesized
tone learns that tone, not the guitar, so every clip varies tuning, tempo, patch, distortion
drive and tone, low-end rolloff, noise, level, and how sloppily it's played - and the patch
is cut at a slightly wrong moment on purpose, because at inference the attack times come
from a detector that is also slightly wrong.

    python training/generate_data.py --clips 200 --out data/strokes
    python training/generate_data.py --clips 20 --out data/holdout --seed 99

Synthetic data alone won't close the gap to a real amp in a real room - real recordings
are worth far more per minute. This is the floor, not the ceiling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabify.features import FRAMES, N_BINS, SAMPLE_RATE, patches_at  # noqa: E402
from tabify.fretting import Position, TabEvent  # noqa: E402
from tabify.notes import Note  # noqa: E402
from tabify.shapes import ROOT_HIGH, ROOT_LOW, SHAPES, positions_for, root_index, shape_index  # noqa: E402
from tabify.synth import synthesize  # noqa: E402
from tabify.tuning import parse_tuning  # noqa: E402

# Weighted toward the tunings and tones this is meant to get right, with enough of everything
# else that the model doesn't only work on drop-tuned metal.
TUNINGS = (
    ("drop-c", 5), ("drop-c#", 3), ("drop-b", 3), ("drop-d", 4), ("drop-a#", 2), ("drop-g#", 2),
    ("standard", 5), ("half-step-down", 3), ("d-standard", 2), ("7-string", 2), ("drop-a", 1), ("open-g", 1),
)
INSTRUMENTS = (("distortion", 5), ("overdrive", 3), ("muted", 2), ("clean", 2), ("jazz", 1), ("steel", 2), ("nylon", 1))
SHAPE_WEIGHTS = (("power3", 4), ("power2", 4), ("single", 5), ("octave", 2), ("triad-minor", 1), ("triad-major", 1))


def _weighted(rng: np.random.Generator, options):
    names = [name for name, _ in options]
    weights = np.array([weight for _, weight in options], dtype=float)
    return names[int(rng.choice(len(names), p=weights / weights.sum()))]


@dataclass
class Clip:
    """One generated riff: what was played, and the audio of it."""

    audio: np.ndarray
    sample_rate: int
    stroke_times: list[float]
    roots: list[int]
    shapes: list[str]
    settings: dict = field(default_factory=dict)


def write_riff(rng: np.random.Generator, tuning, bars: int = 4, max_fret: int = 22):
    """Invent a riff: a list of (beat, root, shape, positions, duration)."""
    grid = float(rng.choice([0.25, 0.5], p=[0.6, 0.4]))  # 16ths or 8ths
    # Most riffs live near the lowest string, but a third of them sit higher up, so the model
    # sees leads and mid-neck chords too rather than only chugging.
    reach = 14 if rng.random() < 0.7 else 26
    low, high = tuning.strings[0], min(tuning.strings[0] + reach, ROOT_HIGH)
    home = int(rng.integers(low, max(low + 1, high - 5)))  # riffs mostly circle a home root
    strokes = []
    beat = 0.0
    while beat < bars * 4:
        if rng.random() < 0.15:  # rests, so the model sees strokes in isolation as well as in runs
            beat += grid
            continue
        root = home if rng.random() < 0.45 else int(np.clip(home + rng.integers(-2, 8), ROOT_LOW, high))
        shape = _weighted(rng, SHAPE_WEIGHTS)
        positions = positions_for(tuning, root, shape, max_fret=max_fret)
        if positions is None:
            positions = positions_for(tuning, root, "single", max_fret=max_fret)
            shape = "single"
        if positions is not None:
            held = grid if rng.random() < 0.8 else grid * float(rng.choice([2, 4]))
            strokes.append((beat, root, shape, positions, held * 0.95))
        beat += grid
    return strokes


def render_clip(rng: np.random.Generator, *, bars: int = 4) -> Clip | None:
    tuning_name = _weighted(rng, TUNINGS)
    tuning = parse_tuning(tuning_name)
    instrument = _weighted(rng, INSTRUMENTS)
    bpm = float(rng.integers(70, 190))
    strokes = write_riff(rng, tuning, bars=bars)
    if len(strokes) < 4:
        return None

    # Played by a human: rushing or dragging across the take, and each stroke slightly off.
    ramp = float(rng.normal(0, 0.05))
    jitter_s = abs(float(rng.normal(0, 0.012)))
    total_beats = max(beat for beat, *_ in strokes) or 1.0
    played, events = [], []
    for i, (beat, _root, _shape, positions, held) in enumerate(strokes):
        fraction = beat / total_beats
        warped = beat * (1 + ramp * (fraction - 0.5)) + rng.normal(0, jitter_s) * bpm / 60
        warped = max(0.0, float(warped))
        played.append(warped)
        pitches = [tuning.strings[s] + f for s, f in positions]
        notes = [Note(warped, held, p) for p in pitches]
        events.append(TabEvent(warped, notes, [Position(s, f) for s, f in positions], palm_mute=rng.random() < 0.4))

    drive = float(rng.uniform(0.0, 1.0)) if instrument in ("distortion", "overdrive", "muted") else float(rng.uniform(0, 0.2))
    audio = synthesize(
        events, tuning, instrument=instrument, bpm=bpm, sample_rate=SAMPLE_RATE,
        drive=drive, tone=float(rng.uniform(0.2, 0.8)),
    ).mean(axis=1)

    high_pass = float(rng.choice([0, 0, 60, 80, 95, 110]))  # some recordings simply have no low end
    if high_pass:
        spectrum = np.fft.rfft(audio)
        spectrum[np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE) < high_pass] *= float(rng.uniform(0.02, 0.15))
        audio = np.fft.irfft(spectrum, len(audio)).astype(np.float32)

    level = float(np.sqrt((audio**2).mean())) or 1e-6
    audio = audio + rng.normal(0, level * 10 ** (rng.uniform(-60, -30) / 20), len(audio)).astype(np.float32)
    audio = (audio / (np.abs(audio).max() + 1e-9) * float(rng.uniform(0.2, 0.95))).astype(np.float32)

    seconds_per_beat = 60.0 / bpm
    return Clip(
        audio, SAMPLE_RATE,
        [t * seconds_per_beat for t in played],
        [root for _beat, root, *_ in strokes],
        [shape for _beat, _root, shape, *_ in strokes],
        {"tuning": tuning_name, "instrument": instrument, "bpm": bpm, "drive": round(drive, 2), "high_pass": high_pass},
    )


def clip_examples(clip: Clip, rng: np.random.Generator, *, detection_error_s: float = 0.025):
    """Cut one patch per stroke, at a slightly wrong time - as a detector would find it."""
    times = [max(0.0, t + float(rng.normal(0, detection_error_s))) for t in clip.stroke_times]
    patches = patches_at(clip.audio, times, clip.sample_rate)
    roots = np.array([root_index(r) for r in clip.roots], dtype=np.int16)
    shapes = np.array([shape_index(s) for s in clip.shapes], dtype=np.int8)
    keep = (roots >= 0) & (roots < ROOT_HIGH - ROOT_LOW + 1)
    return patches[keep].astype(np.float16), roots[keep], shapes[keep]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=int, default=200, help="how many riffs to render (default: 200)")
    parser.add_argument("--bars", type=int, default=4, help="bars per riff (default: 4)")
    parser.add_argument("--out", default="data/strokes", help="where to write the dataset")
    parser.add_argument("--seed", type=int, default=0, help="change this for a second, independent set")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_patches, all_roots, all_shapes, settings = [], [], [], []
    started = time.time()
    for i in range(args.clips):
        clip = render_clip(rng, bars=args.bars)
        if clip is None:
            continue
        patches, roots, shapes = clip_examples(clip, rng)
        if len(patches) == 0:
            continue
        all_patches.append(patches)
        all_roots.append(roots)
        all_shapes.append(shapes)
        settings.append(clip.settings)
        if (i + 1) % 10 == 0:
            done = sum(len(p) for p in all_patches)
            rate = (i + 1) / (time.time() - started)
            print(f"  {i + 1}/{args.clips} clips, {done} strokes ({rate:.1f} clips/s)", file=sys.stderr)

    patches = np.concatenate(all_patches)
    roots = np.concatenate(all_roots)
    shapes = np.concatenate(all_shapes)
    np.savez_compressed(out / "dataset.npz", patches=patches, roots=roots, shapes=shapes)
    (out / "settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")

    size_mb = (out / "dataset.npz").stat().st_size / 1e6
    print(f"\n{len(patches)} strokes from {len(settings)} clips in {time.time() - started:.0f}s -> {out} ({size_mb:.1f} MB)")
    print(f"patch shape: {patches.shape[1:]} (bins x frames), expected ({N_BINS}, {FRAMES})")
    counts = np.bincount(shapes, minlength=len(SHAPES))
    print("shapes: " + ", ".join(f"{name} {count}" for name, count in zip(SHAPES, counts)))
    print(f"roots: {roots.min() + ROOT_LOW} to {roots.max() + ROOT_LOW} (MIDI), {len(set(roots.tolist()))} distinct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
