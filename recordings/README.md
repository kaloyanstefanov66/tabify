# Real recordings

Drop your own playing here. Synthetic data teaches the model what a soundfont sounds like;
real playing through a real amp in a real room is the only thing that teaches it guitar.
A few minutes of this is worth more than hours of generated audio.

Audio files in this folder are ignored by git - they're yours, not the repo's.

## What to record

- **One riff per file**, isolated guitar only: no drums, bass or backing track.
- **Play it through several times in the same take.** Every pick attack is a training
  example, so a 30-second take of a repeated riff is worth far more than one pass.
- **Play it how you play it.** Don't fight for a click - tabify follows a tempo that
  drifts, and that's measured, not hoped for.
- `.wav` or `.flac` preferred, `.mp3` fine. Any sample rate.

## What to send with it

For each recording, a text file with the same name (`riff1.wav` -> `riff1.txt`):

```
tuning: drop-c
tone: distorted        # or: clean, crunch - roughly how it was recorded
notes: played the last repeat sloppy      # anything worth knowing, optional

e|--------------------------------|
B|--------------------------------|
G|--------------------------------|
D|--------------------------------|
A|-0-0-0-----3-3-3-----0-0-0------|
C|-0-0-0-----3-3-3-----0-0-0------|
```

Plain ASCII tab is ideal - exactly what you'd paste into a forum. Bar lines and spacing
don't have to be exact, since the stroke times come from the audio; what matters is the
**order of the strokes and which frets are in each one**. A Guitar Pro or MusicXML export
works too, if you have one.

## What's most useful

Variety beats volume, in roughly this order:

1. **Palm-muted chugs** on the low string - the case tabify gets wrong most often.
2. **Open power chords** and moving power chords.
3. **Single-note lines** and octaves.
4. **A couple of clean (undistorted) takes**, so the model doesn't only learn distortion.
5. **The same riff at two different tunings**, if you can - that directly tests the
   tuning detection.
