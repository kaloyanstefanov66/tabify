# Contributing

Thanks for helping make tabify better!

## Setup

```bash
git clone https://github.com/OWNER/tabify
cd tabify
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,audio]"
pytest
```

## Where things live

| File | Responsibility |
| --- | --- |
| `src/tabify/cli.py` | Argument parsing and the overall pipeline |
| `src/tabify/transcribe.py` | Audio → notes (basic-pitch / pYIN) and beat tracking |
| `src/tabify/fetch.py` | Downloading audio from a link (yt-dlp) or a direct file URL |
| `src/tabify/evaluate.py` | Scoring a transcription against a known tab (see `benchmarks/`) |
| `src/tabify/refine.py` | Cleanup of raw model notes against the audio: pick attacks, merged chugs, missing low roots |
| `src/tabify/techniques.py` | Playing techniques inferred from the tab (palm mutes) |
| `src/tabify/midi_io.py` | MIDI reading and writing |
| `src/tabify/rhythm.py` | Quantization, time signatures, bar alignment |
| `src/tabify/fretting.py` | Picking a string/fret for each note (the Viterbi search) |
| `src/tabify/render.py` | ASCII tab output |

## Good first contributions

- A recording that transcribes badly makes a great bug report. Attach a short clip and the command you ran.
- New tuning presets in `tuning.py`
- Tweaking the cost weights in `FretOptions`, with a test that shows the better fingering
