"""Interactive play-along: the tab scrolls and highlights in sync with audio playback.

Needs `sounddevice` for real-time audio output (`pip install "tabify-cli[play]"`).
Two audio sources are supported, matching what --source picks: the original
recording, or a live FluidSynth render of the transcription (see `tabify.synth`).
Either way, playback position is tracked with a plain wall-clock timer against
the moment audio started, which is simple and accurate as long as the audio
device doesn't glitch - there's no feedback loop reading the sound card's own
clock.
"""

from __future__ import annotations

import importlib.util
import sys
import time

from tabify import TabifyError, term
from tabify.fretting import TabEvent
from tabify.render import TabLayout, build_layout, format_system, system_for_step
from tabify.rhythm import TimeSignature
from tabify.tuning import Tuning

PLAY_HELP = 'play-along mode needs sounddevice for audio output:\n  pip install "tabify-cli[play]"'

# Non-blocking single-keypress reading differs by OS; POSIX needs the terminal
# put into "raw" mode first, restored on exit. Both implementations normalize
# arrow keys to the strings "LEFT"/"RIGHT" so the player loop stays OS-agnostic.
if sys.platform == "win32":
    import msvcrt

    class KeyReader:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> str | None:
            if not msvcrt.kbhit():
                return None
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # extended-key prefix; the real code follows immediately
                return {"K": "LEFT", "M": "RIGHT"}.get(msvcrt.getwch())
            return ch

else:
    import select
    import termios
    import tty

    class KeyReader:
        def __enter__(self):
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            return self

        def __exit__(self, *exc):
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            return False

        def _ready(self, timeout: float = 0.0) -> bool:
            return bool(select.select([sys.stdin], [], [], timeout)[0])

        def read(self) -> str | None:
            if not self._ready():
                return None
            ch = sys.stdin.read(1)
            if ch != "\x1b":
                return ch
            if self._ready(0.01) and sys.stdin.read(1) == "[" and self._ready(0.01):
                return {"C": "RIGHT", "D": "LEFT"}.get(sys.stdin.read(1))
            return "ESC"


def step_at(elapsed_seconds: float, bpm: float, subdivision: int) -> float:
    """The (fractional) grid-step position `elapsed_seconds` into playback."""
    return elapsed_seconds * bpm / 60.0 * subdivision


def load_audio_file(path):
    """Load an audio file into a (frames, 2) float32 array, for playing the original recording."""
    if importlib.util.find_spec("soundfile") is None:
        raise TabifyError(PLAY_HELP)
    import numpy as np
    import soundfile as sf

    try:
        data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception as exc:  # libsndfile raises a plain RuntimeError on an unsupported/corrupt file
        raise TabifyError(f"could not load {path} for playback: {exc}") from exc
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return data, sample_rate


def frame_text(lines: list[str]) -> str:
    """One redraw: overwrite in place from the top-left, clearing leftovers, instead of clearing the
    screen - a full clear on every frame flickers, and some terminals push each one into scrollback."""
    return term.HOME + "".join(line + term.CLEAR_LINE_END + "\n" for line in lines) + term.CLEAR_SCREEN_END


def _draw(layout: TabLayout, header: list[str], step: float, color: bool, status: str) -> None:
    system_index = system_for_step(layout, step)
    lines = [*header, "", *format_system(layout, system_index, color=color, highlight_step=step), "", status]
    sys.stdout.write(frame_text(lines))
    sys.stdout.flush()


def play_along(
    events: list[TabEvent],
    tuning: Tuning,
    audio,  # (frames, channels) float array
    sample_rate: int,
    *,
    bpm: float,
    time_sig: TimeSignature = TimeSignature(),
    subdivision: int = 4,
    capo: int = 0,
    title: str | None = None,
    width: int = 80,
    color: bool = False,
    seek_seconds: float = 5.0,
) -> None:
    if importlib.util.find_spec("sounddevice") is None:
        raise TabifyError(PLAY_HELP)
    import sounddevice as sd

    layout = build_layout(events, tuning, time_sig=time_sig, subdivision=subdivision, width=width)
    if layout is None:
        raise TabifyError("nothing to play - no playable notes were found")

    header = []
    if title:
        header += [title, "=" * len(title)]
    header.append(f"Tuning: {tuning.describe()}" + (f" ({tuning.name})" if tuning.name != "custom" else ""))
    if capo:
        header.append(f"Capo: fret {capo}")
    header.append(f"Time: {time_sig}   Tempo: {round(bpm)} BPM")

    duration = len(audio) / sample_rate
    position = 0.0  # seconds into the track - authoritative while paused
    playing = False
    started_at = 0.0  # perf_counter() timestamp corresponding to track position 0, while playing

    def elapsed() -> float:
        return time.perf_counter() - started_at if playing else position

    def start(from_seconds: float) -> None:
        nonlocal started_at, position, playing
        position = max(0.0, min(from_seconds, duration))
        sd.play(audio[round(position * sample_rate):], sample_rate)
        started_at = time.perf_counter() - position
        playing = True

    def pause() -> None:
        nonlocal position, playing
        position = elapsed()
        sd.stop()
        playing = False

    term.enable_ansi()
    # Play on the alternate screen, so quitting puts the terminal back exactly as it was
    # instead of leaving a wall of old frames behind.
    sys.stdout.write(term.ALT_SCREEN_ON + term.CURSOR_HIDE)
    sys.stdout.flush()
    try:
        with KeyReader() as keys:
            start(0.0)
            while True:
                now = min(elapsed(), duration)
                if playing and now >= duration:
                    pause()
                    position = duration
                    now = duration

                mm, ss = divmod(int(now), 60)
                tmm, tss = divmod(int(duration), 60)
                status = (
                    f"[{'playing' if playing else 'paused '}] {mm:02d}:{ss:02d} / {tmm:02d}:{tss:02d}   "
                    f"space=pause/resume  <-/->=seek {seek_seconds:g}s  q=quit"
                )
                _draw(layout, header, step_at(now, bpm, subdivision), color, status)

                key = keys.read()
                if key in ("q", "Q", "\x03", "ESC"):
                    break
                if key == " ":
                    pause() if playing else start(position)
                elif key in ("LEFT", "RIGHT"):
                    target = max(0.0, min(now + (seek_seconds if key == "RIGHT" else -seek_seconds), duration))
                    if playing:
                        start(target)
                    else:
                        position = target

                time.sleep(0.05)
    finally:
        sd.stop()
        sys.stdout.write(term.CURSOR_SHOW + term.ALT_SCREEN_OFF)
        sys.stdout.flush()
