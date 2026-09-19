"""Running the pitch model directly, without the package that normally wraps it.

tabify hears notes with Spotify's Basic Pitch (https://github.com/spotify/basic-pitch),
Apache-2.0, Copyright 2022 Spotify AB. The model itself is a 0.2 MB ONNX file, vendored
beside this module together with its licence and notice.

The basic-pitch package is not a dependency, because on Python 3.11 and newer it requires
TensorFlow: 1.3 GB of download that tabify never loads, since it runs the ONNX copy of the
same model. Everything that package did for us is reimplemented here in numpy - framing the
audio the way the model expects, and turning its frame-wise output back into note events -
which is the whole of what tabify used. Pitch bends, MIDI writing and sonification are not
reimplemented; tabify never used them.

Predictions are identical, and a test checks that note for note against the real thing
whenever it happens to be installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL = Path(__file__).parent / "models" / "nmp.onnx"

# The model's own geometry: it takes just under two seconds of 22.05 kHz mono and returns
# 172 frames of 88 piano keys.
SAMPLE_RATE = 22050
FFT_HOP = 256
FRAMES_PER_SECOND = SAMPLE_RATE // FFT_HOP  # 86
FRAMES_PER_WINDOW = FRAMES_PER_SECOND * 2  # 172
WINDOW_SAMPLES = SAMPLE_RATE * 2 - FFT_HOP  # 43844
# Windows overlap so a note near a window edge is still seen whole by one of them. Half the
# overlap is trimmed from each end of every window's output before they are joined up.
OVERLAP_FRAMES = 30
OVERLAP_SAMPLES = OVERLAP_FRAMES * FFT_HOP
HOP_SAMPLES = WINDOW_SAMPLES - OVERLAP_SAMPLES
MIDI_OFFSET = 21  # the model's lowest bin is A0
MAX_FREQ_IDX = 87
# How many frames a note may spend below the frame threshold before it is declared over.
ENERGY_TOLERANCE = 11


def _session():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # onnxruntime is chatty about things nobody can act on
    return ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])


def _windows(audio: np.ndarray) -> np.ndarray:
    """Cut the audio into the overlapping fixed-length windows the model expects."""
    padded = np.concatenate([np.zeros(OVERLAP_SAMPLES // 2, dtype=np.float32), audio])
    windows = []
    for start in range(0, len(padded), HOP_SAMPLES):
        window = padded[start : start + WINDOW_SAMPLES]
        if len(window) < WINDOW_SAMPLES:
            window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))
        windows.append(window)
    return np.asarray(windows, dtype=np.float32)[:, :, np.newaxis]


def _unwrap(stacked: np.ndarray, original_samples: int) -> np.ndarray:
    """Join the per-window outputs into one timeline, dropping the overlapping edges."""
    edge = OVERLAP_FRAMES // 2
    trimmed = stacked[:, edge:-edge, :]
    joined = trimmed.reshape(-1, trimmed.shape[2])
    keep = int(np.floor(original_samples * FRAMES_PER_SECOND / SAMPLE_RATE))
    return joined[:keep]


def frame_times(n_frames: int) -> np.ndarray:
    """When each output frame happened, in seconds.

    Windows are trimmed after inference but still numbered as though they were not, which
    leaves a small constant slip per window. Correcting for it is what lines the notes up
    with the audio; the constant is carried over from the original implementation.
    """
    times = np.arange(n_frames) * FFT_HOP / SAMPLE_RATE
    window_number = np.floor(np.arange(n_frames) / FRAMES_PER_WINDOW)
    slip = (FFT_HOP / SAMPLE_RATE) * (FRAMES_PER_WINDOW - WINDOW_SAMPLES / FFT_HOP) + 0.0018
    return times - slip * window_number


def _infer_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    """Add onsets where the frame activations jump, not only where the model says onset."""
    diffs = []
    for n in range(1, n_diff + 1):
        padded = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(padded[n:, :] - padded[:-n, :])
    jump = np.min(diffs, axis=0)
    jump[jump < 0] = 0
    jump[:n_diff, :] = 0
    peak = jump.max()
    if peak:
        jump = onsets.max() * jump / peak
    return np.maximum(onsets, jump)


def _constrain(onsets: np.ndarray, frames: np.ndarray, lowest_hz, highest_hz):
    """Silence the bins outside the range of the instrument being listened for."""
    import librosa

    if highest_hz is not None:
        top = int(np.round(librosa.hz_to_midi(highest_hz) - MIDI_OFFSET))
        onsets[:, top:] = 0
        frames[:, top:] = 0
    if lowest_hz is not None:
        bottom = int(np.round(librosa.hz_to_midi(lowest_hz) - MIDI_OFFSET))
        onsets[:, :bottom] = 0
        frames[:, :bottom] = 0
    return onsets, frames


def _clear(energy: np.ndarray, lo: int, hi: int, bin_index: int) -> None:
    """Spend the energy a note used, and its neighbouring bins, so nothing claims it twice."""
    energy[lo:hi, bin_index] = 0
    if bin_index < MAX_FREQ_IDX:
        energy[lo:hi, bin_index + 1] = 0
    if bin_index > 0:
        energy[lo:hi, bin_index - 1] = 0


def notes_from_output(
    frames: np.ndarray,
    onsets: np.ndarray,
    *,
    onset_threshold: float,
    frame_threshold: float,
    min_note_frames: int,
    lowest_hz: float | None = None,
    highest_hz: float | None = None,
    infer_onsets: bool = True,
    melodia: bool = True,
) -> list[tuple[int, int, int, float]]:
    """Turn frame-wise model output into (start frame, end frame, midi, amplitude) events."""
    import scipy.signal

    onsets, frames = _constrain(onsets.copy(), frames.copy(), lowest_hz, highest_hz)
    if infer_onsets:
        onsets = _infer_onsets(onsets, frames)

    peaks_only = np.zeros_like(onsets)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peaks_only[peaks] = onsets[peaks]
    times, bins = np.where(peaks_only >= onset_threshold)

    n_frames = frames.shape[0]
    energy = frames.copy()
    events: list[tuple[int, int, int, float]] = []

    # Walked backwards in time, so that where two onsets compete for the same energy the
    # later note is not swallowed by an earlier one still ringing through it.
    for start, bin_index in zip(times[::-1], bins[::-1]):
        if start >= n_frames - 1:
            continue
        end, quiet = start + 1, 0
        while end < n_frames - 1 and quiet < ENERGY_TOLERANCE:
            quiet = quiet + 1 if energy[end, bin_index] < frame_threshold else 0
            end += 1
        end -= quiet
        if end - start <= min_note_frames:
            continue
        _clear(energy, start, end, bin_index)
        events.append(
            (int(start), int(end), int(bin_index) + MIDI_OFFSET, float(frames[start:end, bin_index].mean()))
        )

    if melodia:
        # Energy that no onset claimed is still a note someone played - typically one tied
        # into, or one whose attack the model missed. Those grow out from their loudest frame.
        while energy.max() > frame_threshold:
            middle, bin_index = np.unravel_index(np.argmax(energy), energy.shape)
            energy[middle, bin_index] = 0

            end, quiet = middle + 1, 0
            while end < n_frames - 1 and quiet < ENERGY_TOLERANCE:
                quiet = quiet + 1 if energy[end, bin_index] < frame_threshold else 0
                _clear(energy, end, end + 1, bin_index)
                end += 1
            end = end - 1 - quiet

            start, quiet = middle - 1, 0
            while start > 0 and quiet < ENERGY_TOLERANCE:
                quiet = quiet + 1 if energy[start, bin_index] < frame_threshold else 0
                _clear(energy, start, start + 1, bin_index)
                start -= 1
            start = start + 1 + quiet

            if end - start <= min_note_frames:
                continue
            events.append(
                (int(start), int(end), int(bin_index) + MIDI_OFFSET, float(frames[start:end, bin_index].mean()))
            )

    return events


def predict(audio: np.ndarray, *, batch: int = 8) -> dict[str, np.ndarray]:
    """Run the model over mono 22.05 kHz audio, returning its note/onset/contour timelines."""
    session = _session()
    name = session.get_inputs()[0].name
    outputs = [o.name for o in session.get_outputs()]
    windows = _windows(audio)
    chunks = [session.run(None, {name: windows[i : i + batch]}) for i in range(0, len(windows), batch)]
    stacked = {
        output: np.concatenate([chunk[k] for chunk in chunks], axis=0)
        for k, output in enumerate(outputs)
    }
    # Which output is which was settled by comparing tensors against the reference
    # implementation, where they matched exactly. Naming them by position is what made them
    # silently swap once already, so they are picked out by name and width instead, and a
    # model that no longer matches this shape fails here rather than transcribing nonsense.
    named = _label_outputs(stacked)
    return {kind: _unwrap(array, len(audio)) for kind, array in named.items()}


def _label_outputs(stacked: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Work out which model output is the note, onset and contour timeline."""
    contour = [n for n, a in stacked.items() if a.shape[2] == 88 * 3]
    keys = sorted(n for n, a in stacked.items() if a.shape[2] == 88)
    if len(contour) != 1 or len(keys) != 2:
        raise RuntimeError(f"unexpected pitch model outputs: { {n: a.shape for n, a in stacked.items()} }")
    # Of the two 88-wide outputs the graph names the note timeline before the onset one.
    note, onset = keys
    return {"note": stacked[note], "onset": stacked[onset], "contour": stacked[contour[0]]}
