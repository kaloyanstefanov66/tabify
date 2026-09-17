import numpy as np
import pytest

from tabify import TabifyError
from tabify.fretting import assign_frets
from tabify.notes import Note, name_to_midi
from tabify.synth import DEFAULT_DRIVE, RENDER_HELP, _distort, render_audio, resolve_soundfont, schedule
from tabify.tuning import parse_tuning

STANDARD = parse_tuning("standard")


def test_schedule_produces_matching_on_off_pairs_at_the_right_time():
    events = assign_frets([Note(0, 2, name_to_midi("E2"))], STANDARD).events
    sched = schedule(events, STANDARD, capo=0, bpm=120)
    assert sched == [(0.0, True, name_to_midi("E2")), (1.0, False, name_to_midi("E2"))]


def test_schedule_applies_capo_to_sounding_pitch():
    events = assign_frets([Note(0, 1, name_to_midi("E2"))], STANDARD).events
    sched = schedule(events, STANDARD, capo=2, bpm=120)
    assert sched[0][2] == name_to_midi("E2") + 2


def test_schedule_orders_note_off_before_note_on_at_a_tie():
    # Two adjacent notes on the same string: the first note's "off" and the second's "on" land
    # at the same instant, and the off must come first so the note actually retriggers.
    events = assign_frets([Note(0, 1, name_to_midi("E2")), Note(1, 1, name_to_midi("F2"))], STANDARD).events
    sched = schedule(events, STANDARD, capo=0, bpm=60)
    at_the_tie = [ev for ev in sched if ev[0] == 1.0]
    assert [ev[1] for ev in at_the_tie] == [False, True]


def test_chord_produces_simultaneous_events():
    events = assign_frets([Note(0, 1, name_to_midi("E2")), Note(0, 1, name_to_midi("B2"))], STANDARD).events
    sched = schedule(events, STANDARD, capo=0, bpm=60)
    on_times = {t for t, is_on, _ in sched if is_on}
    assert on_times == {0.0}


def test_unknown_instrument_is_rejected_before_touching_fluidsynth():
    with pytest.raises(TabifyError, match="unknown instrument"):
        render_audio([], STANDARD, "out.wav", instrument="kazoo")


def test_missing_fluidsynth_gives_actionable_error(monkeypatch):
    # Simulate FluidSynth not being installed, regardless of whether this
    # particular machine happens to have it.
    monkeypatch.setattr("tabify.synth.importlib.util.find_spec", lambda name: None)
    with pytest.raises(TabifyError, match="FluidSynth"):
        render_audio([], STANDARD, "out.wav", instrument="steel")
    assert "pip install" in RENDER_HELP


def test_resolve_soundfont_missing_explicit_path():
    with pytest.raises(TabifyError, match="not found"):
        resolve_soundfont("does-not-exist.sf2")


def test_resolve_soundfont_uses_explicit_path(tmp_path):
    sf2 = tmp_path / "my.sf2"
    sf2.write_bytes(b"fake soundfont data")
    assert resolve_soundfont(str(sf2)) == sf2


def _pure_tone(freq_hz: float, seconds: float, sample_rate: int = 44100) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    return np.stack([tone, tone], axis=1)


def _spectrum(audio: np.ndarray, sample_rate: int = 44100):
    mono = audio.mean(axis=1)
    mags = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)
    return freqs, mags


def test_zero_drive_leaves_audio_untouched():
    audio = _pure_tone(220.0, 0.5)
    assert np.array_equal(_distort(audio, 44100, drive=0.0), audio)


def test_distortion_adds_odd_harmonics_to_a_pure_tone():
    # A clean sine has essentially all its energy at the fundamental. tanh is an odd (symmetric)
    # function, so soft-clipping a sine generates *odd* harmonics (3rd, 5th, ...) at real,
    # substantial energy - the measurable signature of "this distorts", not just gain - while
    # even harmonics (2nd, 4th) stay near the noise floor, same as an unclipped signal.
    fundamental = 220.0
    clean = _pure_tone(fundamental, 0.5)
    distorted = _distort(clean, 44100, drive=0.8, tone=0.3)

    freqs, clean_mags = _spectrum(clean)
    _, distorted_mags = _spectrum(distorted)

    def energy_near(freqs, mags, target_hz, width=15.0):
        return float(mags[(freqs > target_hz - width) & (freqs < target_hz + width)].sum())

    fundamental_energy = energy_near(freqs, clean_mags, fundamental)
    for harmonic in (3, 5):  # odd harmonics: real, substantial energy after distortion
        assert energy_near(freqs, distorted_mags, fundamental * harmonic) > fundamental_energy * 0.05
    for harmonic in (2, 4):  # even harmonics: stay near the noise floor, same as the clean signal
        assert energy_near(freqs, distorted_mags, fundamental * harmonic) < fundamental_energy * 1e-4


def test_more_drive_reduces_crest_factor():
    # Distortion compresses/squashes a waveform - its peak-to-RMS ratio (crest factor) should
    # drop as drive increases, the same way a squarer, more-clipped wave reads on a meter.
    def crest_factor(audio):
        mono = audio.mean(axis=1)
        return np.abs(mono).max() / np.sqrt((mono**2).mean())

    clean = _pure_tone(220.0, 0.5)
    light = _distort(clean, 44100, drive=0.2, tone=0.5)
    heavy = _distort(clean, 44100, drive=0.9, tone=0.5)
    assert crest_factor(heavy) < crest_factor(light) < crest_factor(clean)


def test_distortion_does_not_clip_on_export():
    loud = _pure_tone(220.0, 0.5) * 2.0  # deliberately over unity, like a hot chord
    distorted = _distort(loud, 44100, drive=1.0, tone=0.5)
    assert np.abs(distorted).max() <= np.abs(loud).max()


def test_default_drive_table_only_covers_real_instruments():
    from tabify.instruments import ALL_PROGRAMS

    assert set(DEFAULT_DRIVE).issubset(ALL_PROGRAMS)
    assert all(0 <= d <= 1 for d in DEFAULT_DRIVE.values())


def test_palm_muted_strokes_are_cut_short_in_the_render():
    from tabify.fretting import Position, TabEvent
    from tabify.synth import PALM_MUTE_SECONDS

    held = TabEvent(0.0, [Note(0.0, 1.0, name_to_midi("E2"))], [Position(0, 0)])
    muted = TabEvent(0.0, [Note(0.0, 1.0, name_to_midi("E2"))], [Position(0, 0)], palm_mute=True)
    assert schedule([held], STANDARD, capo=0, bpm=60)[-1][0] == 1.0
    assert schedule([muted], STANDARD, capo=0, bpm=60)[-1][0] == pytest.approx(PALM_MUTE_SECONDS)
