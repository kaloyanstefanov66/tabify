import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("scipy")

from tabify.refine import (  # noqa: E402
    TimedNote,
    _hz,
    _Spectra,
    cluster_starts,
    detect_onsets,
    low_end_floor,
    overtone_evidence,
    collapse_to_roots,
    snap_to_onsets,
    split_restrikes,
)

SR = 22050
C2, G2, C3, GS2, DS3 = 36, 43, 48, 44, 51


def note(start, end, pitch, velocity=90):
    return TimedNote(start, end, pitch, velocity)


def guitar(pitches, seconds=0.4, drive=6.0, highpass=None):
    """Distorted, optionally high-passed string tones - like an isolated stem with its low end cut."""
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(0)
    x = sum(
        (1 / k) * np.sin(2 * np.pi * _hz(p) * k * t + rng.uniform(0, 6.28))
        for p in pitches for k in range(1, 25) if _hz(p) * k < SR / 2
    )
    x = np.tanh(drive * x / np.abs(x).max())
    if highpass:
        spec = np.fft.rfft(x)
        spec[np.fft.rfftfreq(len(x), 1 / SR) < highpass] *= 0.03
        x = np.fft.irfft(spec, len(x))
    return x.astype(np.float32)


# --- attacks and chord stacking -------------------------------------------------------


def test_cluster_starts_merges_a_staggered_chord_but_not_the_next_stroke():
    notes = cluster_starts([note(1.00, 2, C2), note(1.03, 2, G2), note(1.05, 2, C3), note(1.20, 2, C2)])
    assert [n.start for n in notes] == [1.00, 1.00, 1.00, 1.20]


def test_snap_moves_late_model_onsets_back_to_the_pick_attack():
    # The pitch model reported the chord's notes 20-90 ms late, each by a different amount.
    notes, moved = snap_to_onsets([note(1.02, 1.5, G2), note(1.09, 1.5, C3), note(3.0, 3.4, C2)], onsets=[1.0, 2.0])
    assert [n.start for n in notes] == [1.0, 1.0, 3.0]  # the note with no nearby attack is left alone
    assert moved == 2


def test_split_restrikes_breaks_up_merged_chugs():
    # One long G2 over four attacks where nothing else starts: the model merged four chugs.
    notes, splits = split_restrikes([note(0.0, 1.0, G2)], onsets=[0.0, 0.25, 0.5, 0.75])
    assert [(n.start, n.end) for n in notes] == [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    assert splits == 3


def test_split_restrikes_splits_a_string_whose_chord_mate_was_re_struck():
    notes, _ = split_restrikes([note(0.0, 1.0, G2), note(0.0, 0.4, C3), note(0.5, 0.9, C3)], onsets=[0.0, 0.5])
    assert sorted((n.start, n.pitch) for n in notes) == [(0.0, G2), (0.0, C3), (0.5, G2), (0.5, C3)]


def test_split_restrikes_leaves_a_ringing_note_alone_when_a_different_note_is_picked():
    notes, splits = split_restrikes([note(0.0, 1.0, G2), note(0.5, 0.9, 64)], onsets=[0.0, 0.5])
    assert splits == 0
    assert (0.0, 1.0) in [(n.start, n.end) for n in notes]


def test_detect_onsets_finds_each_distorted_chug():
    truth = [0.2, 0.45, 0.7, 0.95, 1.2, 1.45, 1.7, 1.95]
    y = np.zeros(int(2.4 * SR), dtype=np.float32)
    n = int(0.2 * SR)
    # Decay, then a short fade: chopping a tone off abruptly makes a click that is a real attack.
    envelope = np.exp(-np.arange(n) / (0.06 * SR)) * np.minimum(1.0, np.linspace(n / (0.01 * SR), 0, n))
    for i, t in enumerate(truth):
        hit = guitar([C2, G2], seconds=0.2) * envelope * (1.0 if i % 3 else 0.35)  # some chugs much quieter
        y[int(t * SR) : int(t * SR) + n] += hit
    found = detect_onsets(y, SR)
    assert len(found) == len(truth)
    assert np.max(np.abs(np.asarray(found) - truth)) < 0.03


# --- missing low roots ------------------------------------------------------------------


def test_low_end_floor_sees_a_cut_low_end():
    full = np.concatenate([guitar([C2, G2], seconds=1.0), guitar([C2], seconds=1.0)])
    cut = np.concatenate([guitar([C2, G2], seconds=1.0, highpass=90), guitar([C2], seconds=1.0, highpass=90)])
    assert low_end_floor(full, SR) < _hz(C2)  # the low C is really there
    assert low_end_floor(cut, SR) > _hz(C2)  # the low C is gone, only its overtones remain


@pytest.mark.parametrize(
    "played, heard, candidate, root_really_played",
    [
        ([C2, G2], [G2], C2, True),  # power-chord chug, the model only heard the fifth
        ([C2], [C3], C2, True),  # low C alone, the model heard its octave overtone
        ([G2], [G2], C2, False),  # a genuine single G2
        ([C3], [C3], C2, False),  # a genuine single C3
    ],
)
def test_overtone_evidence_separates_missing_roots_from_real_single_notes(played, heard, candidate, root_really_played):
    evidence = overtone_evidence(_Spectra(guitar(played, highpass=90), SR), 0.015, 0.25, candidate, heard)
    if root_really_played:
        assert evidence >= 0.35
    else:
        assert evidence < 0.1


def test_restores_the_root_of_a_fourth_dyad_below_the_recordings_floor():
    y = guitar([C2, G2, C3], highpass=90)
    notes, added, _ = collapse_to_roots([note(0, 0.4, G2), note(0, 0.4, C3)], y, SR, lowest=C2, floor_hz=89)
    assert added == 1
    assert sorted(n.pitch for n in notes) == [C2, G2, C3]


def test_does_not_invent_a_root_the_recording_could_have_carried():
    y = guitar([G2, C3])
    _, added, _ = collapse_to_roots([note(0, 0.4, G2), note(0, 0.4, C3)], y, SR, lowest=C2, floor_hz=50)
    assert added == 0


def test_does_not_add_roots_to_a_fifth_or_below_the_tunings_lowest_string():
    y = guitar([GS2, DS3], highpass=90)
    _, added, _ = collapse_to_roots([note(0, 0.4, GS2), note(0, 0.4, DS3)], y, SR, lowest=C2, floor_hz=89)
    assert added == 0  # G#2 + D#3 is already root and fifth
    y = guitar([C2, G2, C3], highpass=90)
    _, added, _ = collapse_to_roots([note(0, 0.4, G2), note(0, 0.4, C3)], y, SR, lowest=40, floor_hz=89)
    assert added == 0  # standard tuning's lowest string is E2, so a C2 root isn't playable


def test_restores_a_single_notes_root_only_with_overtone_evidence():
    chug = guitar([C2, G2], highpass=90)
    notes, added, _ = collapse_to_roots([note(0, 0.4, G2)], chug, SR, lowest=C2, floor_hz=89)
    assert added == 1 and sorted(n.pitch for n in notes) == [C2, G2]
    lone_g = guitar([G2], highpass=90)
    _, added, _ = collapse_to_roots([note(0, 0.4, G2)], lone_g, SR, lowest=C2, floor_hz=89)
    assert added == 0


def test_collapses_a_stroke_heard_only_as_overtones_back_to_its_root():
    # A lone drop-C chug reaches the pitch model as C3 + G3 (its 2nd and 3rd harmonics) with
    # the root itself missing. One note was played, so one note should come out: the root
    # comes back and both overtones go. The octave only survives when a fifth came with it,
    # which is what tells a power chord's top string from the root's own 2nd harmonic.
    y = guitar([C2], highpass=90)
    notes, added, dropped = collapse_to_roots(
        [note(0, 0.4, C3), note(0, 0.4, 55)], y, SR, lowest=C2, floor_hz=89
    )
    assert added == 1 and dropped == 2
    assert sorted(n.pitch for n in notes) == [C2]


def test_a_power_chords_octave_survives_because_its_fifth_is_there():
    y = guitar([C2, G2, C3], highpass=90)
    notes, _, _ = collapse_to_roots(
        [note(0, 0.4, G2), note(0, 0.4, C3)], y, SR, lowest=C2, floor_hz=89
    )
    assert sorted(n.pitch for n in notes) == [C2, G2, C3]


def test_a_lead_note_high_on_the_neck_grows_no_bass_note_underneath():
    # A recording of a lead line has no low end either - that must not be read as a
    # missing root, which is exactly the regression an earlier low-end-only rule caused.
    lead = guitar([64])  # E4
    _, added, _ = collapse_to_roots([note(0, 0.4, 64)], lead, SR, lowest=40, floor_hz=250)
    assert added == 0
