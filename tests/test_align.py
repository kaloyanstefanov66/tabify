"""Lining up a transcription against a written tab when they don't match stroke for stroke."""

from tabify.align import agreement, align, similarity


def riff(*strokes) -> list[set[int]]:
    """A sequence of strokes, each written as the pitches it sounds."""
    return [set(s if isinstance(s, (list, tuple)) else [s]) for s in strokes]


def test_identical_riffs_pair_up_completely():
    a = riff(40, 43, (40, 47, 52), 40)
    result = align(a, list(a))
    assert result.pairs == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert not result.only_in_a and not result.only_in_b


def test_an_extra_stroke_does_not_throw_off_everything_after_it():
    written = riff(40, 43, 45, 47)
    played = riff(40, 43, 43, 45, 47)  # one chug played twice
    result = align(written, played)
    assert result.only_in_b == [2]
    assert result.pairs == [(0, 0), (1, 1), (2, 3), (3, 4)]


def test_a_missed_stroke_is_reported_as_missed():
    written = riff(40, 43, 45, 47)
    played = riff(40, 45, 47)  # the model never heard the second one
    result = align(written, played)
    assert result.only_in_a == [1]
    assert result.pairs == [(0, 0), (2, 1), (3, 2)]


def test_a_wrong_note_counts_as_one_missed_and_one_extra():
    written = riff(40, 43, 45)
    played = riff(40, 44, 45)  # a semitone out
    result = align(written, played)
    assert result.pairs == [(0, 0), (2, 2)]
    assert result.only_in_a == [1] and result.only_in_b == [1]


def test_a_chord_missing_one_string_still_pairs_with_its_chord():
    written = riff((40, 47, 52))
    played = riff((40, 47))  # the octave string never came through
    result = align(written, played)
    assert result.pairs == [(0, 0)]
    assert similarity(written[0], played[0]) > 0.5


def test_a_riff_played_twice_pairs_with_the_written_one_once_and_flags_the_rest():
    written = riff(40, 43, 45)
    played = riff(40, 43, 45, 40, 43, 45)
    result = align(written, played)
    assert len(result.pairs) == 3
    assert len(result.only_in_b) == 3


def test_agreement_counts_exact_partial_missed_and_extra():
    written = riff(40, (40, 47, 52), 45, 47)
    played = riff(40, (40, 47), 45, 50)
    result = align(written, played)
    summary = agreement(written, played, result)
    assert summary["exact"] == 2  # the single notes that match
    assert summary["partial"] == 1  # the chord missing a string
    assert summary["missed"] == 1 and summary["extra"] == 1  # the wrong last note
    assert summary["recall"] == 0.5


def test_nothing_in_common_pairs_nothing():
    result = align(riff(40, 41), riff(60, 61))
    assert not result.pairs
    assert result.only_in_a == [0, 1] and result.only_in_b == [0, 1]


def test_empty_sides_are_handled():
    assert align([], riff(40)).only_in_b == [0]
    assert align(riff(40), []).only_in_a == [0]
    assert align([], []).matched == 0
