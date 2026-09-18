from tabify.shapes import ROOT_HIGH, ROOT_LOW, SHAPE_INTERVALS, SHAPES, positions_for, root_index, shape_index
from tabify.tuning import parse_tuning

DROP_C = parse_tuning("drop-c")
STANDARD = parse_tuning("standard")


def pitches(tuning, positions):
    return sorted(tuning.strings[s] + f for s, f in positions)


def test_a_drop_tuning_power_chord_is_one_barred_fret_across_three_strings():
    positions = positions_for(DROP_C, DROP_C.strings[0], "power3")
    assert positions == [(0, 0), (1, 0), (2, 0)]


def test_a_power_chord_higher_up_keeps_the_same_shape():
    root = DROP_C.strings[0] + 5
    assert positions_for(DROP_C, root, "power3") == [(0, 5), (1, 5), (2, 5)]


def test_an_octave_skips_a_string_rather_than_stretching_twelve_frets():
    # Putting the octave on the very next string would need a 12-fret reach; guitarists skip
    # a string instead. Getting this wrong silently turned octaves into single notes.
    positions = positions_for(STANDARD, STANDARD.strings[0] + 5, "octave")
    assert positions is not None
    strings = [s for s, _ in positions]
    assert strings[1] - strings[0] >= 2
    frets = [f for _, f in positions]
    assert max(frets) - min(frets) <= 4


def test_every_shape_sounds_the_intervals_it_claims():
    for shape, intervals in SHAPE_INTERVALS.items():
        root = STANDARD.strings[0] + 5
        positions = positions_for(STANDARD, root, shape)
        assert positions is not None, f"{shape} should be playable in standard tuning"
        assert pitches(STANDARD, positions) == sorted(root + i for i in intervals)


def test_a_root_below_the_lowest_string_is_unplayable():
    assert positions_for(STANDARD, STANDARD.strings[0] - 1, "single") is None


def test_shapes_that_do_not_fit_the_neck_are_refused_rather_than_fudged():
    # Rooted high on the top string, there is no higher string left to put the fifth on.
    assert positions_for(STANDARD, STANDARD.strings[-1] + 15, "power3") is None


def test_label_indexes_line_up_with_the_vocabulary():
    assert [shape_index(s) for s in SHAPES] == list(range(len(SHAPES)))
    assert root_index(ROOT_LOW) == 0
    assert root_index(ROOT_HIGH) == ROOT_HIGH - ROOT_LOW
