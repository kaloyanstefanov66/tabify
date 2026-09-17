import pytest

from tabify import TabifyError
from tabify.notes import midi_to_name, name_to_midi
from tabify.tuning import TUNINGS, parse_tuning


def test_note_names_roundtrip():
    assert name_to_midi("E2") == 40
    assert name_to_midi("F#3") == 54
    assert name_to_midi("Bb1") == 34
    assert midi_to_name(64) == "E4"


def test_standard_labels():
    t = parse_tuning("standard")
    assert t.strings == (40, 45, 50, 55, 59, 64)
    assert t.labels() == ["E", "A", "D", "G", "B", "e"]


def test_preset_name_normalisation():
    assert parse_tuning("Drop D").name == "drop-d"
    assert parse_tuning("drop_d").strings[0] == name_to_midi("D2")


def test_custom_tuning():
    t = parse_tuning("D2, A2, D3, G3, B3, E4")
    assert t.name == "custom"
    assert t.labels() == ["D", "A", "d", "G", "B", "E"]


@pytest.mark.parametrize("spec", ["nonsense", "E4 E2", "H2 A2"])
def test_invalid_tunings(spec):
    with pytest.raises(TabifyError):
        parse_tuning(spec)


def test_all_presets_parse():
    for name in TUNINGS:
        assert parse_tuning(name).strings


def test_every_preset_is_ordered_low_to_high_and_plausible():
    for name in TUNINGS:
        t = parse_tuning(name)
        assert list(t.strings) == sorted(t.strings), f"{name} is not ordered low to high"
        assert 4 <= len(t.strings) <= 8, f"{name} has an odd number of strings"
        assert 20 <= t.strings[0] <= 64, f"{name}'s lowest string is out of range"


def test_every_alias_points_at_a_real_preset():
    from tabify.tuning import ALIASES

    for alias, target in ALIASES.items():
        assert target in TUNINGS, f"{alias} points at missing preset {target}"
        assert parse_tuning(alias).strings == parse_tuning(target).strings


def test_sharp_drop_tunings_exist_and_are_a_semitone_apart():
    steps = ["drop-d", "drop-c#", "drop-c", "drop-b", "drop-a#", "drop-a", "drop-g#", "drop-g", "drop-f#"]
    lowest = [parse_tuning(name).strings[0] for name in steps]
    assert lowest == list(range(lowest[0], lowest[0] - len(steps), -1))


def test_flat_spellings_work_the_same_as_sharps():
    assert parse_tuning("drop-db").strings == parse_tuning("drop-c#").strings
    assert parse_tuning("Drop Bb").strings == parse_tuning("drop-a#").strings
    assert parse_tuning("eb-standard").strings == parse_tuning("half-step-down").strings


def test_drop_tunings_have_a_fifth_between_the_two_lowest_strings():
    for name in [n for n in TUNINGS if n.startswith("drop-")]:
        strings = parse_tuning(name).strings
        assert strings[1] - strings[0] == 7, f"{name} should be a dropped fifth"
