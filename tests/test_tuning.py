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
