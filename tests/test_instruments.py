from tabify.instruments import ALL_PROGRAMS, default_instrument
from tabify.tuning import parse_tuning


def test_all_programs_are_valid_general_midi_numbers():
    assert len(ALL_PROGRAMS) == len(set(ALL_PROGRAMS.values())), "no two names should share a program"
    assert all(0 <= program <= 127 for program in ALL_PROGRAMS.values())


def test_default_instrument_for_guitar_tunings():
    assert default_instrument(parse_tuning("standard")) == "steel"
    assert default_instrument(parse_tuning("drop-d")) == "steel"


def test_default_instrument_for_bass_tunings():
    assert default_instrument(parse_tuning("bass")) == "finger"
    assert default_instrument(parse_tuning("bass-5")) == "finger"
