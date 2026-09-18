"""Reading ASCII tab as people actually write it - including untidily."""

import pytest

from tabify import TabifyError
from tabify.tabfile import parse_tab

SIMPLE = """
e|-----------------|
B|-----------------|
G|-----------------|
D|-----------------|
A|-----2-----------|
E|-0-------3-------|
"""


def test_reads_strokes_in_the_order_they_are_played():
    tab = parse_tab(SIMPLE)
    assert tab.strokes == [[(0, 0)], [(1, 2)], [(0, 3)]]


def test_notes_in_a_column_are_one_stroke():
    chord = """
e|----------|
B|----------|
G|-2--------|
D|-2--------|
A|-0--------|
E|----------|
"""
    assert parse_tab(chord).strokes == [[(1, 0), (2, 2), (3, 2)]]


def test_two_digit_frets_are_read_as_one_number():
    high = """
e|-12--10---|
B|----------|
G|----------|
D|----------|
A|----------|
E|----------|
"""
    assert parse_tab(high).strokes == [[(5, 12)], [(5, 10)]]


def test_several_systems_continue_the_same_riff():
    tab = parse_tab(SIMPLE + "\n" + SIMPLE)
    assert len(tab.strokes) == 6
    assert tab.strokes[3] == [(0, 0)]


def test_techniques_between_notes_are_ignored():
    # Hammer-ons, slides and bends say how a note was played, not which fret it was.
    bends = """
e|--------------|
B|--------------|
G|--------------|
D|--------------|
A|-5h7--9b--7/9-|
E|--------------|
"""
    assert [s[0][1] for s in parse_tab(bends).strokes] == [5, 7, 9, 7, 9]


def test_columns_that_drift_slightly_still_group_together():
    # Handwritten tab rarely lines up perfectly; a stroke is still a stroke.
    drifted = """
e|-----------|
B|-----------|
G|--2--------|
D|-2---------|
A|-0---------|
E|-----------|
"""
    assert len(parse_tab(drifted).strokes) == 1


def test_prose_and_markings_around_the_tab_are_skipped():
    messy = """
Main riff (play it twice)
   P.M. ----|
e|----------|
B|----------|
G|----------|
D|----------|
A|----------|
E|-0-0-3----|

that's the chorus
"""
    assert [s[0][1] for s in parse_tab(messy).strokes] == [0, 0, 3]


def test_tuning_comes_from_the_header_when_given():
    tab = parse_tab("tuning: drop-c\ntone: distorted\n" + SIMPLE)
    assert tab.tuning.name == "drop-c"
    assert tab.metadata["tone"] == "distorted"


def test_tuning_is_read_off_the_string_labels_when_there_is_no_header():
    tab = parse_tab(SIMPLE)
    assert tab.tuning is not None
    assert tab.tuning.describe() == "E2 A2 D3 G3 B3 E4"


def test_a_dropped_string_label_is_understood():
    dropped = """
e|-------|
B|-------|
G|-------|
D|-------|
A|-------|
D|-0-3---|
"""
    assert parse_tab(dropped).tuning.describe().startswith("D2 A2")


def test_text_with_no_tab_in_it_says_so():
    with pytest.raises(TabifyError, match="no tab found"):
        parse_tab("just some notes about a song, no tab here")
