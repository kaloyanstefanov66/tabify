"""Can tabify work out the tuning from the notes alone?"""

import pytest

from tabify.tuning import TUNINGS, parse_tuning, rank_tunings, score_tuning


def played(tuning_name: str, positions: list[tuple[int, int]]) -> list[int]:
    """The pitches you'd hear playing these (string, fret) positions in this tuning."""
    tuning = parse_tuning(tuning_name)
    return [tuning.strings[s] + f for s, f in positions]


# Riffs that lean on the open low string, the way drop tunings are actually used.
OPEN_CHORD_RIFF = [(0, 0), (1, 0), (2, 0)] * 6 + [(0, 0)] * 8 + [(0, 3), (1, 3), (2, 3)]


@pytest.mark.parametrize("truth", ["drop-c", "drop-c#", "drop-b", "drop-d", "drop-g#", "7-string", "standard"])
def test_detects_the_tuning_of_a_riff_that_uses_open_strings(truth):
    assert rank_tunings(played(truth, OPEN_CHORD_RIFF))[0][1].name == truth


def test_detects_bass_rather_than_guitar():
    riff = [(0, 0), (1, 0)] * 4 + [(0, 3), (1, 3)] * 3 + [(0, 5)] * 4
    assert rank_tunings(played("bass", riff))[0][1].name == "bass"


def test_notes_below_a_tunings_lowest_string_rule_it_out():
    # A drop-C riff can't be played in standard tuning at all: C2 is below the low E.
    pitches = played("drop-c", OPEN_CHORD_RIFF)
    assert score_tuning(parse_tuning("standard"), pitches) < score_tuning(parse_tuning("drop-c"), pitches)


def test_a_closed_position_lick_is_ambiguous_and_falls_back_to_the_common_tuning():
    # Nothing here touches an open string, so several tunings fit equally well: the ranking
    # should still be led by the common one, and the margin should be small enough to warn about.
    box = [(0, 5), (0, 8), (1, 5), (1, 7), (2, 5), (2, 7), (3, 5), (3, 7)] * 2
    ranking = rank_tunings(played("standard", box))
    assert ranking[0][1].name == "standard"
    assert ranking[0][0] - ranking[1][0] < 0.15  # i.e. tabify should say it isn't sure


def test_every_preset_can_be_scored_without_blowing_up():
    pitches = played("standard", [(0, 0), (3, 5), (5, 12)])
    assert len(rank_tunings(pitches)) == len(TUNINGS)
    assert all(score >= 0 for score, _ in rank_tunings(pitches))


def test_no_notes_scores_zero_rather_than_guessing():
    assert score_tuning(parse_tuning("standard"), []) == 0.0
