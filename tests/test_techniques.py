from tabify.fretting import Position, TabEvent
from tabify.notes import Note
from tabify.techniques import infer_palm_mutes


def stroke(start, *positions):
    return TabEvent(start, [Note(start, 0.25, 40)], [Position(s, f) for s, f in positions])


def test_fast_low_string_chugs_are_muted_and_the_held_chord_after_them_is_not():
    events = [stroke(0.0, (0, 0)), stroke(0.25, (0, 0)), stroke(0.5, (0, 0)), stroke(1.0, (0, 6), (1, 6), (2, 6))]
    # the last chug is followed by the chord within an 8th note, so it's still part of the run
    marked = infer_palm_mutes(events + [stroke(4.0, (0, 0))])
    assert [e.palm_mute for e in marked] == [True, True, True, False, False]


def test_a_single_quick_note_is_not_a_palm_mute_run():
    marked = infer_palm_mutes([stroke(0.0, (0, 0)), stroke(0.5, (3, 2), (4, 3)), stroke(2.0, (0, 0))])
    assert [e.palm_mute for e in marked] == [False, False, False]


def test_fast_runs_on_high_strings_are_left_alone():
    marked = infer_palm_mutes([stroke(t * 0.25, (4, 5)) for t in range(6)])
    assert not any(e.palm_mute for e in marked)


def test_palm_mutes_are_not_guessed_unless_asked_for(tmp_path, capsys):
    """Measured against a real tab, the guess marked 76% of strokes where 6% were muted."""
    from tabify.cli import build_parser

    args = build_parser().parse_args(["riff.wav"])
    assert args.palm_mute is False


def test_the_old_off_switch_still_works(tmp_path):
    from tabify.cli import build_parser

    args = build_parser().parse_args(["riff.wav", "--no-palm-mute"])
    assert args.no_palm_mute is True and args.palm_mute is False
