import pytest

from tabify.cli import main
from tabify.midi_io import read_midi, write_midi
from tabify.notes import Note
from tabify.rhythm import TimeSignature


def test_demo_runs(capsys):
    assert main(["--demo", "--no-color", "--width", "80"]) == 0
    out = capsys.readouterr().out
    assert "e|" in out and "E|" in out


def test_midi_roundtrip(tmp_path):
    notes = [Note(0, 0.5, 45, 100), Note(0.5, 0.5, 52, 80), Note(1, 2, 57, 90)]
    path = tmp_path / "riff.mid"
    write_midi(path, notes, bpm=100, time_sig=TimeSignature(3, 4))
    content = read_midi(path)
    assert [(n.start, n.duration, n.pitch) for n in content.notes] == [(n.start, n.duration, n.pitch) for n in notes]
    assert round(content.bpm) == 100
    assert content.time_sig == TimeSignature(3, 4)


def test_midi_file_to_tab_file(tmp_path, capsys):
    mid = tmp_path / "riff.mid"
    write_midi(mid, [Note(0, 1, 40), Note(1, 1, 43)])
    out = tmp_path / "riff.txt"
    assert main([str(mid), "-o", str(out), "--tuning", "drop-d"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Tuning: D2 A2 D3 G3 B3 E4 (drop-d)" in text
    assert "D|-2-------5-" in text


def test_missing_file_is_a_clean_error(capsys):
    assert main(["does-not-exist.wav"]) == 1
    assert "file not found" in capsys.readouterr().err


def test_list_tunings(capsys):
    assert main(["--list-tunings"]) == 0
    assert "drop-d" in capsys.readouterr().out


def test_separate_rejects_midi_input(tmp_path, capsys):
    mid = tmp_path / "song.mid"
    write_midi(mid, [Note(0, 1, 40)])
    assert main([str(mid), "--separate"]) == 1
    assert "not MIDI" in capsys.readouterr().err


def test_audio_out_without_fluidsynth_gives_actionable_error(capsys, monkeypatch):
    monkeypatch.setattr("tabify.synth.importlib.util.find_spec", lambda name: None)
    assert main(["--demo", "--audio-out", "out.wav"]) == 1
    assert "FluidSynth" in capsys.readouterr().err


def test_instrument_flag_rejects_unknown_name(capsys):
    with pytest.raises(SystemExit):
        main(["--demo", "--instrument", "kazoo"])
