"""The doctor exists to catch a stale install, so these tests fake stale installs."""

import pytest

from tabify import environment
from tabify.cli import build_parser, run


@pytest.fixture
def installed(monkeypatch):
    """Pretend exactly the named modules are importable."""

    def install(*modules):
        monkeypatch.setattr(environment, "_have", lambda name: name in modules)

    return install


def named(checks, name):
    return next(c for c in checks if c.name == name)


def test_onnxruntime_alone_is_enough_for_chords(installed):
    """The model ships with tabify, so onnxruntime is the only thing that has to be there."""
    installed("librosa", "onnxruntime")
    chords = named(environment.checks(), "chords")
    assert chords.ok and not chords.warn
    assert "TensorFlow" in chords.detail  # named only to say it is not needed


def test_without_onnxruntime_it_falls_back_and_says_so(installed):
    installed("librosa")
    chords = named(environment.checks(), "chords")
    assert not chords.ok
    assert "pYIN" in chords.detail and "onnxruntime" in chords.fix


def test_the_doctor_does_not_import_tensorflow_to_answer(installed, monkeypatch):
    """Finding the ONNX model must not import basic_pitch - that costs a TensorFlow load."""
    installed("librosa", "basic_pitch", "onnxruntime")
    monkeypatch.delitem(__import__("sys").modules, "basic_pitch", raising=False)
    environment.checks()
    assert "basic_pitch" not in __import__("sys").modules


def test_pytorch_separation_works_but_is_flagged_as_the_long_way(installed):
    installed("librosa", "demucs")
    mixes = named(environment.checks(), "full mixes")
    assert mixes.ok and mixes.warn and "demucs-onnx" in mixes.fix


def test_missing_separation_says_what_that_costs(installed):
    installed("librosa")
    mixes = named(environment.checks(), "full mixes")
    assert not mixes.ok and "one part" in mixes.detail


def test_report_mentions_reinstalling_when_something_is_stale(installed):
    installed("librosa", "basic_pitch")
    text = environment.report()
    assert "pipx install --force" in text  # editing a dependency list on its own does nothing


def test_report_is_quiet_when_everything_is_on_the_fast_path(installed, monkeypatch):
    installed("librosa", "basic_pitch", "onnxruntime", "demucs_onnx", "sounddevice", "fluidsynth", "yt_dlp")
    monkeypatch.setattr(environment, "_pitch_engine", lambda: environment.Check("chords", "onnx", ok=True))
    monkeypatch.setattr(environment.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert "fast path" in environment.report()


def test_doctor_exits_zero_without_needing_an_input_file(capsys):
    assert run(build_parser().parse_args(["--doctor"])) == 0
    assert "tabify environment" in capsys.readouterr().out


@pytest.mark.parametrize("version", [(3, 10, 0), (3, 11, 9), (3, 12, 0), (3, 14, 2)])
def test_no_python_version_loses_chords_any_more(installed, monkeypatch, version):
    """Chords used to vanish silently on 3.12+, because basic-pitch had no build there.

    Running the model straight from its own file removes that cliff entirely: the same
    check passes on every version tabify supports.
    """
    installed("librosa", "onnxruntime")
    monkeypatch.setattr(environment.sys, "version_info", (*version, "final", 0))
    assert named(environment.checks(), "chords").ok


def test_a_missing_model_file_is_reported_rather_than_falling_back_quietly(installed, monkeypatch, tmp_path):
    installed("librosa", "onnxruntime")
    from tabify import pitchmodel

    monkeypatch.setattr(pitchmodel, "MODEL", tmp_path / "gone.onnx")
    chords = named(environment.checks(), "chords")
    assert not chords.ok and "missing" in chords.detail
