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


def test_tensorflow_only_is_a_warning_not_a_pass(installed):
    # The exact shape of a stale install: basic-pitch works, but through the 1.2 GB
    # backend, because a dependency list that now asks for onnxruntime was never applied.
    installed("librosa", "basic_pitch")
    chords = named(environment.checks(), "chords")
    assert chords.ok and chords.warn
    assert "TensorFlow" in chords.detail
    assert "onnxruntime" in chords.fix


def test_onnx_is_a_clean_pass(installed, monkeypatch, tmp_path):
    installed("librosa", "basic_pitch", "onnxruntime")
    monkeypatch.setattr(environment, "_bundled_onnx_model", lambda: tmp_path / "nmp.onnx")
    chords = named(environment.checks(), "chords")
    assert chords.ok and not chords.warn


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


def test_a_too_new_python_is_named_as_the_reason_chords_are_missing(installed, monkeypatch):
    """The usual cause, and a silent one: pip leaves basic-pitch out rather than failing."""
    installed("librosa")
    monkeypatch.setattr(environment.sys, "version_info", (3, 14, 2, "final", 0))
    chords = named(environment.checks(), "chords")
    assert not chords.ok
    assert "Python 3.14" in chords.detail
    # Reinstalling cannot help - --force keeps the interpreter, so the venv has to go first.
    assert "pipx uninstall" in chords.fix and "--python 3.11" in chords.fix


def test_on_a_supported_python_it_just_says_to_install_it(installed, monkeypatch):
    installed("librosa")
    monkeypatch.setattr(environment.sys, "version_info", (3, 11, 9, "final", 0))
    chords = named(environment.checks(), "chords")
    assert "pip install basic-pitch" in chords.fix
    assert "uninstall" not in chords.fix
