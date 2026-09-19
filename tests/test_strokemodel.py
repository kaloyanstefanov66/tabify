"""The stroke model's inference path.

These check the path is wired correctly, not that the model is any good: it is trained on
synthesized audio and does not yet transfer to real recordings, which is why nothing in the
transcription pipeline calls it. See the note in the README.
"""

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("librosa")

from tabify import strokemodel  # noqa: E402
from tabify.shapes import ROOT_HIGH, ROOT_LOW, SHAPE_INTERVALS, SHAPES  # noqa: E402


def test_the_model_ships_with_tabify():
    assert strokemodel.available()
    assert strokemodel.MODEL.stat().st_size < 5_000_000


def test_no_times_means_no_predictions():
    assert strokemodel.predict_strokes(np.zeros(1000, dtype=np.float32), 22050, []) == []


def test_every_prediction_is_a_playable_answer():
    sr = 22050
    t = np.arange(sr) / sr
    audio = np.sin(2 * np.pi * 110 * t).astype(np.float32)
    out = strokemodel.predict_strokes(audio, sr, [0.1, 0.4, 0.7])
    assert len(out) == 3
    for stroke in out:
        assert ROOT_LOW <= stroke["root"] <= ROOT_HIGH
        assert stroke["shape"] in SHAPES
        # The pitches must be the shape's intervals over the root, not an arbitrary set.
        assert stroke["pitches"] == {stroke["root"] + i for i in SHAPE_INTERVALS[stroke["shape"]]}
        assert 0.0 <= stroke["confidence"] <= 1.0


def test_the_transcription_pipeline_does_not_use_it_yet():
    """Measured on two real recordings it scored well below the existing pipeline, so it is
    deliberately not wired in. This fails if someone connects it without revisiting that."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "tabify"
    users = [
        p.name for p in src.glob("*.py")
        if p.name not in {"strokemodel.py"} and "strokemodel" in p.read_text(encoding="utf-8")
    ]
    assert users == [], f"strokemodel is now used by {users}; re-measure it on real audio first"
