"""Reading a struck chord off the attack itself.

The pitch model hears notes one at a time and leaves it to later stages to decide which of
them were struck together. This model is asked a different question, once per pick attack:
what note is this rooted on, and what shape was played on top of it. That is the question a
guitarist answers instinctively, and the one the rest of the pipeline kept getting wrong.

Trained on synthesized strokes (see `training/`), shipped as ONNX beside the pitch model and
run the same way, so nothing here needs torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tabify.shapes import ROOT_LOW, SHAPE_INTERVALS, SHAPES

MODEL = Path(__file__).parent / "models" / "stroke.onnx"


def available() -> bool:
    import importlib.util

    return MODEL.exists() and importlib.util.find_spec("onnxruntime") is not None


def _session():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_strokes(y: np.ndarray, sr: int, times) -> list[dict]:
    """What was struck at each of `times`: root, shape, the pitches, and how sure it is.

    Confidence is the product of the two heads' probabilities, because a stroke is only
    right when both are - it is what lets a caller keep the model's answer where it is sure
    and fall back to the pitch model where it is not.
    """
    from tabify.features import patches_at

    if len(times) == 0:
        return []
    patches = patches_at(y, times, sr).astype(np.float32)[:, None, :, :]
    session = _session()
    name = session.get_inputs()[0].name
    roots, shapes = [], []
    for i in range(0, len(patches), 256):
        root_logits, shape_logits = session.run(None, {name: patches[i : i + 256]})
        roots.append(_softmax(root_logits))
        shapes.append(_softmax(shape_logits))
    roots, shapes = np.concatenate(roots), np.concatenate(shapes)

    out = []
    for t, root_p, shape_p in zip(times, roots, shapes):
        root = int(root_p.argmax()) + ROOT_LOW
        shape = SHAPES[int(shape_p.argmax())]
        out.append(
            {
                "time": float(t),
                "root": root,
                "shape": shape,
                "pitches": {root + i for i in SHAPE_INTERVALS[shape]},
                "confidence": float(root_p.max() * shape_p.max()),
            }
        )
    return out
