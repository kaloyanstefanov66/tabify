"""Export a trained stroke model to ONNX, which is what tabify actually runs.

Separate from training because exporting fails on toolchain details more often than it
should, and a training run is too long to repeat for one. The checkpoint is written before
the export for the same reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import N_BINS, N_FRAMES, StrokeNet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1] / "src/tabify/models"
    ap.add_argument("--checkpoint", default=str(root / "stroke.pt"))
    ap.add_argument("--out", default=str(root / "stroke.onnx"))
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    net = StrokeNet(blob["roots"], blob["shapes"], width=blob["width"])
    net.load_state_dict(blob["state_dict"])
    net.eval()

    example = torch.zeros(1, 1, N_BINS, N_FRAMES)
    with torch.no_grad():
        before = [t.clone() for t in net(example)]

    torch.onnx.export(
        net, example, args.out,
        input_names=["patch"], output_names=["root", "shape"],
        dynamic_axes={"patch": {0: "batch"}, "root": {0: "batch"}, "shape": {0: "batch"}},
        opset_version=17, dynamo=True,
        # Weights inside the file, not beside it. The exporter defaults to writing them to a
        # sibling .data file, which works where it was built and fails the moment the model
        # is copied anywhere on its own - including into the wheel.
        external_data=False,
    )

    import shutil
    import tempfile

    import numpy as np
    import onnxruntime as ort

    out = Path(args.out)
    # Checked as a lone file in an empty directory, because that is how it ships.
    with tempfile.TemporaryDirectory() as tmp:
        alone = Path(tmp) / out.name
        shutil.copy2(out, alone)
        session = ort.InferenceSession(str(alone), providers=["CPUExecutionProvider"])
        after = session.run(None, {"patch": example.numpy()})
    worst = max(float(np.abs(a.numpy() - b).max()) for a, b in zip(before, after))
    print(f"exported {out} ({out.stat().st_size/1e6:.2f} MB)")
    print(f"runs standalone; torch vs onnx, largest disagreement: {worst:.2e}")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
