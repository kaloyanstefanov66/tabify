"""Train the stroke model, and say honestly how well it did.

    python training/train.py                  # train on the generated data, report on holdout
    python training/train.py --epochs 60      # longer
    python training/train.py --export out.onnx

The holdout set is clips the training set never saw, generated with their own tuning, tone
and tempo settings. Nothing here trains on it; it is only read at the end. The split that
early stopping watches is carved out of the training set instead, so the number reported at
the end is not one the model was tuned against.

Shapes are badly unbalanced - the triads are about a fortieth as common as a bare note -
so accuracy alone would flatter a model that never predicts them. Per-class recall is
printed for that reason, and the loss is weighted to stop the rare shapes being ignored.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import StrokeNet  # noqa: E402

from tabify.shapes import ROOTS, SHAPES  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def load(split: str):
    blob = np.load(DATA / split / "dataset.npz")
    x = blob["patches"].astype(np.float32)[:, None, :, :]  # (n, 1, bins, frames)
    return x, blob["roots"].astype(np.int64), blob["shapes"].astype(np.int64)


def loaders(x, roots, shapes, batch, shuffle=True):
    data = torch.utils.data.TensorDataset(
        torch.from_numpy(x), torch.from_numpy(roots), torch.from_numpy(shapes)
    )
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=shuffle, drop_last=False)


def evaluate(net, loader, device):
    net.eval()
    root_hits = shape_hits = both = total = 0
    per_shape = np.zeros((len(SHAPES), 2), dtype=np.int64)  # (correct, seen)
    with torch.no_grad():
        for patches, roots, shapes in loader:
            patches = patches.to(device)
            root_logits, shape_logits = net(patches)
            root_pred = root_logits.argmax(1).cpu()
            shape_pred = shape_logits.argmax(1).cpu()
            root_ok, shape_ok = root_pred == roots, shape_pred == shapes
            root_hits += int(root_ok.sum())
            shape_hits += int(shape_ok.sum())
            both += int((root_ok & shape_ok).sum())
            total += len(roots)
            for s in range(len(SHAPES)):
                mask = shapes == s
                per_shape[s, 1] += int(mask.sum())
                per_shape[s, 0] += int((shape_ok & mask).sum())
    return root_hits / total, shape_hits / total, both / total, per_shape


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--val-share", type=float, default=0.1)
    ap.add_argument("--export", default=str(Path(__file__).resolve().parents[1] / "src/tabify/models/stroke.onnx"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    x, roots, shapes = load("strokes")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(x))
    cut = int(len(x) * (1 - args.val_share))
    train_idx, val_idx = order[:cut], order[cut:]
    print(f"train {len(train_idx):,} | val {len(val_idx):,} | holdout {len(load('holdout')[0]):,}")

    train = loaders(x[train_idx], roots[train_idx], shapes[train_idx], args.batch)
    val = loaders(x[val_idx], roots[val_idx], shapes[val_idx], args.batch, shuffle=False)

    net = StrokeNet(len(ROOTS), len(SHAPES), width=args.width).to(device)
    params = sum(p.numel() for p in net.parameters())
    print(f"model: {params:,} parameters")

    counts = np.bincount(shapes[train_idx], minlength=len(SHAPES)).astype(np.float64)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    print("shape counts:", {SHAPES[i]: int(c) for i, c in enumerate(counts)})

    shape_loss = nn.CrossEntropyLoss(weight=weights)
    root_loss = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train))

    best, best_state = 0.0, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        net.train()
        running = 0.0
        for patches, r, s in train:
            patches, r, s = patches.to(device), r.to(device), s.to(device)
            opt.zero_grad(set_to_none=True)
            root_logits, shape_logits = net(patches)
            loss = root_loss(root_logits, r) + shape_loss(shape_logits, s)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss) * len(r)
        root_acc, shape_acc, both_acc, _ = evaluate(net, val, device)
        if both_acc > best:
            best = both_acc
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss {running/len(train_idx):.4f}  "
                  f"val root {root_acc:.1%}  shape {shape_acc:.1%}  both {both_acc:.1%}")
    print(f"trained in {time.perf_counter() - started:.0f}s; best val (both correct) {best:.1%}")

    if best_state:
        net.load_state_dict(best_state)

    hx, hr, hs = load("holdout")
    holdout = loaders(hx, hr, hs, args.batch, shuffle=False)
    root_acc, shape_acc, both_acc, per_shape = evaluate(net, holdout, device)
    print("\nHOLDOUT - clips the model never trained on:")
    print(f"  root correct  {root_acc:.1%}")
    print(f"  shape correct {shape_acc:.1%}")
    print(f"  both correct  {both_acc:.1%}")
    print("  per shape:")
    for i, name in enumerate(SHAPES):
        correct, seen = per_shape[i]
        share = f"{correct/seen:.1%}" if seen else "   n/a"
        print(f"    {name:<12} {share:>7}  ({correct}/{seen})")

    out = Path(args.export)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.eval().cpu()
    # Saved before the export, because exporting is the step most likely to fail on a
    # toolchain detail and a training run is too long to repeat for one.
    checkpoint = out.with_suffix(".pt")
    torch.save({"state_dict": net.state_dict(), "width": args.width,
                "roots": len(ROOTS), "shapes": len(SHAPES)}, checkpoint)
    print(f"saved {checkpoint}")
    torch.onnx.export(
        net,
        torch.zeros(1, 1, hx.shape[2], hx.shape[3]),
        str(out),
        input_names=["patch"],
        output_names=["root", "shape"],
        dynamic_axes={"patch": {0: "batch"}, "root": {0: "batch"}, "shape": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # the TorchScript exporter, which needs no extra toolchain
    )
    print(f"\nexported {out} ({out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
