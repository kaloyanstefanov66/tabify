"""The stroke model: what note is this rooted on, and what shape sits on top of it.

One pick attack in, two answers out. The input is a constant-Q patch of the moment the
string was hit - 144 quarter-tone bins by 20 frames, about a fifth of a second - and the
model says which root it heard and which of the shapes in `tabify.shapes` was played.

Why two heads rather than one label per (root, shape) pair: a power chord on C and a power
chord on F are the same shape at different pitches, and there are forty roots, so a single
flat vocabulary would need 240 classes that mostly never co-occur. Split, each head sees
every example.

Torch lives here rather than in the package because it is only needed to train. What ships
is the exported ONNX file, run through onnxruntime like the pitch model.
"""

from __future__ import annotations

import torch
from torch import nn

N_BINS, N_FRAMES = 144, 20


class StrokeNet(nn.Module):
    """A small convolutional net over the attack's constant-Q patch.

    Pooling is heavier across frequency than time: what distinguishes these shapes is the
    spacing between partials, which survives frequency pooling, while the attack transient
    that says where the stroke begins lives in only a few frames.
    """

    def __init__(self, n_roots: int, n_shapes: int, width: int = 32):
        super().__init__()

        def block(ins, outs, pool):
            return nn.Sequential(
                nn.Conv2d(ins, outs, 3, padding=1),
                nn.BatchNorm2d(outs),
                nn.ReLU(inplace=True),
                nn.Conv2d(outs, outs, 3, padding=1),
                nn.BatchNorm2d(outs),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        self.features = nn.Sequential(
            block(1, width, (2, 1)),          # 144x20 -> 72x20
            block(width, width * 2, (2, 2)),  # -> 36x10
            block(width * 2, width * 4, (2, 2)),  # -> 18x5
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 2))
        hidden = width * 4 * 4 * 2
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.root_head = nn.Linear(256, n_roots)
        self.shape_head = nn.Linear(256, n_shapes)

    def forward(self, x):
        z = self.trunk(self.pool(self.features(x)))
        return self.root_head(z), self.shape_head(z)
