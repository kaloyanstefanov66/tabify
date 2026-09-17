"""ASCII tab rendering."""

from __future__ import annotations

import re

from tabify.fretting import TabEvent
from tabify.rhythm import TimeSignature
from tabify.tuning import Tuning

_FRET_COLOR = "\x1b[1;36m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def _render_bar(steps: dict[int, TabEvent], first_step: int, bar_steps: int, n_strings: int) -> list[str]:
    """Return one text row per string (lowest string first) for a single bar."""
    rows = ["-" for _ in range(n_strings)]
    for step in range(first_step, first_step + bar_steps):
        frets: dict[int, str] = {}
        if step in steps:
            for pos in steps[step].positions:
                frets[pos.string] = str(pos.fret)
        width = max((len(f) for f in frets.values()), default=1)
        for s in range(n_strings):
            rows[s] += frets.get(s, "").ljust(width, "-") + "-"
    return rows


def render_tab(
    events: list[TabEvent],
    tuning: Tuning,
    *,
    time_sig: TimeSignature = TimeSignature(),
    subdivision: int = 4,
    width: int = 80,
    title: str | None = None,
    capo: int = 0,
    bpm: float | None = None,
    color: bool = False,
) -> str:
    header = []
    if title:
        header += [title, "=" * len(title)]
    header.append(f"Tuning: {tuning.describe()}" + (f" ({tuning.name})" if tuning.name != "custom" else ""))
    if capo:
        header.append(f"Capo: fret {capo}")
    header.append(f"Time: {time_sig}" + (f"   Tempo: {round(bpm)} BPM" if bpm else ""))

    if not events:
        return "\n".join(header + ["", "(no playable notes found)"]) + "\n"

    bar_steps = max(1, round(time_sig.bar_length * subdivision))
    steps = {round(e.start * subdivision): e for e in events}
    n_bars = max(steps) // bar_steps + 1

    n_strings = len(tuning.strings)
    labels = tuning.labels()
    label_w = max(len(label) for label in labels)
    bars = [_render_bar(steps, b * bar_steps, bar_steps, n_strings) for b in range(n_bars)]

    # Pack bars into systems (lines) that fit the requested width.
    systems: list[list[int]] = [[]]
    line_len = label_w + 1
    for b, rows in enumerate(bars):
        bar_len = len(rows[0]) + 1
        if systems[-1] and line_len + bar_len > width:
            systems.append([])
            line_len = label_w + 1
        systems[-1].append(b)
        line_len += bar_len

    out = header
    for system in systems:
        out.append("")
        numbers = " " * (label_w + 1)
        for b in system:
            numbers += str(b + 1).ljust(len(bars[b][0]) + 1)
        out.append(numbers.rstrip())
        for s in reversed(range(n_strings)):
            line = labels[s].ljust(label_w) + "|" + "".join(bars[b][s] + "|" for b in system)
            if color:
                prefix, body = line[: label_w + 1], line[label_w + 1:]
                body = re.sub(r"\d+", lambda m: f"{_FRET_COLOR}{m.group()}{_RESET}", body)
                body = re.sub(r"-+", lambda m: f"{_DIM}{m.group()}{_RESET}", body)
                line = prefix + body
            out.append(line)
    return "\n".join(out) + "\n"
