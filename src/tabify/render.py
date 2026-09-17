"""ASCII tab rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tabify.fretting import TabEvent
from tabify.rhythm import TimeSignature
from tabify.tuning import Tuning

_FRET_COLOR = "\x1b[1;36m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"
_HIGHLIGHT = "\x1b[7m"  # reverse video, used to mark the currently-playing column


@dataclass
class TabLayout:
    """A tab laid out into bars and line-wrapped systems, ready to render (optionally with a
    playhead highlight - see `format_system`). Built once per tab; cheap to re-render from.
    """

    bars: list[list[str]]  # bars[b][s] = row string for string s of bar b (no leading '|')
    offsets: list[list[tuple[int, int]]]  # offsets[b][i] = (start_col, width) of grid-step i within bar b
    systems: list[list[int]]  # systems[k] = list of bar indices on line/system k
    bar_steps: int
    tuning: Tuning
    label_w: int


def _render_bar(steps: dict[int, TabEvent], first_step: int, bar_steps: int, n_strings: int):
    """Row strings for one bar, plus each grid-step's (start_col, width) within those rows."""
    rows = ["-" for _ in range(n_strings)]  # leading dash before the bar's first fret, tab convention
    offsets = []
    col = 1
    for step in range(first_step, first_step + bar_steps):
        frets: dict[int, str] = {}
        if step in steps:
            for pos in steps[step].positions:
                frets[pos.string] = str(pos.fret)
        width = max((len(f) for f in frets.values()), default=1)
        offsets.append((col, width))
        for s in range(n_strings):
            rows[s] += frets.get(s, "").ljust(width, "-") + "-"
        col += width + 1
    return rows, offsets


def build_layout(
    events: list[TabEvent], tuning: Tuning, *, time_sig: TimeSignature = TimeSignature(), subdivision: int = 4,
    width: int = 80,
) -> TabLayout | None:
    """Lay out a tab into bars and systems. Returns None if there's nothing to render."""
    if not events:
        return None
    bar_steps = max(1, round(time_sig.bar_length * subdivision))
    steps = {round(e.start * subdivision): e for e in events}
    n_bars = max(steps) // bar_steps + 1
    n_strings = len(tuning.strings)
    label_w = max(len(label) for label in tuning.labels())

    bars, offsets = [], []
    for b in range(n_bars):
        rows, offs = _render_bar(steps, b * bar_steps, bar_steps, n_strings)
        bars.append(rows)
        offsets.append(offs)

    systems: list[list[int]] = [[]]
    line_len = label_w + 1
    for b, rows in enumerate(bars):
        bar_len = len(rows[0]) + 1
        if systems[-1] and line_len + bar_len > width:
            systems.append([])
            line_len = label_w + 1
        systems[-1].append(b)
        line_len += bar_len

    return TabLayout(bars, offsets, systems, bar_steps, tuning, label_w)


def system_for_step(layout: TabLayout, step: float) -> int:
    """Which system index contains grid-step `step` (clamped to the tab's range)."""
    bar = min(max(int(step) // layout.bar_steps, 0), len(layout.bars) - 1)
    for k, system in enumerate(layout.systems):
        if bar in system:
            return k
    return len(layout.systems) - 1


def format_system(
    layout: TabLayout, system_index: int, *, color: bool = False, highlight_step: float | None = None,
) -> list[str]:
    """Render one system (line of bars) as a list of text lines, one per string plus a bar-number line."""
    system = layout.systems[system_index]
    labels = layout.tuning.labels()
    n_strings = len(labels)

    highlight: tuple[int, int] | None = None  # (bar index, offset within that bar's row)
    if highlight_step is not None:
        bar = int(highlight_step) // layout.bar_steps
        if bar in system:
            local = int(highlight_step) - bar * layout.bar_steps
            highlight = (bar, local)

    numbers = " " * (layout.label_w + 1)
    for b in system:
        numbers += str(b + 1).ljust(len(layout.bars[b][0]) + 1)
    out = [numbers.rstrip()]

    for s in reversed(range(n_strings)):
        line = labels[s].ljust(layout.label_w) + "|"
        for b in system:
            raw = layout.bars[b][s]
            if highlight and highlight[0] == b:
                # Split around the highlighted column first, so color regexes never see
                # the ANSI codes and slice boundaries always land on real characters.
                start, w = layout.offsets[b][highlight[1]]
                parts = [raw[:start], raw[start : start + w], raw[start + w :]]
                if color:
                    parts[0] = re.sub(r"\d+", lambda m: f"{_FRET_COLOR}{m.group()}{_RESET}", parts[0])
                    parts[0] = re.sub(r"-+", lambda m: f"{_DIM}{m.group()}{_RESET}", parts[0])
                    parts[2] = re.sub(r"\d+", lambda m: f"{_FRET_COLOR}{m.group()}{_RESET}", parts[2])
                    parts[2] = re.sub(r"-+", lambda m: f"{_DIM}{m.group()}{_RESET}", parts[2])
                row = parts[0] + f"{_HIGHLIGHT}{parts[1]}{_RESET}" + parts[2]
            elif color:
                row = re.sub(r"\d+", lambda m: f"{_FRET_COLOR}{m.group()}{_RESET}", raw)
                row = re.sub(r"-+", lambda m: f"{_DIM}{m.group()}{_RESET}", row)
            else:
                row = raw
            line += row + "|"
        out.append(line)
    return out


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

    layout = build_layout(events, tuning, time_sig=time_sig, subdivision=subdivision, width=width)
    if layout is None:
        return "\n".join(header + ["", "(no playable notes found)"]) + "\n"

    out = header
    for k in range(len(layout.systems)):
        out.append("")
        out.extend(format_system(layout, k, color=color))
    return "\n".join(out) + "\n"
