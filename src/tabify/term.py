"""Terminal escape-code helpers."""

from __future__ import annotations

import os

ALT_SCREEN_ON = "\x1b[?1049h"  # switch to the alternate screen (what vim/less use), leaving scrollback alone
ALT_SCREEN_OFF = "\x1b[?1049l"  # back to the normal screen, exactly as it was before
CURSOR_HIDE = "\x1b[?25l"
CURSOR_SHOW = "\x1b[?25h"
HOME = "\x1b[H"
CLEAR_LINE_END = "\x1b[K"
CLEAR_SCREEN_END = "\x1b[J"


def enable_ansi() -> bool:
    """Make sure escape codes are interpreted rather than printed. Returns False if that's not possible."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return True
    except (AttributeError, OSError):
        return False
