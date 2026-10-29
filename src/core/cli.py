"""
src/core/cli.py
===============
Command-line helpers shared by train.py and postprocess.py.

TableHelpParser is an argparse.ArgumentParser whose --help prints every
argument as a table, followed by usage examples.
"""

import argparse
import shutil
import textwrap


# ==========================================================
# Plain-text table
# ==========================================================

def render_table(headers, rows, max_width=None):
    """
    Render rows as a box-drawn text table.

    The last column is wrapped so the table fits in max_width
    (default: terminal width, capped at 120).
    """
    if max_width is None:
        max_width = min(shutil.get_terminal_size((100, 20)).columns, 120)

    rows   = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]

    # Shrink the last column so the whole table fits
    fixed     = sum(widths[:-1]) + 3 * len(widths) + 1
    widths[-1] = max(20, min(widths[-1], max_width - fixed))

    def line(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def fmt_row(cells):
        wrapped = [textwrap.wrap(c, w) or [""] for c, w in zip(cells, widths)]
        height  = max(len(w) for w in wrapped)
        out = []
        for k in range(height):
            parts = [(cell[k] if k < len(cell) else "").ljust(w)
                     for cell, w in zip(wrapped, widths)]
            out.append("│ " + " │ ".join(parts) + " │")
        return out

    lines = [line("┌", "┬", "┐"), *fmt_row(headers), line("├", "┼", "┤")]
    for r in rows:
        lines += fmt_row(r)
    lines.append(line("└", "┴", "┘"))
    return "\n".join(lines)


# ==========================================================
# argparse parser with a table-formatted --help
# ==========================================================

class TableHelpParser(argparse.ArgumentParser):
    """
    ArgumentParser whose help lists all arguments in a table.

    Args:
        examples : list of (command, explanation) tuples shown after the table
        notes    : optional free text shown at the end
    """

    def __init__(self, *args, examples=None, notes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.examples = examples or []
        self.notes    = notes

    def _value_hint(self, action):
        if action.nargs == 0:
            return ""
        if action.choices:
            return "{" + ",".join(map(str, action.choices)) + "}"
        metavar = action.metavar or action.dest.upper()
        if action.nargs in ("+", "*"):
            return f"{metavar} [...]"
        return metavar

    def format_help(self):
        out = [f"usage: {self.format_usage().split('usage: ', 1)[-1].strip()}", ""]
        if self.description:
            out += [textwrap.fill(self.description, 100), ""]

        rows = []
        for action in self._actions:
            if action.help == argparse.SUPPRESS:
                continue
            help_txt = action.help or ""
            if action.default not in (None, False, argparse.SUPPRESS) and action.nargs != 0:
                default = (" ".join(action.default) if isinstance(action.default, list)
                           else action.default)
                help_txt += f" (default: {default})"
            rows.append([", ".join(action.option_strings) or action.dest,
                         self._value_hint(action), help_txt])
        out += ["Arguments:", render_table(["Argument", "Value", "Description"], rows)]

        # Commands are too long for a table column: list them instead
        if self.examples:
            out += ["", "Examples:"]
            for cmd, what in self.examples:
                out += [f"  # {what}", f"  {cmd}", ""]
            out.pop()
        if self.notes:
            out += ["", self.notes]
        return "\n".join(out) + "\n"
