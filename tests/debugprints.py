"""Regression for the code that only the debug switches reach.

Every switch below is off in both profiles (workflows/dev.yaml), so no launch and no other regression runs these
paths. The script selects Agg after its preamble and before any src submodule import, so the test does not depend on
the selection the plots packages make on import; testkit's children set no MPLBACKEND.
(a) In process, on a scratch config copy (realistic) whose dev copy turns trigger.strategy.printout_io, printout_debug,
    full_output and plot_debug on, a strategy analyses one testkit.sine_frame of 300 daily rows, stdout captured, its
    output written to the copy's trigger store. Expected: the headers TRIGGER - DETECTION and TRIGGER - OUTPUT and the
    completion line; the line "Total triggers: N (Plus: p, Minus: m, Active: a)" with N, p, m and a recomputed from the
    output frame's begin and active flags; N rows in the summary table; one saved figure (counted by a wrapper on
    Figure.savefig) per call the loop made to the three debug plot functions (counted by wrappers that then draw),
    above zero, and under plots/0 beside the trigger store exactly the saved file names (two candidates suppressed at
    one row share a name, so the folder can hold fewer files than calls).
(b) In process, each strategy switch alone: printout_debug prints the detection header, the totals line and the
    completion line and writes no figure; plot_debug writes figures and prints none of the three; printout_io prints
    TRIGGER - OUTPUT and none of the three.
(c) Pipeline children, --task trigger signal on a one-series copy, each with the named switches on and every other
    off, each printing exactly its headers: (1) trigger.read.printout_io: TRIGGER - READ; (2)
    trigger.preprocess.printout_io: TRIGGER - PREPROCESS; (3) trigger.strategy.printout_io with
    trigger.strategy.full_output, and signal.input.printout_io: TRIGGER - OUTPUT, TRIGGER - OUTCOME with "All rows:"
    in its block, SIGNAL - ACCUMULATED TRIGGERED SERIES with "First 10 rows:"; (4) signal.input.printout_io with
    signal.input.full_output, and signal.selection.printout_io: the accumulated block with "All rows:", SIGNAL -
    SELECTION with "First 10 rows:"; (5) signal.selection.printout_io with signal.selection.full_output: SIGNAL -
    SELECTION, its "All rows:" block as many rows as the stored groups table.
A dropped switch removes its header, a crossed one moves a header or an "All rows:" block.

Structure:
1. main
2. In-process strategy runs
3. Pipeline children
4. Printed blocks
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import matplotlib

matplotlib.use("Agg")

import contextlib
import io
import itertools
import re
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from matplotlib.figure import Figure

import src.utils.trigger.realistic as realistic
from src.utils.common.constants import (
    ACTIVE_TRIGGER_MINUS_BEGIN,
    ACTIVE_TRIGGER_PLUS_BEGIN,
    TRIGGER_MINUS_BEGIN,
    TRIGGER_PLUS_BEGIN,
)

ROWS = 300
FIRST_DAY = "2020-01-01"
SERIES = "0"
REALISTIC = {"trigger.replay.mode": "realistic"}
PRINTOUT_IO, PRINTOUT_DEBUG = "trigger.strategy.printout_io", "trigger.strategy.printout_debug"
FULL_OUTPUT, PLOT_DEBUG = "trigger.strategy.full_output", "trigger.strategy.plot_debug"
STRATEGY_SWITCHES = (PRINTOUT_IO, PRINTOUT_DEBUG, FULL_OUTPUT, PLOT_DEBUG)
SWITCHES = ("trigger.read.printout_io", "trigger.preprocess.printout_io", *STRATEGY_SWITCHES,
            "signal.input.printout_io", "signal.input.full_output", "signal.selection.printout_io",
            "signal.selection.full_output")
PLOTS = ("plot_suppressed_trigger", "plot_ended_trigger", "plot_active_trigger")
READ, PREPROCESS, DETECTION = "TRIGGER - READ", "TRIGGER - PREPROCESS", "TRIGGER - DETECTION"
OUTPUT, OUTCOME = "TRIGGER - OUTPUT", "TRIGGER - OUTCOME"
ACCUMULATED, SELECTION = "SIGNAL - ACCUMULATED TRIGGERED SERIES", "SIGNAL - SELECTION"
HEADERS = (READ, PREPROCESS, DETECTION, OUTPUT, OUTCOME, ACCUMULATED, SELECTION)
TOTALS_PREFIX = "Total triggers:"
COMPLETION = "Realistic mode trigger generation complete"
DEBUG_LINES = (DETECTION, TOTALS_PREFIX, COMPLETION)
ALL_ROWS, FIRST_ROWS = "All rows:", "First 10 rows:"
GROUPS_KEY = "/signals"
TABLE_RULE = re.compile(r"^-{100,}$")  # the trigger summary table's rule, longer than every 80-character section rule
INDEX_ROW = re.compile(r"^\d+\s")  # a row of a printed frame under a RangeIndex
# (c): per case, the switches it turns on and its expected headers, each with the marker its blocks must hold
CHILD_CASES = {
    "read": ({"trigger.read.printout_io": True}, {READ: None}),
    "preprocess": ({"trigger.preprocess.printout_io": True}, {PREPROCESS: None}),
    "trigger output": ({PRINTOUT_IO: True, FULL_OUTPUT: True, "signal.input.printout_io": True},
                       {OUTPUT: None, OUTCOME: ALL_ROWS, ACCUMULATED: FIRST_ROWS}),
    "signal input": ({"signal.input.printout_io": True, "signal.input.full_output": True,
                      "signal.selection.printout_io": True}, {ACCUMULATED: ALL_ROWS, SELECTION: FIRST_ROWS}),
    "signal selection": ({"signal.selection.printout_io": True, "signal.selection.full_output": True},
                         {SELECTION: ALL_ROWS}),
}


def main() -> None:
    """The in-process runs with every strategy switch and with each alone, the pipeline children, then the summary
    (return code 1 on any FAIL)."""
    every_switch()
    for switch in (PRINTOUT_DEBUG, PLOT_DEBUG, PRINTOUT_IO):
        one_switch(switch)
    for case, (switches, expected) in CHILD_CASES.items():
        pipeline_case(case, switches, expected)
    KIT.finish()


# =============================================================================
# IN-PROCESS STRATEGY RUNS
# =============================================================================


def every_switch() -> None:
    """(a): every strategy switch on: the headers, the totals line against the output frame, the summary table's
    rows and one figure per debug plot call."""
    run = strategy_run(dict.fromkeys(STRATEGY_SWITCHES, True))
    lines = testkit.plain_lines(run["text"])
    for marker in (DETECTION, OUTPUT, COMPLETION):
        printed = any(marker in line for line in lines)
        KIT.verdict(f"every strategy switch: {marker!r} printed", printed, True, printed)
    frame = run["frame"]
    plus, minus = int(frame[TRIGGER_PLUS_BEGIN].sum()), int(frame[TRIGGER_MINUS_BEGIN].sum())
    active = int(frame[ACTIVE_TRIGGER_PLUS_BEGIN].sum() + frame[ACTIVE_TRIGGER_MINUS_BEGIN].sum())
    totals = f"{TOTALS_PREFIX} {plus + minus} (Plus: {plus}, Minus: {minus}, Active: {active})"
    printed = [line for line in lines if line.startswith(TOTALS_PREFIX)]
    KIT.verdict("every strategy switch: the totals line recomputed from the output frame's begin and active flags",
                printed == [totals], [totals], printed)
    rows = summary_rows(lines)
    KIT.verdict(f"every strategy switch: the summary table holds {plus + minus} rows", len(rows) == plus + minus,
                plus + minus, len(rows))
    calls = sum(run["calls"].values())
    KIT.verdict("every strategy switch: one saved figure per debug plot call, above zero",
                len(run["saved"]) == calls > 0, "saves = calls > 0",
                {"saves": len(run["saved"]), "calls": run["calls"]})
    KIT.verdict("every strategy switch: the files under plots/0 are the saved figures",
                run["figures"] == sorted(set(run["saved"])), sorted(set(run["saved"])), run["figures"])


def one_switch(switch: str) -> None:
    """(b): one strategy switch alone prints or draws only its own output."""
    run = strategy_run({name: name == switch for name in STRATEGY_SWITCHES})
    lines = testkit.plain_lines(run["text"])
    printed = {marker: any(marker in line for line in lines) for marker in (*DEBUG_LINES, OUTPUT)}
    expected = {**{marker: switch == PRINTOUT_DEBUG for marker in DEBUG_LINES}, OUTPUT: switch == PRINTOUT_IO}
    KIT.verdict(f"{switch} alone: the printed headers and lines", printed == expected, expected, printed)
    drawn = bool(run["figures"])
    KIT.verdict(f"{switch} alone: figures {'written' if switch == PLOT_DEBUG else 'none'}",
                drawn == (switch == PLOT_DEBUG), switch == PLOT_DEBUG, {"figures": len(run["figures"]),
                                                                       "calls": run["calls"]})


def strategy_run(switches: dict) -> dict:
    """One strategy over the sine frame on a scratch copy whose dev copy sets the switches, stdout captured: the
    printed text, the output frame, the calls the loop made to each debug plot function (counted, then drawn), the
    file names of the figures saved and the figures under plots/<series> beside the trigger store."""
    originals = {name: getattr(realistic, name) for name in PLOTS}
    calls = dict.fromkeys(PLOTS, 0)
    for name in PLOTS:
        setattr(realistic, name, counted(name, originals[name], calls))
    savefig, saved = Figure.savefig, []
    Figure.savefig = recorded_save(savefig, saved)
    sink = io.StringIO()
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(REALISTIC), dev=dict(switches))
        strategy = testkit.strategy(testkit.validated_config(config_file), testkit.output_file(config_file, "trigger"))
        try:
            with contextlib.redirect_stdout(sink):
                frame = strategy.analyze_series(SERIES, testkit.sine_frame(ROWS, FIRST_DAY))
        finally:
            Figure.savefig = savefig
            for name, original in originals.items():
                setattr(realistic, name, original)
        plots = testkit.output_file(config_file, "trigger").parent / "plots" / SERIES
        figures = sorted(path.name for path in plots.glob("*.png"))
    KIT.log.info("switches %s: %d printed lines, calls %s, %d saves, %d figures", switches,
                 len(sink.getvalue().splitlines()), calls, len(saved), len(figures))
    return {"text": sink.getvalue(), "frame": frame, "calls": calls, "saved": saved, "figures": figures}


def counted(name: str, original: Callable, calls: dict) -> Callable:
    """A debug plot function that counts its call, then draws."""

    def call(*args, **kwargs):
        calls[name] += 1
        return original(*args, **kwargs)
    return call


def recorded_save(savefig: Callable, saved: list) -> Callable:
    """Figure.savefig that records the saved file's name, then saves."""

    def save(figure, path, *args, **kwargs):
        saved.append(Path(path).name)
        return savefig(figure, path, *args, **kwargs)
    return save


def summary_rows(lines: list[str]) -> list[str]:
    """The rows of the trigger summary table: the lines after its rule up to the first blank line."""
    start = next((row for row, line in enumerate(lines) if TABLE_RULE.match(line)), len(lines))
    return list(itertools.takewhile(str.strip, lines[start + 1:]))


# =============================================================================
# PIPELINE CHILDREN
# =============================================================================


def pipeline_case(case: str, switches: dict, expected: dict[str, Optional[str]]) -> None:
    """(c): --task trigger signal on a one-series copy with the switches on and every other off: exactly the
    expected headers, each block holding its marker; a full selection block holds the stored groups table's rows."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES),
                                          dev={**dict.fromkeys(SWITCHES, False), **switches})
        child = testkit.run_pipeline(KIT, folder, case.replace(" ", "_"), config_file, ["trigger", "signal"])
        found = header_blocks(testkit.plain_lines(child.text))
        KIT.verdict(f"{case}: return code 0", child.return_code == 0, 0, child.return_code)
        KIT.verdict(f"{case}: exactly the headers of its switches", set(found) == set(expected), sorted(expected),
                    sorted(found))
        for header, marker in ((header, marker) for header, marker in expected.items() if marker is not None):
            held = [marker in "\n".join(block) for block in found.get(header, [])]
            KIT.verdict(f"{case}: {marker!r} in every {header} block", bool(held) and all(held), "every block", held)
        if expected.get(SELECTION) == ALL_ROWS:
            groups = len(pd.read_hdf(testkit.output_file(config_file, "signal"), GROUPS_KEY))
            printed = [len(full_rows(block)) for block in found.get(SELECTION, [])]
            KIT.verdict(f"{case}: the {SELECTION} block prints the stored groups table's {groups} rows",
                        printed == [groups], [groups], printed)


# =============================================================================
# PRINTED BLOCKS
# =============================================================================


def header_blocks(lines: list[str]) -> dict[str, list[list[str]]]:
    """Per printed header, its blocks: the lines after each header line up to the next header line or the end."""
    starts = [(row, line.strip()) for row, line in enumerate(lines) if line.strip() in HEADERS]
    found: dict = {}
    for (row, header), (end, _) in zip(starts, starts[1:] + [(len(lines), "")]):
        found.setdefault(header, []).append(lines[row + 1:end])
    return found


def full_rows(block: list[str]) -> list[str]:
    """The frame rows a block prints after its "All rows:" line."""
    start = block.index(ALL_ROWS) if ALL_ROWS in block else len(block)
    return [line for line in block[start + 1:] if INDEX_ROW.match(line)]


if __name__ == "__main__":
    main()
