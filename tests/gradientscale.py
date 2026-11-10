"""Regression for the gradient figure's top-axis scale (C-1, R1).

Two synthetic one-series inputs (testkit.sine_frame through testkit.write_raw_input), each read whole by its
config copy: long, 2,200 daily dates, whose curve index (from the effective start, row 49 under the shipped
lookback_window 50) spans about 5.9 calendar years; short, 730 daily dates, about 1.9 years. The copies turn off the
per-series plots and printouts, the gradient printouts and the per-signal plots (max_plots 0); the figure stays on.
Each runs a dual --task all under the dev profile in a child that wraps src.tasks.performance.plot_gradient: the
wrapper calls it with close_figure=False, reads the top axis of the one figure the call adds, prints
"GRADIENT_SCALE <mode> <first date> <last date> <yscale>" and releases the figure. The parent then replots each mode's
performance store with replot_gradient_from_hdf5(..., close_figure=False) and reads the top axis of the one figure that
call adds. Expected: long -> symlog through the pipeline call and through the replot, in both modes; short -> linear
both ways. Non-vacuity: one line per mode per run, the long curve index spanning more than LONG_WINDOW_YEARS calendar
years and the short one less. Before C-1 every top axis is linear (fixed 0.5 and 0.1 locators).

Structure:
1. main
2. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.performance.replot import replot_gradient_from_hdf5

SYNTHETIC_INPUT = Path("input") / "2026_01_01" / "0" / "raw" / "0.h5"
FIRST_DAY = "2015-01-01"
ROWS = {"long": 2200, "short": 730}
EXPECTED_SCALE = {"long": "symlog", "short": "linear"}
LONG_WINDOW_YEARS = 5  # oracle literal: the calendar-year span beyond which the top panel is symmetric-log
DEV_OVERRIDES = {"performance.perseries_trigger.plot_debug": False, "performance.perseries_trigger.printout_io": False,
                 "performance.perseries_signal.plot_debug": False, "performance.perseries_signal.printout_io": False,
                 "performance.gradient.printout_io": False, "performance.gradient.printout_debug": False,
                 "performance.gradient.max_plots": 0}
SCALE_PREFIX = "GRADIENT_SCALE "
CHILD_CODE = testkit.CHILD_BOOTSTRAP + (
    "\nfrom pathlib import Path\n"
    "import matplotlib.pyplot as plt\n"
    "import src.tasks.performance as performance\n"
    "original = performance.plot_gradient\n"
    "def wrapped(*args, **kwargs):\n"
    "    before = set(plt.get_fignums())\n"
    "    original(*args, **{**kwargs, 'close_figure': False})\n"
    "    added = sorted(set(plt.get_fignums()) - before)\n"
    "    scale = plt.figure(added[0]).axes[0].get_yscale() if len(added) == 1 else 'figures:' + str(len(added))\n"
    "    index = kwargs['global_idx']\n"
    f"    mode = next(part for part in Path(kwargs['save_path']).parts if part in {testkit.MODES!r})\n"
    f"    sys.stdout.write('{SCALE_PREFIX}' + ' '.join([mode, str(index[0].date()), str(index[-1].date()), scale]) + '\\n')\n"
    "    sys.stdout.flush()\n"
    "    for number in added:\n"
    "        plt.close(number)\n"
    "performance.plot_gradient = wrapped\n"
    "import pipeline\n"
    "sys.argv = ['pipeline.py', '--task', 'all', '--strategy', sys.argv[4], '--profile', testkit.DEV_PROFILE]\n"
    "pipeline.main()\n")


def main() -> None:
    """Both runs, their scale lines and replots, then the summary (return code 1 on any FAIL)."""
    for label, rows in ROWS.items():
        with testkit.scratch(KIT) as folder:
            config_file = synthetic_copy(folder, rows)
            child = testkit.run_child(KIT, folder, label, ["-c", CHILD_CODE, *testkit.child_args(KIT),
                                                           testkit.strategy_argument(config_file)],
                                      timeout_s=testkit.PIPELINE_TIMEOUT_S)
            KIT.verdict(f"{label}: the dual --task all run returns 0", child.return_code == 0, 0,
                        {"return_code": child.return_code, "timed_out": child.timed_out,
                         "last": testkit.plain_lines(child.text)[-1:]})
            lines = scale_lines(child.text)
            KIT.verdict(f"non-vacuity, {label}: one {SCALE_PREFIX.strip()} line per mode",
                        {mode: len(lines.get(mode, [])) for mode in testkit.MODES} == dict.fromkeys(testkit.MODES, 1),
                        dict.fromkeys(testkit.MODES, 1), lines)
            for mode in testkit.MODES:
                for first, last, scale in lines.get(mode, [])[:1]:
                    spans_long = pd.Timestamp(last) > pd.Timestamp(first) + pd.DateOffset(years=LONG_WINDOW_YEARS)
                    KIT.verdict(f"non-vacuity, {label} {mode}: the curve index {first} .. {last} spans "
                                f"{'more' if label == 'long' else 'less'} than {LONG_WINDOW_YEARS} calendar years",
                                spans_long == (label == "long"), {"spans more": label == "long"},
                                {"spans more": spans_long})
                    KIT.verdict(f"{label} {mode}: the pipeline's figure has a {EXPECTED_SCALE[label]} top axis",
                                scale == EXPECTED_SCALE[label], EXPECTED_SCALE[label], scale)
                replotted = replotted_scale(testkit.mode_output(config_file, mode, "performance"),
                                            folder / f"replot_{mode}.png")
                KIT.verdict(f"{label} {mode}: the replotted figure has a {EXPECTED_SCALE[label]} top axis",
                            replotted == EXPECTED_SCALE[label], EXPECTED_SCALE[label], replotted)
    KIT.finish()


# =============================================================================
# HELPERS
# =============================================================================


def synthetic_copy(folder: Path, rows: int) -> Path:
    """The one-series sine input inside the folder and a config copy that reads all of its dates."""
    raw = folder / SYNTHETIC_INPUT
    frame = testkit.sine_frame(rows, FIRST_DAY)
    testkit.write_raw_input(raw, {"0": frame})
    relative = Path(os.path.relpath(raw, Path.cwd())).as_posix()
    return testkit.config_copy(KIT, folder, strategy={"read.raw_data": f"./{relative}",
                                                      "read.ini_date": frame.index[0].strftime("%Y-%m-%d"),
                                                      "read.end_date": frame.index[-1].strftime("%Y-%m-%d")},
                               dev=dict(DEV_OVERRIDES))


def scale_lines(text: str) -> dict[str, list[tuple[str, str, str]]]:
    """The child's scale lines by mode: (first date, last date, top-axis scale)."""
    found: dict[str, list[tuple[str, str, str]]] = {}
    for line in testkit.plain_lines(text):
        if line.startswith(SCALE_PREFIX):
            mode, first, last, scale = line[len(SCALE_PREFIX):].split()
            found.setdefault(mode, []).append((first, last, scale))
    return found


def replotted_scale(performance_file: Path, save_path: Path) -> str:
    """The top-axis scale of the one figure the replot of the store adds, the figure then released."""
    before = set(plt.get_fignums())
    replot_gradient_from_hdf5(performance_file, save_path=save_path, close_figure=False)
    added = sorted(set(plt.get_fignums()) - before)
    scale = plt.figure(added[0]).axes[0].get_yscale() if len(added) == 1 else f"figures:{len(added)}"
    for number in added:
        plt.close(number)
    return scale


if __name__ == "__main__":
    main()
