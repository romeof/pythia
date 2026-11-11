"""Regression for the observed-date curve index (W5, W8).

(a) The two-series knots of endtoend.py on 36 consecutive calendar days, Sunday 2019-12-29 to Sunday
2020-02-02, as a raw input, and a copy reading those dates with lookback_window 10 and min_distance 2
(endtoend.py's settings), dual --task all: per mode, /total_variation's dates are the input's from row 9,
2020-01-07, to row 35, every calendar day included. (b) --task all in ideal mode on the shipped input:
/total_variation holds 626 rows, its dates are the union, within its first and last date, of the trigger
file's /series_* dates and each reference's observed dates (its var3 dates without missing values in the
signal file's /reference_series_*, the first max_references in sorted sequence), and 2020-01-01, a weekday
no series observes, is absent. (c) The replot of (a)'s ideal performance file in this process: its two date
labels below the bottom axis are 2020-01-07 and 2020-02-02, a Sunday. (d) A copy of that file without
/total_variation replots with the same two labels, now from the reference curves (series 0 is the shipped
reference). A weekday calendar fails all four: it drops (a)'s Saturday and Sunday dates, its last label falls
on 2020-01-31, and it adds 2020-01-01 to (b).

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import os
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.performance.replot import replot_gradient_from_hdf5

SYNTHETIC_INPUT = Path("input") / "2026_01_01" / "0" / "raw" / "0.h5"
DATES = pd.date_range("2019-12-29", periods=36, freq="D")
KNOTS = {"0": [(0, 115.0), (15, 100.0), (30, 130.0), (35, 110.0)], "1": [(0, 130.0), (15, 100.0), (30, 160.0), (35, 120.0)]}
LOOKBACK = 10
CURVE_DATES = pd.date_range("2020-01-07", "2020-02-02", freq="D")
LABELS = ["2020-01-07", "2020-02-02"]
SHIPPED_ROWS = 626
ABSENT_DATE = pd.Timestamp("2020-01-01")
TOTAL_KEY, TOTAL_COLUMN, DATE_COLUMN = "/total_variation", "total_variation", "date"
SERIES_PREFIX, REFERENCE_PREFIX = "/series_", "/reference_series_"
VAR3 = "var3"
DATE_LABEL = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FIGURE_FILE = "gradient_hd5.png"


def main() -> None:
    """The synthetic dual run with its two replots, then the shipped input in ideal mode, then the summary (return
    code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = synthetic_copy(folder)
        child = testkit.run_pipeline(KIT, folder, "dual", config_file, ["all"])
        KIT.verdict("(a) the dual --task all returns 0", child.return_code == 0, 0,
                    {"return_code": child.return_code, "timed_out": child.timed_out})
        if child.return_code == 0:
            every_calendar_day(config_file)
            replot_labels(testkit.mode_output(config_file, "ideal", "performance"), folder / "replot")
    shipped_union()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def every_calendar_day(config_file: Path) -> None:
    """(a) Per mode, /total_variation's dates are the input's from row 9 to row 35, every calendar day included."""
    for mode in testkit.MODES:
        dates = curve_dates(testkit.mode_output(config_file, mode, "performance"))
        KIT.verdict(f"(a) {mode}: {TOTAL_KEY}'s dates are the input's from 2020-01-07 to 2020-02-02, every calendar day "
                    "included",
                    dates.equals(CURVE_DATES), [day(CURVE_DATES[0]), day(CURVE_DATES[-1]), len(CURVE_DATES)],
                    [day(dates[0]), day(dates[-1]), len(dates)] if len(dates) else [])


def replot_labels(source: Path, folder: Path) -> None:
    """(c) The replot's two date labels; (d) the same labels from a copy without /total_variation."""
    folder.mkdir()
    labels = date_labels(source, folder / FIGURE_FILE)
    KIT.verdict("(c) the replot of the ideal performance file labels 2020-01-07 and 2020-02-02", labels == LABELS, LABELS,
                labels)
    copy = folder / "without_total" / source.name
    copy.parent.mkdir()
    shutil.copyfile(source, copy)
    with pd.HDFStore(str(copy), mode="a") as store:
        store.remove(TOTAL_KEY)
    labels = date_labels(copy, copy.parent / FIGURE_FILE)
    KIT.verdict(f"(d) a copy without {TOTAL_KEY} replots with the same labels, from the reference curves", labels == LABELS,
                LABELS, labels)


def shipped_union() -> None:
    """(b) --task all in ideal mode on the shipped input: 626 dates, the union of the observed dates, 2020-01-01 absent."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy={testkit.replay_path(KIT, "mode"): "ideal"})
        child = testkit.run_pipeline(KIT, folder, "ideal", config_file, ["all"])
        KIT.verdict("(b) the ideal --task all on the shipped input returns 0", child.return_code == 0, 0,
                    {"return_code": child.return_code, "timed_out": child.timed_out})
        found = {"rows": None, "union": None, "absent": None}
        if child.return_code == 0:
            dates = curve_dates(testkit.output_file(config_file, "performance"))
            limit = testkit.loaded_config(config_file)["performance"]["gradient"]["max_references"]
            observed = observed_union(testkit.output_file(config_file, "trigger"), testkit.output_file(config_file, "signal"),
                                      limit)
            within = observed[(observed >= dates[0]) & (observed <= dates[-1])] if len(dates) else observed
            found = {"rows": len(dates), "union": dates.equals(within), "absent": ABSENT_DATE not in dates}
        expected = {"rows": SHIPPED_ROWS, "union": True, "absent": True}
        KIT.verdict(f"(b) {TOTAL_KEY}: {SHIPPED_ROWS} rows, the union of the observed dates, {day(ABSENT_DATE)} absent",
                    found == expected, expected, found)


# =============================================================================
# HELPERS
# =============================================================================


def synthetic_copy(folder: Path) -> Path:
    """The two-series input on the 36 calendar days inside the folder and a shipped-configuration copy reading it."""
    raw = folder / SYNTHETIC_INPUT
    frames = {}
    for sid, knots in KNOTS.items():
        levels = np.interp(np.arange(len(DATES)), *zip(*knots))
        frames[sid] = pd.DataFrame({"var0": levels, "var1": levels * 0.99, "var2": levels * 1.01, "var3": levels,
                                    "var4": levels, "var5": np.full(len(DATES), 1.0e6)}, index=DATES)
    testkit.write_raw_input(raw, frames)
    relative = Path(os.path.relpath(raw, Path.cwd())).as_posix()
    return testkit.config_copy(KIT, folder, strategy={
        "read.raw_data": f"./{relative}", "read.ini_date": day(DATES[0]), "read.end_date": day(DATES[-1]),
        testkit.replay_path(KIT, "lookback_window"): LOOKBACK, "trigger.strategy.min_distance": 2})


def curve_dates(path: Path) -> pd.DatetimeIndex:
    """The dates of a performance file's /total_variation."""
    with pd.HDFStore(str(path), mode="r") as store:
        return pd.DatetimeIndex(pd.to_datetime(store.get(TOTAL_KEY)[DATE_COLUMN].to_numpy()))


def observed_union(trigger_file: Path, signal_file: Path, limit: int) -> pd.DatetimeIndex:
    """The sorted union of each processed series' dates (the trigger file's /series_*) and each reference's observed
    dates (its var3 dates without missing values in the signal file, the first limit references in sorted sequence)."""
    with pd.HDFStore(str(trigger_file), mode="r") as store:
        indexes = [store.get(key).index for key in sorted(store.keys()) if key.startswith(SERIES_PREFIX)]
    with pd.HDFStore(str(signal_file), mode="r") as store:
        references = [store.get(key) for key in sorted(store.keys()) if key.startswith(REFERENCE_PREFIX)]
    kept = [frame for frame in references if len(frame) and VAR3 in frame.columns][:limit]
    indexes += [frame[VAR3].dropna().index for frame in kept]
    return pd.DatetimeIndex(sorted({pd.Timestamp(date) for index in indexes for date in index}))


def date_labels(source: Path, save_path: Path) -> list[str]:
    """The date labels below the bottom axis of the one figure a replot of source adds, the figure then released."""
    before = set(plt.get_fignums())
    replot_gradient_from_hdf5(source, save_path=save_path, close_figure=False)
    added = sorted(set(plt.get_fignums()) - before)
    if len(added) != 1:
        return [f"{len(added)} figures added"]
    figure = plt.figure(added[0])
    texts = figure.axes[1].texts if len(figure.axes) > 1 else []
    labels = [text.get_text() for text in texts if DATE_LABEL.match(text.get_text())]
    plt.close(figure)
    return labels


def day(date: object) -> str:
    """A date as YYYY-MM-DD."""
    return pd.Timestamp(date).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
