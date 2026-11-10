"""Regression for the gate columns in reading and cleaning (C3).

In-process on a scratch config copy. (1) Read on a synthetic input whose series stores var0 to var5, var9, var12 and
var15, with the gate columns var12 and var15: the frame holds the value columns and the gate columns, in stored
sequence, without var9. (2) Preprocess.process_series under the shipped validity rules on testkit's daily sine series
with var12 0.9 and var15 0.8 added: a NaN var12 (row 2), a negative var12 (row 5) and a NaN var15 (row 8) keep their
rows and give no report line, while the control rows drop, each under the NaN reason: a NaN var3 (row 11) and a NaN
var5 (row 14; a value column stays cleaned whether or not a gate item names it). A NaN check over every column drops
rows 2 and 8, and a reader that keeps var0 to var5 only drops the gate columns.

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
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.preprocess.preprocess import Preprocess
from src.utils.preprocess.read import Read

ROWS = 20
FIRST_DAY = "2022-01-03"
SYNTHETIC_INPUT = Path("input") / "2026_01_01" / "0" / "raw" / "0.h5"
GATE_COLUMNS = ["var12", "var15"]
STORED = ["var0", "var1", "var2", "var3", "var4", "var5", "var9", "var12", "var15"]
KEPT = ["var0", "var1", "var2", "var3", "var4", "var5", "var12", "var15"]
GATE_VALUES = {"var12": 0.9, "var15": 0.8}
ISSUE_PREFIX = "Date: "
NAN_REASON = "Contains NaN values"
# planted row -> (column, value): the gate-column rows that stay, then the value-column controls that drop
KEPT_PLANTS = {2: ("var12", np.nan), 5: ("var12", -1.0), 8: ("var15", np.nan)}
DROPPED_PLANTS = {11: ("var3", np.nan), 14: ("var5", np.nan)}


def main() -> None:
    """The reader case, then the cleaning case, then the summary (return code 1 on any FAIL)."""
    reader_columns()
    cleaning()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def reader_columns() -> None:
    """Case (1): Read keeps the value columns and the gate columns, in stored sequence, and drops var9."""
    with testkit.scratch(KIT) as folder:
        frame = testkit.sine_frame(ROWS, FIRST_DAY)
        frame["var9"], frame["var12"], frame["var15"] = 1.0, GATE_VALUES["var12"], GATE_VALUES["var15"]
        raw = folder / SYNTHETIC_INPUT
        testkit.write_raw_input(raw, {"0": frame[STORED]})
        relative = Path(os.path.relpath(raw, Path.cwd())).as_posix()
        config = testkit.validated_config(testkit.config_copy(KIT, folder, strategy={
            "read.raw_data": f"./{relative}", "read.ini_date": FIRST_DAY, "read.end_date": str(frame.index[-1].date())}))
        columns = {series_id: list(df.columns) for series_id, df in Read(config.read, GATE_COLUMNS, False).iter_series()}
    KIT.verdict("case (1): Read keeps var0 to var5 and the gate columns var12 and var15, in stored sequence, without var9",
                columns == {"0": KEPT}, {"0": KEPT}, columns)


def cleaning() -> None:
    """Case (2): the gate-column rows stay without a report line; the value-column NaN rows drop under the NaN reason."""
    with testkit.scratch(KIT) as folder:
        frame = planted_frame()
        trigger = testkit.validated_config(testkit.config_copy(KIT, folder)).trigger
        cleaner = Preprocess(trigger.consecutive_rows_with_bad_preprocessed_data, trigger.validity,
                             trigger.preprocess.printout_io, folder / "report" / "0.h5")
        kept = cleaner.process_series("0", frame)
        dropped = sorted(frame.index.get_loc(date) for date in frame.index.difference(kept.index))
        issues = [line for line in cleaner.report_path.read_text(encoding="utf-8").splitlines()
                  if line.startswith(ISSUE_PREFIX)]
    survivors = [row for row in KEPT_PLANTS if row not in dropped]
    reported = [row for row in KEPT_PLANTS if any(str(frame.index[row]) in line for line in issues)]
    KIT.verdict("case (2): a NaN var12, a negative var12 and a NaN var15 keep their rows and give no report line",
                survivors == list(KEPT_PLANTS) and not reported, {"kept": list(KEPT_PLANTS), "reported": []},
                {"kept": survivors, "reported": reported})
    expected = [f"{ISSUE_PREFIX}{frame.index[row]}, Reason: {NAN_REASON}" for row in DROPPED_PLANTS]
    KIT.verdict("case (2): the controls drop, each under the NaN reason: a NaN var3 and a NaN var5 (a value column)",
                dropped == list(DROPPED_PLANTS) and issues == expected, {"dropped": list(DROPPED_PLANTS), "issues": expected},
                {"dropped": dropped, "issues": issues})


# =============================================================================
# HELPERS
# =============================================================================


def planted_frame() -> pd.DataFrame:
    """The sine series with var12 and var15 added and each planted value on its own row."""
    frame = testkit.sine_frame(ROWS, FIRST_DAY)
    for column, value in GATE_VALUES.items():
        frame[column] = value
    for row, (column, value) in {**KEPT_PLANTS, **DROPPED_PLANTS}.items():
        frame.iloc[row, frame.columns.get_loc(column)] = value
    return frame


if __name__ == "__main__":
    main()
