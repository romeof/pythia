"""Regression for the bad-row blocks counted by rows (F-20).

Preprocess.process_series on a scratch config copy (the shipped block threshold of three), each case
in its own folder and report, on a 20-row series from 2022-01-03 (five consecutive dates with a row, then
two without, repeated) whose var5 is zero on three adjacent rows: 2022-01-06, 2022-01-07, 2022-01-10 (dates
missing inside the block) -> one block of three in the report (before F-20: the calendar gap breaks the
block, nothing is reported); 2022-01-04 to 2022-01-06 (calendar-consecutive) -> one block (both trees);
2022-01-04, 2022-01-04, 2022-01-05 on an index that repeats one date -> one block (both trees; the tip
counts rows and makes no date lookup, which a repeated date would break). The report names each block by
its first and last dates.

Structure:
1. main
2. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.preprocess.preprocess import Preprocess

ROWS = 20
FIRST_DAY = "2022-01-03"  # five consecutive dates with a row from here, then two without, repeated
REASON = "Fails nonzero (var5)"
REPORT_FILE = "0_preprocess_report.txt"
BLOCK_PREFIX = "From "
# case: (bad rows, row whose date repeats the row before it, or None)
CASES = {"dates missing inside the block (2022-01-06, 2022-01-07, 2022-01-10)": ([3, 4, 5], None),
         "calendar-consecutive rows (2022-01-04 to 2022-01-06)": ([1, 2, 3], None),
         "a repeated date (2022-01-04, 2022-01-04, 2022-01-05)": ([1, 2, 3], 2)}


def main() -> None:
    """One report per case, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config = testkit.validated_config(testkit.config_copy(KIT, folder))
        for number, (case, (bad_rows, repeated_row)) in enumerate(CASES.items()):
            frame = series(bad_rows, repeated_row)
            report_folder = folder / f"case{number}"
            Preprocess(config.trigger.consecutive_rows_with_bad_preprocessed_data, config.trigger.validity,
                       config.trigger.preprocess.printout_io, report_folder / "0.h5").process_series("s", frame)
            expected = [f"{BLOCK_PREFIX}{frame.index[bad_rows[0]]} to {frame.index[bad_rows[-1]]}, Reason: {REASON}"]
            found = block_lines(report_folder / REPORT_FILE)
            KIT.verdict(f"{case}: one block of three bad rows in the report", found == expected, expected, found)
    KIT.finish()


# =============================================================================
# HELPERS
# =============================================================================


def series(bad_rows: list[int], repeated_row: Optional[int]) -> pd.DataFrame:
    """Twenty rows of rising values, var5 zero on the bad rows; with a repeated row, its date repeats the
    row before it."""
    dates = list(pd.bdate_range(FIRST_DAY, periods=ROWS))
    if repeated_row is not None:
        dates[repeated_row:] = dates[repeated_row - 1:-1]
    values = 100.0 + np.arange(ROWS, dtype=float)
    frame = pd.DataFrame({name: values for name in ("var0", "var3", "var4")}, index=pd.DatetimeIndex(dates))
    frame["var1"], frame["var2"] = values * 0.99, values * 1.01
    frame["var5"] = 1000000.0
    frame.iloc[bad_rows, frame.columns.get_loc("var5")] = 0.0
    return frame


def block_lines(report: Path) -> list[str]:
    """The report's block lines."""
    if not report.is_file():
        return []
    return [line.strip() for line in report.read_text(encoding="utf-8").splitlines() if line.startswith(BLOCK_PREFIX)]


if __name__ == "__main__":
    main()
