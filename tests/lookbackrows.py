"""Regression for the signal phase's series trim: the look-back is counted in rows, never in weekdays.

Slope and correlation read period_back rows before the row of each trigger begin
(_precompute_window_bounds). The trim before them must keep those rows whatever the calendar.
Three series, each with its earliest trigger on its last row:
(a) weekday rows with one weekday missing among the period_back weekdays before the begin: a weekday
    trim keeps period_back - 1 rows there, so the window shrinks;
(b) seven-day rows (every date observed): every window keeps period_back + 1 rows;
(c) a series whose rows start after the earliest begin: nothing before it, so nothing is cut.
Each window after the trim must equal the window on the untrimmed series (non-vacuity: the
untrimmed window holds period_back + 1 rows in (a) and (b)), and the trim must still drop the rows
the windows never read (non-vacuity of the trim itself).

Structure:
1. main
2. The cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import numpy as np
import pandas as pd

from src.utils.common.constants import TRIGGER_BEGIN_DATES, TRIGGER_END_DATES, VAR3
from src.utils.signal.selection import SignalSelection

ROWS = 30
MISSING_WEEKDAY = pd.Timestamp("2021-01-18")


def main() -> None:
    """The three calendars, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config = testkit.validated_config(testkit.config_copy(KIT, folder))
    period_back = config.signal.period_back
    weekdays = pd.bdate_range(end="2021-01-21", periods=ROWS + 1)
    weekdays = weekdays[weekdays != MISSING_WEEKDAY]
    begin = weekdays[-1]
    every_day = pd.date_range(end=begin, periods=ROWS, freq="D")
    later = pd.date_range(start=begin + pd.Timedelta(days=1), periods=ROWS, freq="D")
    KIT.verdict("non-vacuity: the missing weekday lies among the period_back weekdays before the begin",
                MISSING_WEEKDAY > begin - pd.offsets.BDay(period_back), f"> {begin - pd.offsets.BDay(period_back)}",
                MISSING_WEEKDAY)
    windows_survive_trim(config, period_back, begin, {"weekdays with one missing": weekdays, "seven-day rows": every_day,
                                                      "rows after the begin": later})
    KIT.finish()


# =============================================================================
# THE CASES
# =============================================================================


def windows_survive_trim(config, period_back: int, begin: pd.Timestamp, calendars: dict) -> None:
    """Each series' window at the earliest begin is the same before and after the trim."""
    def selection_for() -> SignalSelection:
        selection = SignalSelection(config.signal, config.trigger.replay.mode)
        for name, dates in calendars.items():
            selection.add_series_values(name, pd.DataFrame({VAR3: 100.0 + np.arange(len(dates))}, index=dates))
        return selection

    triggers = pd.DataFrame({TRIGGER_BEGIN_DATES: [begin], TRIGGER_END_DATES: [begin + pd.Timedelta(days=3)]})
    begins = np.array([begin.to_datetime64()], dtype="datetime64[ns]")
    untrimmed, trimmed = selection_for(), selection_for()
    trimmed._trim_series_to_trigger_range(triggers, period_back)
    for name in calendars:
        before = _window(untrimmed, name, begins, period_back)
        after = _window(trimmed, name, begins, period_back)
        expected_rows = 0 if name == "rows after the begin" else period_back + 1
        KIT.verdict(f"non-vacuity: {name}: the untrimmed window holds {expected_rows} rows", len(before) == expected_rows,
                    expected_rows, len(before))
        KIT.verdict(f"{name}: the trimmed window equals the untrimmed one", list(after) == list(before),
                    [str(day.date()) for day in before], [str(day.date()) for day in after])
    kept = len(trimmed._series["seven-day rows"])
    KIT.verdict("seven-day rows: the trim keeps exactly the period_back + 1 rows the window reads", kept == period_back + 1,
                period_back + 1, kept)


def _window(selection: SignalSelection, name: str, begins: np.ndarray, period_back: int) -> pd.DatetimeIndex:
    """The dates of a series' window at the begin, empty when the series has no window there."""
    if name not in selection._series:
        return pd.DatetimeIndex([])
    bounds = selection._precompute_window_bounds([name], begins, period_back)[(name, pd.Timestamp(begins[0]))]
    if bounds is None:
        return pd.DatetimeIndex([])
    return selection._series[name].index[bounds[0]:bounds[1] + 1]


if __name__ == "__main__":
    main()
