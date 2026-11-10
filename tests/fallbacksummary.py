"""Regression for the correlation fallback's summary record (C-6, R5).

In process, a production SignalSelection built as pointintimegrouping.py builds one: two series, A and B, on 30 daily
dates, one plus trigger each (A rows 8..20, B rows 10..22, so B's begin falls inside A's window and B correlates with
A's signal; A's first begin has period_back rows before it, so every window holds its full look-back). B lacks two
dates, rows 14 and 16 (deleted as corrcache.py deletes them), between the first begin and the last end and neither a
trigger's begin or end: the build trims both series to period_back rows before the earliest begin through the latest
end and compares the trimmed indexes, so a gap outside that span would leave one index and no fallback.
build_signal_window_groups() runs under a testkit.RecordCollector at WARNING on the src.utils.signal.selection logger.
Expected: exactly one record, which holds "fell back to date-aligned correlation" and the pair; no record holds the
per-pair text "mismatch between". Control: both series on one index give no record. Non-vacuity: both triggers reach
the groups table. Before C-6 every fallback computation logs its own "DatetimeIndex mismatch between" WARNING.

Structure:
1. main
2. The build
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import logging

import numpy as np
import pandas as pd

from src.utils.common.schema import SignalConfig
from src.utils.signal.selection import SignalSelection

ROWS = 30
FIRST_DAY = "2022-01-03"
DROPPED = [14, 16]
SPANS = {"A": (8, 20), "B": (10, 22)}
CASE_SIGNAL = {"corr_threshold": 0.5, "period_back": 5, "signal_criterion": "earliest_begin_highest_slope"}
CASE_MODE = "ideal"  # eligible signals taken by signal_end, as the case was written
SELECTION_LOGGER = "src.utils.signal.selection"
SUMMARY_TEXT = "fell back to date-aligned correlation"
PER_PAIR_TEXT = "mismatch between"
PAIRS = ("('A', 'B')", "('B', 'A')")


def main() -> None:
    """The build with B's gap and the control on one index, then the summary (return code 1 on any FAIL)."""
    messages, rows = warnings_of_build(dropped=DROPPED)
    KIT.verdict("non-vacuity: both triggers reach the groups table", rows == len(SPANS), len(SPANS), rows)
    KIT.verdict(f"B's gap: exactly one WARNING record, the summary ('{SUMMARY_TEXT}' and the pair)",
                len(messages) == 1 and SUMMARY_TEXT in messages[0] and any(pair in messages[0] for pair in PAIRS),
                f"one record naming {PAIRS[0]} or {PAIRS[1]}", messages)
    KIT.verdict(f"B's gap: no per-pair record ('{PER_PAIR_TEXT}')",
                not any(PER_PAIR_TEXT in message for message in messages), [], messages)
    control, control_rows = warnings_of_build(dropped=[])
    KIT.verdict("control, both series on one index: no WARNING record", not control and control_rows == len(SPANS),
                {"records": [], "rows": len(SPANS)}, {"records": control, "rows": control_rows})
    KIT.finish()


# =============================================================================
# THE BUILD
# =============================================================================


def warnings_of_build(dropped: list[int]) -> tuple[list[str], int]:
    """The WARNING messages of one build of the two series, B without the dropped rows, and the groups table's rows."""
    dates = pd.date_range(FIRST_DAY, periods=ROWS, freq="D")
    steps = np.arange(ROWS, dtype=float)
    levels = {"A": 100.0 + steps, "B": 50.0 + 0.5 * steps + np.sin(steps)}
    selection = SignalSelection(SignalConfig(**CASE_SIGNAL), CASE_MODE)
    selection.add_series_values("A", pd.DataFrame({"var3": levels["A"]}, index=dates))
    kept = np.setdiff1d(np.arange(ROWS), dropped)
    selection.add_series_values("B", pd.DataFrame({"var3": levels["B"][kept]}, index=dates[kept]))
    for series_id, (begin, end) in SPANS.items():
        selection.add_series_triggers(series_id, pd.DataFrame({
            "series_id": [series_id], "trigger_begin_dates": [dates[begin]], "trigger_end_dates": [dates[end]],
            "trigger_start_values": [levels[series_id][begin]], "trigger_end_values": [levels[series_id][end]],
            "is_plus_trigger": [True], "is_true": [True]}))
    collector = testkit.RecordCollector(logging.WARNING)
    logger = logging.getLogger(SELECTION_LOGGER)
    logger.addHandler(collector)
    try:
        groups = selection.build_signal_window_groups()
    finally:
        logger.removeHandler(collector)
    KIT.log.info("dropped %s: records %s; groups %s", dropped, collector.messages,
                 groups[["series_id", "group_id", "is_signal"]].to_dict("records"))
    return collector.messages, len(groups)


if __name__ == "__main__":
    main()
