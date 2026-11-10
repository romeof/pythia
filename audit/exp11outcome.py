"""EXP-11: trigger extraction edge cases (R-T9, trigger.md:247-253).

Unit cases on production TriggerData: nested begins, an end before any begin, a begin and an
end on the same row, begins without ends, and the is_true rule per direction.

Structure:
1. main
2. Cases
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common.constants import (
    IS_PLUS_TRIGGER, IS_TRUE, TRIGGER_BEGIN_DATES, TRIGGER_END_DATES, TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END,
    TRIGGER_PLUS_BEGIN, TRIGGER_PLUS_END, TRIGGER_VALUE,
)
from src.utils.trigger.outcome import TriggerData

ROWS = 10
DAYS = pd.bdate_range("2022-01-03", periods=ROWS)
CASES = {
    "nested begins": ({TRIGGER_PLUS_BEGIN: [1, 3], TRIGGER_PLUS_END: [5, 7]}, [(1, 5, True), (3, 7, True)]),
    "end before any begin": ({TRIGGER_PLUS_BEGIN: [3], TRIGGER_PLUS_END: [1, 5]}, [(3, 5, True)]),
    "begin and end on one row": ({TRIGGER_MINUS_BEGIN: [4], TRIGGER_MINUS_END: [4, 6]}, [(4, 6, False)]),
    "begins without ends": ({TRIGGER_PLUS_BEGIN: [2, 6]}, []),
}


def main() -> None:
    """Each case's extracted (begin, end, direction) against trigger.md:247-253; is_true per direction."""
    _parse_args()
    helper = TriggerData()
    found = {}
    for name, (flags, expected) in CASES.items():
        triggers = helper._extract_triggers_from_trigger_columns(name, frame(flags))
        pairs = [] if triggers is None else [
            (DAYS.get_loc(begin), DAYS.get_loc(end), bool(plus))
            for begin, end, plus in zip(triggers[TRIGGER_BEGIN_DATES], triggers[TRIGGER_END_DATES], triggers[IS_PLUS_TRIGGER])]
        found[name] = sorted(pairs)
        CTX.verdict(f"{name}: pairs as documented", sorted(pairs) == sorted(expected), sorted(expected), sorted(pairs))
    found["is_true"] = check_outcome(helper)
    CTX.log.info("RESULTS %s", json.dumps(found, sort_keys=True))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


def frame(flags: dict) -> pd.DataFrame:
    """A detection frame with the given flag rows and a rising trigger_value."""
    columns = {column: np.zeros(ROWS, dtype=int) for column in (TRIGGER_PLUS_BEGIN, TRIGGER_PLUS_END,
                                                               TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END)}
    for column, rows in flags.items():
        columns[column][rows] = 1
    return pd.DataFrame({**columns, TRIGGER_VALUE: np.arange(ROWS, dtype=float) + 100.0}, index=DAYS)


def check_outcome(helper: TriggerData) -> dict:
    """Plus true when end > start; minus true when end < start; equal values false for both."""
    values = np.array([100.0, 105.0, 105.0, 95.0, 95.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    flags = {TRIGGER_PLUS_BEGIN: [0, 2], TRIGGER_PLUS_END: [1, 4], TRIGGER_MINUS_BEGIN: [5, 7], TRIGGER_MINUS_END: [6, 9]}
    data = frame(flags)
    data[TRIGGER_VALUE] = values
    data.iloc[7, data.columns.get_loc(TRIGGER_VALUE)] = 110.0
    triggers = helper._extract_triggers_from_trigger_columns("outcome", data)
    found = [(DAYS.get_loc(begin), bool(plus), bool(true)) for begin, plus, true in
             zip(triggers[TRIGGER_BEGIN_DATES], triggers[IS_PLUS_TRIGGER], triggers[IS_TRUE])]
    expected = [(0, True, True), (2, True, False), (5, False, False), (7, False, True)]
    CTX.verdict("is_true per direction (equal values are false)", sorted(found) == expected, expected, sorted(found))
    return {"found": sorted(found)}


if __name__ == "__main__":
    main()
