"""Regression for the gate on the previous row, in-process: the begin vector, the applicable items, the pre-filter's
predicate, the missing-column error and the gate item's schema (C1, C5).

begin_allowed on a 6-row frame (var12 NaN, 0.9, 0.5, 0.83, 0.2, 1.0 and var15 0.8 with a NaN on row 1, both float32),
with the item var12 >= 0.83 alone and with var15 >= 0.73 beside it, equals a row-by-row oracle (row i allowed when
i > 0 and row i-1 passes every item, each comparison in the column's dtype) and the listed vector: row 0 is blocked
(no previous row), a NaN fails (row 1 after var12's, row 2 after var15's), and the float32 0.83 of row 3 passes
>= 0.83, so row 4 is allowed. An item var14 <= 5 on a rank column holding 0 on unranked rows passes there.
applicable_items drops the items whose exempt_series names the series; blocking_item returns the first item that
passes on no row, and None when each item passes on some row, though never on one row; begin_allowed on a frame
without var12 and var15 raises ValueError with the exact message. GateItem refuses the op ">", the column value12, a
NaN or infinite threshold and the exempt id 1x2, and accepts var12 exempting "0". A gate on row i itself, a shift that
wraps the last row onto row 0, or a comparison in float64 (0.83 as float32 is 0.82999998) fails a verdict.

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.utils.common.schema import GateItem
from src.utils.trigger.gate import applicable_items, begin_allowed, blocking_item

DATES = pd.date_range("2020-01-01", periods=6, freq="D")
COLUMNS = {"var12": [np.nan, 0.9, 0.5, 0.83, 0.2, 1.0], "var14": [0.0, 3.0, 0.0, 7.0, 0.0, 2.0],
           "var15": [0.8, np.nan, 0.8, 0.8, 0.8, 0.8]}
VAR12_ITEM = {"column": "var12", "op": ">=", "threshold": 0.83}
VAR15_ITEM = {"column": "var15", "op": ">=", "threshold": 0.73}
RANK_ITEM = {"column": "var14", "op": "<=", "threshold": 5.0}
# case: (items, the begin vector the rule gives on the frame)
VECTORS = {
    "var12 >= 0.83": ([VAR12_ITEM], [False, False, True, False, True, False]),
    "var12 >= 0.83 and var15 >= 0.73": ([VAR12_ITEM, VAR15_ITEM], [False, False, False, False, True, False]),
    "var14 <= 5 on a rank column, 0 on unranked rows": ([RANK_ITEM], [False, True, True, True, False, True]),
}
# blocking_item: an item no row passes, and two items that pass on different rows only (var12 on row 5, var14 on row 3)
NOWHERE_ITEM = {"column": "var12", "op": ">=", "threshold": 2.0}
ROW_5_ITEM = {"column": "var12", "op": ">=", "threshold": 0.95}
ROW_3_ITEM = {"column": "var14", "op": ">=", "threshold": 5.0}
MISSING_SERIES = "7"
MISSING_MESSAGE = "series 7 lacks the configured gate columns ['var12', 'var15']"
# case: the fields GateItem refuses
REFUSED = {
    'the op ">"': {**VAR12_ITEM, "op": ">"},
    "the column value12": {**VAR12_ITEM, "column": "value12"},
    "a NaN threshold": {**VAR12_ITEM, "threshold": float("nan")},
    "an infinite threshold": {**VAR12_ITEM, "threshold": float("inf")},
    "a minus infinite threshold": {**VAR12_ITEM, "threshold": float("-inf")},
    "the exempt id 1x2": {**VAR12_ITEM, "exempt_series": ["1x2"]},
}
ACCEPTED = {**VAR12_ITEM, "exempt_series": ["0"]}


def main() -> None:
    """The begin vectors, the exemption, the pre-filter's predicate, the missing columns and the schema, then the
    summary (return code 1 on any FAIL)."""
    frame = gate_frame()
    for name, (items, expected) in VECTORS.items():
        begin_vector(name, frame, [GateItem(**item) for item in items], expected)
    exemption()
    prefilter_predicate(frame)
    missing_columns(frame)
    schema_refusals()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def begin_vector(name: str, frame: pd.DataFrame, items: list, expected: list) -> None:
    """begin_allowed equals the row-by-row oracle and the listed vector."""
    found = begin_allowed("1", frame, items).tolist()
    checked = oracle(frame, items)
    KIT.verdict(f"begin_allowed, {name}: the row-by-row oracle and the listed vector", found == checked == expected,
                {"oracle": checked, "listed": expected}, found)


def exemption() -> None:
    """applicable_items drops the items whose exempt_series names the series."""
    items = [GateItem(**VAR12_ITEM, exempt_series=["0"]), GateItem(**VAR15_ITEM)]
    found = {series_id: [item.column for item in applicable_items(series_id, items)] for series_id in ("0", "1")}
    expected = {"0": ["var15"], "1": ["var12", "var15"]}
    KIT.verdict("applicable_items: the var12 item exempting series 0 applies to series 1 only", found == expected,
                expected, found)


def prefilter_predicate(frame: pd.DataFrame) -> None:
    """blocking_item: the first item no row passes; None when each item passes on some row."""
    nowhere = GateItem(**NOWHERE_ITEM)
    found = blocking_item(frame, [GateItem(**VAR15_ITEM), nowhere])
    KIT.verdict("blocking_item: the item that passes on no row is returned", found == nowhere, nowhere, found)
    separate = [GateItem(**ROW_5_ITEM), GateItem(**ROW_3_ITEM)]
    rows = {item.column: np.flatnonzero([compared(value, item) for value in frame[item.column].to_numpy()]).tolist()
            for item in separate}
    found = blocking_item(frame, separate)
    KIT.verdict("blocking_item: items passing on different rows only (none on one row) return None",
                found is None and rows == {"var12": [5], "var14": [3]}, {"returned": None, "rows": {"var12": [5], "var14": [3]}},
                {"returned": found, "rows": rows})


def missing_columns(frame: pd.DataFrame) -> None:
    """begin_allowed on a frame without the items' columns raises ValueError with the exact message."""
    exc = testkit.raised(lambda: begin_allowed(MISSING_SERIES, frame.drop(columns=["var12", "var15"]),
                                               [GateItem(**VAR12_ITEM), GateItem(**VAR15_ITEM)]))
    KIT.verdict("begin_allowed: a frame without var12 and var15 raises ValueError naming both and the series",
                isinstance(exc, ValueError) and str(exc) == MISSING_MESSAGE, f"ValueError: {MISSING_MESSAGE}", repr(exc))


def schema_refusals() -> None:
    """GateItem refuses each invalid field and accepts var12 exempting series 0."""
    for name, fields in REFUSED.items():
        exc = testkit.raised(lambda fields=fields: GateItem(**fields))
        KIT.verdict(f"GateItem refuses {name}", isinstance(exc, ValidationError), "ValidationError", repr(exc)[:240])
    exc = testkit.raised(lambda: GateItem(**ACCEPTED))
    KIT.verdict('GateItem accepts var12 >= 0.83 exempting "0"', exc is None, None, repr(exc)[:240])


# =============================================================================
# HELPERS
# =============================================================================


def gate_frame() -> pd.DataFrame:
    """The six rows of the gate columns, float32 as the reader stores them."""
    return pd.DataFrame({column: np.asarray(values, dtype="float32") for column, values in COLUMNS.items()}, index=DATES)


def oracle(frame: pd.DataFrame, items: list) -> list[bool]:
    """Row i allowed when i > 0 and row i-1 passes every item, row by row."""
    return [row > 0 and all(compared(frame[item.column].iat[row - 1], item) for item in items) for row in range(len(frame))]


def compared(value: np.floating, item: GateItem) -> bool:
    """value op threshold with the threshold in the value's dtype; a NaN passes no item."""
    threshold = type(value)(item.threshold)
    if np.isnan(value):
        return False
    return bool(value >= threshold) if item.op == ">=" else bool(value <= threshold)


if __name__ == "__main__":
    main()
