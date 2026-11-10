"""Regression for the end reason the ideal cleaning leaves (F-19, R-T8).

On a scratch config copy in ideal mode. Case (a): ideal analyze_series on EXP-01's [max, min, max]
fixture (97 samples of 100 + 10 sin(x), x in [0, 3 pi]: maxima on rows 16 and 80, the minimum on
row 48; var4 = var3, var5 = 1e6), whose plus end on row 16 has no begin before it, so the cleaning
drops it: end_reason is non-empty exactly on the end rows, 48 and 80. Case (b): the cleaning on
an overlapping pair, a plus pair on rows 2 and 8 and a minus pair on rows 5 and 11, each end row
carrying the end reason as the strategy's end generation writes it: the first pair wins, so
end_reason is non-empty exactly on row 8. Ideal analyze_series cannot produce such an overlap
(each signal type's ends are the other type's begins, so its pairs chain), so case (b) calls the
cleaning, _clean_duplicates, directly. The code before F-19 keeps end_reason on the ends the
cleaning drops: rows 16, 48 and 80 in (a), rows 8 and 11 in (b).

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


SAMPLES = 97
VALUE_COLUMNS = ("var0", "var1", "var2", "var3", "var4")
PLUS_BEGIN, PLUS_END = "trigger_plus_begin", "trigger_plus_end"
MINUS_BEGIN, MINUS_END = "trigger_minus_begin", "trigger_minus_end"
END_COLUMN = "end_reason"
IDEAL_REASON = "ideal_mode"
FIXTURE_ENDS = [48, 80]
OVERLAP_ROWS = 14
# (column, row): a plus pair (2, 8) and a minus pair (5, 11) that begins inside it
OVERLAP_FLAGS = ((PLUS_BEGIN, 2), (PLUS_END, 8), (MINUS_BEGIN, 5), (MINUS_END, 11))
OVERLAP_ENDS = [8]


def main() -> None:
    """Both cases, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy={"trigger.replay.mode": "ideal"})
        strategy = testkit.strategy(testkit.validated_config(config_file), testkit.output_file(config_file, "trigger"))
        unmatched_end(strategy)
        overlapping_pair(strategy)
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def unmatched_end(strategy: object) -> None:
    """Case (a): the dropped leading plus end keeps no end reason."""
    levels = (100.0 + 10.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, SAMPLES))).astype("float32")
    frame = pd.DataFrame({column: levels for column in VALUE_COLUMNS}, index=pd.bdate_range("2020-01-01", periods=SAMPLES))
    frame["var5"] = np.full(SAMPLES, 1.0e6)
    reason_rows_equal_ends("case (a): analyze_series", strategy.analyze_series("0", frame), FIXTURE_ENDS)


def overlapping_pair(strategy: object) -> None:
    """Case (b): the removed minus pair keeps no end reason."""
    frame = pd.DataFrame(0, index=pd.bdate_range("2020-01-01", periods=OVERLAP_ROWS),
                         columns=[PLUS_BEGIN, PLUS_END, MINUS_BEGIN, MINUS_END])
    for column, row in OVERLAP_FLAGS:
        frame.iloc[row, frame.columns.get_loc(column)] = 1
    frame[END_COLUMN] = ""
    for column in (PLUS_END, MINUS_END):
        frame.loc[frame[column] == 1, END_COLUMN] = IDEAL_REASON
    reason_rows_equal_ends("case (b): _clean_duplicates", strategy._clean_duplicates(frame), OVERLAP_ENDS)


# =============================================================================
# HELPERS
# =============================================================================


def reason_rows_equal_ends(case: str, output: pd.DataFrame, expected_ends: list[int]) -> None:
    """end_reason is non-empty exactly on the end rows, and those are the expected rows."""
    end_rows = np.flatnonzero(output[[PLUS_END, MINUS_END]].to_numpy().any(axis=1)).tolist()
    reason_rows = np.flatnonzero(output[END_COLUMN].to_numpy() != "").tolist()
    KIT.verdict(f"{case}: end_reason non-empty exactly on the end rows", end_rows == expected_ends == reason_rows,
                {"end rows": expected_ends, "end_reason rows": expected_ends},
                {"end rows": end_rows, "end_reason rows": reason_rows})


if __name__ == "__main__":
    main()
