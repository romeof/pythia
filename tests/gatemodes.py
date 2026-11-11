"""Regression for the gate on the previous row in both replay modes (C2; the recording independence of HYP-03).

analyze_series of the extrema strategy built from a scratch config copy of each concrete mode with the one gate item
var12 >= 0.5, on a 200-row sine (var3 = var4 = 100 + 10 sin(2 pi t / 40), var0 = 0.5 x var3, var5 1e6) whose var12
is 1.0 on every row. Per mode: non-vacuity, two or more begins; with var12 set to 0 on the row before the second
begin r only, the run holds no begin on row r; with var12 set to 0 on row r itself only, the begin on row r stays.
Then var12 alternates in blocks of 15 rows (1.0, then 0.0): non-vacuity, a begin of the first run has a failing
previous row; every begin row's previous row passes; and the decision columns (begins, ends, end_reason) are identical
under the recording modes detection_row and following_row, with begins present. A gate on row i itself fails the removal
and the keeping verdicts; a gate that read a recorded value would make the recording modes differ.

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from pathlib import Path

import numpy as np
import pandas as pd

ROWS, PERIOD = 200, 40
LEVEL, AMPLITUDE = 100.0, 10.0
START = "2020-01-01"
THRESHOLD, PASSING, FAILING = 0.5, 1.0, 0.0
BLOCK_ROWS = 15
SHIPPED_RECORDING = "following_row"
RECORDINGS = ("detection_row", "following_row")
BEGINS = ("trigger_plus_begin", "trigger_minus_begin")
DECISIONS = (*BEGINS, "trigger_plus_end", "trigger_minus_end", "end_reason")


def main() -> None:
    """Both cases in each mode, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        for mode in testkit.MODES:
            config_file = mode_config(folder, mode)
            previous_row(config_file, mode)
            alternate_blocks(config_file, mode)
    KIT.finish()


def mode_config(folder: Path, mode: str) -> Path:
    """The scratch copy of one concrete mode with the one gate item var12 >= 0.5 (outputs inside the scratch folder)."""
    (folder / mode).mkdir()
    return testkit.config_copy(KIT, folder / mode, strategy={
        "trigger.replay.mode": mode, "trigger.gate.items": [testkit.gate_item("var12", ">=", THRESHOLD)]})


# =============================================================================
# CASES
# =============================================================================


def previous_row(config_file: Path, mode: str) -> None:
    """A failing row r-1 removes the begin on row r; a failing row r keeps it."""
    begins = begin_rows(analyze(config_file, SHIPPED_RECORDING, series_frame(np.full(ROWS, PASSING))))
    if not KIT.verdict(f"{mode}: non-vacuity, two or more begins with var12 passing on every row", len(begins) >= 2,
                       ">= 2", begins):
        return
    row = begins[1]
    before = begin_rows(analyze(config_file, SHIPPED_RECORDING, series_frame(failing_on(row - 1))))
    KIT.verdict(f"{mode}: var12 failing on row {row - 1} only removes the begin on row {row}", row not in before,
                f"no begin on row {row}", before)
    same = begin_rows(analyze(config_file, SHIPPED_RECORDING, series_frame(failing_on(row))))
    KIT.verdict(f"{mode}: var12 failing on row {row} only keeps the begin on row {row}", row in same,
                f"a begin on row {row}", same)


def alternate_blocks(config_file: Path, mode: str) -> None:
    """var12 failing on alternate blocks of rows: every begin's previous row passes, and the decisions ignore the
    recording mode."""
    var12 = np.where((np.arange(ROWS) // BLOCK_ROWS) % 2 == 0, PASSING, FAILING)
    unblocked = begin_rows(analyze(config_file, SHIPPED_RECORDING, series_frame(np.full(ROWS, PASSING))))
    blockable = [row for row in unblocked if var12[row - 1] < THRESHOLD]
    KIT.verdict(f"{mode}: alternate blocks, non-vacuity: a begin of the unblocked run has a failing previous row",
                bool(blockable), ">= 1", blockable)
    outputs = {recording: analyze(config_file, recording, series_frame(var12)) for recording in RECORDINGS}
    begins = begin_rows(outputs[SHIPPED_RECORDING])
    failing = [row for row in begins if row == 0 or var12[row - 1] < THRESHOLD]
    KIT.verdict(f"{mode}: alternate blocks, every begin row's previous row passes", bool(begins) and not failing,
                {"begins": ">= 1", "failing previous row": []}, {"begins": begins, "failing previous row": failing})
    counts = {recording: len(begin_rows(output)) for recording, output in outputs.items()}
    differing = decision_differences(outputs[RECORDINGS[0]], outputs[RECORDINGS[1]])
    KIT.verdict(f"{mode}: alternate blocks, decision columns identical under detection_row and following_row, with begins",
                counts[RECORDINGS[0]] > 0 and not differing, {"begins": "> 0", "differences": []},
                {"begins": counts, "first_differences": differing[:4]})


# =============================================================================
# HELPERS
# =============================================================================


def series_frame(var12: np.ndarray) -> pd.DataFrame:
    """The sine series: var3 = var4 the levels, var0 half of them, var5 1e6, and the given var12."""
    levels = LEVEL + AMPLITUDE * np.sin(2.0 * np.pi * np.arange(ROWS) / PERIOD)
    return pd.DataFrame({"var0": 0.5 * levels, "var3": levels, "var4": levels, "var5": np.full(ROWS, 1.0e6),
                         "var12": var12}, index=pd.date_range(START, periods=ROWS, freq="D"))


def failing_on(row: int) -> np.ndarray:
    """var12 passing on every row but the given one."""
    var12 = np.full(ROWS, PASSING)
    var12[row] = FAILING
    return var12


def analyze(config_file: Path, recording: str, frame: pd.DataFrame) -> pd.DataFrame:
    """analyze_series of the copy's strategy under one recording mode (the validated sections with
    trigger.trigger_value replaced), on a fresh copy of the frame."""
    config = testkit.validated_config(config_file)
    trigger = config.trigger.model_copy(update={"trigger_value": recording})
    configured = config.model_copy(update={"trigger": trigger})
    return testkit.strategy(configured, testkit.output_file(config_file, "trigger")).analyze_series("0", frame.copy())


def begin_rows(output: pd.DataFrame) -> list[int]:
    """Rows carrying a begin flag of either direction, ascending."""
    return np.flatnonzero(np.any([output[column].to_numpy() > 0 for column in BEGINS], axis=0)).tolist()


def decision_differences(left: pd.DataFrame, right: pd.DataFrame) -> list:
    """(column, date) of every differing decision cell."""
    return [(column, str(left.index[row].date())) for column in DECISIONS
            for row in np.flatnonzero(left[column].to_numpy() != right[column].to_numpy())]


if __name__ == "__main__":
    main()
