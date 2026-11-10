"""Regression for the newest-candidate rule of the realistic loop (Q7).

EXP-16's series, built inline: 60 rows at 150, then a triangle wave around 100 (step 0.25, period
16 rows), so every wave extremum stays hidden while the plateau is in the rolling window and all
appear on row 109, the first window without it; var5 on row 108 is 1, so row 109's previous bar
fails the gate item var5 >= 2 and its newest candidates are suppressed there. The scratch
config copy pins the shipped values these rows assume (rolling window of 50 rows, find_peaks
parameters) and the gate item, so an edit of workflows/extrema.yaml cannot hide the unfixed
loop's older activation. The realistic loop runs through analyze_series on that copy, with
wrappers on _window_candidates (the newest candidate of each type detected so far),
_check_suppression_conditions (every gate_blocked candidate) and _activate_trigger. Expected:
the gate blocks the newest candidate of each type on row 109 and on no other row (the premise);
every activation's candidate is the newest detected candidate of its type seen so far. The
unfixed loop activates on row 110 the plus candidate of row 88, older than the newest plus
candidate (row 104) suppressed on row 109.

Structure:
1. main
2. The series and the recorded run
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import numpy as np
import pandas as pd

from src.utils.common.constants import SUPPRESSION_GATE_BLOCKED
from src.utils.trigger.realistic import RealisticModeMixin

PLATEAU_ROWS, PLATEAU_LEVEL = 60, 150.0
WAVE_LEVEL, WAVE_STEP, PERIOD, WAVE_ROWS = 100.0, 0.25, 16, 160
STARVED_ROW, STARVED_VAR5 = 108, 1
BLOCKED_ROW = STARVED_ROW + 1  # the row whose previous bar fails the gate item
START = "2019-01-01"
BEGINS = ("trigger_plus_begin", "trigger_minus_begin")
PINNED = {"trigger.replay.mode": "realistic", "trigger.replay.window_type": "rolling",
          "trigger.replay.lookback_window": 50, "trigger.strategy.min_distance": 5,
          "trigger.strategy.prominence_threshold": 0.05, "trigger.strategy.width": 1,
          "trigger.gate.items": [testkit.gate_item("var5", ">=", 2.0)]}


def main() -> None:
    """The recorded run and its verdicts, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(PINNED))
        frame = series_frame()
        frame.iloc[STARVED_ROW, frame.columns.get_loc("var5")] = STARVED_VAR5
        strategy = testkit.strategy(testkit.validated_config(config_file), testkit.output_file(config_file, "trigger"))
        activations, blocked = recorded_run(strategy, frame)
    older = [item for item in activations if item["candidate_row"] < item["newest_row"]]
    premise = sorted((item["row"], item["type"], item["candidate_row"] == item["newest_row"]) for item in blocked)
    expected = sorted((BLOCKED_ROW, kind, True) for kind in BEGINS)
    KIT.verdict("non-vacuity: the loop activated triggers after the starved row",
                any(item["row"] > STARVED_ROW for item in activations), f"activations after row {STARVED_ROW}",
                activations[:4])
    KIT.verdict(f"premise: the gate blocks the newest candidate of each type on row {BLOCKED_ROW} and on no other row",
                premise == expected, expected, blocked)
    KIT.verdict("every activation's candidate is the newest detected candidate of its type seen so far",
                bool(activations) and not older, "none older", older)
    KIT.finish()


# =============================================================================
# THE SERIES AND THE RECORDED RUN
# =============================================================================


def series_frame() -> pd.DataFrame:
    """The plateau, then the triangle wave level + step x (4 - |(t + 4) mod 16 - 8|), as EXP-16 builds it."""
    quarter = PERIOD // 4
    steps = quarter - np.abs((np.arange(WAVE_ROWS) + quarter) % PERIOD - 2 * quarter)
    levels = np.concatenate([np.full(PLATEAU_ROWS, PLATEAU_LEVEL), WAVE_LEVEL + WAVE_STEP * steps]).astype("float32")
    frame = pd.DataFrame({name: levels for name in ("var0", "var3", "var4")},
                         index=pd.bdate_range(START, periods=len(levels)))
    frame["var1"] = (levels * 0.99).astype("float32")
    frame["var2"] = (levels * 1.01).astype("float32")
    frame["var5"] = np.full(len(levels), 1.0e6).astype("int32")
    return frame[["var0", "var1", "var2", "var3", "var4", "var5"]]


def recorded_run(strategy, frame: pd.DataFrame) -> tuple[list, list]:
    """Every activation and every gate_blocked suppression, each with its row, its type, its candidate row and the
    newest candidate row of its type detected so far."""
    newest: dict = {}
    activations: list = []
    blocked: list = []
    extract, activate = RealisticModeMixin._window_candidates, RealisticModeMixin._activate_trigger
    check = RealisticModeMixin._check_suppression_conditions

    def described(i: int, kind: str, candidate: pd.Timestamp) -> dict:
        return {"row": i, "type": kind, "candidate_row": int(frame.index.get_loc(candidate)),
                "newest_row": int(frame.index.get_loc(newest[kind]))}

    def extracting(self, df, start, stop):
        found = extract(self, df, start, stop)
        for kind, detected in zip(BEGINS, found[:2]):
            if detected:
                newest[kind] = max(detected) if kind not in newest else max(newest[kind], max(detected))
        return found

    def checking(self, i, candidate_date, trigger_type, state, effective_start_date=None):
        reason = check(self, i, candidate_date, trigger_type, state, effective_start_date)
        if reason == SUPPRESSION_GATE_BLOCKED:
            blocked.append(described(i, trigger_type, candidate_date))
        return reason

    def activation(self, df, i, current_idx, new_candidate, state):
        kind, candidate = new_candidate
        activations.append(described(i, kind, candidate))
        return activate(self, df, i, current_idx, new_candidate, state)

    RealisticModeMixin._window_candidates, RealisticModeMixin._activate_trigger = extracting, activation
    RealisticModeMixin._check_suppression_conditions = checking
    try:
        strategy.analyze_series("0", frame.copy())
    finally:
        RealisticModeMixin._window_candidates, RealisticModeMixin._activate_trigger = extract, activate
        RealisticModeMixin._check_suppression_conditions = check
    return activations, blocked


if __name__ == "__main__":
    main()
