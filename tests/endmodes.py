"""Regression for the realistic end modes, the minimum duration and the suppression sequence (D-4; Q6, R-T6).

Realistic analyze_series on scratch config copies that pin the shipped find_peaks parameters, the
gate item var5 >= 2, a rolling 20-row window, and one end mode at a time. The V fixture: var3
linear from 140 (row 0) to 100 (row 40), up to 120 (row 60), down to 80 (row 100). The plus
trigger of the minimum on row 40 activates on row 41, begin value 101:
- against_threshold 0.05 -> ends on row 85, the first row with var3 <= 0.95 x 101;
- along_threshold 0.05 -> ends on row 47, the first row with var3 >= 1.05 x 101;
- max_duration 6 -> ends on row 47 = 41 + 6;
- opposite_trigger_as_begin_strategy -> ends on row 61 with "opposite_trigger" while the minus
  trigger of the maximum on row 60 begins on row 61 (the swap);
- different_end_strategy -> ends on row 61, the window's plus end at that maximum;
- against_threshold 0.05 with max_duration 44 -> both fire on row 85; the end reason is the first of
  the fixed checking sequence, against_threshold;
- the minimum duration: along_threshold 0.05 with min_duration_periods 10 -> no end before row 51, so it ends
  on row 51, though the change along the direction holds from row 47.
The suppression sequence: one series (a minimum on row 8, a 150 plateau to row 40, then a wave of
period 8 around 100 with amplitude 0.5, var5 at 1 on row 75, so row 76's previous bar fails the
gate item), no end mode, every suppression recorded through a wrapper on
_check_suppression_conditions: the first of each reason, in sequence, is before_effective_start
(row 19, candidate row 8), temporal_constraint (row 61, candidate 55: the plateau leaves the window
on row 60, the minus candidate of row 59 and the plus candidate of row 55 surface together and the
newer activates), consecutive_same_type (row 68, candidate 67) and gate_blocked (row 76,
candidate 75). The older-candidate activation and the effective start are
newestcandidate.py's and effectivestart.py's. A loop that checked max_duration first would end the
simultaneous case with max_duration.

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.trigger.realistic import RealisticModeMixin

PINNED = {"trigger.replay.mode": "realistic", "trigger.replay.lookback_window": 20,
          "trigger.replay.window_type": "rolling", "trigger.replay.min_duration_periods": 1,
          "trigger.strategy.min_distance": 5, "trigger.strategy.prominence_threshold": 0.05, "trigger.strategy.width": 1,
          "trigger.gate.items": [testkit.gate_item("var5", ">=", 2.0)], "trigger.replay.end.max_duration_periods": None,
          "trigger.replay.end.against_threshold_pct": None, "trigger.replay.end.along_threshold_pct": None,
          "trigger.replay.end.opposite_trigger_as_begin_strategy": False,
          "trigger.replay.end.different_end_strategy": False}
END = "trigger.replay.end."
V_KNOTS = [(0, 140.0), (40, 100.0), (60, 120.0), (100, 80.0)]
ACTIVATION_ROW = 41
# case: (overrides, plus end row, end reason)
END_CASES = {
    "against_threshold": ({END + "against_threshold_pct": 0.05}, 85, "against_threshold"),
    "along_threshold": ({END + "along_threshold_pct": 0.05}, 47, "along_threshold"),
    "max_duration": ({END + "max_duration_periods": 6}, 47, "max_duration"),
    "opposite_trigger": ({END + "opposite_trigger_as_begin_strategy": True}, 61, "opposite_trigger"),
    "different_end_strategy": ({END + "different_end_strategy": True}, 61, "different_end_strategy"),
    "against_threshold and max_duration on one row": ({END + "against_threshold_pct": 0.05, END + "max_duration_periods": 44},
                                                      85, "against_threshold"),
    "minimum duration of 10 rows under along_threshold": ({END + "along_threshold_pct": 0.05,
                                                           "trigger.replay.min_duration_periods": 10}, 51, "along_threshold"),
}
SWAP_ROW = 61
EARLY_KNOTS = [(0, 150.0), (8, 120.0), (15, 150.0)]
PLATEAU_ROWS, PLATEAU_LEVEL = 25, 150.0
WAVE_ROWS, WAVE_LEVEL, WAVE_STEP, WAVE_PERIOD = 80, 100.0, 0.25, 8
REDUCED_ROW = 75
# (row, candidate row, reason) of the first suppression of each reason, in sequence
FIRST_SUPPRESSIONS = [(19, 8, "before_effective_start"), (61, 55, "temporal_constraint"), (68, 67, "consecutive_same_type"),
                      (76, 75, "gate_blocked")]


def main() -> None:
    """The end cases and the suppression sequence, then the summary (return code 1 on any FAIL)."""
    for name, (overrides, end_row, reason) in END_CASES.items():
        end_case(name, overrides, end_row, reason)
    suppression_sequence()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def end_case(name: str, overrides: dict, end_row: int, reason: str) -> None:
    """The plus trigger activates on row 41 and ends on the expected row with the expected end reason."""
    output = analyzed(frame_of(np.interp(np.arange(V_KNOTS[-1][0] + 1), *zip(*V_KNOTS))), overrides)
    begins = np.flatnonzero(output["trigger_plus_begin"].to_numpy() > 0).tolist()
    ends = np.flatnonzero(output["trigger_plus_end"].to_numpy() > 0).tolist()
    found = {"plus begin": begins[:1], "plus end": ends[:1], "end_reason": output["end_reason"].iat[ends[0]] if ends else None}
    expected = {"plus begin": [ACTIVATION_ROW], "plus end": [end_row], "end_reason": reason}
    KIT.verdict(f"{name}: the plus trigger of row {ACTIVATION_ROW} ends on row {end_row} with {reason}", found == expected,
                expected, found)
    if name == "opposite_trigger":
        swapped = int(output["trigger_minus_begin"].iat[SWAP_ROW])
        KIT.verdict(f"{name}: the minus trigger begins on row {SWAP_ROW} (the swap)", swapped == 1, 1, swapped)


def suppression_sequence() -> None:
    """The first suppression of each reason, in sequence, with its row and candidate row."""
    early = np.interp(np.arange(EARLY_KNOTS[-1][0] + 1), *zip(*EARLY_KNOTS))
    quarter = WAVE_PERIOD // 4
    steps = quarter - np.abs((np.arange(WAVE_ROWS) + quarter) % WAVE_PERIOD - 2 * quarter)
    levels = np.concatenate([early, np.full(PLATEAU_ROWS, PLATEAU_LEVEL), WAVE_LEVEL + WAVE_STEP * steps])
    frame = frame_of(levels)
    frame.iloc[REDUCED_ROW, frame.columns.get_loc("var5")] = 1.0
    events: list = []
    check = RealisticModeMixin._check_suppression_conditions

    def recording(self, i, candidate_date, trigger_type, state, effective_start_date=None):
        reason = check(self, i, candidate_date, trigger_type, state, effective_start_date)
        if reason:
            events.append((i, int(frame.index.get_loc(candidate_date)), reason))
        return reason

    RealisticModeMixin._check_suppression_conditions = recording
    try:
        analyzed(frame, {})
    finally:
        RealisticModeMixin._check_suppression_conditions = check
    firsts, seen = [], set()
    for row, candidate_row, reason in events:
        if reason not in seen:
            seen.add(reason)
            firsts.append((row, candidate_row, reason))
    KIT.verdict("the suppression sequence: four distinct reasons, first seen on their expected rows",
                firsts == FIRST_SUPPRESSIONS, FIRST_SUPPRESSIONS, firsts)


# =============================================================================
# HELPERS
# =============================================================================


def frame_of(levels: np.ndarray) -> pd.DataFrame:
    """var3 = var4 = var0 the levels, var1 and var2 1 % below and above, var5 1e6, on weekdays from 2020-01-01."""
    values = np.asarray(levels, dtype="float32")
    return pd.DataFrame({"var0": values, "var1": values * np.float32(0.99), "var2": values * np.float32(1.01), "var3": values,
                         "var4": values, "var5": np.full(len(values), 1.0e6)},
                        index=pd.bdate_range("2020-01-01", periods=len(values)))


def analyzed(frame: pd.DataFrame, overrides: dict) -> Optional[pd.DataFrame]:
    """Realistic analyze_series on a scratch copy with the pinned settings and the case's overrides."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy={**PINNED, **overrides})
        strategy = testkit.strategy(testkit.validated_config(config_file), testkit.output_file(config_file, "trigger"))
        return strategy.analyze_series("0", frame)


if __name__ == "__main__":
    main()
