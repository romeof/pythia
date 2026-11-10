"""Fixture for the opposite extremum seen during the minimum duration (Q6, as coded on both tips).

EXP-16's series, built inline (60 rows at 150, then a triangle wave around 100, step 0.25,
period 16 rows), with trigger.replay.min_duration_periods 8 on a realistic scratch config
copy that also pins the shipped values the rows below assume (rolling window of 50 rows,
find_peaks parameters, end settings; no gate item), through analyze_series with wrappers
on _suppress_pending_opposite (row, candidate and the reason the loop passes, logged as read)
and _activate_trigger. Expected: the plus trigger
activated on row 109 holds its minimum duration of eight rows; the opposite (minus) candidate of row 112,
seen on row 113, is suppressed as the pending opposite within that minimum duration and is never
activated; the trigger ends on row 129, where the opposite trigger begins (the swap). The
decided behaviour is the coded one, so the fixture passes on both tips.

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

from src.utils.trigger.realistic import RealisticModeMixin

PLATEAU_ROWS, PLATEAU_LEVEL = 60, 150.0
WAVE_LEVEL, WAVE_STEP, PERIOD, WAVE_ROWS = 100.0, 0.25, 16, 160
MIN_DURATION = 8
START = "2019-01-01"
ACTIVATION_ROW, SEEN_ROW, LOST_CANDIDATE_ROW, END_ROW = 109, 113, 112, 129
PINNED = {"trigger.replay.mode": "realistic", "trigger.replay.min_duration_periods": MIN_DURATION,
          "trigger.replay.window_type": "rolling", "trigger.replay.lookback_window": 50,
          "trigger.strategy.min_distance": 5, "trigger.strategy.prominence_threshold": 0.05,
          "trigger.strategy.width": 1, "trigger.replay.end.max_duration_periods": 500, "trigger.replay.end.against_threshold_pct": None,
          "trigger.replay.end.along_threshold_pct": None,
          "trigger.replay.end.opposite_trigger_as_begin_strategy": True,
          "trigger.replay.end.different_end_strategy": False}


def main() -> None:
    """The recorded run and its verdicts, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(PINNED))
        strategy = testkit.strategy(testkit.validated_config(config_file), testkit.output_file(config_file, "trigger"))
        output, pending, activations = recorded_run(strategy, series_frame())
    first = activations[0] if activations else {}
    KIT.verdict(f"non-vacuity: the first activation is a plus trigger on row {ACTIVATION_ROW}",
                first.get("row") == ACTIVATION_ROW and first.get("type") == "trigger_plus_begin",
                {"row": ACTIVATION_ROW, "type": "trigger_plus_begin"}, first)
    lost = [item for item in pending if item["row"] == SEEN_ROW and item["candidate_row"] == LOST_CANDIDATE_ROW]
    KIT.verdict(f"the minus candidate of row {LOST_CANDIDATE_ROW}, seen on row {SEEN_ROW}, is suppressed as the pending "
                f"opposite within the minimum duration ({SEEN_ROW} - {ACTIVATION_ROW} < {MIN_DURATION})",
                len(lost) == 1 and lost[0]["type"] == "trigger_minus_begin" and SEEN_ROW - ACTIVATION_ROW < MIN_DURATION,
                "one pending-opposite suppression", lost or pending[:3])
    KIT.verdict(f"the candidate of row {LOST_CANDIDATE_ROW} is never activated",
                bool(activations) and all(item["candidate_row"] != LOST_CANDIDATE_ROW for item in activations),
                "not among the activations", activations[:4])
    swap = [item for item in activations if item["row"] == END_ROW and item["type"] == "trigger_minus_begin"]
    plus_end = int(output["trigger_plus_end"].iat[END_ROW]) if output is not None else None
    KIT.verdict(f"the trigger ends on row {END_ROW} with the swap (plus end and minus begin on that row)",
                len(swap) == 1 and plus_end == 1, {"plus_end": 1, "minus_activation": 1},
                {"plus_end": plus_end, "minus_activations": swap})
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


def recorded_run(strategy, frame: pd.DataFrame) -> tuple:
    """The output, every pending-opposite suppression (row, type, candidate row, reason) and every activation."""
    pending: list = []
    activations: list = []
    suppress, activate = RealisticModeMixin._suppress_pending_opposite, RealisticModeMixin._activate_trigger

    def suppressing(self, series_id, df, window_df, i, current_idx, current_value, pending_opposite, reason, *args):
        pending.append({"row": i, "type": pending_opposite[0], "candidate_row": int(frame.index.get_loc(pending_opposite[1])),
                        "reason": reason})
        return suppress(self, series_id, df, window_df, i, current_idx, current_value, pending_opposite, reason, *args)

    def activation(self, df, i, current_idx, new_candidate, state):
        activations.append({"row": i, "type": new_candidate[0], "candidate_row": int(frame.index.get_loc(new_candidate[1]))})
        return activate(self, df, i, current_idx, new_candidate, state)

    RealisticModeMixin._suppress_pending_opposite, RealisticModeMixin._activate_trigger = suppressing, activation
    try:
        output = strategy.analyze_series("0", frame.copy())
    finally:
        RealisticModeMixin._suppress_pending_opposite, RealisticModeMixin._activate_trigger = suppress, activate
    return output, pending, activations


if __name__ == "__main__":
    main()
