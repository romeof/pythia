"""EXP-04: no future rows in realistic mode (R-T7), length guards (R-T4), the gate's previous-bar dependence (HYP-03).

Inputs come from the raw input (Read + Preprocess), never from persisted tables. Each prefix
df.iloc[:i+1] (a fresh copy) goes through analyze_series in realistic mode, which recomputes
trigger_value and end_reason; decisions (begins, ends, end_reason and
turning_points, the realistic window count of Q14) are
compared with the full run on rows <= i and the recording column on rows <= i-1; the
active_trigger_* flags are excluded (they mark the active trigger at series end by design).
Cases: every series at the cuts in both recording modes; series 0 with an expanding window;
series 0 under following_row with a binding gate: (a2) up to 20 gate rows c at which the item
var4 >= 1.01 x var4[c-1] blocks the full run's begin at c (its gate_blocked suppression recorded,
never spelled); each cut logs its gate input var4[c-1] and the full run's begin flags; synthetic
series without a gate (pipelinekit.GATE_OFF) that exercise every end mode, minimum-duration
suppression and the swap; the length guards and the startup refusal of a lookback window below
min_length (E-7).
Planted-defect control: _window_bounds patched to include row i+1 must be flagged.

Structure:
1. main
2. Prefix comparison
3. Synthetic end-mode series
4. Length guards
5. Planted control
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

import pipelinekit
import synthetic
from src.utils.common.constants import (
    END_REASON, GATE_OP_GE, SUPPRESSION_GATE_BLOCKED, TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END, TRIGGER_PLUS_BEGIN,
    TRIGGER_PLUS_END, TRIGGER_VALUE, TRIGGER_VALUE_FOLLOWING_ROW, TRIGGER_VALUE_DETECTION_ROW, TURNING_POINTS, VAR4,
    WINDOW_EXPANDING,
)
from src.utils.common.schema import AppConfig
from src.utils.trigger.realistic import RealisticModeMixin

EXPERIMENT = "04"
CUTS = (60, 120, 300, 500, 673)
MAX_BINDING_CUTS = 20
BINDING_FACTOR = 1.01
CONTROL_CUTS = 5
DECISIONS = (TRIGGER_PLUS_BEGIN, TRIGGER_MINUS_BEGIN, TRIGGER_PLUS_END, TRIGGER_MINUS_END, END_REASON, TURNING_POINTS)
RECORDING = (TRIGGER_VALUE,)
BEGIN_COLUMNS = (TRIGGER_PLUS_BEGIN, TRIGGER_MINUS_BEGIN)
REFERENCE_SERIES = "0"
SYNTHETIC_SIZE = 320
SYNTHETIC_START = "2019-01-01"
SYNTHETIC_CUTS = (90, 150, 230, 319)
LOOKBACK_REFUSAL = "trigger.replay.lookback_window ({lookback}) < trigger.min_length ({min_length})"  # schema.py's message
END_CASES = {
    "against_threshold": {"trigger.replay.end.against_threshold_pct": 0.02},
    "along_threshold": {"trigger.replay.end.along_threshold_pct": 0.02},
    "max_duration": {"trigger.replay.end.max_duration_periods": 4},
    "different_end_strategy": {"trigger.replay.end.different_end_strategy": True,
                                "trigger.replay.end.opposite_trigger_as_begin_strategy": False},
    "minimum_duration": {"trigger.replay.min_duration_periods": 12},
    "swap": {},
}
EXPECTED_REASONS = {"against_threshold": "against_threshold", "along_threshold": "along_threshold",
                    "max_duration": "max_duration", "different_end_strategy": "different_end_strategy",
                    "minimum_duration": "opposite_trigger", "swap": "opposite_trigger"}


def main() -> None:
    """Real-series cuts, expanding window, binding cuts, synthetic ends, length guards, control."""
    _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    shipped = pipelinekit.load_copy(pipelinekit.write_config_copy(CTX, folder / "config"))
    series = pipelinekit.input_series(shipped, folder / "input")
    results = {}
    for recording in (TRIGGER_VALUE_DETECTION_ROW, TRIGGER_VALUE_FOLLOWING_ROW):
        config = realistic(shipped, recording=recording)
        results[f"cuts/{recording}"] = {sid: prefix_differences(sid, frame, config, CUTS) for sid, frame in sorted(series.items())}
        _verdict(f"cuts, {recording}", results[f"cuts/{recording}"])
    expanding = realistic(shipped, recording=TRIGGER_VALUE_DETECTION_ROW, window=WINDOW_EXPANDING)
    results["expanding"] = {REFERENCE_SERIES: prefix_differences(REFERENCE_SERIES, series[REFERENCE_SERIES], expanding, CUTS)}
    _verdict("expanding window, series 0, detection_row", results["expanding"])
    results["binding"] = check_binding_cuts(series[REFERENCE_SERIES], shipped)
    results["synthetic"] = check_synthetic(shipped)
    check_length_guards(shipped, folder)
    check_control(series, shipped)
    CTX.log.info("RESULTS %s", json.dumps(results, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


def realistic(shipped: AppConfig, *, recording: str, window: str = None, threshold: float = None, overrides: dict = None) -> AppConfig:
    """The shipped config in realistic mode with a recording mode and an optional window, the one gate item
    var4 >= threshold in place of the shipped items, and dotted overrides."""
    values = {"trigger.replay.mode": "realistic", "trigger.trigger_value": recording}
    if window is not None:
        values["trigger.replay.window_type"] = window
    if threshold is not None:
        values["trigger.gate.items"] = [pipelinekit.gate_item(VAR4, GATE_OP_GE, threshold)]
    return pipelinekit.with_values(shipped, {**values, **(overrides or {})})


def _verdict(name: str, found: dict) -> None:
    """One verdict per case: no difference at any cut of any series."""
    differing = {sid: {cut: rows for cut, rows in cuts.items() if rows} for sid, cuts in found.items()}
    differing = {sid: cuts for sid, cuts in differing.items() if cuts}
    compared = sum(len(cuts) for cuts in found.values())
    CTX.verdict(f"{name}: prefixes equal the full run (non-vacuity: {compared} cuts)", compared > 0 and not differing,
                "no difference", differing)


# =============================================================================
# PREFIX COMPARISON
# =============================================================================


def prefix_differences(series_id: str, df: pd.DataFrame, config: AppConfig, cuts: tuple) -> dict:
    """Per cut: decision differences on rows <= cut and recording differences on rows <= cut - 1."""
    return cut_differences(series_id, df, pipelinekit.analyze(series_id, df, config), config, cuts)


def cut_differences(series_id: str, df: pd.DataFrame, full: pd.DataFrame, config: AppConfig, cuts: tuple) -> dict:
    """Per cut, against a given full run: decision differences on rows <= cut, recording differences on rows <= cut - 1."""
    found = {}
    for cut in cuts:
        prefix = pipelinekit.analyze(series_id, df.iloc[:cut + 1], config)
        found[cut] = _differences(full.iloc[:cut + 1], prefix, DECISIONS) + _differences(full.iloc[:cut], prefix.iloc[:cut], RECORDING)
    return found


def _differences(left: pd.DataFrame, right: pd.DataFrame, columns: tuple) -> list:
    """(column, date) of every cell that differs between two aligned frames (NaN equal to NaN)."""
    found = []
    for column in columns:
        a, b = left[column].to_numpy(), right[column].to_numpy()
        differs = ~((a == b) | (pd.isna(a) & pd.isna(b)))
        found += [(column, str(left.index[row].date())) for row in np.flatnonzero(differs)]
    return found


def check_binding_cuts(frame: pd.DataFrame, shipped: AppConfig) -> dict:
    """(a2) Up to MAX_BINDING_CUTS gate rows c of series 0 at which the item var4 >= 1.01 x var4[c-1] blocks the full
    run's begin at c (its gate_blocked suppression recorded); prefix equals full on rows <= c (A4(a), Q3)."""
    rows = pipelinekit.realistic_gate_rows(REFERENCE_SERIES, frame, realistic(shipped, recording=TRIGGER_VALUE_FOLLOWING_ROW))
    found = {}
    for row in rows:
        if len(found) == MAX_BINDING_CUTS:
            break
        if row + 1 >= len(frame):
            continue
        threshold = BINDING_FACTOR * pipelinekit.gate_input(frame, row)
        config = realistic(shipped, recording=TRIGGER_VALUE_FOLLOWING_ROW, threshold=threshold)
        full, failed = gate_failures(REFERENCE_SERIES, frame, config)
        if row not in failed:
            continue
        found[row] = cut_record(frame, full, row, threshold)
        found[row]["differences"] = cut_differences(REFERENCE_SERIES, frame, full, config, (row,))[row]
    CTX.log.info("binding (a2): %d gate rows, cuts %s, records %s", len(rows), sorted(found), json.dumps(found, default=str))
    differing = {row: item["differences"] for row, item in found.items() if item["differences"]}
    CTX.verdict(f"binding (a2), series 0, following_row at the item var4 >= {BINDING_FACTOR} x var4[c-1]: prefixes equal the "
                f"full run (non-vacuity: {len(found)} binding cuts)", bool(found) and not differing,
                ">= 1 cut, no difference", differing or sorted(found))
    return found


def cut_record(frame: pd.DataFrame, full: pd.DataFrame, row: int, threshold: float) -> dict:
    """The logged inputs of one cut: its date, the threshold, the gate input var4[c-1] and the full run's begin
    flags."""
    return {"cut": row, "date": str(frame.index[row].date()), "threshold": threshold,
            "gate_input": pipelinekit.gate_input(frame, row),
            "full_begins": {column: int(full[column].iat[row]) for column in BEGIN_COLUMNS}}


def gate_failures(series_id: str, df: pd.DataFrame, config: AppConfig) -> tuple[pd.DataFrame, set]:
    """The full run and the rows at which its gate blocked a candidate (gate_blocked, recorded run)."""
    original = RealisticModeMixin._check_suppression_conditions
    failed: set = set()

    def recording(self, i, candidate_date, trigger_type, state, effective_start_date=None):
        reason = original(self, i, candidate_date, trigger_type, state, effective_start_date)
        if reason == SUPPRESSION_GATE_BLOCKED:
            failed.add(i)
        return reason

    RealisticModeMixin._check_suppression_conditions = recording
    try:
        full = pipelinekit.analyze(series_id, df, config)
    finally:
        RealisticModeMixin._check_suppression_conditions = original
    return full, failed


# =============================================================================
# SYNTHETIC END-MODE SERIES
# =============================================================================


def check_synthetic(shipped: AppConfig) -> dict:
    """Every end mode, minimum-duration suppression and the swap on a var3-driven synthetic series, without a gate."""
    t = np.arange(SYNTHETIC_SIZE, dtype=float)
    values = 100.0 + 8.0 * np.sin(2.0 * np.pi * t / 37.0) + 3.0 * np.sin(2.0 * np.pi * t / 11.0) + 0.02 * t
    frame = synthetic.daily_frame(values, SYNTHETIC_START)
    found = {}
    for case, overrides in END_CASES.items():
        config = realistic(shipped, recording=TRIGGER_VALUE_DETECTION_ROW, overrides={**pipelinekit.GATE_OFF, **overrides})
        full = pipelinekit.analyze(case, frame, config)
        reasons = sorted(set(full[END_REASON]) - {""})
        found[case] = {"end_reasons": reasons, "cuts": prefix_differences(case, frame, config, SYNTHETIC_CUTS)}
        CTX.log.info("synthetic %s: end reasons %s", case, reasons)
        _verdict(f"synthetic {case}", {case: found[case]["cuts"]})
    fired = {case: EXPECTED_REASONS[case] in found[case]["end_reasons"] for case in END_CASES}
    CTX.verdict("synthetic: each case shows its end reason", all(fired.values()), EXPECTED_REASONS,
                {case: item["end_reasons"] for case, item in found.items()})
    return found


# =============================================================================
# LENGTH GUARDS
# =============================================================================


def check_length_guards(shipped: AppConfig, folder: Path) -> None:
    """n = min_length - 1 and n = lookback - 1 return None (R-T4); lookback < min_length is refused at startup (E-7):
    the Trigger task on such a copy fails with the validation error naming both values, before the task starts. The
    synthetic series run without a gate."""
    config = realistic(shipped, recording=TRIGGER_VALUE_DETECTION_ROW, overrides=pipelinekit.GATE_OFF)
    min_length, lookback = config.trigger.min_length, config.trigger.replay.lookback_window
    values = 100.0 + np.sin(np.arange(SYNTHETIC_SIZE, dtype=float) / 3.0)
    sizes = {f"n = min_length - 1 = {min_length - 1}": min_length - 1, f"n = lookback - 1 = {lookback - 1}": lookback - 1}
    for name, size in sizes.items():
        out = pipelinekit.analyze(name, synthetic.daily_frame(values[:size], SYNTHETIC_START), config)
        CTX.verdict(f"length guard: {name}", out is None, "skipped (None)", "None" if out is None else f"{len(out)} rows")
    outcome = pipelinekit.startup_refusal(CTX, folder / "refusal", {"trigger.min_length": lookback + 1})
    message = LOOKBACK_REFUSAL.format(lookback=lookback, min_length=lookback + 1)
    facts = {"failed": outcome.failed, "validation_error": outcome.validation_error, "task_started": outcome.task_started,
             "message": message in outcome.child.text}
    CTX.verdict(f"length guard: lookback {lookback} < min_length {lookback + 1} refused at startup (E-7)",
                facts == {"failed": True, "validation_error": True, "task_started": False, "message": True},
                {"failed": True, "validation_error": True, "task_started": False, "message": message}, facts)


# =============================================================================
# PLANTED CONTROL
# =============================================================================


def check_control(series: dict[str, pd.DataFrame], shipped: AppConfig) -> None:
    """A window that includes row i+1 must break prefix equality at the patched run's first activations."""
    original = RealisticModeMixin._window_bounds

    def leaky(self, i: int, config: dict) -> tuple[int, int]:
        return original(self, i, config)[0], i + 2

    config = realistic(shipped, recording=TRIGGER_VALUE_DETECTION_ROW)
    frame = series[REFERENCE_SERIES]
    RealisticModeMixin._window_bounds = leaky
    try:
        full = pipelinekit.analyze(REFERENCE_SERIES, frame, config)
        begins = np.flatnonzero((full[TRIGGER_PLUS_BEGIN].to_numpy() > 0) | (full[TRIGGER_MINUS_BEGIN].to_numpy() > 0))
        cuts = tuple(int(row) for row in begins[:CONTROL_CUTS])
        found = prefix_differences(REFERENCE_SERIES, frame, config, cuts)
    finally:
        RealisticModeMixin._window_bounds = original
    CTX.verdict("control: a window including row i+1 is flagged", any(found.values()), "differences found",
                {cut: rows[:3] for cut, rows in found.items()})


if __name__ == "__main__":
    main()
