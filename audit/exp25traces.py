"""EXP-25: the six traces of the reference run, value by value (TR-0 to TR-6).

TR-0 reads series 0 through the production reader and preprocessor (raw rows, the inclusive
date filter, rows dropped, the reports, the saved reference rows). TR-1 replays the realistic loop of
series 0 from var3 and the gate columns (scipy find_peaks on every rolling window, the newest unmarked
and unsuppressed extremum per type, the four suppression checks, the gate among them on the previous
row, the minimum duration and the opposite-trigger end) up to its first swap
and records every input of that activation. TR-2 recomputes the ideal turning points of every
series on the full series. TR-3 recomputes three persisted outcomes from /series_0. TR-4
recomputes two realistic signal groups (eligibility, six-point windows, Pearson, slope
sequence). TR-5 reads the performance terminal values, metrics and breakdowns. TR-6 checks
the orchestration records of RUN-1 and RUN-2. Each recomputed value is judged against the
persisted one and against the anchors measured on run13, each citing the decisions that moved
it from master's value. Non-vacuity: every anchor row
found. Planted control: a perturbed trigger_value must change a TR-3 outcome.

Structure:
1. main
2. TR-0 read, filter and preprocess
3. TR-1 realistic replay
4. TR-2 ideal turning points
5. TR-3 outcomes
6. TR-4 signal groups
7. TR-5 performance
8. TR-6 orchestration
9. Shared readers
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import json
import operator
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import pipelinekit
from src.utils.common.common import tall_to_wide
from src.utils.common.constants import (
    ACTIVE_TRIGGER_MINUS_BEGIN, ACTIVE_TRIGGER_PLUS_BEGIN, BEGIN_COLS, BEGIN_TO_OPPOSITE, CORR_TO_SIGNAL, END_IDEAL_MODE,
    END_OPPOSITE_TRIGGER, END_REASON, GATE_OP_GE, GATE_OP_LE, GRADIENT_STRATEGY_HEADER, GRADIENT_STRATEGY_SENTINEL, GROUP_ID,
    HDF5_GRADIENT_FILTERED_SEQUENCE, HDF5_GRADIENT_METRICS_TABLE, HDF5_GRADIENT_REFERENCE_VARIATION,
    HDF5_GRADIENT_TERMINAL_ATTENUATIONS, HDF5_SIGNALS_BREAKDOWN_TABLE, IS_PLUS_TRIGGER, IS_SIGNAL, IS_TRUE,
    SERIES_ID, SERIES_IDS, SIGNALS_COUNT_LABEL, SLOPE, TRIGGER_BEGIN_DATES, TRIGGER_END_DATES,
    TRIGGER_END_VALUES, TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END, TRIGGER_PLUS_BEGIN, TRIGGER_PLUS_END, TRIGGER_START_VALUES,
    TRIGGER_VALUE, TURNING_POINTS, VAR0, VAR3,
)
from src.utils.common.schema import AppConfig, GateItem, StrategyConfig

EXPERIMENT = "25"
SERIES = "0"
IDEAL, REALISTIC = "ideal", "realistic"
PARAMS = harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR)
REPORT_FILE = "0_preprocess_report.txt"
REPORT_HEADER_LINES = 2
OUTCOME_ROWS = ((REALISTIC, 0), (REALISTIC, 45), (IDEAL, 9))
TR4_GROUPS = (1, 6)
FLOAT32_TOLERANCE = 1e-5
PLANTED_SHIFT = 1000.0
PERFORMANCE_METRICS = ("acr", "std", "fall_dur")
BREAKDOWN_CELLS = ("plus+minus_true+false", "plus+minus_true", "plus+minus_false")
METRICS_COLUMNS = ("metric_key", "column_header", "value")
BREAKDOWN_COLUMNS = ("row_key", "column_header", "value")
CURVE_COLUMNS = {"/total_variation": "total_variation", "/total_variation_attenuated": "total_variation_attenuated"}
DATE_COLUMN, REFERENCE_COLUMN, VARIATION_COLUMN, AMOUNT_COLUMN = "date", "reference_id", "variation", "amount"
CONFIG_COPIES = ("extrema.yaml", "dev.yaml")
RUN_NAMES = ("run1", "run2")
MODE_LINE = "Running mode: "
CLEANED_LINE = "Cleaned previous outputs from "
SUMMARY_LINE = "This progress looks :)"
PLAN_ANCHORS = {
    # reference_first 2019-03-11 -> 2019-03-14 (Q2); rows, first and last as at master
    "TR-0": {"rows": 675, "first": "2019-01-02", "last": "2021-09-03", "reference_first": "2019-03-14"},
    # TR-1 and TR-2 as at master: the --real-data recorder flagged no series at W5's parent (Q7); TR-2's ideal_mode_rows
    # 20 -> 19 (F-19, an earlier revision: the ideal cleaning clears the end reasons of dropped rows; the reference made at an earlier revision)
    "TR-1": {"loop_start": "2019-03-14", "first_begin": [TRIGGER_MINUS_BEGIN, "2019-03-22"], "first_end": "2019-03-29",
             "swap": [TRIGGER_PLUS_BEGIN, "2019-03-29"], "plus_begins": 45, "minus_begins": 46, "opposite_ends": 90,
             "active": [[ACTIVE_TRIGGER_MINUS_BEGIN, "2021-08-26"]]},
    "TR-2": {"turning_points": 20, "maxima": 10, "minima": 10, "first": ["maximum", "2019-05-03"],
             "last": ["minimum", "2021-05-12"], "plus_pairs": 9, "minus_pairs": 10, "ideal_mode_rows": 19, "ends": 19},
    # realistic: signals 233 -> 232 (F-01) -> 231 (Q20); kept 231 -> 230 (F-01) -> 229 (Q20) -> 231 (Q4); first,
    # reference_last and the reference's attenuations amount (Q2); terminal and attenuated (F-01, Q20, Q4; the last-year attenuation
    # is 0.0, so Q17 moves none); acr (F-01, Q2, Q20, Q4); std (F-01, Q2, Q20, Q4) 14.5410 -> 14.5414 and fall_dur
    # (renamed by W3) 396.0 as at master -> 384.0, both by W5's observed-date curve index; breakdown (F-01, Q20;
    # the sweep removes no signal, so Q21 moves none). ideal: signals 64 -> 63, terminal and acr (Q2)
    "TR-5": {REALISTIC: {"signals": 231, "kept": 231, "first": "2019-03-14", "terminal": -0.042369, "attenuated": -0.059964,
                         "acr": -1.734, "std": 14.5414, "fall_dur": 384.0, "breakdown": [231, 97, 134],
                         "attenuations": {GRADIENT_STRATEGY_SENTINEL: 0.0, "0": 0.299498}, "reference_last": {"0": 1.197991}},
             IDEAL: {"signals": 63, "terminal": 3.822976, "acr": 88.8361}},
}
ANCHOR_DECIMALS = 4
# the documented comparison of each gate item op
GATE_COMPARISONS = {GATE_OP_GE: operator.ge, GATE_OP_LE: operator.le}


def main() -> None:
    """Every trace on the reference copy, then the planted control."""
    args = _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    config = pipelinekit.load_copy(pipelinekit.write_config_copy(CTX, folder / "config"))
    trace_read(config, args.ref_copy, folder)
    trace_realistic(config, args.ref_copy)
    trace_ideal(config, args.ref_copy)
    trace_outcomes(args.ref_copy)
    trace_signals(config, args.ref_copy)
    trace_performance(args.ref_copy)
    trace_orchestration(args.run_copy, dict(zip(RUN_NAMES, args.run_logs)))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--ref-copy, --run-copy (RUN-1's labelled copy), --run-logs (the pipeline logs of RUN-1 and RUN-2, file names
    under data\\logs: a harness label is never reused, so a repeated run gets a new log name) and --source-root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref-copy", type=Path, default=harness.REAL_WORKTREE / "data" / "ref" / harness.DATA_OUTPUT_DIR)
    parser.add_argument("--run-copy", type=Path, default=harness.REAL_WORKTREE / "data" / "run1" / harness.DATA_OUTPUT_DIR)
    parser.add_argument("--run-logs", nargs=2, required=True, metavar=("RUN1_LOG", "RUN2_LOG"))
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


# =============================================================================
# TR-0 READ, FILTER AND PREPROCESS
# =============================================================================


def trace_read(config: AppConfig, ref: Path, folder: Path) -> None:
    """Raw rows, the date filter, preprocessing and the saved reference rows of series 0."""
    reader = pipelinekit.reader_of(config)
    with h5py.File(reader.input_file, "r") as source:
        full = reader._read_series(SERIES, source)
    in_range = reader._apply_date_filter(full)
    processed = pipelinekit.preprocessor_of(config, folder / "input.h5").process_series(SERIES, in_range.copy())
    reference = read_key(task_file(ref, REALISTIC, "trigger"), f"/reference_series_{SERIES}")
    found = {"raw_rows": len(full), "raw_first": day(full.index[0]), "raw_last": day(full.index[-1]),
             "rows": len(in_range), "first": day(in_range.index[0]), "last": day(in_range.index[-1]),
             "bounds": [config.read.ini_date, config.read.end_date], "preprocessed_rows": len(processed), "reports": {mode: report_lines(ref, mode) for mode in (IDEAL, REALISTIC)},
             "reference_rows": len(reference), "reference_first": day(reference.index[0])}
    CTX.log.info("TRACE TR-0 %s", json.dumps(found, default=str))
    anchors = PLAN_ANCHORS["TR-0"]
    CTX.verdict("TR-0: in-range rows and dates (inclusive bounds)", [found["rows"], found["first"], found["last"]] ==
                [anchors["rows"], anchors["first"], anchors["last"]], anchors, found)
    CTX.verdict("TR-0: preprocessing drops no row and the reports hold only their header",
                found["preprocessed_rows"] == found["rows"] and set(found["reports"].values()) == {REPORT_HEADER_LINES},
                {"rows": found["rows"], "report_lines": REPORT_HEADER_LINES},
                {"rows": found["preprocessed_rows"], "report_lines": found["reports"]})
    CTX.verdict("TR-0: the reference rows start at the positional effective start (Q2)", found["reference_first"] ==
                anchors["reference_first"], anchors["reference_first"], found["reference_first"])


def report_lines(ref: Path, mode: str) -> int:
    """Non-empty lines of one persisted preprocess report."""
    text = (ref / PARAMS / mode / "trigger" / REPORT_FILE).read_text(encoding="utf-8")
    return sum(bool(line.strip()) for line in text.splitlines())


# =============================================================================
# TR-1 REALISTIC REPLAY
# =============================================================================


def trace_realistic(config: AppConfig, ref: Path) -> None:
    """The independent replay of series 0 up to its first swap, against the persisted flags and the counts."""
    path = task_file(ref, REALISTIC, "trigger")
    frame = read_key(path, f"/series_{SERIES}")
    triggers = read_key(path, f"/triggers_{SERIES}")
    replay = replay_first_swap(frame, config)
    CTX.log.info("TRACE TR-1 replay %s", json.dumps(replay, default=str))
    if replay["end"] is None:
        raise RuntimeError(f"the replay of series {SERIES} found no opposite swap")
    begins = frame.index[(frame[TRIGGER_PLUS_BEGIN] == 1) | (frame[TRIGGER_MINUS_BEGIN] == 1)]
    first, end_row = replay["activation"], replay["end"]
    persisted = {"first_begin": [TRIGGER_MINUS_BEGIN if frame.at[begins[0], TRIGGER_MINUS_BEGIN] == 1 else TRIGGER_PLUS_BEGIN,
                                 day(begins[0])],
                 "first_end_reason": frame.at[pd.Timestamp(end_row["date"]), END_REASON],
                 "swap_flag": int(frame.at[pd.Timestamp(end_row["date"]), end_row["swap"][0]])}
    counts = {"loop_start": day(frame.index[config.trigger.replay.lookback_window - 1]),
              "plus_begins": int(frame[TRIGGER_PLUS_BEGIN].sum()), "minus_begins": int(frame[TRIGGER_MINUS_BEGIN].sum()),
              "opposite_ends": int((frame[END_REASON] == END_OPPOSITE_TRIGGER).sum()),
              "active": active_flags(frame), "triggers_rows": len(triggers),
              "plus_rows_first": bool(triggers[IS_PLUS_TRIGGER].iloc[:int(triggers[IS_PLUS_TRIGGER].sum())].all())}
    CTX.log.info("TRACE TR-1 persisted %s counts %s", json.dumps(persisted, default=str), json.dumps(counts, default=str))
    anchors = PLAN_ANCHORS["TR-1"]
    CTX.verdict("TR-1: the replay's first activation is the persisted first begin",
                [first["kind"], first["date"]] == persisted["first_begin"], persisted["first_begin"], [first["kind"], first["date"]])
    CTX.verdict("TR-1: the replay's first end is the persisted opposite-trigger end and swap",
                persisted["first_end_reason"] == END_OPPOSITE_TRIGGER and persisted["swap_flag"] == 1,
                [END_OPPOSITE_TRIGGER, 1], [persisted["first_end_reason"], persisted["swap_flag"]])
    CTX.verdict("TR-1: the begin records the next row's var0 (following_row)",
                first["values"]["trigger_value"] == first["values"]["next_var0"], "trigger_value == var0[i+1]",
                first["values"])
    observed = {"loop_start": counts["loop_start"], "first_begin": persisted["first_begin"], "first_end": end_row["date"],
                "swap": list(end_row["swap"]), "plus_begins": counts["plus_begins"],
                "minus_begins": counts["minus_begins"], "opposite_ends": counts["opposite_ends"], "active": counts["active"]}
    CTX.verdict("TR-1: the plan's anchors reproduced", observed == anchors, anchors, observed)
    CTX.verdict("TR-1: /triggers_0 lists every plus row before the minus rows", counts["plus_rows_first"], True,
                counts["plus_rows_first"])


def replay_first_swap(frame: pd.DataFrame, config: AppConfig) -> dict:
    """The realistic loop replayed from var3 up to the first opposite swap, with every input of the first activation.

    Mirrors the documented rule (trigger.md:147-198) for the shipped end configuration (opposite-trigger
    end and max_duration only): each rolling window's extrema come from scipy find_peaks; the newest
    extremum of a type seen so far is its only candidate (Q7), taken when the window reports it and it is
    neither marked nor suppressed; the suppression checks run in their documented sequence; an opposite
    candidate within the minimum duration is suppressed for good, after it the active trigger ends and
    the opposite begins on the same row. The gate reads the persisted frame's gate columns on the previous row,
    for the items whose exempt_series does not name series 0.
    """
    replay = config.trigger.replay
    lookback, min_duration = replay.lookback_window, replay.min_duration_periods
    items, dates = [item for item in config.trigger.gate.items if SERIES not in item.exempt_series], frame.index
    effective = dates[lookback - 1]
    marked, suppressed, latest, active, activation = {kind: set() for kind in BEGIN_COLS}, set(), None, None, None
    newest = {kind: None for kind in BEGIN_COLS}
    for i in range(lookback - 1, len(frame)):
        found = window_extrema(frame[VAR3].to_numpy(), i - lookback + 1, i, config.trigger.strategy)
        candidate, opposite = None, None
        for kind in BEGIN_COLS:
            detected = {dates[row] for row in found[kind]}
            if detected:
                newest[kind] = max(detected) if newest[kind] is None else max(newest[kind], max(detected))
            date = newest[kind]
            if date is None or date not in detected or date in marked[kind] or date in suppressed:
                continue
            checks = suppression_checks(frame, i, date, kind, effective, latest, active, items)
            if any(checks.values()):
                suppressed.add(date)
                continue
            if active is not None and kind == BEGIN_TO_OPPOSITE[active["kind"]]:
                opposite = {"kind": kind, "candidate": day(date), "checks": checks}
                continue
            if candidate is None or date > candidate[1]:
                candidate = (kind, date, checks)
        if active is None:
            if candidate is None:
                continue
            kind, date, checks = candidate
            marked[kind].add(date)
            latest = date
            active = {"kind": kind, "row": i}
            activation = activation_record(frame, i, date, kind, checks, found, effective, items)
        if i - active["row"] < min_duration:
            if opposite is not None:
                suppressed.add(pd.Timestamp(opposite["candidate"]))
            continue
        if opposite is not None:
            return {"activation": activation, "end": end_record(frame, i, active, opposite, min_duration)}
    return {"activation": activation, "end": None}


def window_extrema(values: np.ndarray, start: int, end: int, strategy: StrategyConfig) -> dict:
    """Absolute rows of the minima (plus begins) and maxima (minus begins) find_peaks reports on one window."""
    window = values[start:end + 1]
    data_range = float(window.max() - window.min())
    options = {"distance": strategy.min_distance, "prominence": strategy.prominence_threshold * data_range,
               "width": strategy.width}
    maxima, _ = find_peaks(window, **options)
    minima, _ = find_peaks(-window, **options)
    return {TRIGGER_PLUS_BEGIN: [start + int(row) for row in minima], TRIGGER_MINUS_BEGIN: [start + int(row) for row in maxima],
            "bounds": [start, end], "range": data_range, "prominence": options["prominence"]}


def suppression_checks(frame: pd.DataFrame, i: int, date: pd.Timestamp, kind: str, effective: pd.Timestamp,
                       latest: Optional[pd.Timestamp], active: Optional[dict], items: list[GateItem]) -> dict:
    """The four suppression checks in their documented sequence (True = suppressed; gate_blocked is the gate on the
    previous row: row i-1 must pass value op threshold for every applicable item, in the column's dtype, a NaN failing
    and a first row blocked)."""
    passes = i > 0 and all(GATE_COMPARISONS[item.op](frame[item.column].iloc[i - 1], item.threshold) for item in items)
    return {"before_effective_start": bool(date < effective),
            "temporal_constraint": bool(latest is not None and date <= latest),
            "gate_blocked": not passes,
            "consecutive_same_type": bool(active is not None and active["kind"] == kind)}


def activation_record(frame: pd.DataFrame, i: int, date: pd.Timestamp, kind: str, checks: dict, found: dict,
                      effective: pd.Timestamp, items: list[GateItem]) -> dict:
    """Window bounds, find_peaks inputs, candidate lag, suppression inputs (each applicable item's value on the
    previous row) and recorded values of one activation."""
    dates = frame.index
    return {"kind": kind, "row": i, "date": day(dates[i]), "candidate": day(date),
            "lag_rows": int(i - dates.get_loc(date)), "window": [day(dates[found["bounds"][0]]), day(dates[found["bounds"][1]])],
            "data_range": found["range"], "prominence": found["prominence"],
            "extrema": {name: [day(dates[row]) for row in found[name]] for name in BEGIN_COLS},
            "checks": checks, "effective_start": day(effective),
            "gate": {item.column: float(frame[item.column].iloc[i - 1]) for item in items},
            "values": {"trigger_value": float(frame[TRIGGER_VALUE].iloc[i]),
                       "next_var0": float(frame[VAR0].iloc[i + 1]), "var3": float(frame[VAR3].iloc[i])}}


def end_record(frame: pd.DataFrame, i: int, active: dict, opposite: dict, min_duration: int) -> dict:
    """The end-check inputs (var3 only) of the first opposite-trigger end and the swap it begins."""
    return {"row": i, "date": day(frame.index[i]), "duration_rows": i - active["row"], "min_duration": min_duration,
            "begin_var3": float(frame[VAR3].iloc[active["row"]]), "end_var3": float(frame[VAR3].iloc[i]),
            "opposite_candidate": opposite["candidate"], "opposite_checks": opposite["checks"],
            "swap": (opposite["kind"], day(frame.index[i]))}


def active_flags(frame: pd.DataFrame) -> list:
    """(column, date) of every active-trigger flag."""
    return [[column, day(date)] for column in (ACTIVE_TRIGGER_PLUS_BEGIN, ACTIVE_TRIGGER_MINUS_BEGIN)
            for date in frame.index[frame[column] == 1]]


# =============================================================================
# TR-2 IDEAL TURNING POINTS
# =============================================================================


def trace_ideal(config: AppConfig, ref: Path) -> None:
    """Full-series turning points of every series against the persisted ideal flags; pairs and flags of series 0."""
    path = task_file(ref, IDEAL, "trigger")
    strategy = config.trigger.strategy
    per_series, mismatched = {}, []
    with pd.HDFStore(str(path), mode="r") as store:
        frames = {key[len("/series_"):]: store.get(key) for key in sorted(store.keys()) if key.startswith("/series_")}
        triggers = store.get(f"/triggers_{SERIES}")
    for sid, frame in sorted(frames.items()):
        found = window_extrema(frame[VAR3].to_numpy(), 0, len(frame) - 1, strategy)
        points = sorted(found[TRIGGER_PLUS_BEGIN] + found[TRIGGER_MINUS_BEGIN])
        if points != [int(row) for row in np.flatnonzero(frame[TURNING_POINTS].to_numpy() == 1)]:
            mismatched.append(sid)
        maxima, minima = found[TRIGGER_MINUS_BEGIN], found[TRIGGER_PLUS_BEGIN]
        kinds = {row: "maximum" for row in maxima} | {row: "minimum" for row in minima}
        per_series[sid] = {"maxima": len(maxima), "minima": len(minima), "first": [kinds[points[0]], day(frame.index[points[0]])],
                           "last": [kinds[points[-1]], day(frame.index[points[-1]])], "range": found["range"],
                           "prominence": found["prominence"]}
    frame = frames[SERIES]
    found = per_series[SERIES]
    series_zero = {"begins_minima": found["minima"], "ends_maxima": found["maxima"],
                 "first_end_before_first_begin": found["first"][0] == "maximum",
                 "plus_pairs": int(triggers[IS_PLUS_TRIGGER].sum()), "minus_pairs": int((~triggers[IS_PLUS_TRIGGER]).sum()),
                 "active": active_flags(frame), "ideal_mode_rows": int((frame[END_REASON] == END_IDEAL_MODE).sum()),
                 "ends": int(frame[TRIGGER_PLUS_END].sum() + frame[TRIGGER_MINUS_END].sum())}
    CTX.log.info("TRACE TR-2 per series %s; series 0 %s", json.dumps(per_series, default=str), json.dumps(series_zero, default=str))
    CTX.verdict("TR-2: recomputed turning points equal the persisted ones (every series)", not mismatched, "none differ",
                mismatched)
    anchors = PLAN_ANCHORS["TR-2"]
    observed = {"turning_points": found["maxima"] + found["minima"], "maxima": found["maxima"], "minima": found["minima"],
                "first": found["first"], "last": found["last"], "plus_pairs": series_zero["plus_pairs"],
                "minus_pairs": series_zero["minus_pairs"], "ideal_mode_rows": series_zero["ideal_mode_rows"], "ends": series_zero["ends"]}
    CTX.verdict("TR-2: the plan's anchors reproduced", observed == anchors, anchors, observed)


# =============================================================================
# TR-3 OUTCOMES
# =============================================================================


def trace_outcomes(ref: Path) -> None:
    """is_true of three persisted triggers recomputed from /series_0 trigger_value; the effective-start filter."""
    found = {}
    for mode, row in OUTCOME_ROWS:
        frame, triggers = series_and_triggers(ref, mode)
        found[f"{mode}/{row}"] = outcome(frame, triggers.iloc[row])
    first_ideal = first_begin(task_file(ref, IDEAL, "trigger"))
    CTX.log.info("TRACE TR-3 %s first ideal begin %s", json.dumps(found, default=str), first_ideal)
    CTX.verdict("TR-3: recomputed is_true and values equal the persisted ones",
                all(item["recomputed"] == item["persisted"] and item["values_match"] for item in found.values()),
                "equal within float32", found)
    CTX.verdict("TR-3: the first ideal trigger survives the positional effective-start filter (Q2)",
                first_ideal[1] >= PLAN_ANCHORS["TR-0"]["reference_first"], f">= {PLAN_ANCHORS['TR-0']['reference_first']}",
                first_ideal)
    mode, row = OUTCOME_ROWS[0]
    frame, triggers = series_and_triggers(ref, mode)
    planted = frame.copy()
    planted.loc[pd.Timestamp(triggers.iloc[row][TRIGGER_END_DATES]), TRIGGER_VALUE] -= PLANTED_SHIFT
    changed = outcome(planted, triggers.iloc[row])
    CTX.verdict("control: a perturbed trigger_value changes the recomputed outcome", changed["recomputed"] != changed["persisted"],
                "flagged", changed)


def outcome(frame: pd.DataFrame, trigger: pd.Series) -> dict:
    """float64 begin and end values from /series_0, the outcome they give, and the persisted float32 beside them."""
    start = float(frame.at[pd.Timestamp(trigger[TRIGGER_BEGIN_DATES]), TRIGGER_VALUE])
    end = float(frame.at[pd.Timestamp(trigger[TRIGGER_END_DATES]), TRIGGER_VALUE])
    plus = bool(trigger[IS_PLUS_TRIGGER])
    return {"begin": day(trigger[TRIGGER_BEGIN_DATES]), "end": day(trigger[TRIGGER_END_DATES]), "plus": plus,
            "start": start, "end_value": end, "persisted_values": [float(trigger[TRIGGER_START_VALUES]), float(trigger[TRIGGER_END_VALUES])],
            "values_match": bool(np.allclose([start, end], [trigger[TRIGGER_START_VALUES], trigger[TRIGGER_END_VALUES]],
                                             rtol=FLOAT32_TOLERANCE)),
            "recomputed": end > start if plus else end < start, "persisted": bool(trigger[IS_TRUE])}


def first_begin(path: Path) -> list:
    """(series, date) of the earliest persisted trigger begin of a trigger file."""
    begins = []
    with pd.HDFStore(str(path), mode="r") as store:
        for key in sorted(store.keys()):
            if key.startswith("/triggers_"):
                begins.append([day(store.get(key)[TRIGGER_BEGIN_DATES].min()), key[len("/triggers_"):]])
    date, sid = min(begins)
    return [sid, date]


# =============================================================================
# TR-4 SIGNAL GROUPS
# =============================================================================


def trace_signals(config: AppConfig, ref: Path) -> None:
    """Eligibility, six-point windows, Pearson and slope sequence of two realistic groups."""
    period_back = config.signal.period_back
    signals = read_key(task_file(ref, REALISTIC, "signal"), "/signals")
    with pd.HDFStore(str(task_file(ref, REALISTIC, "trigger")), mode="r") as store:
        var3 = {key[len("/series_"):]: store.get(key)[VAR3] for key in sorted(store.keys()) if key.startswith("/series_")}
    found = {gid: group_trace(signals[signals[GROUP_ID] == gid], var3, period_back) for gid in TR4_GROUPS}
    CTX.log.info("TRACE TR-4 %s", json.dumps(found, default=str))
    CTX.verdict("TR-4: non-vacuity: both groups found with their members", all(item["members"] for item in found.values()),
                list(TR4_GROUPS), {gid: len(item["members"]) for gid, item in found.items()})
    CTX.verdict("TR-4: one signal per group, every member eligible, Pearson and slope sequence recomputed",
                all(item["one_signal"] and item["eligible"] and item["pearson_match"] and item["slope_sequence"]
                    for item in found.values()), "all hold",
                {gid: {k: item[k] for k in ("one_signal", "eligible", "pearson_match", "slope_sequence")} for gid, item in found.items()})


def group_trace(rows: pd.DataFrame, var3: dict, period_back: int) -> dict:
    """One group: its signal row, each member's eligibility, window and recomputed Pearson, the slope sequence."""
    signal = rows[rows[IS_SIGNAL]].iloc[0]
    begin, end, sid = pd.Timestamp(signal[TRIGGER_BEGIN_DATES]), pd.Timestamp(signal[TRIGGER_END_DATES]), signal[SERIES_ID]
    members = []
    for _, row in rows.sort_values([TRIGGER_BEGIN_DATES, SERIES_ID]).iterrows():
        member_begin = pd.Timestamp(row[TRIGGER_BEGIN_DATES])
        window = var3[sid].loc[:member_begin].tail(period_back + 1)
        pair = np.corrcoef(window.to_numpy(), var3[row[SERIES_ID]].reindex(window.index).to_numpy())[0, 1]
        members.append({"series": row[SERIES_ID], "begin": day(member_begin), "end": day(row[TRIGGER_END_DATES]),
                        "is_signal": bool(row[IS_SIGNAL]), "slope": float(row[SLOPE]),
                        "eligible": bool(begin <= member_begin < end), "window": [day(window.index[0]), day(window.index[-1]), len(window)],
                        "pearson": float(pair), "persisted": float(row[CORR_TO_SIGNAL])})
    slopes = rows.groupby(SERIES_ID)[SLOPE].max().sort_values(ascending=False, kind="mergesort")
    return {"signal": [sid, day(begin), day(end)], "members": members, "series_ids": signal[SERIES_IDS],
            "one_signal": int(rows[IS_SIGNAL].sum()) == 1, "eligible": all(item["eligible"] for item in members),
            "pearson_match": all(abs(item["pearson"] - item["persisted"]) <= FLOAT32_TOLERANCE for item in members),
            "slope_sequence": ",".join(slopes.index) == signal[SERIES_IDS]}


# =============================================================================
# TR-5 PERFORMANCE
# =============================================================================


def trace_performance(ref: Path) -> None:
    """Signal counts, terminal values, attenuations, metrics and the breakdown's first cells of both modes."""
    found = {mode: performance_values(task_file(ref, mode, "performance")) for mode in (IDEAL, REALISTIC)}
    CTX.log.info("TRACE TR-5 %s", json.dumps(found, default=str))
    for mode, anchors in sorted(PLAN_ANCHORS["TR-5"].items()):
        observed = {name: found[mode][name] for name in anchors}
        CTX.verdict(f"TR-5 {mode}: the plan's anchors reproduced (rounded to {ANCHOR_DECIMALS} decimals)",
                    rounded(observed) == rounded(anchors), anchors, observed)


def performance_values(path: Path) -> dict:
    """The values TR-5 names, read from one performance file (tall tables pivoted by the production tall_to_wide)."""
    with pd.HDFStore(str(path), mode="r") as store:
        signals, kept = store.get("/signals"), store.get(HDF5_GRADIENT_FILTERED_SEQUENCE)
        curves = {key: store.get(key) for key in CURVE_COLUMNS}
        metrics = tall_to_wide(store.get(HDF5_GRADIENT_METRICS_TABLE), *METRICS_COLUMNS)
        breakdown = tall_to_wide(store.get(HDF5_SIGNALS_BREAKDOWN_TABLE), *BREAKDOWN_COLUMNS)
        attenuations = store.get(HDF5_GRADIENT_TERMINAL_ATTENUATIONS)
        reference = store.get(HDF5_GRADIENT_REFERENCE_VARIATION)
    total, attenuated = (curves[key][column] for key, column in CURVE_COLUMNS.items())
    references = reference.sort_values(DATE_COLUMN).groupby(REFERENCE_COLUMN)[VARIATION_COLUMN].last()
    return {"signals": int(signals[IS_SIGNAL].sum()), "kept": len(kept),
            "first": day(curves["/total_variation"][DATE_COLUMN].iloc[0]),
            "terminal": float(total.iloc[-1]), "attenuated": float(attenuated.iloc[-1]),
            **{name: float(metrics.at[name, GRADIENT_STRATEGY_HEADER]) for name in PERFORMANCE_METRICS},
            "breakdown": [float(breakdown.at[SIGNALS_COUNT_LABEL, cell]) for cell in BREAKDOWN_CELLS],
            "attenuations": {str(sid): float(amount) for sid, amount in zip(attenuations[SERIES_ID], attenuations[AMOUNT_COLUMN])},
            "reference_last": {str(rid): float(value) for rid, value in references.items()}}


def rounded(value: object) -> object:
    """Floats rounded to the anchor precision, through lists and dicts."""
    if isinstance(value, dict):
        return {name: rounded(item) for name, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return round(value, ANCHOR_DECIMALS) if isinstance(value, float) else value


# =============================================================================
# TR-6 ORCHESTRATION
# =============================================================================


def trace_orchestration(run_copy: Path, run_logs: dict) -> None:
    """Mode sequence, cleanup lines, config copies, git_info.txt and Luigi's summaries of RUN-1/RUN-2."""
    lines = {name: harness.plain_lines((CTX.roots.logs / log).read_text(encoding="utf-8")) for name, log in sorted(run_logs.items())}
    modes = {name: [line.split(MODE_LINE, 1)[1].strip() for line in text if MODE_LINE in line] for name, text in lines.items()}
    cleaned = {name: [line.split(CLEANED_LINE, 1)[1].strip() for line in text if CLEANED_LINE in line] for name, text in lines.items()}
    summaries = {name: sum(SUMMARY_LINE in line for line in text) for name, text in lines.items()}
    params = run_copy / PARAMS
    copies = {name: (params / name).read_bytes() == (harness.REAL_WORKTREE / "workflows" / name).read_bytes() for name in CONFIG_COPIES}
    git_info = (params / "git_info.txt").read_text(encoding="utf-8").splitlines()
    found = {"modes": modes, "cleaned": cleaned, "summaries": summaries, "copies_identical": copies, "git_info": git_info}
    CTX.log.info("TRACE TR-6 %s", json.dumps(found, default=str))
    CTX.verdict("TR-6: dual runs ideal, then realistic (both runs)", all(item == [IDEAL, REALISTIC] for item in modes.values()),
                [IDEAL, REALISTIC], modes)
    CTX.verdict("TR-6: over existing outputs, one cleanup line per task, _run_task cleaning before each build (Q12) (RUN-2: "
                "six; RUN-1 started on a fresh folder: none)",
                [len(cleaned["run1"]), len(cleaned["run2"])] == [0, 6], [0, 6], [len(cleaned["run1"]), len(cleaned["run2"])])
    CTX.verdict("TR-6: the parameters folder's config copies equal workflows/*.yaml byte for byte", all(copies.values()),
                True, copies)
    CTX.verdict("TR-6: the default quiet Luigi prints no summary; --quiet-luigi False prints twelve",
                [summaries["run1"], summaries["run2"]] == [0, 12], [0, 12], [summaries["run1"], summaries["run2"]])


# =============================================================================
# SHARED READERS
# =============================================================================


def task_file(root: Path, mode: str, task: str) -> Path:
    """The task output of one mode under a copy of the reference tree."""
    return root / PARAMS / mode / task / "0.h5"


def read_key(path: Path, key: str) -> pd.DataFrame:
    """One key of a store read in mode r."""
    with pd.HDFStore(str(path), mode="r") as store:
        return store.get(key)


def series_and_triggers(ref: Path, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """/series_0 and /triggers_0 of one mode's trigger file."""
    path = task_file(ref, mode, "trigger")
    return read_key(path, f"/series_{SERIES}"), read_key(path, f"/triggers_{SERIES}")


def day(value: object) -> str:
    """ISO date of a timestamp-like value."""
    return str(pd.Timestamp(value).date())


if __name__ == "__main__":
    main()
