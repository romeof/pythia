"""EXP-21: Read's branches and the persisted data contracts (standard 05 data tests; R-S4, R-P7, R-T1, R-T10, R-T14).

(a) R-T1 and R-T14 on synthetic files under data/exp/21 in the read.py layout (built by
synthetic.py, read in mode r): a missing and an unknown index_encoding raise ValueError (each in
a child of this script, judged by return code and text); one series per valid encoding; a scalars
group (columns_map, var0) stored and ignored, no attributes on the frame; a skip_series item skipped;
"none" bounds keep every row; inclusive bounds; the value columns and var12, the one gate column the
copies configure, kept (the extra var9, no gate column, dropped).
(b) data/run1, both modes: documented keys present; dtypes per the task schemas
(trigger.py:227-231, signal.py:174-184, performance.py:713-718) and the performance.md table;
storage formats against that table; DatetimeIndex unique and increasing; reconciliations:
/triggers_i equals the pairs derived from /series_i's flags (R-T9, after the filter at the
series' positional effective start /series_i.index[lookback_window - 1], Q2), signal /signals
rows equal the trigger rows, one is_signal row per group, performance /signals equals signal
/signals plus enrichment columns, /gradient_filtered_sequence lies within the signal rows,
reference rows start at their series' positional effective start (Q2).
Planted control: one dropped /triggers_0 row must be flagged.

Structure:
1. main
2. Read branches
3. Run contracts
4. Reconciliations
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

import h5py
import numpy as np
import pandas as pd

import pipelinekit
import synthetic
from src.utils.common.constants import (
    GATE_OP_GE, GROUP_ID, IS_PLUS_TRIGGER, IS_SIGNAL, IS_TRUE, SERIES_ID, TRIGGER_BEGIN_DATES, TRIGGER_END_DATES,
    TRIGGER_END_VALUES, TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END, TRIGGER_PLUS_BEGIN, TRIGGER_PLUS_END, TRIGGER_START_VALUES,
    TRIGGER_VALUE,
)
from src.utils.trigger.gate import gate_columns

EXPERIMENT = "21"
CHILD_READ = "--child-read"
MODES = ("ideal", "realistic")
SERIES_IDS = ("0", "1", "2", "3", "4", "5")
REFERENCE_IDS = ("0",)
START, ROWS = "2019-01-02", 60
CUT_FIRST, CUT_LAST = 5, 50
EXTRA_COLUMN, GATE_COLUMN = "var9", "var12"
COLUMNS_MAP = np.array([b"a", b"b", b"c", b"d", b"e", b"f"])
SCALARS_GROUP = "scalars"
PLANTED_SCALARS = {"columns_map": COLUMNS_MAP, "var0": np.float64(530.48)}
NONE_BOUND = "none"
READ_ERRORS = {"missing": (None, "index_encoding attribute not found"), "unknown": ("epoch_weeks", "Unknown index encoding")}
# signal.md:229: the processed frame's 15 columns; a signal /series_<id> holds the 15 columns plus any gate column
# outside them (F-26)
SIGNAL_SERIES_COLUMNS = ("var0", "var1", "var2", "var3", "var4", "var5", "trigger_value", "end_reason",
                         "trigger_plus_begin", "trigger_minus_begin", "trigger_plus_end", "trigger_minus_end", "turning_points",
                         "active_trigger_plus_begin", "active_trigger_minus_begin")
TRIGGER_DTYPES = {TRIGGER_BEGIN_DATES: "datetime64[ns]", TRIGGER_END_DATES: "datetime64[ns]", TRIGGER_START_VALUES: "float32",
                  TRIGGER_END_VALUES: "float32", IS_PLUS_TRIGGER: "bool", IS_TRUE: "bool"}
SIGNAL_DTYPES = {TRIGGER_START_VALUES: "float32", TRIGGER_END_VALUES: "float32", "slope": "float32", "corr_to_signal": "float32",
                 IS_PLUS_TRIGGER: "bool", IS_TRUE: "bool", IS_SIGNAL: "bool", GROUP_ID: "int32", "num_series": "int32",
                 "series_ids": "object", SERIES_ID: "object"}
# performance.md:504-517: key -> (documented format, documented column dtypes); six keys written as table (F-26)
PERFORMANCE_DOC = {
    "/signals": ("table", {}),
    "/strategy_name": ("table", {}),
    "/total_variation": ("fixed", {"total_variation": "float32"}),
    "/total_variation_attenuated": ("fixed", {"total_variation_attenuated": "float32"}),
    "/gradient_reference_variation": ("table", {"reference_id": "object", "variation": "float32"}),
    "/gradient_signal_segments": ("fixed", {"group_id": "int32", "pct": "float32"}),
    "/gradient_filtered_sequence": ("fixed", {"sequence_index": "int32", "group_id": "int32"}),
    "/gradient_terminal_attenuations": ("table", {"series_id": "object", "amount": "float32"}),
    "/gradient_params": ("table", {}),
    "/gradient_param_sequences": ("table", {"kind": "object", "sequence_index": "int32", "name": "object"}),
    "/gradient_metrics_table": ("table", {"metric_key": "object", "column_header": "object", "value": "float32"}),
    "/signals_breakdown_table": ("table", {"row_key": "object", "column_header": "object", "value": "float32"}),
}
DIRECTIONS = ((True, TRIGGER_PLUS_BEGIN, TRIGGER_PLUS_END), (False, TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END))


def main() -> None:
    """A child read, or the read branches, both modes' contracts and the planted control."""
    args = _parse_args()
    if args.child_read:
        child_read(args.child_read)
        return
    folder = CTX.label_dir(EXPERIMENT)
    config = pipelinekit.load_copy(pipelinekit.write_config_copy(CTX, folder / "config"))
    lookback = config.trigger.replay.lookback_window
    series_columns = sorted(set(SIGNAL_SERIES_COLUMNS).union(gate_columns(config.trigger.gate.items)))
    found = {"read": read_branches(folder)}
    for mode in MODES:
        found[mode] = run_contracts(mode, args.run_copy / harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR) / mode,
                                    lookback, series_columns)
    found["planted"] = planted_control(args.run_copy, lookback)
    CTX.log.info("RESULTS %s", json.dumps(found, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--run-copy, --source-root and the child mode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-copy", type=Path, default=harness.REAL_WORKTREE / "data" / "run1" / harness.DATA_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    parser.add_argument(CHILD_READ, type=Path, help="child mode: read every series of this config copy's input")
    return parser.parse_args()


def child_read(config_file: Path) -> None:
    """Child: Read every series of the copy's input; an error propagates and ends the child non-zero."""
    for series_id, frame in pipelinekit.raw_series(pipelinekit.load_copy(config_file)).items():
        CTX.log.info("READ %s rows %d", series_id, len(frame))


# =============================================================================
# READ BRANCHES
# =============================================================================


def read_branches(folder: Path) -> dict:
    """The valid file through Read in-process; each invalid encoding in a child."""
    frame = synthetic.daily_frame(100.0 + np.arange(ROWS, dtype=float), START)
    wide = frame.assign(**{EXTRA_COLUMN: frame["var3"], GATE_COLUMN: frame["var3"]})
    valid = write_file(folder / "valid", {
        "days": synthetic.SyntheticSeries(frame, PLANTED_SCALARS),
        "seconds": synthetic.SyntheticSeries(frame, {}, synthetic.ENCODING_SECONDS),
        "skipped": synthetic.SyntheticSeries(frame),
        "wide": synthetic.SyntheticSeries(wide)})
    bounded = read_copy(folder / "bounded", valid, str(frame.index[CUT_FIRST].date()), str(frame.index[CUT_LAST].date()))
    unbounded = read_copy(folder / "unbounded", valid, NONE_BOUND, NONE_BOUND)
    found = {"bounded": {sid: [str(f.index[0].date()), str(f.index[-1].date()), len(f)] for sid, f in bounded.items()},
             "unbounded_rows": {sid: len(f) for sid, f in unbounded.items()}}
    CTX.verdict("skip_series item skipped", sorted(bounded) == ["days", "seconds", "wide"], ["days", "seconds", "wide"], sorted(bounded))
    CTX.verdict("both encodings decode to the written index", all(unbounded[sid].index.equals(frame.index) for sid in ("days", "seconds")),
                "equal indexes", {sid: str(unbounded[sid].index[0]) for sid in ("days", "seconds")})
    with h5py.File(valid, "r") as source:
        scalars = {"stored": sorted(source["days"].get(SCALARS_GROUP, ())), "attrs": unbounded["days"].attrs}
    ignored = {"stored": sorted(PLANTED_SCALARS), "attrs": {}}
    CTX.verdict("a scalars group ignored: no attributes on the frame", scalars == ignored, ignored, scalars)
    kept = [*synthetic.VALUE_COLUMNS, GATE_COLUMN]
    CTX.verdict(f"the value columns and the gate column {GATE_COLUMN} kept ({EXTRA_COLUMN}, no gate column, dropped)",
                list(unbounded["wide"].columns) == kept, kept, list(unbounded["wide"].columns))
    CTX.verdict('"none" bounds keep every row', all(rows == ROWS for rows in found["unbounded_rows"].values()), ROWS,
                found["unbounded_rows"])
    expected = [str(frame.index[CUT_FIRST].date()), str(frame.index[CUT_LAST].date()), CUT_LAST - CUT_FIRST + 1]
    CTX.verdict("date bounds inclusive", all(item == expected for item in found["bounded"].values()), expected, found["bounded"])
    found["errors"] = {name: read_error(folder / name, name, encoding, text) for name, (encoding, text) in READ_ERRORS.items()}
    return found


def write_file(folder: Path, series: dict) -> Path:
    """A synthetic input file at <folder>/<date>/<dataset>/raw/0.h5."""
    return synthetic.write_input(CTX, folder / "2019_01_02" / "0" / "raw" / "0.h5", series)


def read_copy(folder: Path, path: Path, ini_date: str, end_date: str) -> dict:
    """Every series Read yields for a config copy pointing at the file with these bounds."""
    return pipelinekit.raw_series(pipelinekit.load_copy(config_copy(folder, path, {"read.ini_date": ini_date,
                                                                                    "read.end_date": end_date})))


def config_copy(folder: Path, path: Path, overrides: dict) -> Path:
    """A config copy whose read.raw_data names the synthetic file, skip_series ["skipped"], one gate item on var12."""
    rel = path.resolve().relative_to(harness.REAL_WORKTREE).as_posix()
    return pipelinekit.write_config_copy(CTX, folder / "config", strategy={
        "read.raw_data": f"./{rel}", "read.skip_series": ["skipped"],
        "trigger.gate.items": [pipelinekit.gate_item(GATE_COLUMN, GATE_OP_GE, 0.0)], **overrides})


def read_error(folder: Path, name: str, encoding: object, text: str) -> dict:
    """A one-series file with this index_encoding, read in a child: non-zero return code with the ValueError text."""
    path = write_file(folder, {"bad": synthetic.SyntheticSeries(synthetic.daily_frame(np.ones(ROWS), START), {}, encoding)})
    child = CTX.launch_child(f"read{name}", [str(Path(__file__).resolve()), CHILD_READ, str(config_copy(folder, path, {}))],
                             timeout_s=harness.CHILD_TIMEOUT_S, hashseed=harness.DEFAULT_HASHSEED, audit_child=True)
    raised = child.return_code not in (0, None) and "ValueError" in child.text and text in child.text
    CTX.verdict(f"{name} index_encoding -> ValueError (trigger.md:44-45)", raised, f"return != 0, ValueError: {text}",
                {"return_code": child.return_code, "log": child.log_path.name})
    return {"return_code": child.return_code, "raised": raised}


# =============================================================================
# RUN CONTRACTS
# =============================================================================


def run_contracts(mode: str, base: Path, lookback: int, series_columns: list[str]) -> dict:
    """Keys, dtypes, formats, indexes and reconciliations of one mode's three task files; series_columns the signal
    /series_<id> columns expected."""
    trigger, signal, performance = (read_store(base / task / "0.h5") for task in ("trigger", "signal", "performance"))
    signal_series = sorted(set(signal["frames"]["/signals"].loc[signal["frames"]["/signals"][IS_SIGNAL], SERIES_ID]))
    expected_keys = {
        "trigger": {f"/{prefix}_{sid}" for sid in SERIES_IDS for prefix in ("series", "triggers")}
        | {f"/reference_series_{rid}" for rid in REFERENCE_IDS},
        "signal": {"/signals", "/strategy_name"} | {f"/series_{sid}" for sid in signal_series}
        | {f"/reference_series_{rid}" for rid in REFERENCE_IDS},
        "performance": set(PERFORMANCE_DOC)}
    for task, store in (("trigger", trigger), ("signal", signal), ("performance", performance)):
        CTX.verdict(f"{mode} {task}: documented keys present, no other", set(store["frames"]) == expected_keys[task],
                    sorted(expected_keys[task]), {"missing": sorted(expected_keys[task] - set(store["frames"])),
                                                  "extra": sorted(set(store["frames"]) - expected_keys[task])})
    found = {"dtypes": check_dtypes(mode, trigger, signal, performance), "formats": check_formats(mode, performance),
             "indexes": check_indexes(mode, (trigger, signal)),
             "signal_series_columns": signal_series_columns(mode, signal, series_columns)}
    found["reconciliations"] = reconcile(mode, trigger["frames"], signal["frames"], performance["frames"], lookback)
    return found


def read_store(path: Path) -> dict:
    """Every frame of a task file and each key's storage format, read in mode r."""
    with pd.HDFStore(str(path), mode="r") as store:
        keys = sorted(store.keys())
        return {"frames": {key: store.get(key) for key in keys},
                "formats": {key: store.get_storer(key).format_type for key in keys}}


def check_dtypes(mode: str, trigger: dict, signal: dict, performance: dict) -> dict:
    """Column dtypes against the task schemas and the performance.md table."""
    wrong = {}
    for key, frame in sorted(trigger["frames"].items()):
        if key.startswith("/triggers_"):
            wrong.update(dtype_mismatches(f"trigger{key}", frame, TRIGGER_DTYPES))
    wrong.update(dtype_mismatches("signal/signals", signal["frames"]["/signals"], SIGNAL_DTYPES))
    wrong.update(dtype_mismatches("performance/signals", performance["frames"]["/signals"], SIGNAL_DTYPES))
    for key, (_, dtypes) in sorted(PERFORMANCE_DOC.items()):
        wrong.update(dtype_mismatches(f"performance{key}", performance["frames"][key], dtypes))
    CTX.verdict(f"{mode}: dtypes per the task schemas and performance.md", not wrong, "no mismatch", wrong)
    return wrong


def dtype_mismatches(name: str, frame: pd.DataFrame, expected: dict) -> dict:
    """{name.column: [expected, actual]} for every listed column whose dtype differs or is absent."""
    return {f"{name}.{column}": [dtype, str(frame[column].dtype) if column in frame.columns else "absent"]
            for column, dtype in sorted(expected.items())
            if column not in frame.columns or str(frame[column].dtype) != dtype}


def check_formats(mode: str, performance: dict) -> dict:
    """Storage formats against performance.md's table (R-P7 lists six keys written as table)."""
    differing = {key: [documented, performance["formats"][key]] for key, (documented, _) in sorted(PERFORMANCE_DOC.items())
                 if performance["formats"][key] != documented}
    CTX.verdict(f"{mode}: performance storage formats as documented (performance.md:504-517)", not differing, "as documented",
                differing)
    return differing


def check_indexes(mode: str, stores: tuple) -> dict:
    """Every DatetimeIndex unique and increasing."""
    checked = {}
    for store in stores:
        for key, frame in sorted(store["frames"].items()):
            if isinstance(frame.index, pd.DatetimeIndex):
                checked[key] = bool(frame.index.is_unique and frame.index.is_monotonic_increasing)
    CTX.verdict(f"{mode}: every DatetimeIndex unique and increasing", bool(checked) and all(checked.values()),
                f"{len(checked)} indexes", sorted(key for key, ok in checked.items() if not ok))
    return {"indexes": len(checked)}


def signal_series_columns(mode: str, signal: dict, expected: list[str]) -> dict:
    """Signal /series_<id> columns against signal.md:229 (the processed frame's 15 columns plus any gate column outside
    them, sorted)."""
    columns = {key: sorted(frame.columns) for key, frame in sorted(signal["frames"].items()) if key.startswith("/series_")}
    CTX.verdict(f"{mode}: signal /series_<id> hold the 15 columns plus any gate column outside them (signal.md:229)",
                all(expected == item for item in columns.values()), expected,
                {key: len(item) for key, item in columns.items()})
    return {key: len(item) for key, item in columns.items()}


# =============================================================================
# RECONCILIATIONS
# =============================================================================


def reconcile(mode: str, trigger: dict, signal: dict, performance: dict, lookback: int) -> dict:
    """The six reconciliations of one mode."""
    differing = {sid: trigger_differences(trigger[f"/series_{sid}"], trigger[f"/triggers_{sid}"], lookback)
                 for sid in SERIES_IDS}
    rows = sum(len(trigger[f"/triggers_{sid}"]) for sid in SERIES_IDS)
    CTX.verdict(f"{mode}: non-vacuity: trigger rows reconciled", rows > 0, "> 0", rows)
    CTX.verdict(f"{mode}: /triggers_i equal the pairs derived from /series_i flags (R-T9)", not any(differing.values()),
                "no difference", {sid: items[:3] for sid, items in differing.items() if items})
    signals = signal["/signals"]
    CTX.verdict(f"{mode}: signal /signals rows equal the trigger rows", len(signals) == rows, rows, len(signals))
    groups = signals[GROUP_ID].nunique()
    CTX.verdict(f"{mode}: one is_signal row per group", int(signals[IS_SIGNAL].sum()) == groups and
                signals.loc[signals[IS_SIGNAL], GROUP_ID].is_unique, groups, int(signals[IS_SIGNAL].sum()))
    enriched = performance["/signals"]
    sort_keys = [GROUP_ID, SERIES_ID, TRIGGER_BEGIN_DATES]
    shared = [column for column in signals.columns if column in enriched.columns]
    same = (shared == list(signals.columns) and
            signals.sort_values(sort_keys).reset_index(drop=True).equals(enriched[shared].sort_values(sort_keys).reset_index(drop=True)))
    CTX.verdict(f"{mode}: performance /signals = signal /signals plus enrichment columns", same, "equal on the signal columns",
                {"added": sorted(set(enriched.columns) - set(signals.columns)), "rows": [len(signals), len(enriched)]})
    sequence = set(performance["/gradient_filtered_sequence"][GROUP_ID])
    outside = sorted(sequence - set(signals.loc[signals[IS_SIGNAL], GROUP_ID]))
    CTX.verdict(f"{mode}: /gradient_filtered_sequence within the signal rows", not outside, "none outside", outside[:10])
    starts = {key: str(frame.index[0].date()) for store in (trigger, signal) for key, frame in sorted(store.items())
              if key.startswith("/reference_series_")}
    positional = {key: str(trigger[f"/series_{key.split('_')[-1]}"].index[lookback - 1].date()) for key in starts}
    CTX.verdict(f"{mode}: reference rows start at their series' positional effective start (Q2)",
                bool(starts) and starts == positional, positional, starts)
    return {"trigger_rows": rows, "signal_rows": len(signals), "groups": groups, "sequence": len(sequence), "reference_starts": starts}


def trigger_differences(series: pd.DataFrame, triggers: pd.DataFrame, lookback: int) -> list:
    """Persisted trigger rows against the pairs and values derived from the series' flags."""
    derived = derived_triggers(series, lookback)
    persisted = [(pd.Timestamp(row[TRIGGER_BEGIN_DATES]), pd.Timestamp(row[TRIGGER_END_DATES]), bool(row[IS_PLUS_TRIGGER]),
                  float(row[TRIGGER_START_VALUES]), float(row[TRIGGER_END_VALUES]), bool(row[IS_TRUE]))
                 for _, row in triggers.iterrows()]
    return [str(item) for item in sorted(set(derived) ^ set(persisted))]


def derived_triggers(series: pd.DataFrame, lookback: int) -> list:
    """R-T9: each begin with the first strictly later unused end of its direction, values at both rows, the
    outcome per direction, begins before the series' positional effective start index[lookback - 1] dropped (Q2)."""
    values = series[TRIGGER_VALUE].to_numpy().astype("float32")
    derived = []
    for plus, begin_col, end_col in DIRECTIONS:
        ends = list(np.flatnonzero(series[end_col].to_numpy() > 0))
        cursor = 0
        for begin in np.flatnonzero(series[begin_col].to_numpy() > 0):
            while cursor < len(ends) and ends[cursor] <= begin:
                cursor += 1
            if cursor == len(ends):
                break
            start, end = float(values[begin]), float(values[ends[cursor]])
            derived.append((series.index[begin], series.index[ends[cursor]], plus, start, end,
                            end > start if plus else end < start))
            cursor += 1
    return [item for item in derived if item[0] >= series.index[lookback - 1]]


def planted_control(run_copy: Path, lookback: int) -> dict:
    """/triggers_0 of the realistic run with its first row dropped must be flagged."""
    path = run_copy / harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR) / MODES[-1] / "trigger" / "0.h5"
    with pd.HDFStore(str(path), mode="r") as store:
        series, triggers = store.get("/series_0"), store.get("/triggers_0")
    differences = trigger_differences(series, triggers.iloc[1:], lookback)
    CTX.verdict("planted control: one dropped /triggers_0 row is flagged", len(differences) == 1, 1, differences)
    return {"differences": differences}


if __name__ == "__main__":
    main()
