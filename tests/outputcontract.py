"""Regression for the data contract of a run's outputs (D-8; R-T9, R-S4, R-P7; EXP-21's reconciliations).

One dual --task all on a two-series copy (series 0 and 1, reference 0), then per mode: the key
set of each task file (trigger: /series_<id>, /triggers_<id>, /reference_series_0; signal:
/signals, /strategy_name, /series_<id> of the signal series, /reference_series_0; performance:
the twelve keys of performance.md's table) and its dtypes (the task schemas and that table);
/triggers_<id> equal the pairs derived from /series_<id>'s flags (each begin with the first
strictly later unused end of its direction, the trigger values at both rows, the outcome per
direction, begins before the series' effective start index[lookback_window - 1] dropped); the
signal /signals rows equal the trigger rows; one is_signal row per group; every
/gradient_filtered_sequence group lies within the signal rows. A grouping that marked two
signals in one group breaks the one-signal rule.

Structure:
1. main
2. Contracts
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import numpy as np
import pandas as pd

TWO_SERIES = {"read.skip_series": ["2", "3", "4", "5"], "read.reference_series": ["0"]}
SERIES_IDS = ("0", "1")
REFERENCE_KEYS = {"/reference_series_0"}
LOOKBACK = 50
TRIGGER_DTYPES = {"trigger_begin_dates": "datetime64[ns]", "trigger_end_dates": "datetime64[ns]",
                  "trigger_start_values": "float32", "trigger_end_values": "float32", "is_plus_trigger": "bool",
                  "is_true": "bool"}
SIGNAL_DTYPES = {"trigger_start_values": "float32", "trigger_end_values": "float32", "slope": "float32",
                 "corr_to_signal": "float32", "is_plus_trigger": "bool", "is_true": "bool", "is_signal": "bool",
                 "group_id": "int32", "num_series": "int32", "series_ids": "object", "series_id": "object"}
# performance.md's table: key -> its documented column dtypes
PERFORMANCE_DTYPES = {
    "/signals": {}, "/strategy_name": {}, "/total_variation": {"total_variation": "float32"},
    "/total_variation_attenuated": {"total_variation_attenuated": "float32"},
    "/gradient_reference_variation": {"reference_id": "object", "variation": "float32"},
    "/gradient_signal_segments": {"group_id": "int32", "pct": "float32"},
    "/gradient_filtered_sequence": {"sequence_index": "int32", "group_id": "int32"},
    "/gradient_terminal_attenuations": {"series_id": "object", "amount": "float32"}, "/gradient_params": {},
    "/gradient_param_sequences": {"kind": "object", "sequence_index": "int32", "name": "object"},
    "/gradient_metrics_table": {"metric_key": "object", "column_header": "object", "value": "float32"},
    "/signals_breakdown_table": {"row_key": "object", "column_header": "object", "value": "float32"},
}
DIRECTIONS = ((True, "trigger_plus_begin", "trigger_plus_end"), (False, "trigger_minus_begin", "trigger_minus_end"))


def main() -> None:
    """The dual run and both modes' contracts, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(TWO_SERIES))
        child = testkit.run_pipeline(KIT, folder, "dual", config_file, ["all"])
        KIT.verdict("the dual --task all returns 0", child.return_code == 0, 0, {"return_code": child.return_code, "timed_out": child.timed_out})
        for mode in testkit.MODES:
            stores = {task: testkit.store_frames(testkit.mode_output(config_file, mode, task)) for task in testkit.TASKS}
            contracts(mode, stores)
    KIT.finish()


# =============================================================================
# CONTRACTS
# =============================================================================


def contracts(mode: str, stores: dict) -> None:
    """Key sets, dtypes and the four reconciliations of one mode."""
    trigger, signal, performance = (stores[task] for task in testkit.TASKS)
    signals = signal["/signals"]
    signal_series = sorted(set(signals.loc[signals["is_signal"], "series_id"].astype(str)))
    expected = {"trigger": {f"/{prefix}_{sid}" for sid in SERIES_IDS for prefix in ("series", "triggers")} | REFERENCE_KEYS,
                "signal": {"/signals", "/strategy_name"} | {f"/series_{sid}" for sid in signal_series} | REFERENCE_KEYS,
                "performance": set(PERFORMANCE_DTYPES)}
    for task in testkit.TASKS:
        found = set(stores[task])
        KIT.verdict(f"{mode} {task}: the documented keys, no other", found == expected[task], sorted(expected[task]),
                    {"missing": sorted(expected[task] - found), "extra": sorted(found - expected[task])})
    wrong = {}
    for sid in SERIES_IDS:
        wrong.update(dtype_mismatches(f"trigger/triggers_{sid}", trigger[f"/triggers_{sid}"], TRIGGER_DTYPES))
    wrong.update(dtype_mismatches("signal/signals", signals, SIGNAL_DTYPES))
    wrong.update(dtype_mismatches("performance/signals", performance["/signals"], SIGNAL_DTYPES))
    for key, dtypes in PERFORMANCE_DTYPES.items():
        wrong.update(dtype_mismatches(f"performance{key}", performance[key], dtypes))
    KIT.verdict(f"{mode}: dtypes per the task schemas and performance.md", not wrong, "no mismatch", wrong)
    differing = {sid: trigger_differences(trigger[f"/series_{sid}"], trigger[f"/triggers_{sid}"]) for sid in SERIES_IDS}
    rows = sum(len(trigger[f"/triggers_{sid}"]) for sid in SERIES_IDS)
    KIT.verdict(f"{mode}: /triggers_<id> equal the pairs derived from /series_<id> flags ({rows} rows)",
                rows > 0 and not any(differing.values()), "no difference",
                {sid: items[:3] for sid, items in differing.items() if items} or rows)
    KIT.verdict(f"{mode}: the signal /signals rows equal the trigger rows", len(signals) == rows, rows, len(signals))
    groups = signals["group_id"].nunique()
    marked = signals.loc[signals["is_signal"], "group_id"]
    KIT.verdict(f"{mode}: one is_signal row per group", len(marked) == groups and marked.is_unique,
                {"groups": groups, "signals": groups}, {"groups": groups, "signals": len(marked)})
    outside = sorted(set(performance["/gradient_filtered_sequence"]["group_id"]) - set(marked))
    KIT.verdict(f"{mode}: every /gradient_filtered_sequence group lies within the signal rows", not outside, [], outside[:10])


# =============================================================================
# HELPERS
# =============================================================================


def dtype_mismatches(name: str, frame: pd.DataFrame, expected: dict) -> dict:
    """{name.column: [expected, actual]} for every listed column whose dtype differs or is absent."""
    return {f"{name}.{column}": [dtype, str(frame[column].dtype) if column in frame.columns else "absent"]
            for column, dtype in sorted(expected.items())
            if column not in frame.columns or str(frame[column].dtype) != dtype}


def trigger_differences(series: pd.DataFrame, triggers: pd.DataFrame) -> list:
    """Persisted trigger rows against the pairs and values derived from the series' flags."""
    persisted = [(pd.Timestamp(row["trigger_begin_dates"]), pd.Timestamp(row["trigger_end_dates"]), bool(row["is_plus_trigger"]),
                  float(row["trigger_start_values"]), float(row["trigger_end_values"]), bool(row["is_true"]))
                 for _, row in triggers.iterrows()]
    return [str(item) for item in sorted(set(derived_triggers(series)) ^ set(persisted))]


def derived_triggers(series: pd.DataFrame) -> list:
    """Each begin with the first strictly later unused end of its direction, the trigger values at both rows as float32,
    the outcome per direction; begins before index[lookback_window - 1] dropped."""
    values = series["trigger_value"].to_numpy().astype("float32")
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
            derived.append((series.index[begin], series.index[ends[cursor]], plus, start, end, end > start if plus else end < start))
            cursor += 1
    return [item for item in derived if item[0] >= series.index[LOOKBACK - 1]]


if __name__ == "__main__":
    main()
