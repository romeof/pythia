"""Regression for point-in-time grouping in realistic mode (Q20).

(1) EXP-08's precedence-sensitive case through production SignalSelection: X rises over
[d0, d25), Y dips over [d2, d10), Z (Y's dip plus a small ripple) begins at d5; corr_threshold 0.5,
period_back 5, earliest_begin_highest_slope. Both X and Y are eligible signals at d5 and both pass
the threshold. With the replay mode realistic, eligible signals are taken by (signal_begin,
group_id), so Z joins X's group; with ideal, by signal_end, so Z joins Y's group (Y ends first).
(2) The selection built from each mode copy's validated configuration (case (1)'s signal
parameters set on the copy) groups as case (1): realistic Z with X; ideal and the shipped dual Z
with Y. (3) A begin still active at the data end yields no trigger row through TriggerData, so it
never reaches grouping. The unfixed SignalSelection takes eligible signals by signal_end in every
mode (Z joins Y's group). (4) The Signal task passes its configuration's concrete mode to
SignalSelection: in process, per mode copy on series 0, each configuration _run_pipeline builds (the
dual copy's two, ideal then realistic) goes through pipeline._create_task and luigi.build, Trigger then
Signal, with src.tasks.signal's SignalSelection wrapped by a recorder of its replay_mode. Case (2)
builds the selection itself, so it cannot see that argument.

Structure:
1. main
2. Cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import luigi
import numpy as np
import pandas as pd

import pipeline
import src.tasks.signal as signal_task

ROWS = 30
CASE_SIGNAL = {"corr_threshold": 0.5, "period_back": 5, "signal_criterion": "earliest_begin_highest_slope"}
MODE_KEY = "trigger.replay.mode"
MODE_COPIES = ("realistic", "ideal", "dual")
TASK_MODES = {"realistic": ["realistic"], "ideal": ["ideal"], "dual": ["ideal", "realistic"]}  # passed per mode copy
BUILT_TASKS = ("trigger", "signal")
LUIGI_LOG_LEVEL = "WARNING"  # keeps Luigi's failure records, drops its per-task DEBUG lines that print the whole config


def main() -> None:
    """The four cases, then the summary (return code 1 on any FAIL)."""
    precedence_case()
    mode_copies()
    active_at_end()
    task_modes()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def precedence_case() -> None:
    """Case (1): Z joins X's group in realistic mode (point in time) and Y's group in ideal mode (signal_end)."""
    from src.utils.common.schema import SignalConfig

    groups = {mode: group_of_z(SignalConfig(**CASE_SIGNAL), mode) for mode in ("realistic", "ideal")}
    KIT.verdict("case (1): realistic: Z joins the group of X (taken by signal_begin, group_id)",
                groups["realistic"]["Z"] == groups["realistic"]["X"], "Z with X", groups["realistic"])
    KIT.verdict("case (1): ideal: Z joins the group of Y (taken by signal_end)",
                groups["ideal"]["Z"] == groups["ideal"]["Y"], "Z with Y", groups["ideal"])


def group_of_z(signal, replay_mode: str) -> dict:
    """series_id -> group_id of the three triggers grouped by production SignalSelection."""
    from src.utils.signal.selection import SignalSelection

    dates = pd.bdate_range("2022-01-03", periods=ROWS)
    rising = np.linspace(100.0, 130.0, ROWS)
    dip = 100.0 + np.concatenate([[0.0, -10.0, 0.0], np.arange(1.0, ROWS - 2.0)])
    levels = {"X": rising, "Y": dip, "Z": dip + 0.01 * np.sin(np.arange(ROWS))}
    selection = SignalSelection(signal, replay_mode)
    for series_id, values in levels.items():
        selection.add_series_values(series_id, pd.DataFrame({"var3": values}, index=dates))
    spans = {"X": (0, 25), "Y": (2, 10), "Z": (5, 20)}
    for series_id, (begin, end) in spans.items():
        selection.add_series_triggers(series_id, pd.DataFrame({
            "series_id": [series_id], "trigger_begin_dates": [dates[begin]], "trigger_end_dates": [dates[end]],
            "trigger_start_values": [levels[series_id][begin]], "trigger_end_values": [levels[series_id][end]],
            "is_plus_trigger": [True], "is_true": [True]}))
    groups = selection.build_signal_window_groups()
    return {str(series_id): int(group) for series_id, group in zip(groups["series_id"], groups["group_id"])}


def mode_copies() -> None:
    """Case (2): each mode copy's validated configuration groups as case (1); the shipped dual copy is not realistic."""
    groups = {}
    with testkit.scratch(KIT) as folder:
        for mode in MODE_COPIES:
            (folder / mode).mkdir()
            overrides = {f"signal.{key}": value for key, value in CASE_SIGNAL.items()}
            overrides.update({} if mode == "dual" else {MODE_KEY: mode})
            config = testkit.validated_config(testkit.config_copy(KIT, folder / mode, strategy=overrides))
            groups[mode] = group_of_z(config.signal, config.trigger.replay.mode)
    KIT.verdict("case (2): the realistic copy: Z joins the group of X", groups["realistic"]["Z"] == groups["realistic"]["X"],
                "Z with X", groups)
    KIT.verdict("case (2): the ideal copy and the shipped dual copy: Z joins the group of Y",
                all(groups[mode]["Z"] == groups[mode]["Y"] for mode in ("ideal", "dual")), "Z with Y", groups)


def active_at_end() -> None:
    """Case (3): a plus begin with no later plus end gives no trigger row; the completed pair gives one."""
    from src.utils.trigger.outcome import TriggerData

    frame = pd.DataFrame(0, index=pd.bdate_range("2022-01-03", periods=12),
                         columns=["trigger_plus_begin", "trigger_plus_end", "trigger_minus_begin", "trigger_minus_end"])
    frame["trigger_value"] = np.linspace(100.0, 111.0, 12)
    frame.iloc[2, frame.columns.get_loc("trigger_plus_begin")] = 1
    frame.iloc[5, frame.columns.get_loc("trigger_plus_end")] = 1
    frame.iloc[8, frame.columns.get_loc("trigger_plus_begin")] = 1
    rows = TriggerData()._extract_triggers_from_trigger_columns("0", frame)
    begins = [str(date.date()) for date in rows["trigger_begin_dates"]] if rows is not None else []
    KIT.verdict("case (3): the begin active at the data end yields no trigger row (only the completed pair)",
                begins == [str(frame.index[2].date())], [str(frame.index[2].date())], begins)


def task_modes() -> None:
    """Case (4): per mode copy on series 0, the replay_mode the Signal task passes to SignalSelection in each
    configuration _run_pipeline builds."""
    found = {}
    with testkit.scratch(KIT) as folder:
        for mode in MODE_COPIES:
            (folder / mode).mkdir()
            config_file = testkit.config_copy(KIT, folder / mode, strategy={**testkit.ONE_SERIES, MODE_KEY: mode})
            found[mode] = recorded_modes(testkit.validated_config(config_file))
    built = {mode: record["built"] for mode, record in found.items()}
    KIT.verdict("case (4) precondition: every Trigger and Signal build returns True",
                all(all(results) for results in built.values()), True, built)
    passed = {mode: record["modes"] for mode, record in found.items()}
    KIT.verdict("case (4): the Signal task passes its configuration's concrete mode to SignalSelection (the dual copy: "
                "ideal, then realistic)", passed == TASK_MODES, TASK_MODES, passed)


def recorded_modes(config) -> dict:
    """The luigi.build results and the replay_mode of every SignalSelection built while each configuration
    _run_pipeline builds from the copy (pipeline._resolve_modes, _clone_config_with_mode) goes through
    pipeline._create_task and luigi.build, Trigger then Signal; src.tasks.signal's SignalSelection is wrapped by a
    recorder for the builds."""
    original = signal_task.SignalSelection
    record = {"built": [], "modes": []}

    def recorder(signal, replay_mode: str):
        record["modes"].append(replay_mode)
        return original(signal, replay_mode)

    modes = pipeline._resolve_modes(config)
    signal_task.SignalSelection = recorder
    try:
        for mode in modes:
            mode_config = pipeline._clone_config_with_mode(config, mode) if len(modes) > 1 else config
            record["built"] += [bool(luigi.build([pipeline._create_task(mode_config, name)], local_scheduler=True,
                                                 log_level=LUIGI_LOG_LEVEL)) for name in BUILT_TASKS]
    finally:
        signal_task.SignalSelection = original
    return record


if __name__ == "__main__":
    main()
