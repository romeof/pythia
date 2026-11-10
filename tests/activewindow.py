"""Regression for the window of the trigger still active at series end (C-3).

Since C-3 the debug plot of a trigger still active at the series end receives the window of the last row as every
row's debug plots receive theirs, _select_window(df, n - 1, config); the removed _create_active_trigger_window
restated it. No test or launch reached that plot: both profiles ship trigger.strategy.plot_debug false, and
plotfailure's debug case fails at the first debug plot. In process, per window type (rolling, then expanding), on a
scratch config copy (realistic; trigger.strategy.plot_debug true in its dev copy), recorders that draw nothing replace
the three debug plot functions src.utils.trigger.realistic imports, and a strategy analyses a synthetic one-series
frame (testkit.sine_frame, 300 daily rows) to the end. Expected per window type: the loop ends with an active trigger,
so plot_active_trigger is called once, with n = 300, and its window holds the frame's rows n - lookback_window .. n - 1
(rolling) or 0 .. n - 1 (expanding); a verdict shows a window as [first row, last row, row count].

Structure:
1. main
2. Recorded run
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import inspect

import pandas as pd

import src.utils.trigger.realistic as realistic

ROWS = 300
FIRST_DAY = "2020-01-01"
# The strategy-copy overrides per window type
WINDOWS = {window_type: {"trigger.replay.mode": "realistic", "trigger.replay.window_type": window_type}
           for window_type in ("rolling", "expanding")}
DEBUG_PLOTS = {"trigger.strategy.plot_debug": True}
PLOTS = ("plot_suppressed_trigger", "plot_ended_trigger", "plot_active_trigger")


def main() -> None:
    """Per window type, the recorded run against the rows of the last row's window; then the summary (return code 1
    on any FAIL)."""
    frame = testkit.sine_frame(ROWS, FIRST_DAY)
    for window_type, overrides in WINDOWS.items():
        with testkit.scratch(KIT) as folder:
            config_file = testkit.config_copy(KIT, folder, strategy=dict(overrides), dev=dict(DEBUG_PLOTS))
            config = testkit.validated_config(config_file)
            strategy = testkit.strategy(config, testkit.output_file(config_file, "trigger"))
            active = recorded_run(strategy, frame)["plot_active_trigger"]
        first_row = ROWS - config.trigger.replay.lookback_window if window_type == "rolling" else 0
        lengths = [call["n"] for call in active]
        KIT.verdict(f"{window_type}: an active trigger at the end: plot_active_trigger called once, n = {ROWS}",
                    lengths == [ROWS], [ROWS], lengths)
        windows = [call["window_df"].index for call in active]
        KIT.verdict(f"{window_type}: the window of the last row holds rows {first_row} .. {ROWS - 1}",
                    len(windows) == 1 and windows[0].equals(frame.index[first_row:]),
                    [span(frame.index, frame.index[first_row:])], [span(frame.index, window) for window in windows])
    KIT.finish()


# =============================================================================
# RECORDED RUN
# =============================================================================


def recorded_run(strategy, frame: pd.DataFrame) -> dict:
    """By debug plot function, the arguments (by parameter name) of each call while the strategy analyses a copy of the
    frame to the end, the three functions src.utils.trigger.realistic imports replaced by recorders that draw
    nothing."""
    originals = {name: getattr(realistic, name) for name in PLOTS}
    calls: dict = {name: [] for name in PLOTS}

    def recorder(name: str):
        signature = inspect.signature(originals[name])

        def record(*args, **kwargs):
            calls[name].append(signature.bind(*args, **kwargs).arguments)
        return record

    for name in PLOTS:
        setattr(realistic, name, recorder(name))
    try:
        strategy.analyze_series("0", frame.copy())
    finally:
        for name, original in originals.items():
            setattr(realistic, name, original)
    KIT.log.info("recorded calls: %s", {name: len(found) for name, found in calls.items()})
    return calls


def span(frame_index: pd.Index, window: pd.Index) -> list:
    """The first and last frame row a window holds and its row count; empty for an empty window."""
    rows = frame_index.get_indexer(window)
    return [int(rows[0]), int(rows[-1]), len(rows)] if len(rows) else []


if __name__ == "__main__":
    main()
