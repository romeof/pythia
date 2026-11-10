"""Regression for the realistic loop's frame copies and config reads (C-3, C-4; R3, R4).

Both changes leave the outputs identical, so only a count can show them undone. In process, on a scratch config copy
(realistic, rolling; trigger.strategy.plot_debug and trigger.replay.end.different_end_strategy false, as shipped and in
the dev profile), counters wrap RealisticModeMixin._select_window and RealisticModeMixin._load_realistic_config
before the strategy is built (testkit.strategy, its output file a constructor argument); one strategy analyses two
synthetic one-series frames (testkit.sine_frame, 300 daily rows). Expected: _select_window is never called (no
reader of the window frame is on); _load_realistic_config is called once for the two series. Control: with
different_end_strategy true the window frame is built on every loop row of both frames, 2 x (300 - lookback_window
+ 1) calls. Before C-3 the loop builds a frame per row; before C-4 each series reads both configs again.

Structure:
1. main
2. Counted run
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from collections import Counter
from pathlib import Path

from src.utils.trigger.realistic import RealisticModeMixin

ROWS = 300
FIRST_DAYS = {"0": "2020-01-01", "1": "2021-01-01"}
REALISTIC_ROLLING = {"trigger.replay.mode": "realistic", "trigger.replay.window_type": "rolling"}
DIFFERENT_END = "trigger.replay.end.different_end_strategy"
COUNTED = ("_select_window", "_load_realistic_config")


def main() -> None:
    """The counted run without and with a frame reader, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(REALISTIC_ROLLING))
        lookback = testkit.validated_config(config_file).trigger.replay.lookback_window
        counts = counted_run(config_file)
        KIT.verdict("no reader of the window frame on: _select_window is never called", counts["_select_window"] == 0,
                    0, counts["_select_window"])
        KIT.verdict(f"_load_realistic_config once for {len(FIRST_DAYS)} series (once per strategy)",
                    counts["_load_realistic_config"] == 1, 1, counts["_load_realistic_config"])
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy={**REALISTIC_ROLLING, DIFFERENT_END: True})
        rows_per_series = ROWS - lookback + 1
        frames = counted_run(config_file)["_select_window"]
        KIT.verdict(f"control, {DIFFERENT_END} true: the window frame on every loop row "
                    f"({len(FIRST_DAYS)} x {rows_per_series})", frames == len(FIRST_DAYS) * rows_per_series,
                    len(FIRST_DAYS) * rows_per_series, frames)
    KIT.finish()


# =============================================================================
# COUNTED RUN
# =============================================================================


def counted_run(config_file: Path) -> Counter:
    """The calls of the two counted names while one strategy, built after the wrapping, analyses both frames."""
    counts: Counter = Counter()
    originals = {"_select_window": RealisticModeMixin._select_window,
                 "_load_realistic_config": RealisticModeMixin._load_realistic_config}

    def counting(name: str):
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return originals[name](*args, **kwargs)
        return wrapper

    RealisticModeMixin._select_window = counting("_select_window")
    RealisticModeMixin._load_realistic_config = counting("_load_realistic_config")
    try:
        strategy = testkit.strategy(testkit.validated_config(config_file), testkit.output_file(config_file, "trigger"))
        for series_id, first_day in FIRST_DAYS.items():
            strategy.analyze_series(series_id, testkit.sine_frame(ROWS, first_day))
    finally:
        RealisticModeMixin._select_window = originals["_select_window"]
        RealisticModeMixin._load_realistic_config = originals["_load_realistic_config"]
    KIT.log.info("counted calls: %s", {name: counts[name] for name in COUNTED})
    return counts


if __name__ == "__main__":
    main()
