"""Regression for the realistic turning_points column (Q14).

A synthetic sine series (200 rows, period 20 rows, var5 constant) through the realistic
analyze_series of the extrema strategy on a scratch config copy (window type pinned to rolling,
of trigger.replay.lookback_window rows L). Expected: for every row i >= L - 1,
turning_points[i] equals the turning points the strategy's own find_peaks wrapper
(_generate_trigger_begin, pinned by plateauextrema.py) reports on the rolling window ending at
row i, a value known at row i; rows before L - 1 hold 0; non-vacuity: some window holds more
than one turning point, which no mark on the extremum's own row can equal. The unfixed loop
leaves the column at 0 on every row.

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import numpy as np
import pandas as pd


ROWS, PERIOD = 200, 20
TURNING = "turning_points"
ROLLING = {"trigger.replay.mode": "realistic", "trigger.replay.window_type": "rolling"}


def main() -> None:
    """The realistic run against the window counts, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(ROLLING))
        config = testkit.validated_config(config_file)
        lookback = config.trigger.replay.lookback_window
        levels = 100.0 + 10.0 * np.sin(2.0 * np.pi * np.arange(ROWS) / PERIOD)
        frame = pd.DataFrame({name: levels for name in ("var0", "var3", "var4")}, index=pd.bdate_range("2021-01-04", periods=ROWS))
        frame["var1"], frame["var2"], frame["var5"] = levels * 0.99, levels * 1.01, np.full(ROWS, 1.0e6)
        strategy = testkit.strategy(config, testkit.output_file(config_file, "trigger"))
        output = strategy.analyze_series("0", frame.copy())
    persisted = output[TURNING].to_numpy()
    expected = np.zeros(ROWS, dtype=int)
    for row in range(lookback - 1, ROWS):
        expected[row] = int(strategy._generate_trigger_begin(frame.iloc[row - lookback + 1:row + 1].copy())[TURNING].sum())
    differing = [[int(row), int(persisted[row]), int(expected[row])] for row in np.flatnonzero(persisted != expected)]
    KIT.verdict(f"non-vacuity: some rolling window of {lookback} rows holds more than one turning point",
                int(expected.max()) > 1, "> 1", int(expected.max()))
    KIT.verdict(f"rows before {lookback - 1} hold 0", not persisted[:lookback - 1].any(), 0,
                int(np.abs(persisted[:lookback - 1]).sum()))
    KIT.verdict("turning_points[i] equals the turning points of the window ending at row i",
                not differing, "no row differs", {"differing [row, persisted, window count]": differing[:6],
                                                 "differing_rows": len(differing)})
    KIT.finish()


if __name__ == "__main__":
    main()
