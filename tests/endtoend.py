"""End-to-end oracle on a synthetic input with hand-derived values, both modes (D-3; F-01, F-02, R-S2, R-P2..P6).

One dual --task all run on a scratch copy of the shipped configuration that reads a two-series
input written by testkit.write_raw_input (36 rows dated from 2020-01-01, the read window those
dates), with lookback_window 10 and min_distance 2; the shipped trigger_value following_row, weight
0.2, multiplicative_total_before_active_signals and yearly_attenuation_fraction 0.25 stay. var3 is piecewise
linear: series 0 falls 1 per row from 115 (row 0) to its minimum 100 (row 15), rises 2 per row to
its maximum 130 (row 30) and falls 4 per row to 110 (row 35); series 1 is 2 x series 0 - 100 (130,
100, 160 and 120 on those rows); var0 = var4 = var3, var1 and var2 1 % below and above, var5 1e6,
so preprocessing keeps every row; the copy sets no gate item. Hand-derived, per mode:
- trigger_value is the next row's var0, and var3 on the last row (the following_row rule);
- ideal: find_peaks on the whole series reports the minimum of row 15 and the maximum of row 30,
  so each series holds one plus trigger from row 15 to row 30 ("ideal_mode"), the minus begin of
  row 30 having no end; values 102 -> 126 (series 0) and 104 -> 152 (series 1);
- realistic: the 10-row rolling window reports each extremum on the row after it, so the plus
  trigger begins on row 16 and ends on row 31 by the opposite trigger, whose minus trigger begins
  there (the swap; still active at the end, so never extracted); values 104 -> 122 and 108 -> 144;
- every trigger is true (end above begin); the slope is the absolute least-squares slope of
  var3 / var3(begin) - 1 over the six rows ending on the begin (period_back 5): ideal 0.01 and
  0.02, realistic (4/7)/102 and (8/7)/104;
- the two triggers share their begin and correlate (r = 1: series 1 is an affine copy), so they
  form one group, whose signal is the steeper series 1 (num_series 2, series_ids "1,0");
- the gradient index holds the observed dates from row 9 (index[lookback_window - 1]) to row 35;
  the group's segment is (tv / tv(begin) - 1) x 100 on the trigger's rows, tv series 1's
  trigger_value; the curve is w x V x (tv / tv(begin) - 1) there, V = 1 (no event ended before
  it), 0 before and flat after, so its final value is 0.2 x (152/104 - 1) ideal and
  0.2 x (144/108 - 1) realistic; the index spans one calendar year, whose attenuation falls on the last
  item and takes 0.25 x the final value: the attenuated curve's only difference and the strategy's
  terminal attenuation;
- acr = ((1 + final) ^ (365.25 / 36) - 1) x 100 (36 days from row 9 to row 35), acr_after_attenuation the
  same on 0.75 x final; the breakdown counts one signal, plus and true.
_build_next_row_vector returning the same row's var0 breaks every value verdict.

Structure:
1. main
2. Mode verdicts
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common.common import variation_pct

SYNTHETIC_INPUT = Path("input") / "2026_01_01" / "0" / "raw" / "0.h5"
DATES = pd.bdate_range("2020-01-01", periods=36)
KNOTS = {"0": [(0, 115.0), (15, 100.0), (30, 130.0), (35, 110.0)], "1": [(0, 130.0), (15, 100.0), (30, 160.0), (35, 120.0)]}
LOOKBACK = 10
OVERRIDES = {"read.ini_date": DATES[0].strftime("%Y-%m-%d"), "read.end_date": DATES[-1].strftime("%Y-%m-%d"),
             "trigger.replay.lookback_window": LOOKBACK, "trigger.strategy.min_distance": 2}
# mode: (begin row, end row, end reason, {series: (begin value, end value)}, {series: slope})
EXPECTED = {
    "ideal": (15, 30, "ideal_mode", {"0": (102.0, 126.0), "1": (104.0, 152.0)}, {"0": 0.01, "1": 0.02}),
    "realistic": (16, 31, "opposite_trigger", {"0": (104.0, 122.0), "1": (108.0, 144.0)},
                  {"0": 4 / 7 / 102, "1": 8 / 7 / 104}),
}
SIGNAL_SERIES = "1"
WEIGHT = 0.2
ATTENUATION_RATE = 0.25
COUNTS = {"plus+minus_true+false": 1.0, "plus+minus_true": 1.0, "plus+minus_false": 0.0, "plus_true+false": 1.0,
          "plus_true": 1.0, "plus_false": 0.0, "minus_true+false": 0.0, "minus_true": 0.0, "minus_false": 0.0}
TOLERANCE = 1e-5


def main() -> None:
    """The dual run and both modes' verdicts, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = synthetic_copy(folder)
        child = testkit.run_pipeline(KIT, folder, "dual", config_file, ["all"])
        KIT.verdict("the dual --task all returns 0", child.return_code == 0, 0, {"return_code": child.return_code, "timed_out": child.timed_out})
        for mode in testkit.MODES:
            mode_verdicts(mode, {task: testkit.store_frames(testkit.mode_output(config_file, mode, task)) for task in testkit.TASKS})
    KIT.finish()


# =============================================================================
# MODE VERDICTS
# =============================================================================


def mode_verdicts(mode: str, stores: dict) -> None:
    """Trigger values, triggers, the group and its signal, the segment, the curves, acr and the counts of one mode."""
    trigger, signal, performance = (stores[task] for task in testkit.TASKS)
    begin_row, end_row, reason, values, slopes = EXPECTED[mode]
    for sid in KNOTS:
        series = trigger[f"/series_{sid}"]
        found = series["trigger_value"].to_numpy(dtype=float)
        KIT.verdict(f"{mode} series {sid}: trigger_value is the next row's var0, var3 on the last row",
                    np.array_equal(found, next_row_values(sid)), next_row_values(sid)[-3:].tolist(), found[-3:].tolist())
        rows = [(pd.Timestamp(row["trigger_begin_dates"]), pd.Timestamp(row["trigger_end_dates"]), bool(row["is_plus_trigger"]),
                 float(row["trigger_start_values"]), float(row["trigger_end_values"]), bool(row["is_true"]))
                for _, row in trigger[f"/triggers_{sid}"].iterrows()]
        expected = [(DATES[begin_row], DATES[end_row], True, *values[sid], True)]
        KIT.verdict(f"{mode} series {sid}: one true plus trigger, row {begin_row} to row {end_row}, {values[sid][0]} to "
                    f"{values[sid][1]}", rows == expected, [str(item) for item in expected], [str(item) for item in rows])
        flags = {"end_reason": series["end_reason"].iat[end_row], "minus begin": int(series["trigger_minus_begin"].iat[end_row])}
        wanted = {"end_reason": reason, "minus begin": int(mode == "realistic")}
        KIT.verdict(f"{mode} series {sid}: row {end_row} ends the trigger with {reason}"
                    + (" and begins the minus trigger (the swap)" if mode == "realistic" else ""), flags == wanted, wanted, flags)
    group = signal["/signals"].assign(series_id=lambda frame: frame["series_id"].astype(str)).set_index("series_id")
    found = {"group_id": sorted(set(group["group_id"].astype(int))), "signal": sorted(group.index[group["is_signal"]]),
             "num_series": sorted(set(group["num_series"].astype(int))), "series_ids": sorted(set(group["series_ids"]))}
    wanted = {"group_id": [1], "signal": [SIGNAL_SERIES], "num_series": [2], "series_ids": ["1,0"]}
    measured = {sid: [float(group.at[sid, "slope"]), float(group.at[sid, "corr_to_signal"])] for sid in KNOTS}
    matching = all(np.isclose(measured[sid], [slopes[sid], 1.0], rtol=TOLERANCE).all() for sid in KNOTS)
    KIT.verdict(f"{mode}: one group of both triggers, signal the steeper series {SIGNAL_SERIES}, slopes and correlations",
                found == wanted and matching, {**wanted, "slope, corr": {sid: [slopes[sid], 1.0] for sid in KNOTS}},
                {**found, "slope, corr": measured})
    curve_verdicts(mode, performance, begin_row, end_row, values[SIGNAL_SERIES])


def curve_verdicts(mode: str, performance: dict, begin_row: int, end_row: int, values: tuple) -> None:
    """The segment, the pure and attenuated curves, the terminal attenuation, acr and acr_after_attenuation, the signal counts."""
    begin_value, end_value = values
    tv = next_row_values(SIGNAL_SERIES)
    segments = performance["/gradient_signal_segments"]
    segment = segments[segments["group_id"] == 1]
    expected_pct = variation_pct(tv[begin_row:end_row + 1], begin_value)
    KIT.verdict(f"{mode}: the segment is (tv / {begin_value} - 1) x 100 on rows {begin_row} to {end_row}",
                list(pd.DatetimeIndex(segment["date"])) == list(DATES[begin_row:end_row + 1])
                and np.allclose(segment["pct"].to_numpy(dtype=float), expected_pct, rtol=TOLERANCE),
                expected_pct[[0, -1]].tolist(), segment["pct"].to_numpy(dtype=float)[[0, -1]].tolist() if len(segment) else [])
    rows = np.arange(LOOKBACK - 1, len(DATES))
    event = np.where(rows < begin_row, 0.0, tv[np.clip(rows, begin_row, end_row)] / begin_value - 1.0)
    final = WEIGHT * (end_value / begin_value - 1.0)
    total = performance["/total_variation"]
    KIT.verdict(f"{mode}: the curve is w x (tv / {begin_value} - 1) from row {begin_row}, flat at {final:.6f} after row {end_row}",
                list(pd.DatetimeIndex(total["date"])) == list(DATES[LOOKBACK - 1:])
                and np.allclose(total["total_variation"].to_numpy(dtype=float), WEIGHT * event, rtol=TOLERANCE, atol=1e-7),
                final, total["total_variation"].to_numpy(dtype=float)[-1:].tolist())
    attenuated_expected = WEIGHT * event
    attenuated_expected[-1] -= ATTENUATION_RATE * final
    attenuated = performance["/total_variation_attenuated"]["total_variation_attenuated"].to_numpy(dtype=float)
    attenuations = performance["/gradient_terminal_attenuations"].set_index("series_id")["amount"]
    KIT.verdict(f"{mode}: the attenuation takes {ATTENUATION_RATE} x the final value on the last item only, the strategy's terminal attenuation",
                np.allclose(attenuated, attenuated_expected, rtol=TOLERANCE, atol=1e-7)
                and np.isclose(float(attenuations.get("__strategy__", np.nan)), ATTENUATION_RATE * final, rtol=TOLERANCE),
                {"last": attenuated_expected[-1], "terminal": ATTENUATION_RATE * final},
                {"last": attenuated[-1:].tolist(), "terminal": float(attenuations.get("__strategy__", np.nan))})
    years = (DATES[-1] - DATES[LOOKBACK - 1]).days / 365.25
    expected_acr = {"acr": ((1.0 + final) ** (1.0 / years) - 1.0) * 100.0,
                    "acr_after_attenuation": ((1.0 + (1.0 - ATTENUATION_RATE) * final) ** (1.0 / years) - 1.0) * 100.0}
    table = performance["/gradient_metrics_table"]
    strategy = table[table["column_header"] == "strategy"].set_index("metric_key")["value"]
    found_acr = {key: float(strategy.get(key, np.nan)) for key in expected_acr}
    KIT.verdict(f"{mode}: acr and acr_after_attenuation of the strategy", all(
        np.isclose(found_acr[key], value, rtol=TOLERANCE) for key, value in expected_acr.items()), expected_acr, found_acr)
    breakdown = performance["/signals_breakdown_table"]
    counts = breakdown[breakdown["row_key"] == "Signals"].set_index("column_header")["value"].astype(float).to_dict()
    KIT.verdict(f"{mode}: the breakdown counts one signal, plus and true", counts == COUNTS, COUNTS, counts)


# =============================================================================
# HELPERS
# =============================================================================


def synthetic_copy(folder: Path) -> Path:
    """The two-series input inside the folder and a shipped-configuration copy that reads it."""
    raw = folder / SYNTHETIC_INPUT
    frames = {}
    for sid in KNOTS:
        levels = series_levels(sid)
        frames[sid] = pd.DataFrame({"var0": levels, "var1": levels * 0.99, "var2": levels * 1.01, "var3": levels, "var4": levels,
                                    "var5": np.full(len(DATES), 1.0e6)}, index=DATES)
    testkit.write_raw_input(raw, frames)
    relative = Path(os.path.relpath(raw, Path.cwd())).as_posix()
    return testkit.config_copy(KIT, folder, strategy={"read.raw_data": f"./{relative}", **OVERRIDES})


def series_levels(sid: str) -> np.ndarray:
    """The piecewise-linear var3 of one series."""
    return np.interp(np.arange(len(DATES)), *zip(*KNOTS[sid]))


def next_row_values(sid: str) -> np.ndarray:
    """The following_row trigger values of one series: the next row's var0 (= var3), var3 on the last row."""
    levels = series_levels(sid)
    return np.append(levels[1:], levels[-1])


if __name__ == "__main__":
    main()
