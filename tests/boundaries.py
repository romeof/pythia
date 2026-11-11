"""Regression for boundary inputs (D-6; F-01, F-20, F-21, R-T16).

(a) A tie at a turning point: ideal analyze_series on EXP-01's plateau fixture (97 samples of
    100 + 10 sin(x), x in [0, 3 pi], sample 17 set equal to sample 16, a two-sample plateau at the
    first maximum): the pairs (minus, 16, 48) and (plus, 48, 80), the plateau typed as a maximum.
(b) A single-row and an empty series: --task trigger on a synthetic input in the reader's layout
    (index_days, columns) with a 120-row series 0, a one-row series 1 and an empty
    series 2: the run returns 0, the store holds /series_0 only, and each short series logs its
    skip WARNING.
(c) NaN rows: Preprocess.process_series drops exactly the rows holding a NaN.
(d) An attenuation date at the index end: build_attenuation_schedule on the observed dates from 2020-01-02 to
    2021-06-01 with attenuation_month 6: the 2020 item is attenuated on 2021-06-01, the first index date of
    June that the index ends on, and the last item also on the index's last date.
(e) The last row under following_row: select_trigger_value_vector takes each row's next var0 and
    falls back to the last row's own var3. Dates missing inside a bad block are badblocks.py's case.
A vector without the last-row fallback leaves NaN on the last row.

Structure:
1. main
2. Cases
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

from src.utils.common.common import select_trigger_value_vector
from src.utils.performance.gradient_.attenuation import build_attenuation_schedule
from src.utils.preprocess.preprocess import Preprocess

SAMPLES, PLATEAU = 97, 16
PLATEAU_PAIRS = {("minus", 16, 48), ("plus", 48, 80)}
FLAGS = {"plus": ("trigger_plus_begin", "trigger_plus_end"), "minus": ("trigger_minus_begin", "trigger_minus_end")}
SYNTHETIC_INPUT = Path("input") / "2026_01_01" / "0" / "raw" / "0.h5"
FIRST_DAY = "2020-01-02"
SERIES_ROWS = {"0": 120, "1": 1, "2": 0}
SKIP_WARNING = "insufficient data length"
NAN_ROWS = [3, 7]
ATTENUATION_INDEX = ("2020-01-02", "2021-06-01")
ATTENUATION_MONTH = 6
ATTENUATION_DATE = "2021-06-01"


def main() -> None:
    """The five cases, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy={"trigger.replay.mode": "ideal"})
        config = testkit.validated_config(config_file)
        plateau_tie(testkit.strategy(config, testkit.output_file(config_file, "trigger")))
        nan_rows(config, folder)
    short_series()
    attenuation_at_index_end()
    last_row_fallback()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def plateau_tie(strategy) -> None:
    """Case (a): the plateau keeps its maximum type, so the pairs are EXP-01's."""
    frame = sine_frame()
    frame.iloc[PLATEAU + 1, :5] = frame.iloc[PLATEAU, :5].to_numpy()
    output = strategy.analyze_series("0", frame)
    pairs = {(direction, int(begin), int(end)) for direction, (begin_col, end_col) in FLAGS.items()
             for begin, end in zip(np.flatnonzero(output[begin_col].to_numpy() > 0),
                                   np.flatnonzero(output[end_col].to_numpy() > 0))}
    KIT.verdict("case (a): a two-sample plateau at a maximum keeps the pairs", pairs == PLATEAU_PAIRS, sorted(PLATEAU_PAIRS),
                sorted(pairs))


def nan_rows(config, folder: Path) -> None:
    """Case (c): the rows holding a NaN are dropped, the others kept."""
    frame = sine_frame()
    frame.iloc[NAN_ROWS, frame.columns.get_loc("var3")] = np.nan
    kept = Preprocess(config.trigger.consecutive_rows_with_bad_preprocessed_data, config.trigger.validity,
                      config.trigger.preprocess.printout_io, folder / "nan" / "0.h5").process_series("s", frame)
    dropped = sorted(frame.index.get_loc(date) for date in frame.index.difference(kept.index))
    KIT.verdict("case (c): exactly the NaN rows are dropped", dropped == NAN_ROWS, NAN_ROWS, dropped)


def short_series() -> None:
    """Case (b): a one-row and an empty series are skipped and the run completes."""
    with testkit.scratch(KIT) as folder:
        raw = folder / SYNTHETIC_INPUT
        testkit.write_raw_input(raw, {series_id: synthetic_frame(rows) for series_id, rows in SERIES_ROWS.items()})
        relative = Path(os.path.relpath(raw, Path.cwd())).as_posix()
        config_file = testkit.config_copy(KIT, folder, strategy={"read.raw_data": f"./{relative}", "read.reference_series": [],
                                                                 "trigger.replay.mode": "realistic"})
        child = testkit.run_pipeline(KIT, folder, "short", config_file, ["trigger"])
        keys = testkit.store_keys(testkit.output_file(config_file, "trigger")) or []
        series_keys = [key for key in keys if key.startswith("/series_")]
        skipped = sorted(series_id for series_id in ("1", "2")
                         if any(SKIP_WARNING in line and f"'{series_id}'" in line for line in testkit.plain_lines(child.text)))
        KIT.verdict("case (b): the one-row and the empty series skipped with their WARNING, return code 0, /series_0 stored",
                    child.return_code == 0 and series_keys == ["/series_0"] and skipped == ["1", "2"],
                    {"return_code": 0, "series keys": ["/series_0"], "skipped": ["1", "2"]},
                    {"return_code": child.return_code, "timed_out": child.timed_out, "series keys": series_keys, "skipped": skipped})


def attenuation_at_index_end() -> None:
    """Case (d): an attenuation date on the index's last date."""
    schedule = build_attenuation_schedule(pd.bdate_range(*ATTENUATION_INDEX), ATTENUATION_MONTH)
    attenuations = [str(item["attenuation_date"].date()) for item in schedule]
    KIT.verdict(f"case (d): both attenuation items fall on the index's last date {ATTENUATION_DATE}", attenuations == [ATTENUATION_DATE, ATTENUATION_DATE],
                [ATTENUATION_DATE, ATTENUATION_DATE], attenuations)


def last_row_fallback() -> None:
    """Case (e): next-row var0, the last row falling back to its own var3."""
    frame = sine_frame()
    values = select_trigger_value_vector(frame, "following_row", primary_col="var3", start_col="var0")
    expected = np.append(frame["var0"].to_numpy()[1:], frame["var3"].to_numpy()[-1])
    KIT.verdict("case (e): following_row takes the next row's var0 and the last row's var3",
                bool(np.array_equal(values, expected)), {"last": float(expected[-1])},
                {"last": float(values[-1]), "equal": bool(np.array_equal(values, expected))})


# =============================================================================
# HELPERS
# =============================================================================


def sine_frame() -> pd.DataFrame:
    """EXP-01's fixture: 97 samples of var3 = var4 = 100 + 10 sin(x), var0 = var3 + 0.5, var1 and var2 1 % below and
    above var3 (so no row reads as constant), var5 = 1e6."""
    levels = (100.0 + 10.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, SAMPLES))).astype("float32")
    return pd.DataFrame({"var0": levels + np.float32(0.5), "var1": levels * np.float32(0.99), "var2": levels * np.float32(1.01),
                         "var3": levels, "var4": levels, "var5": np.full(SAMPLES, 1.0e6)},
                        index=pd.bdate_range("2020-01-01", periods=SAMPLES))


def synthetic_frame(rows: int) -> pd.DataFrame:
    """A series of the synthetic input: three periods of a sine over the rows, its columns as in sine_frame."""
    levels = 100.0 + 10.0 * np.sin(np.linspace(0.0, 6.0 * np.pi, rows))
    return pd.DataFrame({"var0": levels + 0.5, "var1": levels * 0.99, "var2": levels * 1.01, "var3": levels, "var4": levels,
                         "var5": np.full(rows, 1.0e6)}, index=pd.bdate_range(FIRST_DAY, periods=rows))


if __name__ == "__main__":
    main()
