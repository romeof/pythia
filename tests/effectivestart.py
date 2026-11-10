"""Regression for the one positional effective start (Q2, F-02).

A dual run (--task all) on a copy that keeps series 0 and 4 (read.skip_series 1, 2, 3, 5) of
the real input. Both series' effective start is index[lookback_window - 1] = 2019-03-14 of
their date-filtered, preprocessed frame, three rows after the calendar date
read.ini_date + (lookback_window - 1) weekdays = 2019-03-11 that the unfixed code used for the
extraction filter, the reference trimming and the curve index. Expected in the outputs: the
ideal /triggers_4 holds no begin 2019-03-13 (inside that gap); /reference_series_0 starts
2019-03-14 in both modes; /total_variation starts 2019-03-14 in both modes; the
effective_start column of /gradient_params holds "2019-03-14" in both modes; and
replot_gradient_from_hdf5 on each performance file writes its figure in a child (return code 0).
Without a run, build_reference_variation_series_map on a reference that holds three rows before
the curve index start, as an untrimmed reference (its preprocessed frame shorter than the
lookback) does: its curve is 0 on its first row on the index and compounds from that row, and a
reference whose rows all precede the index gets no curve.
The unfixed code keeps the 2019-03-13 trigger, starts the reference rows and the curve at
2019-03-11 and persists no effective_start column; its curve builder takes the base from the
reference's first row, so the curve starts with the move from the last row before the index start,
and a reference without a row on the index gets a flat zero curve.

Structure:
1. main
2. Checks
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from pathlib import Path

import numpy as np
import pandas as pd

KEPT_SERIES = {"trigger.replay.mode": "dual", "read.skip_series": ["1", "2", "3", "5"]}
START = "2019-03-14"
GAP_SERIES, GAP_BEGIN = "4", "2019-03-13"
REFERENCE_KEY, TRIGGERS_KEY = "/reference_series_0", f"/triggers_{GAP_SERIES}"
CURVE_KEY, PARAMS_KEY, START_COLUMN = "/total_variation", "/gradient_params", "effective_start"
REPLOT_CODE = testkit.CHILD_BOOTSTRAP + (
    "from pathlib import Path\n"
    "from src.utils.performance.replot import replot_gradient_from_hdf5\n"
    "replot_gradient_from_hdf5(Path(sys.argv[4]), save_path=Path(sys.argv[5]))\n")
CURVE_INDEX = pd.bdate_range("2021-01-04", periods=5)
# Reference rows: three before the curve index start (2020-12-30, 2020-12-31, 2021-01-01), then one per index date
REFERENCE_ROWS = pd.Series([100.0, 125.0, 120.0, 132.0, 108.0, 126.0, 117.0, 120.0],
                           index=pd.bdate_range("2020-12-30", periods=8))
UNTRIMMED_ID, EARLIER_ID = "untrimmed", "earlier"


def main() -> None:
    """The dual run, its outputs per mode and the replot, the untrimmed reference curve, then the summary
    (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(KEPT_SERIES))
        child = testkit.run_pipeline(KIT, folder, "dual", config_file, ["all"])
        KIT.verdict("non-vacuity: the dual run completed", child.return_code == 0, 0,
                    {"return_code": child.return_code, "timed_out": child.timed_out})
        gap_trigger(mode_file(config_file, "ideal", "trigger"))
        for mode in testkit.MODES:
            first_dates(mode, mode_file(config_file, mode, "trigger"), mode_file(config_file, mode, "performance"))
            replot(folder, mode, mode_file(config_file, mode, "performance"))
    untrimmed_reference()
    KIT.finish()


def mode_file(config_file: Path, mode: str, task: str) -> Path:
    """The task output of one concrete mode of the dual copy."""
    from src.utils.common.outputs import get_output_file_name

    config = testkit.validated_config(config_file)
    trigger = config.trigger.model_copy(update={"replay": config.trigger.replay.model_copy(update={"mode": mode})})
    return Path(get_output_file_name(config.read, trigger, task))


# =============================================================================
# CHECKS
# =============================================================================


def gap_trigger(trigger_file: Path) -> None:
    """The ideal trigger of series 4 beginning inside the gap is filtered out at the positional start."""
    begins = [str(date.date()) for date in pd.to_datetime(read_key(trigger_file, TRIGGERS_KEY)["trigger_begin_dates"])]
    KIT.verdict(f"non-vacuity: ideal {TRIGGERS_KEY} holds triggers", bool(begins), "> 0 rows", len(begins))
    KIT.verdict(f"ideal {TRIGGERS_KEY} holds no begin {GAP_BEGIN} (before the effective start {START})",
                bool(begins) and GAP_BEGIN not in begins, f"no {GAP_BEGIN}",
                {"holds_gap_begin": GAP_BEGIN in begins, "earliest": min(begins, default=None)})


def first_dates(mode: str, trigger_file: Path, performance_file: Path) -> None:
    """The reference rows, the curve and the persisted effective start of one mode begin at the positional start."""
    reference = read_key(trigger_file, REFERENCE_KEY)
    KIT.verdict(f"{mode}: {REFERENCE_KEY} starts at {START}", str(reference.index[0].date()) == START, START,
                {"first": str(reference.index[0].date()), "rows": len(reference)})
    curve = read_key(performance_file, CURVE_KEY)
    first = str(pd.Timestamp(curve["date"].iloc[0]).date())
    KIT.verdict(f"{mode}: {CURVE_KEY} starts at {START}", first == START, START, {"first": first, "rows": len(curve)})
    params = read_key(performance_file, PARAMS_KEY)
    persisted = str(params[START_COLUMN].iloc[0]) if START_COLUMN in params.columns else "column absent"
    KIT.verdict(f"{mode}: {PARAMS_KEY}.{START_COLUMN} is {START}", persisted == START, START, persisted)


def replot(folder: Path, mode: str, performance_file: Path) -> None:
    """replot_gradient_from_hdf5 on the persisted performance file writes its figure in a child."""
    figure = folder / f"replot_{mode}.png"
    child = testkit.run_child(KIT, folder, f"replot{mode}", ["-c", REPLOT_CODE, *testkit.child_args(KIT),
                                                             str(performance_file), str(figure)])
    KIT.verdict(f"{mode}: the replot writes its figure (return code 0)", child.return_code == 0 and figure.is_file(),
                "return code 0 and the figure", {"return_code": child.return_code, "figure": figure.is_file(),
                                          "last": testkit.plain_lines(child.text)[-1:]})


def untrimmed_reference() -> None:
    """A reference with rows before the curve index start: its curve is 0 on its first row on the index and
    compounds from that row; a reference whose rows all precede the index gets no curve."""
    from src.utils.common.common import variation_pct
    from src.utils.performance.gradient import build_reference_variation_series_map

    earlier = REFERENCE_ROWS[REFERENCE_ROWS.index < CURVE_INDEX[0]]
    curves = build_reference_variation_series_map({UNTRIMMED_ID: REFERENCE_ROWS, EARLIER_ID: earlier}, CURVE_INDEX)
    on_index = REFERENCE_ROWS[REFERENCE_ROWS.index >= CURVE_INDEX[0]]
    expected = variation_pct(on_index, on_index.iloc[0]) / 100.0
    curve = curves.get(UNTRIMMED_ID)
    matches = curve is not None and curve.index.equals(CURVE_INDEX) and np.allclose(curve.to_numpy(), expected.to_numpy())
    KIT.verdict(f"untrimmed reference: {len(earlier)} rows before the curve index start, the curve is 0 on "
                f"{CURVE_INDEX[0].date()} and compounds from that row", matches, [round(value, 6) for value in expected],
                None if curve is None else [round(value, 6) for value in curve])
    left = curves.get(EARLIER_ID)
    KIT.verdict("a reference whose rows all precede the curve index gets no curve", left is None, None,
                None if left is None else [round(value, 6) for value in left])


def read_key(path: Path, key: str) -> pd.DataFrame:
    """One key of a store read in mode r."""
    with pd.HDFStore(str(path), mode="r") as store:
        return store.get(key)


if __name__ == "__main__":
    main()
