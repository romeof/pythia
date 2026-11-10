"""Regression for the helpers that used to swallow a failure or substitute a default (G-9, F-14 groups B-D, G;
group E's config-reads diagnostics left with the reads log, group F's scalar decoder with the scalars read).

In-process cases observe the expected exception through testkit.raised, on a scratch config copy
(one series, realistic):
B  add_series_values on a frame whose index cannot become a DatetimeIndex raises (before G-9: the
   series is silently stored nowhere); np.polyfit, then np.corrcoef, patched to raise make
   build_signal_window_groups raise on a two-trigger fixture (before: a NaN slope, no
   correlation, the grouping goes on).
C  normalize_for_hdf5 with a float32 schema on a text column raises (before: a WARNING and a
   partial frame); normalize_dataframe with a text metric column raises (before: a WARNING and
   the raw frame); plot_gradient, its four leading arguments positional, with a text start_val
   raises a TypeError in the segment label (before: a "nan%" label).
D  the five dead helpers are gone: HDF5Manager.put_dataframe_safe and get_all_keys,
   PlotBase.plots_dir_from_store and save_figure, and _resolve_trigger_begin_date of the
   realistic printouts (before: present); no planted failure can reach deleted code.
G  _add_active_trigger_flags of a create_strategy instance with a pre-cleaned label absent from
   the index raises the lookup's own KeyError naming that label (before: the ValueError of the
   None its lookup helper substituted).

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

import src.utils.trigger.prints.realistic as realistic_prints
from src.utils.common.constants import (
    GROUP_ID,
    IS_PLUS_TRIGGER,
    IS_SIGNAL,
    IS_TRUE,
    SERIES_ID,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
    TRIGGER_END_VALUES,
    TRIGGER_START_VALUES,
    VAR3,
)
from src.utils.common.hdf5manager import HDF5Manager, hdf5_manager
from src.utils.performance.perseries_.metrics import get_metric_config, normalize_dataframe
from src.utils.performance.plots.common import PlotBase
from src.utils.performance.plots.gradient import plot_gradient
from src.utils.signal.selection import SignalSelection

ROWS = 40
DATES = pd.bdate_range("2022-01-03", periods=ROWS)
# (series id, scale, begin row, end row): two overlapping plus triggers on one index
FIXTURE_TRIGGERS = (("a", 1.0, 10, 20), ("b", 2.0, 12, 22))
UNDATED_INDEX = ["not a date", "nor this"]
TEXT = "text"
METRIC_COLUMN = "variation"
UNSUPPORTED_OPERAND = "unsupported operand"
DEAD_HELPERS = ((HDF5Manager, "put_dataframe_safe"), (HDF5Manager, "get_all_keys"), (PlotBase, "plots_dir_from_store"),
                (PlotBase, "save_figure"), (realistic_prints, "_resolve_trigger_begin_date"))
ABSENT_DATE = "2030-01-02"


def main() -> None:
    """Rows B, C, D and G, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        signal_computations(testkit.validated_config(config_file))
        normalisation()
        dead_helpers()
        ideal_flags(config_file)
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def signal_computations(config) -> None:
    """Row B: the date conversion, the slope fit and the correlation raise."""
    selection = SignalSelection(config.signal, config.trigger.replay.mode)
    undated = pd.DataFrame({VAR3: [1.0, 2.0]}, index=UNDATED_INDEX)
    exc = testkit.raised(lambda: selection.add_series_values("x", undated))
    KIT.verdict("row B: add_series_values raises on an index that cannot become a DatetimeIndex", exc is not None,
                "an exception", repr(exc))
    for name in ("polyfit", "corrcoef"):
        with planted_numpy(name):
            exc = testkit.raised(two_trigger_selection(config).build_signal_window_groups)
        KIT.verdict(f"row B: build_signal_window_groups raises the planted np.{name} error",
                    isinstance(exc, OSError) and testkit.PLANTED_ERROR in str(exc), testkit.PLANTED_ERROR, repr(exc))


def normalisation() -> None:
    """Row C: the HDF5 and the metric normalisation and the segment label raise."""
    text_frame = pd.DataFrame({METRIC_COLUMN: [TEXT, TEXT]})
    exc = testkit.raised(lambda: hdf5_manager.normalize_for_hdf5(text_frame, schema={"float32": [METRIC_COLUMN]}))
    KIT.verdict("row C: normalize_for_hdf5 raises on a text column under a float32 schema", exc is not None,
                "an exception", repr(exc))
    exc = testkit.raised(lambda: normalize_dataframe("0", text_frame, get_metric_config()))
    KIT.verdict("row C: normalize_dataframe raises on a text metric column", exc is not None, "an exception", repr(exc))
    signals_data = {1: {"pct": pd.Series([0.0, 1.0, 2.0], index=DATES[:3]), "start_val": TEXT, "end_val": 1.0,
                        SERIES_ID: "a", IS_PLUS_TRIGGER: True, IS_TRUE: True, TRIGGER_START_VALUES: 1.0,
                        TRIGGER_END_VALUES: 1.0}}
    filtered = pd.DataFrame({GROUP_ID: [1], IS_SIGNAL: [True], IS_TRUE: [True], IS_PLUS_TRIGGER: [True]})
    exc = testkit.raised(lambda: plot_gradient(signals_data, filtered, None, DATES))
    KIT.verdict("row C: plot_gradient raises a TypeError on a text start_val",
                isinstance(exc, TypeError) and UNSUPPORTED_OPERAND in str(exc), f"TypeError {UNSUPPORTED_OPERAND}",
                repr(exc))


def dead_helpers() -> None:
    """Row D: the five dead helpers are absent."""
    present = [f"{getattr(owner, '__name__', owner)}.{name}" for owner, name in DEAD_HELPERS if hasattr(owner, name)]
    KIT.verdict("row D: the five dead helpers are gone", not present, [], present)


def ideal_flags(config_file: Path) -> None:
    """Row G: a pre-cleaned label absent from the index raises the lookup's own KeyError."""
    strategy = testkit.strategy(testkit.validated_config(config_file), testkit.output_file(config_file, "trigger"))
    frame = pd.DataFrame({VAR3: np.arange(5, dtype=float)}, index=DATES[:5])
    exc = testkit.raised(lambda: strategy._add_active_trigger_flags(frame, pd.Timestamp(ABSENT_DATE), None))
    KIT.verdict(f"row G: _add_active_trigger_flags raises the KeyError naming {ABSENT_DATE}",
                isinstance(exc, KeyError) and ABSENT_DATE in str(exc), f"KeyError {ABSENT_DATE}", repr(exc))


# =============================================================================
# HELPERS
# =============================================================================


def two_trigger_selection(config) -> SignalSelection:
    """Two linear series on one index, each with one plus trigger, the second beginning inside the first."""
    selection = SignalSelection(config.signal, config.trigger.replay.mode)
    for series_id, scale, begin, end in FIXTURE_TRIGGERS:
        values = scale * (100.0 + np.arange(ROWS, dtype=float))
        selection.add_series_values(series_id, pd.DataFrame({VAR3: values}, index=DATES))
        selection.add_series_triggers(series_id, pd.DataFrame({
            SERIES_ID: [series_id], TRIGGER_BEGIN_DATES: [DATES[begin]], TRIGGER_END_DATES: [DATES[end]],
            TRIGGER_START_VALUES: [values[begin]], TRIGGER_END_VALUES: [values[end]], IS_PLUS_TRIGGER: [True],
            IS_TRUE: [True]}))
    return selection


@contextmanager
def planted_numpy(name: str) -> Iterator[None]:
    """numpy's function of that name raises the planted OSError inside the block."""
    original = getattr(np, name)

    def planted(*args: object, **kwargs: object) -> object:
        raise OSError(testkit.PLANTED_ERROR)

    setattr(np, name, planted)
    try:
        yield
    finally:
        setattr(np, name, original)


if __name__ == "__main__":
    main()
