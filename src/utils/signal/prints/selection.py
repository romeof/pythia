"""
Print utilities for signal selection output.

Provides printout functions for accumulated triggers and signal groups,
delegating to the shared print frame print_dataframe_output (src/utils/common/printframe.py).
"""

import pandas as pd

from ...common.constants import (
    CORR_TO_SIGNAL,
    GROUP_ID,
    IS_PLUS_TRIGGER,
    IS_SIGNAL,
    IS_TRUE,
    NUM_SERIES,
    SERIES_ID,
    SERIES_IDS,
    SLOPE,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
    TRIGGER_END_VALUES,
    TRIGGER_START_VALUES,
)
from ...common.printframe import print_dataframe_output

_TRIGGER_BASE_COLUMNS = [
    SERIES_ID,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
    TRIGGER_START_VALUES,
    TRIGGER_END_VALUES,
    IS_PLUS_TRIGGER,
    IS_TRUE,
]

_SIGNAL_GROUP_COLUMNS = [
    GROUP_ID,
    IS_SIGNAL,
    SLOPE,
    CORR_TO_SIGNAL,
    SERIES_IDS,
    NUM_SERIES,
]


def print_accumulated(all_triggers: pd.DataFrame, full_output: bool = False) -> None:
    """Print accumulated triggers after concatenation and sorting."""
    if all_triggers is None or all_triggers.empty:
        print("\nSIGNAL - ACCUMULATED TRIGGERED SERIES")
        print("=" * 80)
        print("No triggers accumulated.")
        print("=" * 80)
        print("")
        return

    present_cols = [col for col in _TRIGGER_BASE_COLUMNS if col in all_triggers.columns]

    print_dataframe_output(
        task_name="SIGNAL",
        sub_task_name="ACCUMULATED TRIGGERED SERIES",
        series_id=None,
        df=all_triggers[present_cols],
        full_output=full_output,
    )


def print_groups(groups_df: pd.DataFrame, full_output: bool = False) -> None:
    """Print signal selection groups DataFrame."""
    present_base = [col for col in _TRIGGER_BASE_COLUMNS if col in groups_df.columns]
    present_group = [col for col in _SIGNAL_GROUP_COLUMNS if col in groups_df.columns]

    print_dataframe_output(
        task_name="SIGNAL",
        sub_task_name="SELECTION",
        series_id=None,
        df=groups_df[present_base + present_group],
        full_output=full_output,
    )
