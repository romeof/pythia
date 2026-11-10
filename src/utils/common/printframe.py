"""
Shared print frame of the task printouts.

Provides the shared print frame function used by the perseries, gradient and signal
printouts, following the same format as trigger/prints/detection.py.

Structure:
1. print_dataframe_output (the shared print frame)
2. Internal helpers (_print_dataframe_body, _print_series_body)
"""

from typing import Optional, Union

import pandas as pd


def print_dataframe_output(
    series_id: Optional[str],
    df: Union[pd.DataFrame, pd.Series],
    task_name: str,
    sub_task_name: str,
    full_output: bool = False,
) -> None:
    """Print a DataFrame or Series following the standard printout format.

    Format:
        {task_name} - {sub_task_name}
        ================================================================================
        SERIES ID: {series_id}  (if provided)
        ================================================================================
        Shape: N rows x M columns  (DataFrame) / Length: N  (Series)
        Date range: ... to ...  (if DatetimeIndex)
        Columns: [...]  (DataFrame only)

        First 10 rows:      (or full DataFrame when full_output=True)
        ...

        Last 10 rows:       (omitted when full_output=True)
        ...
        ================================================================================

    Args:
        task_name: Name of the task (e.g. "PERFORMANCE PERSERIES")
        sub_task_name: Name of the sub-task (e.g. "EVENTS")
        series_id: Series identifier (omitted from output if None)
        df: DataFrame or Series to print
        full_output: If True, print all rows instead of first/last 10
    """
    print(f"\n{task_name} - {sub_task_name}")
    print(f"{'=' * 80}")
    if series_id is not None:
        print(f"SERIES ID: {series_id}")
        print(f"{'=' * 80}")

    if isinstance(df, pd.DataFrame):
        _print_dataframe_body(df, full_output=full_output)
    else:
        _print_series_body(df, full_output=full_output)

    print(f"{'=' * 80}\n")


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _print_dataframe_body(df: pd.DataFrame, full_output: bool = False) -> None:
    """Print shape, date range, columns, and head/tail (or full) for a DataFrame."""
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {df.columns.tolist()}")
    with pd.option_context(
        "display.max_columns", None,
        "display.max_rows", None if full_output else 20,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        if full_output:
            print(f"\nAll rows:")
            print(df)
        else:
            print(f"\nFirst 10 rows:")
            print(df.head(10))
            print(f"\nLast 10 rows:")
            print(df.tail(10))


def _print_series_body(series: pd.Series, full_output: bool = False) -> None:
    """Print length, date range, and head/tail (or full) for a Series."""
    print(f"Length: {len(series)}")
    if isinstance(series.index, pd.DatetimeIndex) and len(series) > 0:
        print(f"Date range: {series.index.min()} to {series.index.max()}")
    with pd.option_context(
        "display.max_columns", None,
        "display.max_rows", None if full_output else 20,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        if full_output:
            print(f"\nAll items:")
            print(series)
        else:
            print(f"\nFirst 10 items:")
            print(series.head(10))
            print(f"\nLast 10 items:")
            print(series.tail(10))
