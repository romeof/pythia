"""
Detection print utilities for trigger output.

Provides the shared print function used by both realistic and ideal modes
for outputting trigger results.

Structure:
1. print_trigger_output (trigger I/O summary)
2. print_trigger_outcome_output (outcome I/O summary)
3. _print_section_header (shared header block)
4. _print_trigger_rows (trigger rows display)
5. _print_dataframe_head_tail (shared head/tail display)
"""

import pandas as pd

from ...common.constants import (
    END_REASON,
    ACTIVE_TRIGGER_MINUS_BEGIN,
    ACTIVE_TRIGGER_PLUS_BEGIN,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
    TRIGGER_MINUS_BEGIN,
    TRIGGER_MINUS_END,
    TRIGGER_PLUS_BEGIN,
    TRIGGER_PLUS_END,
    TURNING_POINTS,
)


def print_trigger_output(series_id: str, df: pd.DataFrame) -> None:
    """Print trigger output summary."""
    _print_section_header(series_id, "TRIGGER - OUTPUT")
    print(f"Shape (after trigger detection): {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {df.columns.tolist()}")

    _print_trigger_rows(df)
    _print_dataframe_head_tail(df)

    print("=" * 80 + "\n")


def print_trigger_outcome_output(
    series_id: str, df: pd.DataFrame, full_output: bool = False
) -> None:
    """Print outcome output summary."""
    _print_section_header(series_id, "TRIGGER - OUTCOME")
    print(f"Shape (triggers): {df.shape[0]} rows x {df.shape[1]} columns")
    if len(df) > 0:
        print(f"Date range: {df[TRIGGER_BEGIN_DATES].min()} to {df[TRIGGER_END_DATES].max()}")
    print(f"Columns: {df.columns.tolist()}")

    _print_dataframe_head_tail(df, full_output=full_output)

    print("=" * 80 + "\n")


def _print_section_header(series_id: str, title: str) -> None:
    """Print a standardized section header with title and series ID."""
    print(f"\n{title}")
    print("=" * 80)
    print(f"SERIES ID: {series_id}")
    print("=" * 80)


def _print_trigger_rows(df: pd.DataFrame) -> None:
    """Print rows where any trigger column is active."""
    trigger_cols = [
        END_REASON,
        TRIGGER_PLUS_BEGIN,
        TRIGGER_MINUS_BEGIN,
        TURNING_POINTS,
        ACTIVE_TRIGGER_PLUS_BEGIN,
        ACTIVE_TRIGGER_MINUS_BEGIN,
        TRIGGER_PLUS_END,
        TRIGGER_MINUS_END,
    ]
    existing_trigger_cols = [col for col in trigger_cols if col in df.columns]

    if not existing_trigger_cols:
        print("\nNo trigger plus/minus begin has been found")
        return

    # Create mask for rows where any trigger column is non-zero/non-empty
    trigger_mask = pd.Series(False, index=df.index)
    for col in existing_trigger_cols:
        if col == END_REASON:
            trigger_mask |= (df[col] != "") & (df[col].notna())
        else:
            trigger_mask |= (df[col] != 0) & (df[col].notna())

    trigger_rows = df[trigger_mask]

    if len(trigger_rows) == 0:
        print("\nNo trigger plus/minus begin has been found")
        return

    if (
        END_REASON in trigger_rows.columns
        and ACTIVE_TRIGGER_PLUS_BEGIN in trigger_rows.columns
        and ACTIVE_TRIGGER_MINUS_BEGIN in trigger_rows.columns
    ):
        trigger_rows = trigger_rows.copy()
        active_mask = (trigger_rows[ACTIVE_TRIGGER_PLUS_BEGIN] > 0) | (
            trigger_rows[ACTIVE_TRIGGER_MINUS_BEGIN] > 0
        )
        trigger_rows.loc[active_mask, END_REASON] = "none, active"

    print(f"\nRows with triggers ({len(trigger_rows)} rows):")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_colwidth",
        None,
    ):
        print(trigger_rows)


def _print_dataframe_head_tail(df: pd.DataFrame, full_output: bool = False) -> None:
    """Print first and last 10 rows of a DataFrame with full-width display."""
    with pd.option_context(
        "display.max_columns", None,
        "display.max_rows", None if full_output else 20,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        if full_output:
            print("\nAll rows:")
            print(df)
        else:
            print("\nFirst 10 rows:")
            print(df.head(10))
            print("\nLast 10 rows:")
            print(df.tail(10))
