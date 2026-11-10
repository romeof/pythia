"""
Read print utilities for trigger pipeline.

Provides print functions for HDF5 reading stage output.

Structure:
1. print_read_file_summary (file-level summary)
2. print_read_series_input (per-series input summary)
3. print_read_date_filter_result (post-filter summary)
"""

from typing import List

import pandas as pd


def print_read_file_summary(
    input_file: str,
    all_series: List[str],
    series_to_process: List[str],
    ini_date: str,
    end_date: str,
) -> None:
    """Print summary of HDF5 file contents."""
    print(f"\n{'=' * 80}")
    print(f"READING HDF5 FILE: {input_file}")
    print(f"{'=' * 80}")
    print(f"Series found in file: {all_series}")
    print(f"Series to process: {series_to_process} (excluding skip_series)")
    print(f"Total series to process: {len(series_to_process)}")
    print(f"Date filter: {ini_date} to {end_date}")
    print(f"{'=' * 80}\n")


def print_read_series_input(series_id: str, df: pd.DataFrame) -> None:
    """Print series input summary."""
    print(f"\nTRIGGER - READ")
    print(f"{'-' * 80}")
    print(f"SERIES ID: {series_id}")
    print(f"{'-' * 80}")
    print(f"Shape (before date filter): {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 10 rows:")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_colwidth",
        None,
    ):
        print(df.head(10))
    print(f"\nLast 10 rows:")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_colwidth",
        None,
    ):
        print(df.tail(10))


def print_read_date_filter_result(
    df: pd.DataFrame, ini_date: str, end_date: str
) -> None:
    """Print result after date filtering."""
    print(
        f"\nShape (after date filter {ini_date} to {end_date}): {df.shape[0]} rows x {df.shape[1]} columns"
    )
    print(f"{'-' * 80}\n")
