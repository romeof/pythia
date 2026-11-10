"""
Preprocess print utilities for trigger pipeline.

Provides the print function for preprocessing output summary.

Structure:
1. print_preprocess_output (preprocessing I/O summary)
"""

from typing import Dict

import pandas as pd


def print_preprocess_output(series_id: str, df: pd.DataFrame, report: Dict) -> None:
    """Print preprocessing output summary."""
    print(f"\nTRIGGER - PREPROCESS")
    print(f"{'=' * 80}")
    print(f"SERIES ID: {series_id}")
    print(f"{'=' * 80}")
    print(f"Shape (after cleaning): {df.shape[0]} rows x {df.shape[1]} columns")

    if report["issues"]:
        num_perseries = len(report["perseries_issues"])
        num_consecutive = len(report["consecutive_issues"])
        print(
            f"Issues found: {num_perseries} perseries, {num_consecutive} consecutive blocks"
        )
    else:
        print(f"No issues found - data is clean")

    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 10 rows (preprocessed):")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_colwidth",
        None,
    ):
        print(df.head(10))
    print(f"\nLast 10 rows (preprocessed):")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_colwidth",
        None,
    ):
        print(df.tail(10))
    print(f"{'=' * 80}\n")
