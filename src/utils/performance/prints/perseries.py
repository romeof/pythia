"""
Print utilities for perseries performance output.

Provides the printout function for perseries performance rows,
delegating to the shared print frame print_dataframe_output (src/utils/common/printframe.py).

Structure:
1. print_perseries_rows (main starting point)
"""

from typing import Dict, List

import pandas as pd

from ...common.constants import (
    IS_PLUS_TRIGGER,
    IS_TRUE,
    SERIES_ID,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
    TRIGGER_END_VALUES,
    TRIGGER_START_VALUES,
)
from ...common.printframe import print_dataframe_output
from .summary import generate_summary_table
from ..perseries_.metrics import resolve_signal_selection


def print_perseries_rows(
    series_id: str,
    rows_df: pd.DataFrame,
    display_order_signals: List[str],
    metric_config: Dict,
    sub_task_name: str = "TRIGGERED SERIES METRICS",
    full_output: bool = False,
) -> None:
    """Print a perseries performance rows DataFrame and aggregated metrics summary.

    Args:
        series_id: Series identifier
        rows_df: Rows DataFrame to print (trigger or signal rows)
        display_order_signals: Ordered list of composite signal metric keys
        metric_config: Metric configuration from metrics.json
        sub_task_name: Sub-task label shown in output header
        full_output: If True, print all rows instead of first/last 10
    """
    # Derive the sequence of unique metric column names from the composite key selection.
    # Deduplicates by column name (e.g. variation_avg/std/std_dw all map to "variation").
    selection = resolve_signal_selection(metric_config, display_order_signals)
    seen: set = set()
    metric_col_sequence: List[str] = []
    for metric_name, _, _ in selection:
        col = metric_config[metric_name]["column"]
        if col not in seen:
            seen.add(col)
            metric_col_sequence.append(col)

    # Arrange columns: base columns first, then the metric column sequence
    base_cols = [
        SERIES_ID,
        TRIGGER_BEGIN_DATES,
        TRIGGER_END_DATES,
        TRIGGER_START_VALUES,
        TRIGGER_END_VALUES,
        IS_PLUS_TRIGGER,
        IS_TRUE,
    ]
    present_base_cols = [col for col in base_cols if col in rows_df.columns]
    present_metrics = [col for col in metric_col_sequence if col in rows_df.columns]
    rows_df = rows_df[present_base_cols + present_metrics]

    print_dataframe_output(
        task_name="PERFORMANCE",
        sub_task_name=sub_task_name,
        series_id=series_id,
        df=rows_df,
        full_output=full_output,
    )

    summary = generate_summary_table(rows_df, metric_config, display_order_signals)
    print(summary)
