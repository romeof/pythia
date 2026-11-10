"""
Computation module for per-series performance analysis.

This module contains the unified per-series processing logic for both trigger and signal phases:
metrics calculation, normalization, printing, and plotting.

Structure:
1. process_perseries (main starting point)
"""

from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

from .perseries_.metrics import (
    calculate_metrics,
    normalize_dataframe,
)
from .plots.perseries import plot_series
from .prints.perseries import print_perseries_rows


def process_perseries(
    series_id: str,
    series_df: pd.DataFrame,
    rows_df: pd.DataFrame,
    strategy_name: str,
    metric_config: Dict[str, Any],
    display_order_signals: List[str],
    needed_columns: Set[str],
    plots_dir: Path,
    printout_io: bool,
    plot_debug: bool,
    print_sub_task_name: str = "TRIGGERED SERIES METRICS",
    full_output: bool = False,
) -> None:
    """Process a single series: compute metrics, normalize, print, and plot.

    Designed to be called identically for both trigger and signal phases.
    The caller controls behavior via printout_io and plot_debug flags,
    which come from the phase-specific config section.

    Args:
        series_id: Series identifier
        series_df: Original time series DataFrame
        rows_df: Rows (trigger or signal) for this series
        strategy_name: Strategy name for plot titles
        metric_config: Metric configuration from metrics.json
        display_order_signals: Ordered composite signal metric keys for display
        needed_columns: Set of DataFrame column names to compute (from _extract_needed_columns)
        plots_dir: Directory to save plot files
        printout_io: Whether to print per-series output
        plot_debug: Whether to generate per-series plots
        print_sub_task_name: Sub-task label shown in print output
        full_output: If True, print all rows instead of first/last 10
    """
    if rows_df is None or len(rows_df) == 0:
        return

    rows_df = calculate_metrics(series_df, rows_df, needed_columns)
    rows_df = normalize_dataframe(series_id, rows_df, metric_config)

    if printout_io:
        print_perseries_rows(
            series_id, rows_df, display_order_signals, metric_config,
            sub_task_name=print_sub_task_name,
            full_output=full_output,
        )
    if plot_debug:
        plot_file = plots_dir / f"series_id_{series_id}.png"
        plot_series(
            series_id,
            series_df,
            rows_df,
            strategy_name,
            metric_config,
            save_path=str(plot_file),
            show=False,
            include_summary=True,
            display_order_signals=display_order_signals,
        )
