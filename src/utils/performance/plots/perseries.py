"""
Plotting and visualization module for perseries performance analysis.

This module handles all plotting and visualization logic, and is designed to be
completely generic - it works with any metrics defined in METRIC_CONFIG.

Structure:
1. plot_series (main starting point)
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...common.constants import (
    IS_PLUS_TRIGGER,
    IS_TRUE,
    ACTIVE_TRIGGER_MINUS_BEGIN,
    ACTIVE_TRIGGER_PLUS_BEGIN,
    VAR3,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
)
from ..prints.summary import generate_summary_table
from .common import PlotBase

logger = logging.getLogger(__name__)


def plot_series(
    series_id: str,
    original_df: pd.DataFrame,
    triggers_df: pd.DataFrame,
    strategy_name: str = "Strategy",
    metric_config: Dict = None,
    trigger_numbers: list = None,
    save_path: Optional[str] = None,
    show: bool = True,
    include_summary: bool = False,
    display_order_signals: Optional[List[str]] = None,
):
    """
    Plot time series with trigger segments using colored lines based on trigger type.
    Annotate each segment with trigger number and variation.
    Always show all legend labels.
    """
    if triggers_df.empty:
        logger.warning(f"No triggers to plot for series {series_id}")
        return

    # Prepare figure
    if include_summary:
        fig, (ax_plot, ax_text) = plt.subplots(
            2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 2]}
        )
    else:
        fig, ax_plot = plt.subplots(1, 1, figsize=(15, 8))

    # Ensure datetime index and select the correct value column
    if not isinstance(original_df.index, pd.DatetimeIndex):
        original_df = original_df.copy()
        original_df.index = pd.to_datetime(original_df.index)

    # Expect var3 column - required for plotting
    if VAR3 not in original_df.columns:
        raise ValueError(
            f"Expected 'var3' column in dataframe for series {series_id}, but found columns: {list(original_df.columns)}"
        )

    value_col = VAR3

    value_series = original_df[value_col].astype(float)

    # Plot the original time series in black (rasterized for large datasets)
    ax_plot.plot(
        original_df.index,
        value_series,
        color="black",
        linewidth=1.5,
        alpha=0.8,
        label="Time Series",
        rasterized=True,
    )

    # Overlay star markers at active trigger begin flags (blue for plus, red for minus)
    plus_color = PlotBase.style_for(True, True).color
    minus_color = PlotBase.style_for(False, True).color
    if ACTIVE_TRIGGER_PLUS_BEGIN in original_df.columns:
        plus_idx = original_df.index[
            original_df[ACTIVE_TRIGGER_PLUS_BEGIN].fillna(0).astype(int) == 1
        ]
        if len(plus_idx) > 0:
            ax_plot.scatter(
                plus_idx,
                value_series.loc[plus_idx],
                marker="*",
                s=60,
                color=plus_color,
                zorder=4,
                label=None,
            )
    if ACTIVE_TRIGGER_MINUS_BEGIN in original_df.columns:
        minus_idx = original_df.index[
            original_df[ACTIVE_TRIGGER_MINUS_BEGIN].fillna(0).astype(int) == 1
        ]
        if len(minus_idx) > 0:
            ax_plot.scatter(
                minus_idx,
                value_series.loc[minus_idx],
                marker="*",
                s=60,
                color=minus_color,
                zorder=4,
                label=None,
            )

    # Sort triggers by Begin Date for consistent processing
    triggers_df_sorted = triggers_df.sort_values(TRIGGER_BEGIN_DATES).reset_index(
        drop=True
    )

    # Prepare legend handles for all possible trigger types
    legend_handles = {}
    added_labels = set()

    # Use trigger_numbers if provided, else use DataFrame index + 1
    if trigger_numbers is None:
        trigger_numbers = list(range(1, len(triggers_df_sorted) + 1))

    # Plot trigger segments and annotate (array-based iteration)
    # Note: dates should already be datetime from normalize_dataframe, but we ensure it here for safety
    begin_series = pd.to_datetime(triggers_df_sorted[TRIGGER_BEGIN_DATES])
    end_series = pd.to_datetime(triggers_df_sorted[TRIGGER_END_DATES])
    begin_dt = begin_series.to_numpy()
    end_dt = end_series.to_numpy()
    is_plus_arr = triggers_df_sorted[IS_PLUS_TRIGGER].to_numpy()
    is_true_arr = triggers_df_sorted[IS_TRUE].to_numpy()
    variation_arr = (
        triggers_df_sorted["variation"].to_numpy()
        if "variation" in triggers_df_sorted.columns
        else np.zeros(len(triggers_df_sorted))
    )

    # Pre-compute segment boundaries with np.searchsorted (avoids per-trigger boolean mask)
    idx_values = original_df.index.values
    seg_starts = np.searchsorted(idx_values, begin_dt, side="left")
    seg_ends = np.searchsorted(idx_values, end_dt, side="right")

    # Vectorized processing of triggers
    for idx, (is_plus, is_true, variation) in enumerate(
        zip(is_plus_arr, is_true_arr, variation_arr)
    ):
        sty = PlotBase.style_for(bool(is_plus), bool(is_true))
        color, linestyle, label = sty.color, sty.linestyle, sty.label

        # Get the data for this time segment using pre-computed boundaries
        segment_data = original_df.iloc[seg_starts[idx] : seg_ends[idx]]
        segment_values = (
            segment_data[value_col].astype(float)
            if not segment_data.empty
            else segment_data.iloc[:, 0]
        )

        # Trigger number and variation for annotation
        trigger_number = trigger_numbers[idx]
        variation_str = f"({trigger_number}) {float(variation):.0f}%"

        # Plot the segment (rasterized for performance with many triggers)
        handle = None
        if label not in added_labels:
            (handle,) = ax_plot.plot(
                segment_data.index,
                segment_values,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                alpha=0.9,
                label=label,
                rasterized=True,
            )
            legend_handles[label] = handle
            added_labels.add(label)
        else:
            ax_plot.plot(
                segment_data.index,
                segment_values,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                alpha=0.9,
                rasterized=True,
            )

        # Annotate the segment
        if len(segment_data) > 0:
            x_pos = segment_data.index[len(segment_data) // 2]
            seg_min = float(segment_values.min()) if len(segment_values) else 0.0
            seg_max = float(segment_values.max()) if len(segment_values) else 0.0
            y_pos = seg_min if not is_plus else seg_max
            y_offset = (float(value_series.max()) - float(value_series.min())) * 0.03
            if is_plus:
                y_pos = y_pos + y_offset
                va = "bottom"
            else:
                y_pos = y_pos - y_offset
                va = "top"
            ax_plot.annotate(
                variation_str,
                xy=(x_pos, y_pos),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=10,
                color=color,
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                ),
            )

    # Add dummy handles for missing legend labels
    for lbl in PlotBase.legend_sequence:
        if lbl not in legend_handles:
            sty = PlotBase.style_for(lbl.startswith("Plus"), lbl.endswith("True"))
            (dummy,) = ax_plot.plot(
                [],
                [],
                color=sty.color,
                linestyle=sty.linestyle,
                linewidth=2.5,
                alpha=0.9,
                label=lbl,
            )
            legend_handles[lbl] = dummy

    # Use centralized legend handling
    PlotBase.add_legend_once(ax_plot, legend_handles)
    ax_plot.set_title(f"{strategy_name} - Series {series_id}")
    ax_plot.set_xlabel("Date")
    ax_plot.set_ylabel("Values")
    ax_plot.grid(True, alpha=0.3)

    # Add textual summary table below if requested
    if include_summary:
        summary_text = generate_summary_table(
            triggers_df, metric_config, display_order_signals or []
        )
        ax_text.axis("off")
        if summary_text:
            ax_text.text(
                0.01, 0.99, summary_text, va="top", ha="left", family="monospace"
            )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
