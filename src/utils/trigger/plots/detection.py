"""
Detection debug plotting utilities for trigger detection algorithms.

Provides visualization for trigger detection debugging.

Structure:
1. plot_window_with_triggers (starting point)
2. _render_subplot (draws data line, markers, and date lines on one axis)
3. _draw_series_and_markers (draws data line + trigger scatter markers)
4. _draw_date_lines (draws vertical lines for begin/end dates)
5. _add_legend (builds full legend for subplot 1)
6. _apply_zoom (computes and applies zoom bounds for subplot 2)
7. _build_title (constructs plot title string)
8. _save_plot (formats and saves figure to disk)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from ...common.constants import VAR3, TRIGGER_MINUS_BEGIN, TRIGGER_PLUS_BEGIN, DIRECTION_PLUS

# Marker style lookup for trigger types
TRIGGER_MARKER_STYLES: Dict[str, Tuple[str, str, str]] = {
    TRIGGER_PLUS_BEGIN: ("blue", "v", "Plus Candidate"),
    TRIGGER_MINUS_BEGIN: ("red", "^", "Minus Candidate"),
}


def plot_window_with_triggers(
    series_id: str,
    window_df: pd.DataFrame,
    trigger_cols: List[str],
    iteration: int,
    current_date: pd.Timestamp,
    debug_dir: str,
    detection_date: pd.Timestamp = None,
    detection_end_date: pd.Timestamp = None,
    trigger_end_date: pd.Timestamp = None,
    duration: int = None,
    begin_value: float = None,
    end_value: float = None,
    variation_pct: float = None,
    is_active: bool = False,
    trigger_type: str = None,
    end_reason: str = None,
    detection_begin_idx: int = None,
    current_idx: int = None,
    detection_end_idx: int = None,
    trigger_end_idx: int = None,
    trigger_count: int = None,
    is_suppressed: bool = False,
    suppression_reason: str = None,
) -> None:
    """Plot the current window with detected triggers for debugging purposes.

    Creates a two-subplot figure:
    - Subplot 1: Full window with legend (all date lines, metrics, suppression or end reason)
    - Subplot 2: Zoomed view around the detection region

    Args:
        series_id: Series identifier
        window_df: DataFrame with 'var3' column and trigger columns
        trigger_cols: Trigger column names to plot as markers
        iteration: Current loop iteration (for filename)
        current_date: Trigger Begin Date (when algorithm marked/activated the trigger)
        debug_dir: Output directory for plot files
        detection_date: Detection Begin Date (when pattern was first detected)
        detection_end_date: Detection End Date (when detection ended)
        trigger_end_date: Trigger End Date (when trigger ended)
        duration: Trigger duration in periods
        begin_value: Value at trigger begin
        end_value: Value at trigger end
        variation_pct: Percentage variation
        is_active: Whether this is an active trigger, not yet ended
        trigger_type: 'trigger_plus_begin' or 'trigger_minus_begin'
        end_reason: Reason for trigger end
        detection_begin_idx: DataFrame row index when trigger pattern was first detected
        current_idx: Iteration index when trigger was marked/activated (trigger begin)
        detection_end_idx: DataFrame row index when trigger pattern ended
        trigger_end_idx: DataFrame row index when trigger ended
        trigger_count: Counter for this trigger type
        is_suppressed: Whether this is a suppressed trigger
        suppression_reason: Reason for suppression
    """
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)

    if VAR3 not in window_df.columns:
        raise ValueError(
            f"Iteration {iteration}: 'var3' column not found in window_df"
        )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    value_series = window_df[VAR3].astype(float)

    # === SUBPLOT 1: Full view with legend ===
    _render_subplot(
        ax1, window_df, value_series, trigger_cols,
        detection_date, current_date, detection_end_date, trigger_end_date, is_active,
    )
    _add_legend(
        ax1,
        detection_date,
        current_date,
        detection_end_date,
        trigger_end_date,
        detection_begin_idx,
        current_idx,
        detection_end_idx,
        trigger_end_idx,
        duration,
        begin_value,
        end_value,
        variation_pct,
        is_active,
        is_suppressed,
        suppression_reason,
        end_reason,
    )
    title = _build_title(series_id, trigger_type, is_suppressed, trigger_count)
    ax1.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel(VAR3, fontsize=10)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # === SUBPLOT 2: Zoomed view ===
    _render_subplot(
        ax2, window_df, value_series, trigger_cols,
        detection_date, current_date, detection_end_date, trigger_end_date, is_active,
    )
    _apply_zoom(
        ax2,
        window_df,
        value_series,
        detection_date,
        detection_end_date,
        trigger_end_date,
        current_date,
        duration,
    )
    ax2.set_title("Zoom on Detection", fontsize=11, fontweight="bold", loc="left")
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel(VAR3, fontsize=10)
    ax2.grid(True, alpha=0.3)

    # === Save ===
    _save_plot(fig, debug_path, current_idx, iteration, trigger_count, is_suppressed)


def _render_subplot(
    ax: plt.Axes,
    window_df: pd.DataFrame,
    value_series: pd.Series,
    trigger_cols: List[str],
    detection_date: Optional[pd.Timestamp],
    current_date: pd.Timestamp,
    detection_end_date: Optional[pd.Timestamp],
    trigger_end_date: Optional[pd.Timestamp],
    is_active: bool,
) -> None:
    """Draw data line, trigger markers, and date lines on a single axis."""
    _draw_series_and_markers(ax, window_df, value_series, trigger_cols)
    _draw_date_lines(ax, detection_date, current_date, detection_end_date, trigger_end_date, is_active)


def _draw_series_and_markers(
    ax: plt.Axes,
    window_df: pd.DataFrame,
    value_series: pd.Series,
    trigger_cols: List[str],
) -> None:
    """Draw the var3 line and trigger candidate markers on an axis."""
    ax.plot(
        window_df.index,
        value_series,
        color="black",
        linewidth=1.5,
        alpha=0.7,
        marker="o",
        markersize=3,
    )

    for col in trigger_cols:
        if col not in window_df.columns or col not in TRIGGER_MARKER_STYLES:
            continue
        color, marker, _ = TRIGGER_MARKER_STYLES[col]
        trigger_mask = window_df[col] == 1
        if trigger_mask.any():
            ax.scatter(
                window_df.index[trigger_mask],
                value_series[trigger_mask],
                color=color,
                marker=marker,
                s=100,
                zorder=5,
                edgecolors="black",
                linewidths=1.5,
            )


def _draw_date_lines(
    ax: plt.Axes,
    detection_date: Optional[pd.Timestamp],
    current_date: pd.Timestamp,
    detection_end_date: Optional[pd.Timestamp],
    trigger_end_date: Optional[pd.Timestamp],
    is_active: bool,
) -> None:
    """Draw vertical date lines: begin dates (dotted) and end dates (dashed).

    Color convention:
    - Purple/violet: Detection Begin Date (dotted) and Detection End Date (dashed)
    - Green: Trigger Begin Date (dotted) and Trigger End Date (dashed)
    """
    # Detection Begin Date: purple dotted
    if detection_date is not None:
        ax.axvline(
            x=detection_date,
            color="purple",
            linestyle=":",
            linewidth=2.5,
            alpha=0.8,
            zorder=5,
        )
    # Trigger Begin Date: green dotted
    ax.axvline(
        x=current_date, color="green", linestyle=":", linewidth=2.5, alpha=0.8, zorder=5
    )
    # Trigger End Date: green dashed (drawn first so Signal End overlays it)
    if trigger_end_date is not None and not is_active:
        ax.axvline(
            x=trigger_end_date,
            color="green",
            linestyle="--",
            linewidth=2.5,
            alpha=0.7,
            zorder=6,
        )
    # Detection End Date: purple dashed (drawn last, higher zorder for visibility when overlapping)
    if detection_end_date is not None and not is_active:
        ax.axvline(
            x=detection_end_date,
            color="purple",
            linestyle="--",
            linewidth=3.0,
            alpha=0.9,
            zorder=7,
        )


def _add_legend(
    ax: plt.Axes,
    detection_date: Optional[pd.Timestamp],
    current_date: pd.Timestamp,
    detection_end_date: Optional[pd.Timestamp],
    trigger_end_date: Optional[pd.Timestamp],
    detection_begin_idx: Optional[int],
    current_idx: Optional[int],
    detection_end_idx: Optional[int],
    trigger_end_idx: Optional[int],
    duration: Optional[int],
    begin_value: Optional[float],
    end_value: Optional[float],
    variation_pct: Optional[float],
    is_active: bool,
    is_suppressed: bool,
    suppression_reason: Optional[str],
    end_reason: Optional[str],
) -> None:
    """Build the full legend for subplot 1: date lines, metrics, suppression or end reason, data series."""
    # Date line items
    if detection_date is not None:
        detection_begin_label = f"Detection Begin Date: {detection_date.strftime('%Y-%m-%d')}"
        if detection_begin_idx is not None:
            detection_begin_label += f" (idx {detection_begin_idx})"
        ax.plot(
            [], [], color="purple", linestyle=":", linewidth=2, label=detection_begin_label
        )

    current_label = f"Trigger Begin Date: {current_date.strftime('%Y-%m-%d')}"
    if current_idx is not None:
        current_label += f" (idx {current_idx})"
    ax.plot([], [], color="green", linestyle=":", linewidth=2, label=current_label)

    if detection_end_date is not None and not is_active:
        detection_end_label = f"Detection End Date: {detection_end_date.strftime('%Y-%m-%d')}"
        if detection_end_idx is not None:
            detection_end_label += f" (idx {detection_end_idx})"
        ax.plot(
            [], [], color="purple", linestyle="--", linewidth=2, label=detection_end_label
        )

    if trigger_end_date is not None and not is_active:
        trigger_end_label = f"Trigger End Date: {trigger_end_date.strftime('%Y-%m-%d')}"
        if trigger_end_idx is not None:
            trigger_end_label += f" (idx {trigger_end_idx})"
        ax.plot(
            [], [], color="green", linestyle="--", linewidth=2, label=trigger_end_label
        )

    # Metrics
    if duration is not None:
        ax.plot([], [], " ", label=f"Duration: {duration} periods")
    if begin_value is not None:
        ax.plot([], [], " ", label=f"Begin Value: {begin_value:.2f}")
    if end_value is not None:
        val_label = (
            f"Current Value: {end_value:.2f}"
            if is_active
            else f"End Value: {end_value:.2f}"
        )
        ax.plot([], [], " ", label=val_label)
    if variation_pct is not None:
        ax.plot([], [], " ", label=f"Variation: {variation_pct:+.2f}%")

    # Suppression or end reason
    if is_suppressed and suppression_reason:
        ax.plot([], [], " ", label=f"Suppression Reason: {suppression_reason}")
    elif is_active:
        ax.plot([], [], " ", label="End Reason: none, active")
    elif end_reason is not None:
        ax.plot([], [], " ", label=f"End Reason: {end_reason}")

    # Data series markers
    ax.plot(
        [],
        [],
        color="black",
        linewidth=1.5,
        marker="o",
        markersize=3,
        label=VAR3,
    )
    ax.scatter(
        [],
        [],
        color="blue",
        marker="v",
        s=100,
        edgecolors="black",
        linewidths=1.5,
        label="Plus Candidate",
    )
    ax.scatter(
        [],
        [],
        color="red",
        marker="^",
        s=100,
        edgecolors="black",
        linewidths=1.5,
        label="Minus Candidate",
    )


def _apply_zoom(
    ax: plt.Axes,
    window_df: pd.DataFrame,
    value_series: pd.Series,
    detection_date: Optional[pd.Timestamp],
    detection_end_date: Optional[pd.Timestamp],
    trigger_end_date: Optional[pd.Timestamp],
    current_date: pd.Timestamp,
    duration: Optional[int],
) -> None:
    """Compute and apply zoom bounds for the zoomed subplot around the detection region."""
    if detection_date is None:
        return

    zoom_end = trigger_end_date if trigger_end_date is not None else current_date

    if hasattr(detection_date, "to_pydatetime") and hasattr(zoom_end, "to_pydatetime"):
        time_span = (zoom_end - detection_date).days
    else:
        time_span = duration if duration is not None else 50

    padding = max(int(time_span * 0.2), 10)

    if hasattr(detection_date, "to_pydatetime"):
        padding_delta = pd.Timedelta(days=padding)
        x_min = detection_date - padding_delta
        x_max = zoom_end + padding_delta
    else:
        x_min = detection_date - padding
        x_max = zoom_end + padding

    ax.set_xlim(x_min, x_max)

    zoomed_mask = (window_df.index >= x_min) & (window_df.index <= x_max)
    if zoomed_mask.any():
        zoomed_values = value_series[zoomed_mask]
        y_min_val, y_max_val = zoomed_values.min(), zoomed_values.max()
        y_range = y_max_val - y_min_val
        y_padding = (
            y_range * 0.05
            if y_range > 0
            else (abs(y_min_val) * 0.05 if y_min_val != 0 else 1)
        )
        ax.set_ylim(y_min_val - y_padding, y_max_val + y_padding)


def _build_title(
    series_id: str,
    trigger_type: Optional[str],
    is_suppressed: bool,
    trigger_count: Optional[int],
) -> str:
    """Construct the plot title from series_id, trigger type, suppression status, and trigger count."""
    if trigger_type is None:
        return ""
    label = "plus_trigger" if DIRECTION_PLUS in trigger_type.lower() else "minus_trigger"
    if is_suppressed:
        label = f"{label}_suppressed"
    if trigger_count is not None:
        title = f"{label} ({trigger_count})"
    else:
        title = label

    # Prepend series_id if provided
    if series_id:
        return f"series {series_id}: {title}"
    return title


def _save_plot(
    fig: plt.Figure,
    debug_path: Path,
    current_idx: Optional[int],
    iteration: int,
    trigger_count: Optional[int],
    is_suppressed: bool,
) -> None:
    """Format axes and save figure to disk."""
    for ax in fig.axes:
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    fig.tight_layout()

    file_idx = current_idx if current_idx is not None else iteration
    suffix = f"_s{trigger_count}" if is_suppressed else f"_e{trigger_count}"
    if trigger_count is None:
        suffix = "_s" if is_suppressed else "_e"

    plot_filename = debug_path / f"{file_idx}{suffix}_th.png"
    fig.savefig(plot_filename, dpi=100)
    plt.close(fig)
