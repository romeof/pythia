"""
Plotting module for grouping performance analysis.

This module handles all visualization for grouping performance,
focusing purely on matplotlib operations and figure styling.

Structure:
1. Main starting point (plot_gradient, which receives pre-computed data)
2. Helpers in invocation sequence:
   create_gradient_figure, _plot_signal_variation, draw_signal_line,
   _emit_signal_plot_from_pct, _marker_x_step, _draw_terminal_marker, plot_top_panel, _compute_padded_ylim,
   _set_top_axis_scale, _update_ylim_for_all_series, format_and_finalize_bottom_axis
3. Summary table (append_summary_table: figure text placement; metric columns built via
   compute_gradient_columns when precomputed DataFrames are not supplied)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.pyplot import AutoLocator, MultipleLocator

from ...common.constants import (
    GRADIENT_LONG_WINDOW_YEARS,
    GROUP_ID,
    IS_PLUS_TRIGGER,
    IS_SIGNAL,
    IS_TRUE,
    SERIES_ID,
    TRIGGER_END_VALUES,
    TRIGGER_START_VALUES,
)
from ..gradient import compute_variation_pct
from ..gradient_.metrics import compute_gradient_columns
from ..perseries_.metrics import get_metric_config
from ..prints.summary import generate_summary_table
from .common import PlotBase

# Terminal attenuation marker styling
_MARKER_SIZE = 1.75
_MARKER_X_STEP_SCALE = 1.8       # spacing multiplier: marker-widths between overlapping markers
_MARKER_X_STEP_FALLBACK_DAYS = 7.0  # fallback spacing when axes geometry is unavailable

# Date annotation offset below x-axis (points, negative = downward)
_DATE_ANNOTATION_OFFSET_PTS = -22


# =============================================================================
# MAIN STARTING POINT
# =============================================================================


def plot_gradient(
    signals_data: Dict[int, Dict[str, Any]],
    filtered_signal_sequence: pd.DataFrame,
    gradient_series: Optional[pd.Series],
    global_idx: pd.DatetimeIndex,
    *,
    reference_variation_map: Optional[Dict[str, pd.Series]] = None,
    strategy_name: Optional[str] = None,
    emit_signal_plots: bool = False,
    signal_plots_whitelist: Optional[List[str]] = None,
    max_signal_plots: int = 100,
    plots_dir: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    display_order_signals: Optional[List[str]] = None,
    display_order_gradient: Optional[List[str]] = None,
    attenuation_adjusted_gradient: Optional[pd.Series] = None,
    terminal_attenuation_amount: float = 0.0,
    reference_terminal_attenuations: Optional[Dict[str, float]] = None,
    mpv: float = 0.0,
    gradient_metrics_table: Optional[pd.DataFrame] = None,
    signals_breakdown_table: Optional[pd.DataFrame] = None,
    close_figure: bool = True,
) -> None:
    """
    Create ONE figure with 2 aligned panels sharing the x-axis:
      - Top: Actual variation (%) according to the used variation_method
      - Bottom: all signal segments (%), styled by Plus/Minus and True/False with annotations.
      - Text table with summary statistics of the signal series.

    When attenuation is active, the top panel shows both pure gradient (solid) and
    attenuation-adjusted gradient (dashed black).

    This function is a pure plotting function. All computation (signal variation,
    weight validation, gradient aggregation) must be done before calling this.

    Args:
        signals_data: Pre-computed dict mapping group_id to signal data (pct, sign, etc.)
        filtered_signal_sequence: DataFrame of signal rows after weight filtering, sorted by begin date
        gradient_series: Pre-computed cumulative variation series for the top panel
        global_idx: Global curve index (the observed dates) for x-axis alignment
        reference_variation_map: Pre-computed dict of reference cumulative variation series
        strategy_name: Name displayed as the gradient legend label
        emit_signal_plots: Whether to emit per-signal standalone plots
        signal_plots_whitelist: Whitelist of series_ids to emit standalone plots for
        max_signal_plots: Maximum number of standalone signal plots
        plots_dir: Directory for standalone signal plots
        save_path: Path to save the gradient figure
        show: Whether to show the figure interactively
        display_order_signals: Ordered composite signal metric keys for summary table
        display_order_gradient: Ordered gradient metric keys for summary table
        attenuation_adjusted_gradient: Optional attenuation-adjusted cumulative variation series
            (every calendar year's attenuation applied, the last year's at its last item)
        terminal_attenuation_amount: Strategy last-year attenuation amount, already in the
            attenuated line's last point; when positive, a filled black circle
            marks that point
        reference_terminal_attenuations: Per-reference attenuation amounts; each rendered
            as a filled circle in the reference's color
        gradient_metrics_table: Precomputed gradient metrics DataFrame (index=metric_key,
            columns=column headers, dtype float32). When supplied, rendered directly
            without recomputing metric columns (eliminates double computation).
        signals_breakdown_table: Precomputed signals breakdown DataFrame (index=row_key,
            columns=direction_subdivision, dtype float32). Same pass-through semantics.
        close_figure: Whether to close the figure after saving. Default True.
            Pass False in tests to keep the figure alive for artist inspection.
    """
    if signals_data is None or not signals_data:
        return
    if filtered_signal_sequence is None or filtered_signal_sequence.empty:
        return

    long_window = global_idx[-1] > global_idx[0] + pd.DateOffset(years=GRADIENT_LONG_WINDOW_YEARS)
    fig, ax_top, ax_bottom = create_gradient_figure()
    # sharex=True ensures both panels adopt this xlim
    ax_bottom.set_xlim(global_idx[0], global_idx[-1])

    # Ensure legend items appear only once (bottom panel) and collect handles
    legend_drawn = {
        PlotBase.LABEL_PLUS_TRUE: False,
        PlotBase.LABEL_PLUS_FALSE: False,
        PlotBase.LABEL_MINUS_TRUE: False,
        PlotBase.LABEL_MINUS_FALSE: False,
    }
    legend_handles: Dict[str, Any] = {}

    # Plot filtered signal segments
    plotted = 0
    plus_label_rank = 0
    minus_label_rank = 0
    whitelist_set = set(map(str, (signal_plots_whitelist or [])))

    for i in range(len(filtered_signal_sequence)):
        row = filtered_signal_sequence.iloc[i]
        gid = int(row[GROUP_ID])
        gdata = signals_data.get(gid)
        if gdata is None:
            continue
        pct = gdata.get("pct")
        if pct is None:
            continue

        _plot_signal_variation(
            idx=i,
            pct=pct,
            start_val=gdata["start_val"],
            end_val=gdata["end_val"],
            series_id=gdata[SERIES_ID],
            is_plus=gdata[IS_PLUS_TRIGGER],
            is_true=gdata[IS_TRUE],
            label_rank=(
                plus_label_rank if gdata[IS_PLUS_TRIGGER] else minus_label_rank
            ),
            ax_bottom=ax_bottom,
            legend_drawn=legend_drawn,
            legend_handles=legend_handles,
        )
        if gdata[IS_PLUS_TRIGGER]:
            plus_label_rank += 1
        else:
            minus_label_rank += 1

        # Optionally emit a dedicated per-signal plot
        series_id = str(gdata[SERIES_ID])
        if emit_signal_plots and PlotBase.should_emit_signal_plot(
            series_id=series_id,
            emit_signal_plots=emit_signal_plots,
            whitelist=whitelist_set,
            plotted_so_far=plotted,
            max_signal_plots=max_signal_plots,
        ):
            plotted += _emit_signal_plot_from_pct(
                pct=pct,
                gid=gid,
                idx=i,
                series_id=series_id,
                is_plus=gdata[IS_PLUS_TRIGGER],
                is_true=gdata[IS_TRUE],
                start_val=gdata[TRIGGER_START_VALUES],
                end_val=gdata[TRIGGER_END_VALUES],
                plots_dir=plots_dir,
                gradient_save_path=save_path,
            )

    # Plot the top panel if we have data
    if gradient_series is not None:
        label_top = str(strategy_name) if strategy_name else "gradient"
        plot_top_panel(
            ax_top, gradient_series, label_top,
            long_window=long_window,
            attenuation_adjusted_series=attenuation_adjusted_gradient,
        )
    else:
        # Set labels/limits to produce a valid (empty) gradient figure
        ax_top.set_ylabel("gradient")
        ax_top.set_ylim(-5, 20)
        _set_top_axis_scale(ax_top, long_window)
        ax_top.set_xlim(global_idx[0], global_idx[-1])
        ax_top.grid(True, linestyle="--", alpha=0.3)

    # Overlay one curve per reference on the top panel for comparison.
    # Lines are clean (no periodic markers); the post-attenuation terminal value is
    # rendered as a single filled circle in the matching color.
    reference_series_map = reference_variation_map or {}
    ref_colors: Dict[str, str] = {}
    if reference_series_map:
        colors = plt.rcParams.get("axes.prop_cycle", None)
        color_list = colors.by_key().get("color", []) if colors else []
        if not color_list:
            color_list = ["gray"]
        for idx, (ref_id, ref_series) in enumerate(
            sorted(reference_series_map.items(), key=lambda kv: kv[0])
        ):
            if ref_series is None or len(ref_series) == 0:
                continue
            color = color_list[idx % len(color_list)]
            ref_colors[ref_id] = color
            ax_top.plot(
                ref_series.index,
                ref_series.values,
                color=color,
                linestyle="--",
                linewidth=1.2,
                label=f"reference {ref_id}",
                rasterized=True,
            )

    # Post-attenuation terminal markers — same-sized filled circles, offset
    # horizontally so overlapping markers stay distinct. Strategy is at
    # the exact last date; each reference shifts one marker-step further
    # right in display space.
    x_step = _marker_x_step(ax_top, _MARKER_SIZE)

    if attenuation_adjusted_gradient is not None and terminal_attenuation_amount > 0.0:
        _draw_terminal_marker(
            ax_top, attenuation_adjusted_gradient.index[-1], float(attenuation_adjusted_gradient.iloc[-1]),
            color="black", markersize=_MARKER_SIZE, zorder=5,
        )

    for i, (ref_id, attenuation_amount) in enumerate(
        sorted((reference_terminal_attenuations or {}).items())
    ):
        ref_series = reference_series_map.get(ref_id)
        if ref_series is not None and attenuation_amount > 0.0:
            _draw_terminal_marker(
                ax_top, ref_series.index[-1], float(ref_series.iloc[-1]) - attenuation_amount,
                color=ref_colors.get(ref_id, "gray"), markersize=_MARKER_SIZE, zorder=5,
                x_offset_days=(i + 1) * x_step,
            )

    if reference_series_map:
        handles, labels = ax_top.get_legend_handles_labels()
        if handles and labels:
            ax_top.legend(loc="upper left")

    # Recalculate y-axis limits to fit all series (gradient + references + attenuation) with 10% padding
    _update_ylim_for_all_series(
        ax_top, gradient_series, reference_series_map, attenuation_adjusted_gradient
    )

    # Bottom axis formatting and legend
    format_and_finalize_bottom_axis(
        ax_bottom=ax_bottom,
        signals_data=signals_data,
        legend_handles=legend_handles,
        global_idx=global_idx,
    )

    # Append summary table text computed from is_signal rows
    append_summary_table(
        fig=fig,
        ax_bottom=ax_bottom,
        signals_df=filtered_signal_sequence,
        display_order_signals=display_order_signals,
        display_order_gradient=display_order_gradient,
        gradient_series=gradient_series,
        reference_variation_map=reference_variation_map,
        attenuation_adjusted_gradient=attenuation_adjusted_gradient,
        reference_terminal_attenuations=reference_terminal_attenuations,
        mpv=mpv,
        gradient_metrics_df=gradient_metrics_table,
        signals_breakdown_df=signals_breakdown_table,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    if close_figure:
        plt.close(fig)


# =============================================================================
# HELPERS (in invocation sequence)
# =============================================================================


def create_gradient_figure():
    """Create gradient figure with two vertically stacked axes sharing x-axis."""
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2.2]},
    )
    return fig, ax_top, ax_bottom


def _plot_signal_variation(
    series_id: str,
    idx: int,
    pct: pd.Series,
    start_val: float,
    end_val: float,
    is_plus: bool,
    is_true: bool,
    label_rank: int,
    ax_bottom: Any,
    legend_drawn: Dict[str, bool],
    legend_handles: Dict[str, Any],
) -> None:
    """
    Plot a single signal's percent-change segment on ax_bottom with annotations.

    Args:
        series_id: Series identifier
        idx: Index of the signal (for annotation numbering)
        pct: Percent-change series
        start_val: Starting value for variation calculation
        end_val: Ending value for variation calculation
        is_plus: Whether this is a plus trigger
        is_true: Whether this is a true trigger
        label_rank: Zero-based label rank inside plus/minus group for staggering
        ax_bottom: Bottom axis to plot on
        legend_drawn: Dictionary tracking which legend labels have been drawn
        legend_handles: Dictionary of legend handles
    """
    pct_clip = pct.clip(lower=-50, upper=50)
    sty = PlotBase.style_for(is_plus, is_true)
    plot_label = sty.label if not legend_drawn.get(sty.label, False) else None

    draw_signal_line(
        ax_bottom,
        pct.index,
        pct_clip.values,
        color=sty.color,
        linestyle=sty.linestyle,
        linewidth=1.4,
        plot_label=plot_label,
        label_key=sty.label,
        legend_drawn=legend_drawn,
        legend_handles=legend_handles,
    )

    variation_pct = compute_variation_pct(start_val, end_val)
    start_x = pct.index[0]
    start_y = pct_clip.values[0]
    text = f"({idx + 1}) [{series_id}] {variation_pct:.0f}%"
    # Place labels at segment start and stagger as up/down/up/down.
    x_offset = 6
    if is_plus:
        y_anchor = max(start_y, 0.0)
        # 3-level cycle for plus labels: bottom -> middle -> top (repeat).
        plus_offsets = (8, 20, 32)
        y_offset = plus_offsets[label_rank % 3]
        va = "bottom"
    else:
        y_anchor = min(start_y, 0.0)
        # 3-level cycle for minus labels: bottom -> middle -> top (repeat).
        # Negative offsets place labels below the zero baseline.
        minus_offsets = (-32, -20, -8)
        y_offset = minus_offsets[label_rank % 3]
        va = "top"
    ax_bottom.annotate(
        text,
        xy=(start_x, y_anchor),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        color=sty.color,
        fontsize=8,
        ha="left",
        va=va,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.8},
    )


def draw_signal_line(
    ax_bottom: Any,
    x_index: Any,
    y_values: Any,
    *,
    color: str,
    linestyle: str,
    linewidth: float = 1.4,
    plot_label: Optional[str],
    label_key: str,
    legend_drawn: Dict[str, bool],
    legend_handles: Dict[str, Any],
) -> None:
    """Draw a signal line on the bottom axis and update legend handles if needed."""
    n_points = len(x_index) if x_index is not None else 0
    is_short_segment = n_points <= 3
    (line,) = ax_bottom.plot(
        x_index,
        y_values,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth if not is_short_segment else (linewidth + 0.2),
        alpha=1.0 if not is_short_segment else 0.9,
        label=plot_label,
        rasterized=True,
    )
    if plot_label is not None:
        legend_handles[label_key] = line
        legend_drawn[label_key] = True


def _emit_signal_plot_from_pct(
    *,
    pct: pd.Series,
    gid: int,
    idx: int,
    series_id: str,
    is_plus: bool,
    is_true: bool,
    start_val: float,
    end_val: float,
    plots_dir: Optional[str],
    gradient_save_path: Optional[str],
) -> int:
    """
    Create and save a standalone signal plot using an already computed pct series.
    Returns 1 if a plot is saved, 0 for an empty pct series; the figure is released in every case.
    """
    if pct is None or len(pct) == 0:
        return 0
    norm = 1.0 + (pct / 100.0)
    sty = PlotBase.style_for(is_plus, is_true)
    color, style, label = sty.color, sty.linestyle, sty.label
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        ax.plot(
            pct.index,
            norm.values,
            linestyle=style,
            color=color,
            linewidth=1.5,
            label=label,
            rasterized=True,
        )
        ax.set_title(f"Signal - Group {gid} (Series {series_id})")
        ax.set_xlabel("Date")
        ax.set_ylabel("percentage variation")
        span_days = int((pct.index.max() - pct.index.min()).days) if len(pct.index) > 1 else 0
        if span_days <= 45:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        elif span_days <= 180:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=30, ha="right")
        ax.legend(loc="upper left")
        variation_pct = compute_variation_pct(start_val, end_val)
        ax.text(
            pct.index[-1],
            norm.values[-1],
            f"({idx + 1}) [{series_id}] {variation_pct:.0f}%",
            color=color,
            fontsize=8,
            ha="left",
            va="bottom",
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        out_dir = (
            Path(plots_dir)
            if plots_dir
            else (Path(gradient_save_path).parent if gradient_save_path else Path.cwd())
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_file = out_dir / f"group_{gid}_signal_{series_id}.png"
        plt.savefig(plot_file, bbox_inches="tight", dpi=120)
    finally:
        plt.close(fig)
    return 1


def _marker_x_step(
    ax: Any,
    markersize_pts: float,
    scale: float = _MARKER_X_STEP_SCALE,
) -> float:
    """Return an x-offset in matplotlib date-number units (days since epoch).

    Converts scale * markersize_pts of display space to data units so that
    successive markers are spaced by roughly one marker-width, independently
    of the figure size and date range.
    """
    fig_width_pts = ax.figure.get_figwidth() * 72.0
    ax_width_pts = fig_width_pts * ax.get_position().width
    x_min, x_max = ax.get_xlim()  # matplotlib date numbers (days since epoch)
    if ax_width_pts <= 0 or x_max <= x_min:
        return _MARKER_X_STEP_FALLBACK_DAYS
    pts_per_data = ax_width_pts / (x_max - x_min)
    return scale * markersize_pts / pts_per_data


def _draw_terminal_marker(
    ax: Any,
    last_date: pd.Timestamp,
    y_value: float,
    *,
    color: str,
    markersize: float,
    zorder: int,
    x_offset_days: float = 0.0,
    **kwargs: Any,
) -> None:
    """Draw a single filled-circle post-attenuation marker at (last_date, y_value).

    x_offset_days shifts the marker right by that many calendar days so
    markers drawn on the same date don't overlap. Extra kwargs
    (e.g., markeredgecolor, markeredgewidth) are forwarded to ax.plot.
    """
    x_base = mdates.date2num(last_date)
    ax.plot(
        [mdates.num2date(x_base + x_offset_days)],
        [y_value],
        color=color,
        marker="o",
        markersize=markersize,
        linestyle="None",
        zorder=zorder,
        clip_on=False,
        **kwargs,
    )


def plot_top_panel(
    ax_top: Any,
    series_to_plot: pd.Series,
    label_top: Optional[str],
    long_window: bool,
    attenuation_adjusted_series: Optional[pd.Series] = None,
) -> None:
    """Plot the top panel time series and apply axis styling.

    When attenuation_adjusted_series is provided, a second dashed black line is drawn
    representing the attenuation-adjusted gradient. long_window selects the axis scale
    (_set_top_axis_scale).
    """
    ax_top.plot(
        series_to_plot.index,
        series_to_plot.values,
        color="black",
        linewidth=1.6,
        label=label_top if label_top else None,
        rasterized=True,
    )
    if attenuation_adjusted_series is not None:
        ax_top.plot(
            attenuation_adjusted_series.index,
            attenuation_adjusted_series.values,
            color="black",
            linewidth=1.6,
            linestyle="--",
            label="after attenuation",
            rasterized=True,
        )
    ax_top.set_ylabel("gradient")
    ymin = float(series_to_plot.min()) if len(series_to_plot) else 0.0
    ymax = float(series_to_plot.max()) if len(series_to_plot) else 0.0
    if attenuation_adjusted_series is not None and len(attenuation_adjusted_series) > 0:
        ymin = min(ymin, float(attenuation_adjusted_series.min()))
        ymax = max(ymax, float(attenuation_adjusted_series.max()))
    lower, upper = _compute_padded_ylim(ymin, ymax)
    ax_top.set_ylim(lower, upper)
    # A bounded tick count for any range; mirror labels to the right
    _set_top_axis_scale(ax_top, long_window)
    ax_top.tick_params(
        axis="y",
        which="both",
        left=True,
        right=True,
        labelleft=True,
        labelright=True,
    )
    ax_top.grid(True, which="major", linestyle="--", alpha=0.3)
    ax_top.grid(True, which="minor", linestyle=":", alpha=0.2)
    if label_top:
        ax_top.legend(loc="upper left")


def _compute_padded_ylim(ymin: float, ymax: float) -> Tuple[float, float]:
    """
    Compute y-limits with 25% padding beyond min/max; clamp lower to 0 if min is non-negative.
    Ensure non-degenerate limits.
    """
    lower = ymin - 0.25 * abs(ymin) if ymin < 0 else 0.0
    upper = ymax + 0.25 * abs(ymax) if ymax > 0 else 0.0
    if lower == upper:
        upper = lower + 1e-6
    return lower, upper


def _set_top_axis_scale(ax_top: Any, long_window: bool) -> None:
    """Bound the top panel's ticks, keeping its limits: a symmetric-log axis when the curve index spans more than
    GRADIENT_LONG_WINDOW_YEARS calendar years (it accepts the zero every curve begins at and the values below it),
    else the linear axis with automatic major ticks and the linear scale's automatic minor ticks (minorticks_on)."""
    if long_window:
        ax_top.set_yscale("symlog")
        return
    ax_top.yaxis.set_major_locator(AutoLocator())
    ax_top.yaxis.minorticks_on()


def _update_ylim_for_all_series(
    ax_top: Any,
    gradient_series: Optional[pd.Series],
    reference_series_map: Dict[str, pd.Series],
    attenuation_adjusted_gradient: Optional[pd.Series] = None,
) -> None:
    """
    Update y-axis limits to fit all series (gradient + references + attenuation) with 10% padding on top.

    Args:
        ax_top: Top axis to update
        gradient_series: Main gradient series
        reference_series_map: Dictionary of reference series
        attenuation_adjusted_gradient: Optional attenuation-adjusted gradient series
    """
    all_values = []

    # Collect gradient series values
    if gradient_series is not None and len(gradient_series) > 0:
        all_values.extend(gradient_series.values)

    # Collect attenuation-adjusted gradient values
    if attenuation_adjusted_gradient is not None and len(attenuation_adjusted_gradient) > 0:
        all_values.extend(attenuation_adjusted_gradient.values)

    # Collect reference series values
    for ref_series in reference_series_map.values():
        if ref_series is not None and len(ref_series) > 0:
            all_values.extend(ref_series.values)

    if all_values:
        ymin = float(np.min(all_values))
        ymax = float(np.max(all_values))

        # Set lower limit: 0 if ymin >= 0, otherwise add 10% padding below
        lower = 0.0 if ymin >= 0 else ymin * 1.1

        # Set upper limit: max * 1.1 (10% padding above)
        upper = ymax * 1.1 if ymax > 0 else 0.0

        # Ensure non-degenerate limits
        if lower == upper:
            upper = lower + 1e-6

        ax_top.set_ylim(lower, upper)


def format_and_finalize_bottom_axis(
    ax_bottom: Any,
    signals_data: Dict[int, Dict[str, Any]],
    legend_handles: Dict[str, Any],
    global_idx: Optional[pd.DatetimeIndex] = None,
) -> None:
    """Format the bottom axis with labels, limits, grid, and legend.

    Adds monthly minor x-ticks between the existing major (auto) ticks and,
    when global_idx is provided, annotates the actual begin and end dates
    just below the x-axis.
    """
    ax_bottom.set_xlabel("Date")
    ax_bottom.set_ylabel("signal variation (%)")
    ax_bottom.set_ylim(-50, 50)
    ax_bottom.axhline(0, color="gray", linewidth=1.0, alpha=0.6, linestyle="--")

    # Major ticks every 20%, minor ticks every 5%; mirror labels to the right
    ax_bottom.yaxis.set_major_locator(MultipleLocator(20))
    ax_bottom.yaxis.set_minor_locator(MultipleLocator(5))
    ax_bottom.tick_params(
        axis="y",
        which="both",
        left=True,
        right=True,
        labelleft=True,
        labelright=True,
    )

    # Monthly minor ticks add granularity without changing major label text
    ax_bottom.xaxis.set_minor_locator(mdates.MonthLocator())
    ax_bottom.grid(True, which="major", linestyle="--", alpha=0.3)
    ax_bottom.grid(True, which="minor", linestyle=":", alpha=0.2)

    # Annotate actual begin and end dates just below the x-axis
    if global_idx is not None and len(global_idx) > 0:
        for x_pos, ha in ((global_idx[0], "left"), (global_idx[-1], "right")):
            ax_bottom.annotate(
                pd.Timestamp(x_pos).strftime("%Y-%m-%d"),
                xy=(x_pos, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, _DATE_ANNOTATION_OFFSET_PTS),
                textcoords="offset points",
                ha=ha,
                va="top",
                fontsize=8,
                color="gray",
            )

    for lbl in PlotBase.legend_sequence:
        if lbl not in legend_handles:
            sty = PlotBase.style_for(lbl.startswith("Plus"), lbl.endswith("True"))
            legend_handles[lbl] = Line2D(
                [], [], color=sty.color, linestyle=sty.linestyle, linewidth=1.4, label=lbl
            )
    PlotBase.add_legend_once(ax_bottom, legend_handles)


# =============================================================================
# SUMMARY TABLE
# =============================================================================


def append_summary_table(
    fig: Any,
    ax_bottom: Any,
    signals_df: pd.DataFrame,
    display_order_signals: Optional[List[str]] = None,
    display_order_gradient: Optional[List[str]] = None,
    gradient_series: Optional[pd.Series] = None,
    reference_variation_map: Optional[Dict[str, pd.Series]] = None,
    attenuation_adjusted_gradient: Optional[pd.Series] = None,
    reference_terminal_attenuations: Optional[Dict[str, float]] = None,
    mpv: float = 0.0,
    gradient_metrics_df: Optional[pd.DataFrame] = None,
    signals_breakdown_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Append summary statistics table to the bottom of the figure.

    When precomputed DataFrames are supplied (gradient_metrics_df,
    signals_breakdown_df), they are forwarded directly to generate_summary_table
    without recomputing metric columns — the fast path used by the task. When
    absent, metric columns are built via compute_gradient_columns (fallback).

    Args:
        fig: Matplotlib figure
        ax_bottom: Bottom axis to anchor the table
        signals_df: DataFrame with signal data (must have is_signal column)
        display_order_signals: Ordered composite signal metric keys for the table
        display_order_gradient: Ordered gradient metric keys for the table
        gradient_series: Pure (pre-attenuation) cumulative gradient series (decimal)
        reference_variation_map: Pre-attenuation reference cumulative variation series (decimal)
        attenuation_adjusted_gradient: Gradient with every calendar year's attenuation applied,
            the last year's at its last item (decimal). None when attenuation is inactive.
        reference_terminal_attenuations: Terminal attenuation amount per reference (decimal),
            applied once on its total accumulated increase.
        mpv: Minimal positive variation rate (decimal). Default 0
        gradient_metrics_df: Precomputed gradient metrics DataFrame (fast path).
        signals_breakdown_df: Precomputed signals breakdown DataFrame (fast path).
    """
    with pd.option_context(
        "display.max_columns", None,
        "display.max_rows", None,
        "display.expand_frame_repr", False,  # prevent multi-line blocks
    ):
        signal_only = signals_df[signals_df[IS_SIGNAL]]
        clean = signal_only.dropna(subset=[IS_TRUE, IS_PLUS_TRIGGER], how="any")
        metric_config = get_metric_config()
        gradient_keys = display_order_gradient or []

        # Build gradient columns only when DataFrame not precomputed (fallback path).
        # compute_gradient_columns handles pct scaling and terminal attenuation adjustments.
        preattenuation_columns = None
        postattenuation_columns = None
        if gradient_metrics_df is None:
            preattenuation_columns, postattenuation_columns = compute_gradient_columns(
                gradient_series,
                attenuation_adjusted_gradient,
                reference_variation_map,
                reference_terminal_attenuations,
                selection=gradient_keys,
                mpv=mpv,
            )

        summary_text = generate_summary_table(
            clean,
            metric_config,
            display_order_signals or [],
            preattenuation_columns=preattenuation_columns,
            postattenuation_columns=postattenuation_columns,
            display_order_gradient=gradient_keys,
            gradient_metrics_df=gradient_metrics_df,
            signals_breakdown_df=signals_breakdown_df,
        )

        # If summary_text is empty, do nothing
        if isinstance(summary_text, str) and summary_text.strip():
            # Scale bottom margin to number of text lines (fontsize=8 ~0.014 fig units per line)
            n_lines = summary_text.count("\n") + 1
            bottom_margin = 0.037 + n_lines * 0.014
            PlotBase.place_monospace_text(fig, ax_bottom, summary_text, y=0.037)
            plt.subplots_adjust(bottom=bottom_margin, hspace=0.15)
