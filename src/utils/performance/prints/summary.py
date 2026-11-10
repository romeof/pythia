"""
The summary table of the per-series and gradient outputs, shared by their prints and plots.

Structure:
1. Summary table, transposed layout (generate_summary_table, the main summary starting point;
   _render_transposed_signal_table, signals DataFrame to formatted rows; _build_column_header, the two-line
   header; _format_transposed_row, one row)
2. Gradient metrics table, n-column layout (_render_gradient_metrics_table, gradient metrics DataFrame to
   formatted rows; _format_cell, one right-aligned cell; _is_nan_or_none, the null check)
"""

import math
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...common.constants import (
    DIRECTION_SUBDIVISION_COLUMNS,
    DIRECTIONS,
    GRADIENT_METRICS_SECTION,
    SIGNAL_METRICS_SECTION,
    SIGNALS_COUNT_LABEL,
    SUBDIVISIONS,
)
from ..gradient_.metrics import (
    GradientColumn,
    build_gradient_metrics_dataframe,
    get_gradient_metric_config,
    shared_annualization,
)
from ..perseries_.metrics import build_signals_breakdown_dataframe, resolve_signal_selection


# Column widths for the transposed table
_LABEL_WIDTH = 28
_COL_WIDTH = 11
_SEPARATOR_WIDTH = 3


# =============================================================================
# SUMMARY TABLE (TRANSPOSED LAYOUT)
# =============================================================================


def generate_summary_table(
    rows_df: pd.DataFrame,
    metric_config: Dict,
    display_order_signals: List[str],
    preattenuation_columns: Optional[List[GradientColumn]] = None,
    postattenuation_columns: Optional[List[GradientColumn]] = None,
    display_order_gradient: Optional[List[str]] = None,
    gradient_metrics_df: Optional[pd.DataFrame] = None,
    signals_breakdown_df: Optional[pd.DataFrame] = None,
) -> str:
    """Build formatted summary table with signal and gradient metrics.

    Signal metrics are displayed in a transposed layout: metrics as rows,
    direction x subdivision as columns. Gradient metrics appear below in an
    n-column layout (strategy + one column per reference), with a dedicated
    'acr (after attenuation) (%)' row when attenuation is active.

    When precomputed DataFrames are supplied, they are rendered directly (replot
    path). When absent, DataFrames are built internally from the existing
    column inputs (task path) — same numerical result, no double computation.

    Args:
        rows_df: DataFrame with signal-level metrics (pre-filtered to IS_SIGNAL)
        metric_config: Perseries metric configuration from metrics.json
        display_order_signals: Ordered list of composite signal metric keys
        preattenuation_columns: Pre-attenuation gradient columns (used when gradient_metrics_df
            is None; ignored when gradient_metrics_df is supplied)
        postattenuation_columns: Post-attenuation gradient columns (idem)
        display_order_gradient: Ordered list of gradient metric keys
        gradient_metrics_df: Precomputed wide numeric gradient metrics DataFrame
            (index=metric_key, columns=column headers, dtype float32)
        signals_breakdown_df: Precomputed wide numeric signals breakdown DataFrame
            (index=row_key, columns=direction_subdivision, dtype float32)

    Returns:
        Formatted summary table string.
    """
    if rows_df.empty:
        return "No rows found across all series."

    selection = resolve_signal_selection(metric_config, display_order_signals)
    gradient_keys = display_order_gradient or []

    # Build signals breakdown from the rows when not precomputed
    if signals_breakdown_df is None:
        signals_breakdown_df = build_signals_breakdown_dataframe(
            rows_df, metric_config, display_order_signals
        )

    signal_table_width = _LABEL_WIDTH + len(DIRECTIONS) * (
        len(SUBDIVISIONS) * _COL_WIDTH + _SEPARATOR_WIDTH
    )
    total_width = signal_table_width

    gradient_table = ""
    # Build gradient metrics DataFrame from columns when not precomputed
    if gradient_metrics_df is None and preattenuation_columns:
        gradient_metrics_df = build_gradient_metrics_dataframe(
            preattenuation_columns, postattenuation_columns, gradient_keys
        )

    if gradient_metrics_df is not None and not gradient_metrics_df.empty:
        gradient_metric_config = get_gradient_metric_config()
        n_cols = len(gradient_metrics_df.columns)
        gradient_table_width = (
            _LABEL_WIDTH + n_cols * _COL_WIDTH + (n_cols - 1) * _SEPARATOR_WIDTH
        )
        total_width = max(signal_table_width, gradient_table_width)
        gradient_table = _render_gradient_metrics_table(
            gradient_metrics_df, gradient_metric_config, gradient_keys
        )

    output = ["=" * total_width]
    output.append(
        _render_transposed_signal_table(signals_breakdown_df, metric_config, selection)
    )
    if gradient_table:
        output.append("")
        output.append(gradient_table)
    output.append("=" * total_width)
    return "\n".join(output)


def _render_transposed_signal_table(
    signals_breakdown_df: pd.DataFrame,
    metric_config: Dict,
    selection: List[Tuple[str, str, str]],
) -> str:
    """Render signal metrics DataFrame in transposed layout (metrics as rows).

    Columns: 3 directions × 3 subdivisions = 9 data columns.
    Rows: Signals count, then one row per (metric_name, operation) in selection.

    Args:
        signals_breakdown_df: Wide numeric DataFrame from build_signals_breakdown_dataframe.
            Index: row_key; columns: "{direction}_{subdivision}".
        metric_config: Perseries metric configuration for label lookups.
        selection: Pre-resolved list of (metric_name, operation, composite_key).
    """
    col_sequence = DIRECTION_SUBDIVISION_COLUMNS

    output = []
    output.append(_build_column_header())

    # --- Signals count row ---
    if SIGNALS_COUNT_LABEL in signals_breakdown_df.index:
        counts = [
            float(signals_breakdown_df.loc[SIGNALS_COUNT_LABEL, col])
            for col in col_sequence
        ]
    else:
        counts = [0.0] * len(col_sequence)
    output.append(_format_transposed_row(SIGNALS_COUNT_LABEL, counts, fmt="d"))

    # --- Metric rows ---
    for metric_name, operation, composite_key in selection:
        label = metric_config[metric_name]["labels"][operation]
        if composite_key in signals_breakdown_df.index:
            values = [
                float(signals_breakdown_df.loc[composite_key, col])
                for col in col_sequence
            ]
        else:
            values = [0.0] * len(col_sequence)
        output.append(_format_transposed_row(label, values))

    return "\n".join(output)


def _build_column_header() -> str:
    """Build two-line column header for transposed signal table.

    Line 1: direction names centered over their 3 sub-columns.
    Line 2: subdivision names repeated for each direction.
    """
    group_width = len(SUBDIVISIONS) * _COL_WIDTH + _SEPARATOR_WIDTH

    # Line 1: direction names
    line1 = " " * _LABEL_WIDTH
    for direction in DIRECTIONS:
        line1 += f"{direction:^{group_width}}"

    # Line 2: subdivision names
    line2 = " " * _LABEL_WIDTH
    for _ in DIRECTIONS:
        for sub in SUBDIVISIONS:
            line2 += f"{sub:>{_COL_WIDTH}}"
        line2 += " " * _SEPARATOR_WIDTH

    return f"\n {SIGNAL_METRICS_SECTION}\n{line1}\n{line2}"


def _format_transposed_row(
    label: str,
    values: List[float],
    fmt: str = ".2f",
) -> str:
    """Format a single transposed row: label + 9 values grouped by direction.

    Args:
        label: Row label (left-aligned).
        values: 9 values in order [dir1_tf, dir1_t, dir1_f, dir2_tf, ...].
        fmt: Format spec for values ('d' for integers, '.2f' for floats).
    """
    row = f"{label:<{_LABEL_WIDTH}}"
    for i, val in enumerate(values):
        if fmt == "d":
            row += f"{int(val):>{_COL_WIDTH}}"
        else:
            row += f"{val:>{_COL_WIDTH}{fmt}}"
        # Add separator after each direction group (every 3 values)
        if (i + 1) % len(SUBDIVISIONS) == 0 and i < len(values) - 1:
            row += " " * _SEPARATOR_WIDTH
    return row


# =============================================================================
# GRADIENT METRICS TABLE (n-COLUMN LAYOUT)
# =============================================================================


def _render_gradient_metrics_table(
    gradient_metrics_df: pd.DataFrame,
    gradient_metric_config: Dict,
    display_order_gradient: List[str],
) -> str:
    """Render gradient metrics DataFrame in n-column layout (strategy + references).

    Args:
        gradient_metrics_df: Wide numeric DataFrame from build_gradient_metrics_dataframe.
            Index: metric_key; columns: column headers (strategy + references).
        gradient_metric_config: Gradient metric configuration from metrics.json.
        display_order_gradient: Ordered metric keys to render as rows.
    """
    col_headers = list(gradient_metrics_df.columns)
    n_cols = len(col_headers)

    output = []
    output.append(f" {GRADIENT_METRICS_SECTION}")

    # Annualization info block — present when sentinel rows were written by
    # build_gradient_metrics_dataframe. Renders four labeled rows (N-1, years,
    # m, √m) aligned with the column widths, then a blank separator line.
    annualization = shared_annualization(gradient_metrics_df)
    if annualization is not None:
        ref_n, ref_y = annualization
        m_val = ref_n / ref_y if ref_y > 1e-6 else 0.0
        sqm_val = math.sqrt(m_val) if m_val > 0.0 else 0.0
        output.append(
            f" N-1: {int(ref_n)};"
            f" years: {ref_y:.2f};"
            f" m: {m_val:.1f};"
            f" \u221am: {sqm_val:.2f}"
        )
        output.append("")  # blank separator before column headers

    # Column-header line: blank label cell + each column header right-aligned
    col_header = " " * _LABEL_WIDTH
    for i, header in enumerate(col_headers):
        col_header += f"{header:>{_COL_WIDTH}}"
        if i < n_cols - 1:
            col_header += " " * _SEPARATOR_WIDTH
    output.append(col_header)

    # Body rows — one per metric key in display order
    for key in display_order_gradient:
        if key not in gradient_metrics_df.index:
            continue
        cfg = gradient_metric_config.get(key, {})
        label = cfg.get("label", key)
        is_int = cfg.get("format") == "int"

        row = f"{label:<{_LABEL_WIDTH}}"
        for i, header in enumerate(col_headers):
            val = gradient_metrics_df.loc[key, header]
            row += _format_cell(val, is_int, _COL_WIDTH)
            if i < n_cols - 1:
                row += " " * _SEPARATOR_WIDTH
        output.append(row)

    return "\n".join(output)


def _format_cell(value: Optional[float], is_int: bool, width: int) -> str:
    """Right-aligned cell string of exactly `width` characters; blank if None or NaN."""
    if _is_nan_or_none(value):
        return " " * width
    if is_int:
        return f"{int(value):>{width}}"
    return f"{value:.2f}".rjust(width)


def _is_nan_or_none(val: Optional[float]) -> bool:
    """Check if a value is None or NaN (handles numpy float32 as well as Python float)."""
    return val is None or bool(pd.isna(val))
