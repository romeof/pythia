"""
Metric orchestration for per-series performance analysis.

Contains orchestration and configuration loading for perseries metrics.
All formula implementations live in metricscalculations.py.

Structure (strict top-down — callee immediately after first caller):
1. _extract_needed_columns    calls resolve_signal_selection
2. resolve_signal_selection   utility (also used by printing code)
3. calculate_metrics          uses needed_columns, calls trials_to_true_per_direction
4. trials_to_true_per_direction  calls _trials_by_begin (also used by the Performance task)
5. _trials_by_begin           also used by prepare_direction_subset
6. normalize_dataframe
7. get_metric_config
8. build_signals_breakdown_dataframe  calls calculate_direction_metrics
9. calculate_direction_metrics  calls prepare_direction_subset, _calculate_metric_for_subsets
10. prepare_direction_subset
11. _calculate_metric_for_subsets
"""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ...common.config import load_json_config
from ...common.constants import (
    DIRECTION_MINUS,
    DIRECTION_PLUS,
    DIRECTION_PLUS_MINUS,
    DIRECTION_SUBDIVISION_COLUMNS,
    DIRECTIONS,
    IS_PLUS_TRIGGER,
    IS_TRUE,
    SERIES_ID,
    SIGNALS_COUNT_LABEL,
    SUBDIVISIONS,
    TRIALS_TO_TRUE,
    TRIGGER_BEGIN_DATES,
    FROM_DICT_ORIENT,
    OPERATION_AVGT,
    SUBDIVISION_FALSE,
    SUBDIVISION_TRUE,
)
from .metricscalculations import (
    calculate_variations as _calculate_variations,
    calculate_durations as _calculate_durations,
    calculate_trials_to_true as _calculate_trials_to_true,
    _apply_operation,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COMPOSITE KEY UTILITIES
# =============================================================================


def _extract_needed_columns(
    metric_config: Dict, display_order_signals: List[str]
) -> Set[str]:
    """Return the set of DataFrame columns needed by the selected composite keys.

    Delegates parsing to resolve_signal_selection — single source of truth.
    """
    selection = resolve_signal_selection(metric_config, display_order_signals)
    return {metric_config[metric_name]["column"] for metric_name, _, _ in selection}


def resolve_signal_selection(
    metric_config: Dict, display_order_signals: List[str]
) -> List[Tuple[str, str, str]]:
    """Parse composite keys into (metric_name, operation, composite_key) tuples.

    For each composite key (e.g. 'variation_avg', 'trials_to_true_avgt'),
    finds the metric_name from metric_config that is a prefix of the key
    (separated by '_') and whose remaining suffix is a valid operation.

    Raises:
        ValueError: For any composite key that does not match a known metric + operation.
    Preserves the user-specified sequence.
    """
    # One-time O(N×ops) build — each display_order lookup is then O(1).
    composite_map: Dict[str, Tuple[str, str]] = {
        f"{m}_{o}": (m, o)
        for m, cfg in metric_config.items()
        for o in cfg["operations"]
    }
    result: List[Tuple[str, str, str]] = []
    for composite_key in display_order_signals:
        if composite_key not in composite_map:
            raise ValueError(
                f"Unknown signal metric composite key {composite_key!r} in "
                f"display_order_signals. Valid keys: {sorted(composite_map)}"
            )
        metric_name, op = composite_map[composite_key]
        result.append((metric_name, op, composite_key))
    return result


# =============================================================================
# MAIN STARTING POINT
# =============================================================================


def calculate_metrics(
    original_df: pd.DataFrame,
    rows_df: pd.DataFrame,
    needed_columns: Set[str],
) -> pd.DataFrame:
    """Calculate selected metrics for a rows DataFrame.

    Only columns in needed_columns are computed, skipping unused calculations.

    Args:
        original_df: Original time series DataFrame
        rows_df: DataFrame of trigger or signal rows
        needed_columns: Set of column names to compute (from _extract_needed_columns)

    Returns:
        DataFrame with only the requested metric columns added.
    """
    if rows_df.empty:
        logger.warning("No rows found for metrics calculation")
        return rows_df

    df = rows_df.sort_values(TRIGGER_BEGIN_DATES).reset_index(drop=True)
    if "variation" in needed_columns:
        df["variation"] = _calculate_variations(df)
    if "duration" in needed_columns:
        df["duration"] = _calculate_durations(df, original_df)
    if TRIALS_TO_TRUE in needed_columns:
        df[TRIALS_TO_TRUE] = trials_to_true_per_direction(df)
    return df


def trials_to_true_per_direction(rows_df: pd.DataFrame) -> np.ndarray:
    """Trials-to-true counts per direction, aligned to the rows of rows_df.

    Plus rows and minus rows are counted separately, each through _trials_by_begin.
    """
    counts = np.zeros(len(rows_df), dtype=np.float32)
    plus = rows_df[IS_PLUS_TRIGGER].to_numpy(dtype=bool)
    for direction_mask in (plus, ~plus):
        if direction_mask.any():
            counts[direction_mask] = _trials_by_begin(rows_df[direction_mask])
    return counts


def _trials_by_begin(rows_df: pd.DataFrame) -> np.ndarray:
    """Trials-to-true counts aligned to the rows of rows_df, counted in trigger begin sequence.

    A stable sort by TRIGGER_BEGIN_DATES, then SERIES_ID when that column exists (the signal
    grouping's processing key), fixes the counting sequence; the counts are scattered back to
    the input rows, so the caller's row sequence never matters.
    """
    keys = [TRIGGER_BEGIN_DATES, SERIES_ID] if SERIES_ID in rows_df.columns else [TRIGGER_BEGIN_DATES]
    sequence = rows_df.reset_index(drop=True).sort_values(keys, kind="mergesort").index.to_numpy()
    outcomes = np.ascontiguousarray(rows_df[IS_TRUE].to_numpy()[sequence], dtype=np.int8)
    counts = np.empty(len(rows_df), dtype=np.float32)
    counts[sequence] = _calculate_trials_to_true(outcomes)
    return counts


# =============================================================================
# SCHEMA NORMALIZATION
# =============================================================================


def normalize_dataframe(
    series_id: str, df: pd.DataFrame, metric_config: Dict
) -> pd.DataFrame:
    """
    Normalize a rows DataFrame schema for consistent HDF5 storage.

    Ensures consistent dtypes for series_id (string), booleans, and metric floats
    across both perseries and gradient performance flows.

    Args:
        series_id: Series identifier to ensure string type
        df: Rows DataFrame to normalize
        metric_config: Metric configuration from metrics.json

    Returns:
        Normalized DataFrame with consistent dtypes
    """
    normalized_df = df.copy()
    normalized_df[SERIES_ID] = str(series_id)

    for bcol in (IS_TRUE, IS_PLUS_TRIGGER):
        if bcol in normalized_df.columns:
            normalized_df[bcol] = normalized_df[bcol].astype("bool")

    metric_cols = {v.get("column") for v in metric_config.values() if "column" in v}
    for fcol in metric_cols:
        if fcol in normalized_df.columns:
            normalized_df[fcol] = normalized_df[fcol].astype("float32")

    return normalized_df


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================


def get_metric_config() -> Dict:
    """Load metric configuration from metrics.json located alongside this module."""
    return load_json_config(__file__)


# =============================================================================
# SIGNALS BREAKDOWN DATAFRAME BUILDER
# =============================================================================


def build_signals_breakdown_dataframe(
    signal_only_df: pd.DataFrame,
    metric_config: Dict,
    display_order_signals: List[str],
) -> pd.DataFrame:
    """Build a wide numeric DataFrame representing the signals breakdown table.

    Rows: SIGNALS_COUNT_LABEL (first) + one composite_key per selection item.
    Columns: "{direction}_{subdivision}" for each direction × subdivision (9 total),
    e.g. "plus+minus_true+false", "plus+minus_true", "plus+minus_false", …

    Values are float32. Count rows (SIGNALS_COUNT_LABEL) are stored as float32
    (exact for integers up to 2²⁴ — well above any realistic signal count).

    Args:
        signal_only_df: Rows DataFrame pre-filtered to IS_SIGNAL rows.
        metric_config: Perseries metric configuration from metrics.json.
        display_order_signals: Ordered list of composite signal metric keys.

    Returns:
        DataFrame with index name "row_key", 9 columns, dtype float32.
        Empty DataFrame when signal_only_df is empty or selection is empty.
    """
    if signal_only_df.empty:
        return pd.DataFrame()

    selection: List[Tuple[str, str, str]] = resolve_signal_selection(
        metric_config, display_order_signals
    )

    col_names = DIRECTION_SUBDIVISION_COLUMNS

    dir_metrics: Dict[str, Dict] = {}
    for direction in DIRECTIONS:
        dir_metrics[direction] = calculate_direction_metrics(
            signal_only_df, metric_config, direction, selection
        )

    rows: Dict[str, Dict[str, float]] = {}

    # Signals count row
    count_row: Dict[str, float] = {}
    for direction in DIRECTIONS:
        m = dir_metrics[direction]
        count_row[f"{direction}_{SUBDIVISIONS[0]}"] = float(
            m.get("combined_rows", (None, 0))[1]
        )
        count_row[f"{direction}_{SUBDIVISIONS[1]}"] = float(
            m.get("true_rows", (None, 0))[1]
        )
        count_row[f"{direction}_{SUBDIVISIONS[2]}"] = float(
            m.get("false_rows", (None, 0))[1]
        )
    rows[SIGNALS_COUNT_LABEL] = count_row

    # Metric rows
    for metric_name, operation, composite_key in selection:
        row: Dict[str, float] = {}
        for direction in DIRECTIONS:
            m = dir_metrics[direction]
            combined_key = f"combined_rows_{metric_name}_{operation}"
            true_key = f"true_rows_{metric_name}_{operation}"
            false_key = f"false_rows_{metric_name}_{operation}"
            row[f"{direction}_{SUBDIVISIONS[0]}"] = float(
                m.get(combined_key, (None, 0.0))[1]
            )
            row[f"{direction}_{SUBDIVISIONS[1]}"] = float(
                m.get(true_key, (None, 0.0))[1]
            )
            row[f"{direction}_{SUBDIVISIONS[2]}"] = float(
                m.get(false_key, (None, 0.0))[1]
            )
        rows[composite_key] = row

    df = pd.DataFrame.from_dict(rows, orient=FROM_DICT_ORIENT, columns=col_names)
    df.index.name = "row_key"
    return df.astype("float32")


# =============================================================================
# DIRECTION-BASED AGGREGATION
# =============================================================================


def calculate_direction_metrics(
    rows_df: pd.DataFrame,
    metric_config: Dict,
    direction: str,
    selection: List[Tuple[str, str, str]],
) -> Dict[str, Tuple[str, float]]:
    """Calculate metrics for a specific direction (plus, minus, or plus+minus).

    Only (metric_name, operation) pairs in `selection` are computed.
    The Signals count row is always included regardless of selection.

    Args:
        rows_df: DataFrame of rows with calculated metrics
        metric_config: Configuration dictionary defining metrics and operations
        direction: 'plus', 'minus', or 'plus+minus'
        selection: Pre-resolved list of (metric_name, operation, composite_key)
            from resolve_signal_selection

    Returns:
        Dictionary of metric keys and (label, value) tuples
    """
    if rows_df.empty:
        return {}

    subset_df = prepare_direction_subset(rows_df, direction)

    if subset_df.empty:
        return {}

    true_subset = subset_df[IS_TRUE]
    false_subset = ~subset_df[IS_TRUE]
    true_count = int(np.sum(true_subset))
    false_count = int(np.sum(false_subset))

    result = {
        "combined_rows": (SIGNALS_COUNT_LABEL, true_count + false_count),
        "true_rows": (SIGNALS_COUNT_LABEL, true_count),
        "false_rows": (SIGNALS_COUNT_LABEL, false_count),
    }

    for metric_name, operation, _ in selection:
        config = metric_config[metric_name]
        column = config["column"]
        label = config["labels"][operation]

        if column in subset_df.columns:
            values = subset_df[column].values
            metric_result = _calculate_metric_for_subsets(
                values, true_subset, false_subset, operation, metric_name, label
            )
            result.update(metric_result)

    return result


def prepare_direction_subset(
    rows_df: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    """Filter rows by direction, sort chronologically, recompute trials-to-true.

    Args:
        rows_df: DataFrame with IS_PLUS_TRIGGER and IS_TRUE columns
        direction: 'plus', 'minus', or 'plus+minus'

    Returns:
        Direction-filtered copy sorted by TRIGGER_BEGIN_DATES with
        trials-to-true recomputed on the subset.
    """
    if direction == DIRECTION_PLUS:
        subset = rows_df[rows_df[IS_PLUS_TRIGGER]]
    elif direction == DIRECTION_MINUS:
        subset = rows_df[~rows_df[IS_PLUS_TRIGGER]]
    elif direction == DIRECTION_PLUS_MINUS:
        subset = rows_df
    else:
        raise ValueError(f"Unknown direction: {direction}")

    subset = subset.sort_values(TRIGGER_BEGIN_DATES).reset_index(drop=True).copy()

    if not subset.empty and TRIALS_TO_TRUE in subset.columns:
        subset[TRIALS_TO_TRUE] = _trials_by_begin(subset)

    return subset


def _calculate_metric_for_subsets(
    values: np.ndarray,
    true_subset: pd.Series,
    false_subset: pd.Series,
    operation: str,
    metric_name: str,
    label: str,
) -> Dict[str, Tuple[str, float]]:
    """Calculate metric values for combined, true, and false subsets.

    The "combined" subset aggregates across both true and false rows.
    For most operations it is computed by applying the operation directly to
    the whole values array, which yields the mathematically correct combined
    statistic (weighted mean for avg, exact combined std for std, pooled
    semi-dispersion for std_dw, geometric compound for mult, etc.).

    The only exception is "avgt" (trials-to-true): false rows carry a
    sentinel value of 0 that would dilute a naive mean, so the combined
    value is taken as the mean over the true subset only.

    Args:
        label: Row label from metric_config["labels"][operation] (JSON source of truth).
    """
    result = {}

    if operation == OPERATION_AVGT:
        combined_subset = np.asarray(true_subset, dtype=bool)
    else:
        combined_subset = np.ones(len(values), dtype=bool)

    subsets = [
        ("combined", combined_subset),
        (SUBDIVISION_TRUE, true_subset),
        (SUBDIVISION_FALSE, false_subset),
    ]

    for subset_name, subset in subsets:
        if np.any(subset):
            subset_values = np.ascontiguousarray(values[subset], dtype=np.float32)
            value = _apply_operation(operation, subset_values)
        else:
            value = 0.0

        key = f"{subset_name}_rows_{metric_name}_{operation}"
        result[key] = (label, value)

    return result
