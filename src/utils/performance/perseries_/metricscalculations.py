"""
Formula implementations for per-series metric calculations.

Contains row-level metric computations (variation, duration, trials-to-true)
and aggregation operations (mult, avg, std, std_dw) used by the perseries
metrics orchestrator.

Structure:
1. Row-level metrics (calculate_variations, calculate_durations, calculate_trials_to_true and its JIT
   helper _calculate_per_row_wait_times_jit)
2. Aggregation operations (_apply_operation, the dispatcher, and the JIT compound _multiplicative_compound)
"""

import numba
import numpy as np
import pandas as pd

from ...common.common import variation_pct
from ...common.constants import (
    IS_PLUS_TRIGGER,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
    TRIGGER_END_VALUES,
    TRIGGER_START_VALUES,
    OPERATION_AVG,
    OPERATION_AVGT,
    OPERATION_MULT,
    OPERATION_STD,
    OPERATION_STD_DW,
    OPERATION_SUM,
)


# =============================================================================
# ROW-LEVEL METRICS
# =============================================================================


def calculate_variations(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate percentage variation for each trigger.

    For plus triggers:  variation = variation_pct(end, start)
    For minus triggers: variation = -variation_pct(end, start)

    Args:
        df: Triggers DataFrame with trigger values and direction flags

    Returns:
        Contiguous float32 array of percentage variations
    """
    start_vals = np.ascontiguousarray(
        df[TRIGGER_START_VALUES].values, dtype=np.float32
    )
    end_vals = np.ascontiguousarray(df[TRIGGER_END_VALUES].values, dtype=np.float32)
    is_plus = df[IS_PLUS_TRIGGER].values

    plus_variations = variation_pct(end_vals, start_vals)
    minus_variations = -plus_variations

    return np.where(is_plus, plus_variations, minus_variations).astype(np.float32)


def calculate_durations(df: pd.DataFrame, original_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate duration (in rows of the series index) for each trigger.

    Args:
        df: Triggers DataFrame with begin/end dates
        original_df: Original time series with DatetimeIndex

    Returns:
        Contiguous float32 array of durations
    """
    from ...common.common import ensure_datetime_series

    begin_col = df[TRIGGER_BEGIN_DATES]
    end_col = df[TRIGGER_END_DATES]
    begin_dates = ensure_datetime_series(begin_col)
    end_dates = ensure_datetime_series(end_col)

    start_idx = original_df.index.get_indexer(begin_dates)
    end_idx = original_df.index.get_indexer(end_dates)

    unresolved = (start_idx == -1) | (end_idx == -1)
    durations = (end_idx - start_idx).astype(np.float32)
    durations[unresolved] = np.nan
    return np.ascontiguousarray(durations)


def calculate_trials_to_true(is_true: np.ndarray) -> np.ndarray:
    """
    Calculate per-row wait counts until (and including) each true event.

    True rows receive the count of consecutive rows (including themselves)
    since the previous true event. False rows always receive 0.0.

    Args:
        is_true: Boolean array of true/false outcomes (as int8: 1=true, 0=false)

    Returns:
        Float32 array of length len(is_true). True rows hold their wait count,
        false rows hold 0.0.
    """
    if is_true.size == 0:
        return np.zeros(0, dtype=np.float32)

    arr = np.ascontiguousarray(is_true, dtype=np.int8)

    if arr.sum() == 0:
        return np.zeros(len(arr), dtype=np.float32)

    return _calculate_per_row_wait_times_jit(arr)


@numba.njit(cache=True, fastmath=True)
def _calculate_per_row_wait_times_jit(arr: np.ndarray) -> np.ndarray:
    """
    JIT-compiled per-row wait-count calculation.

    For each true row, records how many rows (including itself) have elapsed
    since the last true event. False rows are recorded as 0.0.

    Args:
        arr: Boolean array as int8 (1 for true, 0 for false)

    Returns:
        Float32 array of same length as arr
    """
    n = len(arr)
    result = np.zeros(n, dtype=np.float32)
    counter = 1

    for i in range(n):
        if arr[i] == 1:
            result[i] = float(counter)
            counter = 1
        else:
            counter += 1

    return result


# =============================================================================
# AGGREGATION OPERATIONS
# =============================================================================


def _apply_operation(operation: str, values: np.ndarray) -> float:
    """
    Apply aggregation operation to values.

    Centralizes operation dispatch logic used across metrics calculations.

    Args:
        operation: Operation type ('sum', 'avg', 'avgt', 'mult', 'std', 'std_dw')
        values: Contiguous float32 array of values

    Returns:
        Aggregated value as float
    """
    if operation == OPERATION_SUM:
        return float(values.sum())
    elif operation in (OPERATION_AVG, OPERATION_AVGT):
        return float(values.mean())
    elif operation == OPERATION_MULT:
        return _multiplicative_compound(values)
    elif operation == OPERATION_STD:
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    elif operation == OPERATION_STD_DW:
        if len(values) < 1:
            return 0.0
        # RMS of raw negative values — no ddof=1 because Bessel's correction only applies when
        # deviations are computed around the sample mean (std), where dividing by n underestimates
        # dispersion. Here we square raw negative values (not deviations from a mean), so there is no such bias.
        # No len>1 guard either: divides by n via mean, so n=1 is valid unlike ddof=1 which needs n>1.
        downside = np.minimum(values, 0.0)
        return float(np.sqrt(np.mean(downside ** 2)))
    else:
        raise ValueError(f"Unknown operation: {operation}")


@numba.njit(cache=True)
def _multiplicative_compound(variations: np.ndarray) -> float:
    """
    JIT-compiled multiplicative (geometric) variation calculation.

    Formula: [(1 + r1/100) * (1 + r2/100) * ... * (1 + rn/100) - 1] * 100

    Uses log-space calculation for numerical stability:
    prod(1 + ri) = exp(sum(log(1 + ri)))

    Args:
        variations: Array of percentage variations.

    Returns:
        Multiplicative (geometric) variation as percentage (float).
    """
    if variations.size == 0:
        return 0.0

    log_sum = 0.0
    for i in range(variations.size):
        r_decimal = variations[i] / 100.0

        if r_decimal <= -1.0:
            return -100.0

        log_sum += np.log1p(r_decimal)

    compounded = np.expm1(log_sum) * 100.0
    return float(compounded)
