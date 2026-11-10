"""
Common utilities for data handling (the configuration lives in config.py, the output paths in outputs.py).

This module provides shared utilities used across the pythia pipeline:
- Trigger value selection for performance metrics
- Datetime utilities and the effective start
- DataFrame pivot utilities (tall_to_wide, wide_to_tall)

Structure:
1. Trigger value selection
2. Datetime utilities
3. Mathematical utilities (variation_pct)
4. DataFrame pivot utilities
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from .constants import (
    TRIGGER_VALUE_FOLLOWING_ROW,
    TRIGGER_VALUE_DETECTION_ROW,
)


# =============================================================================
# TRIGGER VALUE SELECTION
# =============================================================================


def select_trigger_value_vector(
    df: pd.DataFrame,
    mode: str,
    *,
    primary_col: str,
    start_col: str,
) -> np.ndarray:
    """
    Return trigger values according to the configured mode.

    Args:
        df: DataFrame containing the relevant columns
        mode: 'detection_row' | 'following_row'
        primary_col: Column name for same-row values
        start_col: Column name for next-row var0

    Returns:
        Array of trigger values
    """
    if mode == TRIGGER_VALUE_DETECTION_ROW:
        return _detection_row_values(df, primary_col)
    elif mode == TRIGGER_VALUE_FOLLOWING_ROW:
        return _following_row_values(df, primary_col, start_col)
    else:
        raise ValueError(f"Unknown trigger_value mode: {mode}")


def _detection_row_values(
    df: pd.DataFrame,
    primary_col: str,
) -> np.ndarray:
    """Get same-row var3 values."""
    return df[primary_col].to_numpy(copy=False)


def _following_row_values(
    df: pd.DataFrame,
    primary_col: str,
    start_col: str,
) -> np.ndarray:
    """Get next-row var0 values with fallback to var3."""
    # Fallback if start_col is absent
    if start_col not in df.columns:
        return _detection_row_values(df, primary_col)

    var0_values = df[start_col].to_numpy(copy=False)

    base = df[primary_col].to_numpy(copy=False)
    n = var0_values.shape[0]

    return _build_next_row_vector(var0_values, base, n)


def _build_next_row_vector(
    var0_values: np.ndarray, base: np.ndarray, n: int
) -> np.ndarray:
    """Build next-row vector for all rows."""
    next_var0 = np.full(n, np.nan, dtype=float)
    if n > 1:
        next_var0[:-1] = var0_values[1:]
    return np.where(np.isnan(next_var0), base, next_var0)


# =============================================================================
# DATETIME UTILITIES
# =============================================================================


def ensure_datetime_series(series: Union[pd.Series, np.ndarray, list]) -> pd.Series:
    """Convert a series or array to datetime if not already."""
    from pandas.api.types import is_datetime64_any_dtype

    if isinstance(series, pd.Series):
        if is_datetime64_any_dtype(series):
            return series
        return pd.to_datetime(series)
    elif isinstance(series, (list, tuple)):
        return pd.Series(pd.to_datetime(series))
    else:
        # NumPy array
        if is_datetime64_any_dtype(series):
            return pd.Series(series)
        return pd.Series(pd.to_datetime(series))


def effective_start(index: pd.DatetimeIndex, lookback: int) -> Optional[pd.Timestamp]:
    """First row on which a series may emit a trigger: index[lookback - 1] of its date-filtered, preprocessed frame.

    Returns None for an index shorter than lookback (such a series never reaches detection).
    """
    if len(index) < lookback:
        return None
    return index[lookback - 1]


# =============================================================================
# MATHEMATICAL UTILITIES
# =============================================================================


def variation_pct(current, var0):
    """Percentage variation: (current / var0 - 1) * 100.

    var0 is always the reference starting value
    from which the variation is measured.
    """
    return (current / var0 - 1.0) * 100.0


# =============================================================================
# DATAFRAME PIVOT UTILITIES
# =============================================================================


def wide_to_tall(
    df: pd.DataFrame,
    row_name: str,
    col_name: str,
    value_name: str,
) -> pd.DataFrame:
    """Pivot a wide DataFrame (index = row key, columns = col keys) to tall form.

    Each (row_key, col_key, value) triple becomes one row in the output.

    Args:
        df: Wide DataFrame; index is the row-key axis, columns are col-key axis.
        row_name: Column name for the row-key dimension in the output.
        col_name: Column name for the column-key dimension in the output.
        value_name: Column name for the value dimension in the output.

    Returns:
        Tall DataFrame with columns [row_name, col_name, value_name], dtype float32
        for the value column, string for the key columns.
    """
    tall = df.rename_axis(row_name).reset_index().melt(
        id_vars=row_name,
        var_name=col_name,
        value_name=value_name,
    )
    tall[value_name] = tall[value_name].astype("float32")
    tall[row_name] = tall[row_name].astype(str)
    tall[col_name] = tall[col_name].astype(str)
    return tall.reset_index(drop=True)


def tall_to_wide(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Pivot a tall DataFrame back to wide form (inverse of wide_to_tall).

    Args:
        df: Tall DataFrame with at least [row_col, col_col, value_col] columns.
        row_col: Column containing the row-key dimension (becomes the index).
        col_col: Column containing the column-key dimension (becomes columns).
        value_col: Column containing the float32 values.

    Returns:
        Wide DataFrame with index = row keys (name = row_col), columns = col keys,
        dtype float32.

    Raises:
        ValueError: If (row_col, col_col) pairs are not unique (ambiguous pivot).
    """
    n_unique = df[[row_col, col_col]].drop_duplicates().shape[0]
    if n_unique != len(df):
        raise ValueError(
            f"tall_to_wide: duplicate ({row_col!r}, {col_col!r}) pairs detected. "
            f"Each pair must identify exactly one value."
        )

    wide = df.pivot(index=row_col, columns=col_col, values=value_col)
    wide.columns.name = None
    # Restore insertion order (pivot sorts alphabetically by default)
    col_sequence = list(dict.fromkeys(df[col_col]))
    wide = wide[[c for c in col_sequence if c in wide.columns]]
    wide = wide.astype("float32")
    return wide
