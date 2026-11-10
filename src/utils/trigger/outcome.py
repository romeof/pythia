"""
Trigger extraction from time series trigger data.

This module provides the TriggerData class that extracts triggers from
trigger columns, pairing begin/end dates per direction
and computing trigger metadata (direction, outcome).

Structure:
1. TriggerData class with trigger extraction
2. Trigger date pairing helper
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common.constants import (
    IS_PLUS_TRIGGER,
    IS_TRUE,
    SERIES_ID,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
    TRIGGER_END_VALUES,
    TRIGGER_MINUS_BEGIN,
    TRIGGER_MINUS_END,
    TRIGGER_PLUS_BEGIN,
    TRIGGER_PLUS_END,
    TRIGGER_START_VALUES,
    TRIGGER_VALUE,
)

logger = logging.getLogger(__name__)


class TriggerData:
    """Extracts trigger data from trigger columns in HDF5 DataFrames."""

    # =========================================================================
    # TRIGGER EXTRACTION
    # =========================================================================

    def _extract_triggers_from_trigger_columns(
        self, series_id: str, df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Extract trigger data from trigger columns in the DataFrame.

        Reads the trigger_value column (precomputed during trigger detection) at begin/end dates
        to get the mode-aware trigger start/end values without redundant recomputation.

        Args:
            series_id: Series identifier
            df: DataFrame with trigger columns and trigger_value column

        Returns:
            DataFrame with trigger data in the format expected by existing analysis functions
        """
        # Extract dates from trigger columns
        (
            trigger_plus_begin_dates,
            trigger_plus_end_dates,
            trigger_minus_begin_dates,
            trigger_minus_end_dates,
        ) = self._get_begin_end_dates_from_trigger_columns(df)

        # Combine all triggers
        trigger_begin_dates = trigger_plus_begin_dates + trigger_minus_begin_dates
        trigger_end_dates = trigger_plus_end_dates + trigger_minus_end_dates

        if not trigger_begin_dates or not trigger_end_dates:
            logger.warning(f"No triggers found for series {series_id}")
            return None

        begin_dt = pd.DatetimeIndex(trigger_begin_dates)
        end_dt = pd.DatetimeIndex(trigger_end_dates)
        begin_pos = df.index.get_indexer(begin_dt)
        end_pos = df.index.get_indexer(end_dt)

        # Create direction flag (aligned to current number of triggers)
        is_plus_trigger = [True] * len(trigger_plus_begin_dates) + [False] * len(
            trigger_minus_begin_dates
        )
        is_plus_trigger = np.array(is_plus_trigger)

        # Read trigger values from precomputed column (single source of truth from trigger detection)
        trigger_value_arr = df[TRIGGER_VALUE].to_numpy(copy=False)
        trigger_start_values = trigger_value_arr[begin_pos]
        trigger_end_values = trigger_value_arr[end_pos]

        # Create true/false flag
        # For plus triggers: true is end > start and false is end <= start
        plus_true = trigger_end_values > trigger_start_values
        # For minus triggers: true is end < start and false is end >= start
        minus_true = trigger_end_values < trigger_start_values
        is_true = np.where(is_plus_trigger, plus_true, minus_true)

        # Create the trigger data in the format expected by existing functions
        trigger_data = {
            SERIES_ID: [series_id] * len(begin_dt),
            TRIGGER_BEGIN_DATES: begin_dt,
            TRIGGER_END_DATES: end_dt,
            TRIGGER_START_VALUES: trigger_start_values,
            TRIGGER_END_VALUES: trigger_end_values,
            IS_PLUS_TRIGGER: is_plus_trigger.astype(np.bool_),
            IS_TRUE: is_true.astype(np.bool_),
        }

        # Create DataFrame
        triggers_df = pd.DataFrame(trigger_data)
        return triggers_df

    # =========================================================================
    # TRIGGER DATE PAIRING
    # =========================================================================

    def _get_begin_end_dates_from_trigger_columns(
        self, df: pd.DataFrame
    ) -> Tuple[List, List, List, List]:
        """Extract trigger dates from trigger columns with vectorized 1:1 pairing per direction.
        Each begin pairs with the first subsequent end in the SAME direction.
        Ends are used at most once. Begins with no subsequent end are dropped.
        Ends before any begin are ignored. Begin==End at the same timestamp do NOT pair.
        This pairing logic might be redundant after _clean_duplicates in ideal.py, but it provides robustness against edge cases and data inconsistencies.
        """
        index = df.index.to_numpy()

        def pair_direction(begin_col: str, end_col: str):
            beg_idx = np.flatnonzero(df[begin_col].to_numpy(copy=False) > 0)
            end_idx = np.flatnonzero(df[end_col].to_numpy(copy=False) > 0)

            if beg_idx.size == 0 or end_idx.size == 0:
                return [], []

            # For each begin, first candidate end strictly after it
            pos = np.searchsorted(end_idx, beg_idx, side="right")

            # Enforce 1:1 usage of ends: make chosen end rows strictly increasing
            # adj[i] = max(pos[i], adj[i-1] + 1) via cumulative maximum trick
            adj = np.maximum.accumulate(pos - np.arange(pos.size)) + np.arange(pos.size)

            # Keep only begins that still have an available end
            valid = adj < end_idx.size
            if not np.any(valid):
                return [], []

            matched_beg_idx = beg_idx[valid]
            matched_end_idx = end_idx[adj[valid]]

            return index[matched_beg_idx].tolist(), index[matched_end_idx].tolist()

        trigger_plus_begin_dates, trigger_plus_end_dates = pair_direction(
            TRIGGER_PLUS_BEGIN, TRIGGER_PLUS_END
        )
        trigger_minus_begin_dates, trigger_minus_end_dates = pair_direction(
            TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END
        )

        return (
            trigger_plus_begin_dates,
            trigger_plus_end_dates,
            trigger_minus_begin_dates,
            trigger_minus_end_dates,
        )
