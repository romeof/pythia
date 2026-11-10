"""
Extrema (turning points) trigger detection.

This module provides the ExtremaTrigger class that detects
local minima and maxima in time series data using scipy's find_peaks.

Structure:
1. ExtremaTrigger class with __init__
2. _generate_trigger_begin (abstract impl)
3. _begin_candidate_rows (abstract impl; called 1st by _generate_trigger_begin, and by the realistic loop)
4. _find_extrema (called by _begin_candidate_rows)
5. _initialize_columns (called 2nd, shared by begin and end)
6. _mark_triggers (called 3rd)
7. _generate_trigger_end (abstract impl)
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ...common.constants import (
    END_IDEAL_MODE,
    END_REASON,
    VAR3,
    TRIGGER_MINUS_BEGIN,
    TRIGGER_MINUS_END,
    TRIGGER_PLUS_BEGIN,
    TRIGGER_PLUS_END,
    TURNING_POINTS,
)
from ...common.schema import TriggerConfig
from ..detection import BaseDetection
BEGIN_COLUMNS = [TRIGGER_PLUS_BEGIN, TRIGGER_MINUS_BEGIN, TURNING_POINTS]
END_COLUMNS = [TRIGGER_PLUS_END, TRIGGER_MINUS_END]


class ExtremaTrigger(BaseDetection):
    """Turning points (extrema) detection algorithm.

    Detects local minima (trigger_plus_begin) and maxima (trigger_minus_begin)
    using scipy's find_peaks with configurable parameters.
    """

    def __init__(self, trigger: TriggerConfig, output_file: Path):
        """Initialize extrema trigger algorithm from trigger.strategy, then the base strategy's cache.

        Reads:
            min_distance: Minimum horizontal separation between maxima
            prominence_threshold: Minimum required prominence of maxima
            width: Turning-point width constraint
        """
        self.min_distance = trigger.strategy.min_distance
        self.prominence_threshold = trigger.strategy.prominence_threshold
        self.width = trigger.strategy.width

        super().__init__(trigger, output_file)

    # =========================================================================
    # TRIGGER GENERATION (Abstract method implementations)
    # =========================================================================

    def _generate_trigger_begin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate begin trigger columns for triggered triggers.

        Detects:
        - Minima -> trigger_plus_begin
        - Maxima -> trigger_minus_begin
        - All turning points -> turning_points column
        """
        minima_indices, maxima_indices, turning_points = self._begin_candidate_rows(df[VAR3].values)

        df = self._initialize_columns(df, BEGIN_COLUMNS)
        df = self._mark_triggers(df, minima_indices, maxima_indices, turning_points)

        return df

    def _begin_candidate_rows(self, var3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The rows of the three begin columns: minima are the plus rows, maxima the minus rows, and their union the
        turning points."""
        minima, maxima = self._find_extrema(var3)
        return minima, maxima, np.union1d(minima, maxima)

    def _find_extrema(self, ts_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find minima and maxima using scipy's find_peaks.

        Each turning point takes its type from the call that reports it: maxima of -var3 are minima,
        maxima of var3 maxima, so a flat extremum (find_peaks reports its middle sample) keeps its type.

        Args:
            ts_data: Time series values array

        Returns:
            Tuple of (minima_indices, maxima_indices) as NumPy arrays
        """
        data_range = np.max(ts_data) - np.min(ts_data)
        min_prominence = self.prominence_threshold * data_range

        # Find local maxima
        maxima, _ = find_peaks(
            ts_data,
            distance=self.min_distance,
            prominence=min_prominence,
            width=self.width,
        )

        # Find local minima by inverting the data
        minima, _ = find_peaks(
            -ts_data,
            distance=self.min_distance,
            prominence=min_prominence,
            width=self.width,
        )

        return minima, maxima

    def _initialize_columns(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Initialize trigger columns to zero."""
        for col in cols:
            df[col] = 0
        return df

    def _mark_triggers(
        self,
        df: pd.DataFrame,
        minima_indices: np.ndarray,
        maxima_indices: np.ndarray,
        turning_points: np.ndarray,
    ) -> pd.DataFrame:
        """Mark detected triggers in the DataFrame."""
        if len(minima_indices) > 0:
            df.iloc[minima_indices, df.columns.get_loc(TRIGGER_PLUS_BEGIN)] = 1

        if len(maxima_indices) > 0:
            df.iloc[maxima_indices, df.columns.get_loc(TRIGGER_MINUS_BEGIN)] = 1

        if len(turning_points) > 0:
            df.iloc[turning_points, df.columns.get_loc(TURNING_POINTS)] = 1

        return df

    def _generate_trigger_end(
        self, df: pd.DataFrame, active_trigger=None
    ) -> pd.DataFrame:
        """Generate end trigger columns based on begin triggers.

        For the extrema algorithm:
        - Plus end occurs at maxima (where minus begin occurs)
        - Minus end occurs at minima (where plus begin occurs)
        """
        df = self._initialize_columns(df, END_COLUMNS)

        # Derive end triggers from begin triggers (opposite extrema)
        df[TRIGGER_PLUS_END] = df[TRIGGER_MINUS_BEGIN].values
        df[TRIGGER_MINUS_END] = df[TRIGGER_PLUS_BEGIN].values

        # Set end_reason for all end triggers
        df.loc[df[TRIGGER_PLUS_END] == 1, END_REASON] = END_IDEAL_MODE
        df.loc[df[TRIGGER_MINUS_END] == 1, END_REASON] = END_IDEAL_MODE

        return df
