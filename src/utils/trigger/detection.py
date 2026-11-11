"""
Core abstract base class for trigger detection.

This module provides the foundational abstract class for trigger detection using diverse strategies that inherit from it.
It handles dispatch to realistic/ideal modes.

Structure:
1. Base detection class (BaseDetection): the public starting point, validation, the abstract methods the
   strategies implement, I/O
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from ..common.common import select_trigger_value_vector
from ..common.hdf5manager import hdf5_manager
from ..common.schema import TriggerConfig
from ..common.constants import (
    END_REASON,
    HDF5_SERIES_PREFIX,
    REPLAY_DUAL,
    REPLAY_IDEAL,
    REPLAY_REALISTIC,
    VAR3,
    VAR0,
    TRIGGER_VALUE,
)
from .gate import applicable_items, begin_allowed
from .ideal import IdealModeMixin
from .prints.detection import print_trigger_output
from .realistic import RealisticModeMixin

logger = logging.getLogger(__name__)

# =============================================================================
# BASE DETECTION CLASS
# =============================================================================


class BaseDetection(RealisticModeMixin, IdealModeMixin, ABC):
    """Abstract base class for all trigger strategies.

    Expects already preprocessed time series data as input.
    Dispatches to realistic or ideal mode based on configuration.
    """

    def __init__(self, trigger: TriggerConfig, output_file: Path):
        """Cache the values the per-series calls read, once per strategy (one strategy per task and mode): the base
        values of the trigger section, the recording mode (trigger.trigger_value), the gate items (trigger.gate.items),
        the trigger output file and, in realistic mode, last, the realistic loop's config."""
        self._min_length = trigger.min_length
        self._lookback = trigger.replay.lookback_window
        self._replay_mode = trigger.replay.mode
        self._trigger_value_mode = trigger.trigger_value
        self._printout_io = trigger.strategy.printout_io
        self._gate_items = list(trigger.gate.items)
        self._output_file = output_file
        if self._replay_mode == REPLAY_REALISTIC:
            self._realistic_config = self._load_realistic_config(trigger)

    # =========================================================================
    # PUBLIC STARTING POINT
    # =========================================================================

    def analyze_series(
        self, series_id: str, df: pd.DataFrame, store: Optional[pd.HDFStore] = None
    ) -> Optional[pd.DataFrame]:
        """Main method to analyze a preprocessed time series DataFrame and save trigger columns to HDF5.

        Returns:
            The Trigger DataFrame, or None if validation failed.
        """
        if not self._validate_data_length(series_id, df):
            logger.warning(
                f"Skipping series '{series_id}': insufficient data length for analysis."
            )
            return None

        df = self._add_trigger_value_column(df)
        # The gate on the previous row, once per series: both modes read allowed[i] at a begin's row i
        self._begin_allowed = begin_allowed(series_id, df, applicable_items(series_id, self._gate_items))
        # Initialize end_reason column early to ensure it exists with correct dtype
        if END_REASON not in df.columns:
            df[END_REASON] = ""
        df[END_REASON] = df[END_REASON].astype("object").fillna("")

        if self._replay_mode == REPLAY_REALISTIC:
            df = self._analyze_series_realistic(series_id, df)
        elif self._replay_mode == REPLAY_IDEAL:
            df = self._analyze_series_ideal(series_id, df)
        else:
            raise ValueError(
                f"Unresolved replay mode '{self._replay_mode}' reached detection. "
                f"'{REPLAY_DUAL}' must be expanded to individual modes before task creation."
            )

        self._save_to_hdf5(series_id, df, store=store)
        return df

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    def _validate_data_length(self, series_id: str, df: pd.DataFrame) -> bool:
        """Check if data has sufficient length for meaningful analysis."""
        n = len(df)

        if n < self._min_length:
            logger.warning(
                f"Series '{series_id}': has {n} data points but trigger.min_length is {self._min_length}. "
                f"Series too short for analysis."
            )
            return False

        if n < self._lookback:
            logger.warning(
                f"Series '{series_id}': has {n} data points but lookback_window is {self._lookback}. "
                f"Series too short for analysis with this lookback period."
            )
            return False

        return True

    def _add_trigger_value_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and store the trigger_value column based on configured mode.

        - "detection_row": trigger_value[D] = var3[D] for every date D.
        - "following_row": trigger_value[D] = var0[D+1] (the next row's var0).
          Fallback to var3[D] if D is the last date or if var0[D+1] is NaN.
        """
        df[TRIGGER_VALUE] = select_trigger_value_vector(
            df,
            self._trigger_value_mode,
            primary_col=VAR3,
            start_col=VAR0,
        )
        return df

    # =========================================================================
    # ABSTRACT METHODS (STRATEGY-SPECIFIC)
    # =========================================================================

    @abstractmethod
    def _generate_trigger_begin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate begin trigger columns - strategy-specific implementation.

        Should generate trigger_plus_begin, trigger_minus_begin, and turning_points columns.
        """
        pass

    @abstractmethod
    def _begin_candidate_rows(self, var3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The window rows _generate_trigger_begin marks in its three columns - plus begins, minus begins, turning
        points - from the window's var3 values; the realistic loop reads a window's candidates and turning-point
        count through it, so no frame is built per row.
        """
        pass

    @abstractmethod
    def _generate_trigger_end(
        self, df: pd.DataFrame, active_trigger: Optional[Tuple[str, Any, float]] = None
    ) -> pd.DataFrame:
        """Generate end trigger columns - strategy-specific implementation for different_end_strategy mode.

        Args:
            df: DataFrame window to analyze (preprocessed)
            active_trigger: Current active trigger tuple (trigger_type, begin_date, begin_value) or None

        Returns:
            DataFrame with trigger_plus_end and trigger_minus_end columns populated
        """
        pass

    # =========================================================================
    # I/O METHODS
    # =========================================================================

    def _save_to_hdf5(
        self, series_id: str, df: pd.DataFrame, *, store: Optional[pd.HDFStore] = None
    ) -> str:
        """Save complete DataFrame to HDF5 file in strategy-specific folder."""
        if self._printout_io:
            print_trigger_output(series_id, df)

        Path(self._output_file).parent.mkdir(parents=True, exist_ok=True)

        if store is None:
            with hdf5_manager.open_store(self._output_file, "a") as store_local:
                hdf5_manager.put_dataframe_normalized(
                    store_local, f"{HDF5_SERIES_PREFIX}{series_id}", df, format="table"
                )
        else:
            hdf5_manager.put_dataframe_normalized(
                store, f"{HDF5_SERIES_PREFIX}{series_id}", df, format="table"
            )

        return str(self._output_file)
