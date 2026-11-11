"""
Realistic mode analysis for trigger strategies.

This module provides the RealisticModeMixin class that implements incremental
trigger detection with no future rows.

Structure:
1. Realistic mode mixin (RealisticModeMixin): the starting point _analyze_series_realistic, the main
   incremental loop, the phase handlers (trigger detection, end monitoring), the shared end and suppression
   helpers
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..common.common import effective_start
from ..common.constants import (
    BEGIN_COLS,
    BEGIN_TO_END,
    BEGIN_TO_ACTIVE,
    BEGIN_TO_OPPOSITE,
    END_COLS,
    END_DIFFERENT_END_STRATEGY,
    END_MAX_DURATION,
    END_ALONG_THRESHOLD,
    END_AGAINST_THRESHOLD,
    END_OPPOSITE_TRIGGER,
    END_OPPOSITE_TRIGGER_AS_BEGIN,
    END_OPPOSITE_TRIGGER_SWAPPED,
    END_REASON,
    ACTIVE_COLS,
    SUPPRESSION_BEFORE_EFFECTIVE_START,
    SUPPRESSION_CONSECUTIVE_SAME_TYPE,
    SUPPRESSION_GATE_BLOCKED,
    SUPPRESSION_MINIMUM_DURATION,
    SUPPRESSION_NO_END_CONDITION,
    SUPPRESSION_TEMPORAL_CONSTRAINT,
    VAR3,
    TRIGGER_MINUS_BEGIN,
    TRIGGER_PLUS_BEGIN,
    TRIGGER_VALUE,
    TURNING_POINTS,
    WINDOW_EXPANDING,
    WINDOW_ROLLING,
)
from ..common.schema import EndConfig, TriggerConfig
from .plots.realistic import (
    plot_ended_trigger,
    plot_active_trigger,
    plot_suppressed_trigger,
    setup_debug_dir,
    should_plot_suppressed,
)
from .prints.realistic import print_finalize_realistic_triggers, print_trigger_summary

logger = logging.getLogger(__name__)


# =============================================================================
# REALISTIC MODE MIXIN
# =============================================================================


class RealisticModeMixin:
    """Mixin providing realistic mode analysis methods.

    Processes incrementally with no future rows:
    - Rolling/expanding window trigger detection
    - The gate on the previous row at the activation row
    - Consecutive same-type trigger handling
    - Threshold monitoring against and along the trigger direction
    - Inline trigger_end generation
    """

    # =========================================================================
    # STARTING POINT
    # =========================================================================

    def _analyze_series_realistic(
        self, series_id: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Realistic mode analysis starting point."""
        df = self._realistic_incremental_loop(series_id, df)
        df = self._finalize_realistic_triggers(series_id, df)
        return df

    def _realistic_incremental_loop(
        self, series_id: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Unified incremental loop for realistic (no future rows) processing.

        Processes one window at a time with no future rows.
        End modes are checked in a fixed sequence (first match wins).
        """
        config = self._realistic_config

        trigger_cols, trigger_begin_cols = self._discover_trigger_columns(
            df, config["min_length"]
        )
        df = self._initialize_trigger_columns(df, trigger_cols)
        state = self._init_loop_state(effective_start(df.index, config["lookback"]))
        debug_dir = setup_debug_dir(series_id, self._output_file, config["plot_debug"])

        # Pre-extract column arrays for positional access (avoids per-iteration df.loc)
        self._var3s = df[VAR3].to_numpy(copy=False)

        # Pre-compute column indices for fast .iat writes
        self._col_indices = {
            col: df.columns.get_loc(col) for col in trigger_cols + [END_REASON]
        }

        n = len(df)
        for i in range(config["lookback"] - 1, n):
            self._process_iteration(
                series_id,
                df,
                i,
                config,
                trigger_cols,
                trigger_begin_cols,
                state,
                debug_dir,
            )

        self._handle_active_trigger_at_end(
            series_id, df, config, trigger_cols, state, debug_dir, n
        )

        if config["printout_debug"]:
            print_trigger_summary(series_id, df, state["all_marked_triggers"])

        return df

    # =========================================================================
    # CONFIG & SETUP
    # =========================================================================

    def _load_realistic_config(self, trigger: TriggerConfig) -> Dict[str, Any]:
        """Load all configuration parameters for realistic mode, once per strategy (BaseDetection.__init__): lookback
        and min_length are the base values that method cached first, never read twice.

        End modes are derived from perseries parameters:
        non-null thresholds and true booleans are active.
        The checking sequence is fixed: against_threshold > along_threshold > opposite_trigger > different_end > max_duration.
        """
        end: EndConfig = trigger.replay.end
        max_duration_periods = end.max_duration_periods
        against_threshold_pct = end.against_threshold_pct
        along_threshold_pct = end.along_threshold_pct
        opposite_trigger = end.opposite_trigger_as_begin_strategy
        different_end = end.different_end_strategy

        end_mode_list = []
        if against_threshold_pct is not None:
            end_mode_list.append(END_AGAINST_THRESHOLD)
        if along_threshold_pct is not None:
            end_mode_list.append(END_ALONG_THRESHOLD)
        if opposite_trigger:
            end_mode_list.append(END_OPPOSITE_TRIGGER_AS_BEGIN)
        if different_end:
            end_mode_list.append(END_DIFFERENT_END_STRATEGY)
        if max_duration_periods is not None:
            end_mode_list.append(END_MAX_DURATION)
        plot_debug = trigger.strategy.plot_debug

        return {
            "lookback": self._lookback,
            "window_type": trigger.replay.window_type,
            "min_length": self._min_length,
            "end_mode_list": end_mode_list,
            "min_duration_periods": trigger.replay.min_duration_periods,
            "plot_debug": plot_debug,
            # the window frame's only readers: the debug plots and the different-end mode
            "window_frame": plot_debug or END_DIFFERENT_END_STRATEGY in end_mode_list,
            "printout_debug": trigger.strategy.printout_debug,
            "max_duration_periods": max_duration_periods,
            "against_threshold_pct": against_threshold_pct,
            "along_threshold_pct": along_threshold_pct,
        }

    def _discover_trigger_columns(
        self, df: pd.DataFrame, min_length: int
    ) -> Tuple[List[str], List[str]]:
        """Discover trigger columns from strategy sample and return all trigger cols + trigger cols."""
        before_cols = set(df.columns)
        sample_df = self._generate_trigger_begin(df.iloc[0:min_length].copy())
        all_trigger_cols = [col for col in sample_df.columns if col not in before_cols]

        for col in ACTIVE_COLS + END_COLS:
            if col not in all_trigger_cols:
                all_trigger_cols.append(col)

        trigger_begin_cols = list(BEGIN_COLS)
        return all_trigger_cols, trigger_begin_cols

    def _initialize_trigger_columns(
        self, df: pd.DataFrame, trigger_cols: List[str]
    ) -> pd.DataFrame:
        """Initialize all trigger columns to 0.

        Note: end_reason column is already initialized by BaseDetection.analyze_series.
        """
        for col in trigger_cols:
            df[col] = 0
        return df

    def _init_loop_state(self, effective_start_date: Optional[pd.Timestamp]) -> Dict[str, Any]:
        """Initialize all state variables for the incremental loop, the series' effective start among them."""
        return {
            "effective_start_date": effective_start_date,
            "marked_triggers": {
                TRIGGER_PLUS_BEGIN: set(),
                TRIGGER_MINUS_BEGIN: set(),
            },
            "suppressed_triggers_cache": set(),
            "newest_candidates": {TRIGGER_PLUS_BEGIN: None, TRIGGER_MINUS_BEGIN: None},
            "active_trigger": None,
            "most_recent_marked_date": None,
            "all_marked_triggers": [],
            "trigger_counters": {TRIGGER_PLUS_BEGIN: 0, TRIGGER_MINUS_BEGIN: 0},
            "suppressed_counters": {TRIGGER_PLUS_BEGIN: 0, TRIGGER_MINUS_BEGIN: 0},
        }

    # =========================================================================
    # MAIN LOOP ITERATION
    # =========================================================================

    def _process_iteration(
        self,
        series_id: str,
        df: pd.DataFrame,
        i: int,
        config: Dict[str, Any],
        trigger_cols: List[str],
        trigger_begin_cols: List[str],
        state: Dict[str, Any],
        debug_dir: Optional[Path],
    ) -> None:
        """Process a single iteration of the incremental loop."""
        current_idx = df.index[i]
        current_value = self._var3s[i]

        # Strategy-agnostic candidates from the window's var3 values (the strategy's _begin_candidate_rows); the window
        # frame is built only for its readers (config["window_frame"]).
        start, stop = self._window_bounds(i, config)
        plus_candidates, minus_candidates, turning_count = self._window_candidates(df, start, stop)
        # turning_points of row i: the turning points of the window ending at i, known at i
        df.iat[i, self._col_indices[TURNING_POINTS]] = turning_count
        window_df = self._generate_trigger_begin(self._select_window(df, i, config)) if config["window_frame"] else None

        new_candidate, pending_opposite = self._detect_candidates_from_sets(
            series_id,
            df,
            window_df,
            i,
            current_idx,
            current_value,
            config,
            trigger_cols,
            trigger_begin_cols,
            state,
            debug_dir,
            plus_candidates,
            minus_candidates,
        )

        if not state["active_trigger"]:
            if new_candidate:
                self._activate_trigger(df, i, current_idx, new_candidate, state)
            else:
                return

        self._monitor_end_conditions(
            series_id,
            df,
            i,
            current_idx,
            current_value,
            window_df,
            config,
            trigger_cols,
            state,
            debug_dir,
            pending_opposite,
        )

    def _select_window(
        self, df: pd.DataFrame, i: int, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """The analysis window ending at row i (_window_bounds), as a frame copy."""
        start, stop = self._window_bounds(i, config)
        return df.iloc[start:stop].copy()

    def _window_bounds(self, i: int, config: Dict[str, Any]) -> Tuple[int, int]:
        """The positional [start, stop) of the analysis window ending at row i, by window type."""
        if config["window_type"] == WINDOW_EXPANDING:
            return 0, i + 1
        if config["window_type"] == WINDOW_ROLLING:
            return i - config["lookback"] + 1, i + 1
        raise ValueError(f"Unknown window_type: {config['window_type']}")

    def _window_candidates(self, df: pd.DataFrame, start: int, stop: int) -> Tuple[set, set, int]:
        """The plus and minus begin candidate dates of the window [start, stop) and its turning-point count, from the
        window's var3 values through the strategy's _begin_candidate_rows."""
        plus_rows, minus_rows, turning_rows = self._begin_candidate_rows(self._var3s[start:stop])
        return set(df.index[start + plus_rows]), set(df.index[start + minus_rows]), len(turning_rows)

    # =========================================================================
    # CANDIDATE DETECTION
    # =========================================================================

    def _detect_candidates_from_sets(
        self,
        series_id: str,
        df: pd.DataFrame,
        window_df: Optional[pd.DataFrame],
        i: int,
        current_idx,
        current_value: float,
        config: Dict[str, Any],
        trigger_cols: List[str],
        trigger_begin_cols: List[str],
        state: Dict[str, Any],
        debug_dir: Optional[Path],
        plus_candidates: set,
        minus_candidates: set,
    ) -> Tuple[Optional[Tuple[str, Any]], Optional[Tuple[str, Any]]]:
        """Detect trigger candidates from pre-computed candidate sets.

        Mapping: plus candidates -> trigger_plus_begin, minus candidates -> trigger_minus_begin.
        A type's only candidate is the newest candidate of that type detected so far, taken
        when the window reports it and it is neither marked nor suppressed: once a newer
        candidate has been seen, an older one never activates.

        Returns:
            Tuple of (new_candidate, pending_opposite) where each is (trigger_type, date) or None.
        """
        # Local variables — no carry-over from previous iteration
        new_candidate = None
        pending_opposite = None

        candidate_map = {
            TRIGGER_PLUS_BEGIN: plus_candidates,
            TRIGGER_MINUS_BEGIN: minus_candidates,
        }

        for trigger_type in trigger_begin_cols:
            detected = candidate_map[trigger_type]
            candidate_date = state["newest_candidates"][trigger_type]
            if detected:
                candidate_date = max(detected) if candidate_date is None else max(candidate_date, max(detected))
                state["newest_candidates"][trigger_type] = candidate_date

            if (
                candidate_date is None
                or candidate_date not in detected
                or candidate_date in state["marked_triggers"][trigger_type]
                or candidate_date in state["suppressed_triggers_cache"]
            ):
                continue

            suppression_reason = self._check_suppression_conditions(
                i, candidate_date, trigger_type, state, state["effective_start_date"],
            )

            if suppression_reason:
                self._suppress_trigger(state, trigger_type, candidate_date)
                if config["plot_debug"] and should_plot_suppressed(
                    candidate_date, state["active_trigger"]
                ):
                    plot_suppressed_trigger(
                        series_id,
                        df,
                        window_df,
                        i,
                        current_idx,
                        current_value,
                        candidate_date,
                        trigger_type,
                        suppression_reason,
                        trigger_cols,
                        state,
                        debug_dir,
                    )
                continue

            if state["active_trigger"] and trigger_type == BEGIN_TO_OPPOSITE[
                state["active_trigger"][0]
            ]:
                pending_opposite = (trigger_type, candidate_date)
                continue

            # Reachable only when no active trigger: keep most recent candidate between trigger_plus_begin and trigger_minus_begin
            if new_candidate is None or candidate_date > new_candidate[1]:
                new_candidate = (trigger_type, candidate_date)

        return new_candidate, pending_opposite

    def _check_suppression_conditions(
        self,
        i: int,
        candidate_date,
        trigger_type: str,
        state: Dict[str, Any],
        effective_start_date=None,
    ) -> Optional[str]:
        """Check if a candidate should be suppressed. Returns reason or None.

        The gate reads the begin vector of the series (BaseDetection.analyze_series) at the activation row i: row i-1
        must pass every item that applies to the series, whatever trigger_value records.
        """
        if effective_start_date is not None and candidate_date < effective_start_date:
            return SUPPRESSION_BEFORE_EFFECTIVE_START
        if (
            state["most_recent_marked_date"]
            and candidate_date <= state["most_recent_marked_date"]
        ):
            return SUPPRESSION_TEMPORAL_CONSTRAINT
        if not self._begin_allowed[i]:
            return SUPPRESSION_GATE_BLOCKED
        if state["active_trigger"] and state["active_trigger"][0] == trigger_type:
            return SUPPRESSION_CONSECUTIVE_SAME_TYPE
        return None

    def _suppress_trigger(
        self, state: Dict[str, Any], trigger_type: str, detection_date
    ) -> None:
        """Record a suppression in state tracking."""
        state["suppressed_triggers_cache"].add(detection_date)
        state["suppressed_counters"][trigger_type] += 1

    # =========================================================================
    # TRIGGER ACTIVATION
    # =========================================================================

    def _activate_trigger(
        self,
        df: pd.DataFrame,
        i: int,
        current_idx,
        new_candidate: Tuple[str, Any],
        state: Dict[str, Any],
    ) -> None:
        """Activate a new trigger: mark in DataFrame and update state.

        Active trigger tuple: (trigger_type, detection_date, trigger_begin_value,
                               current_idx, begin_value, begin_row)
        - begin_value: cached var3 at activation (avoids repeated df.loc lookup)
        - begin_row: row number i at activation (avoids df.index.get_loc in duration)
        """
        trigger_type, detection_date = new_candidate

        df.iat[i, self._col_indices[trigger_type]] = 1
        state["marked_triggers"][trigger_type].add(detection_date)
        trigger_begin_value = df.loc[current_idx, TRIGGER_VALUE]
        begin_value = self._var3s[i]

        state["active_trigger"] = (
            trigger_type,
            detection_date,
            trigger_begin_value,
            current_idx,
            begin_value,
            i,
        )
        state["most_recent_marked_date"] = detection_date
        state["trigger_counters"][trigger_type] += 1
        state["all_marked_triggers"].append(
            (
                trigger_type,
                detection_date,
                current_idx,
                trigger_begin_value,
                i,
                state["trigger_counters"][trigger_type],
            )
        )

    # =========================================================================
    # END MONITORING
    # =========================================================================

    def _monitor_end_conditions(
        self,
        series_id: str,
        df: pd.DataFrame,
        i: int,
        current_idx,
        current_value: float,
        window_df: Optional[pd.DataFrame],
        config: Dict[str, Any],
        trigger_cols: List[str],
        state: Dict[str, Any],
        debug_dir: Optional[Path],
        pending_opposite: Optional[Tuple[str, Any]],
    ) -> None:
        """Monitor active trigger for end conditions."""
        begin_row = state["active_trigger"][5]
        duration = i - begin_row

        if duration < config["min_duration_periods"]:
            if pending_opposite:
                self._suppress_pending_opposite(
                    series_id,
                    df,
                    window_df,
                    i,
                    current_idx,
                    current_value,
                    pending_opposite,
                    SUPPRESSION_MINIMUM_DURATION,
                    trigger_cols,
                    state,
                    debug_dir,
                    config["plot_debug"],
                )
            return

        # Check end modes in their fixed sequence (first match wins)
        ended, end_reason = False, None
        for end_mode in config["end_mode_list"]:
            ended, end_reason = self._check_single_end_mode(
                series_id,
                df,
                end_mode,
                i,
                current_idx,
                current_value,
                window_df,
                config,
                trigger_cols,
                state,
                debug_dir,
                pending_opposite,
                duration,
            )
            if ended:
                break

        if ended:
            # "opposite_trigger (swapped)" means active_trigger was already replaced
            if end_reason != END_OPPOSITE_TRIGGER_SWAPPED:
                state["active_trigger"] = None
        elif pending_opposite:
            self._suppress_pending_opposite(
                series_id,
                df,
                window_df,
                i,
                current_idx,
                current_value,
                pending_opposite,
                SUPPRESSION_NO_END_CONDITION,
                trigger_cols,
                state,
                debug_dir,
                config["plot_debug"],
            )

    def _suppress_pending_opposite(
        self,
        series_id: str,
        df: pd.DataFrame,
        window_df: Optional[pd.DataFrame],
        i: int,
        current_idx,
        current_value: float,
        pending_opposite: Tuple[str, Any],
        reason: str,
        trigger_cols: List[str],
        state: Dict[str, Any],
        debug_dir: Optional[Path],
        plot_debug: bool,
    ) -> None:
        """Suppress a pending opposite trigger."""
        pend_type, pend_date = pending_opposite
        self._suppress_trigger(state, pend_type, pend_date)
        if plot_debug:
            plot_suppressed_trigger(
                series_id,
                df,
                window_df,
                i,
                current_idx,
                current_value,
                pend_date,
                pend_type,
                reason,
                trigger_cols,
                state,
                debug_dir,
            )

    def _check_single_end_mode(
        self,
        series_id: str,
        df: pd.DataFrame,
        end_mode: str,
        i: int,
        current_idx,
        current_value: float,
        window_df: Optional[pd.DataFrame],
        config: Dict[str, Any],
        trigger_cols: List[str],
        state: Dict[str, Any],
        debug_dir: Optional[Path],
        pending_opposite: Optional[Tuple[str, Any]],
        duration: int,
    ) -> Tuple[bool, Optional[str]]:
        """Check a single end mode. Returns (ended, reason)."""
        trigger_type, begin_date, _, _, begin_value, _ = state["active_trigger"]

        if end_mode == END_MAX_DURATION:
            if duration >= config["max_duration_periods"]:
                return self._conclude_trigger(
                    series_id,
                    df,
                    i,
                    current_idx,
                    window_df,
                    config,
                    trigger_cols,
                    state,
                    debug_dir,
                    duration,
                    END_MAX_DURATION,
                )

        elif end_mode == END_AGAINST_THRESHOLD:
            if self._check_against_threshold(
                begin_value, current_value, trigger_type, config["against_threshold_pct"]
            ):
                return self._conclude_trigger(
                    series_id,
                    df,
                    i,
                    current_idx,
                    window_df,
                    config,
                    trigger_cols,
                    state,
                    debug_dir,
                    duration,
                    END_AGAINST_THRESHOLD,
                )

        elif end_mode == END_ALONG_THRESHOLD:
            if self._check_along_threshold(
                begin_value, current_value, trigger_type, config["along_threshold_pct"]
            ):
                return self._conclude_trigger(
                    series_id,
                    df,
                    i,
                    current_idx,
                    window_df,
                    config,
                    trigger_cols,
                    state,
                    debug_dir,
                    duration,
                    END_ALONG_THRESHOLD,
                )

        elif end_mode == END_OPPOSITE_TRIGGER_AS_BEGIN:
            if pending_opposite:
                return self._handle_opposite_trigger_end(
                    series_id,
                    df,
                    i,
                    current_idx,
                    window_df,
                    config,
                    trigger_cols,
                    state,
                    debug_dir,
                    pending_opposite,
                    duration,
                )

        elif end_mode == END_DIFFERENT_END_STRATEGY:
            window_df_with_end = self._generate_trigger_end(
                window_df, state["active_trigger"]
            )
            end_col = BEGIN_TO_END[trigger_type]
            if self._trigger_exists_after(window_df_with_end, end_col, begin_date):
                return self._conclude_trigger(
                    series_id,
                    df,
                    i,
                    current_idx,
                    window_df,
                    config,
                    trigger_cols,
                    state,
                    debug_dir,
                    duration,
                    END_DIFFERENT_END_STRATEGY,
                )

        return False, None

    def _conclude_trigger(
        self,
        series_id: str,
        df: pd.DataFrame,
        i: int,
        current_idx,
        window_df: Optional[pd.DataFrame],
        config: Dict[str, Any],
        trigger_cols: List[str],
        state: Dict[str, Any],
        debug_dir: Optional[Path],
        duration: int,
        end_reason: str,
        detection_end_date=None,
    ) -> Tuple[bool, str]:
        """End the active trigger, optionally plot, and return the end result."""
        self._end_trigger(df, i, state["active_trigger"][0], end_reason)
        if config["plot_debug"]:
            plot_ended_trigger(
                series_id,
                df,
                window_df,
                i,
                current_idx,
                trigger_cols,
                state,
                debug_dir,
                duration,
                end_reason,
                detection_end_date=detection_end_date,
            )
        return True, end_reason

    def _end_trigger(
        self, df: pd.DataFrame, i: int, trigger_type: str, reason: str
    ) -> None:
        """End the active trigger by marking its end column."""
        end_col = BEGIN_TO_END[trigger_type]
        df.iat[i, self._col_indices[end_col]] = 1
        df.iat[i, self._col_indices[END_REASON]] = reason

    def _check_against_threshold(
        self,
        begin_value: float,
        current_value: float,
        trigger_type: str,
        threshold_pct: float,
    ) -> bool:
        """Check if the move against the trigger direction exceeds threshold.

        Uses var3 for decision logic (not trigger_value).
        For plus triggers: change against the direction = value DOWN
        For minus triggers: change against the direction = value UP
        """
        move_pct = (current_value - begin_value) / begin_value
        if trigger_type == TRIGGER_PLUS_BEGIN:
            return move_pct <= -abs(threshold_pct)
        return move_pct >= abs(threshold_pct)

    def _check_along_threshold(
        self,
        begin_value: float,
        current_value: float,
        trigger_type: str,
        threshold_pct: float,
    ) -> bool:
        """Check if the move along the trigger direction exceeds threshold.

        Uses var3 for decision logic (not trigger_value).
        For plus triggers: change along the direction = value UP
        For minus triggers: change along the direction = value DOWN
        """
        if trigger_type == TRIGGER_PLUS_BEGIN:
            move_pct = (current_value - begin_value) / begin_value
        else:
            move_pct = (begin_value - current_value) / begin_value
        return move_pct >= abs(threshold_pct)

    def _handle_opposite_trigger_end(
        self,
        series_id: str,
        df: pd.DataFrame,
        i: int,
        current_idx,
        window_df: Optional[pd.DataFrame],
        config: Dict[str, Any],
        trigger_cols: List[str],
        state: Dict[str, Any],
        debug_dir: Optional[Path],
        pending_opposite: Tuple[str, Any],
        duration: int,
    ) -> Tuple[bool, str]:
        """Handle an opposite trigger as the end of the active one."""
        trigger_type = state["active_trigger"][0]
        end_reason = END_OPPOSITE_TRIGGER_SWAPPED

        if config["plot_debug"]:
            plot_ended_trigger(
                series_id,
                df,
                window_df,
                i,
                current_idx,
                trigger_cols,
                state,
                debug_dir,
                duration,
                end_reason,
                detection_end_date=pending_opposite[1],
            )

        self._end_trigger(df, i, trigger_type, END_OPPOSITE_TRIGGER)
        self._activate_trigger(df, i, current_idx, pending_opposite, state)

        return True, end_reason

    def _trigger_exists_after(
        self, window_df: pd.DataFrame, trigger_col: str, after_date
    ) -> bool:
        """Check if any trigger exists after a given date."""
        if trigger_col not in window_df.columns:
            return False
        return (window_df.loc[window_df.index > after_date, trigger_col] == 1).any()

    # =========================================================================
    # ACTIVE TRIGGER / FINALIZATION
    # =========================================================================

    def _handle_active_trigger_at_end(
        self,
        series_id: str,
        df: pd.DataFrame,
        config: Dict[str, Any],
        trigger_cols: List[str],
        state: Dict[str, Any],
        debug_dir: Optional[Path],
        n: int,
    ) -> None:
        """Handle an active trigger remaining at series end."""
        if not state["active_trigger"]:
            return

        trigger_type, begin_date, _, _, _, begin_row = state["active_trigger"]
        active_col = BEGIN_TO_ACTIVE[trigger_type]
        df.iat[begin_row, self._col_indices[active_col]] = 1

        if config["plot_debug"]:
            window_df_active = self._generate_trigger_begin(self._select_window(df, n - 1, config))
            plot_active_trigger(
                series_id, df, window_df_active, trigger_cols, state, debug_dir, n
            )

    def _finalize_realistic_triggers(
        self, series_id: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Final cleanup for realistic mode."""
        if self._realistic_config["printout_debug"]:
            print_finalize_realistic_triggers(series_id, df)
        return df
