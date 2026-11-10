"""
Signal selection for grouped time series triggers.

This module provides the SignalSelection class that:
- Accumulates per-series values and triggers from trigger HDF5 output
- Groups triggers by signal-seeded fixed windows with time overlap and correlation-based membership
- Computes slope, signal marking, and group aggregates

Structure:
1. SignalSelection class (public API: add_series_values, add_series_triggers, build_signal_window_groups)
2. Group building helpers (_prepare_all_triggers, _assign_group_ids, _verify_signal_update, _recompute_correlations_for_updated_signals)
3. Computation helpers (_precompute_bounds_and_slopes, _precompute_window_bounds, _precompute_slopes, _pair_cache_key, _compute_corr_from_bounds)
"""

import bisect
import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ..common.common import ensure_datetime_series
from ..common.constants import (
    CORR_TO_SIGNAL,
    GROUP_ID,
    IS_PLUS_TRIGGER,
    IS_SIGNAL,
    IS_TRUE,
    NUM_SERIES,
    REPLAY_REALISTIC,
    VAR3,
    SERIES_ID,
    SERIES_IDS,
    SIGNAL_EARLIEST_BEGIN,
    SIGNAL_EARLIEST_BEGIN_HIGHEST_SLOPE,
    SELECTION_FALLBACK_PAIRS_LOGGED,
    SIGNAL_TABLE_COLUMNS,
    SLOPE,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
    TRIGGER_END_VALUES,
    TRIGGER_START_VALUES,
    TRIGGER_VALUE,
)
from ..common.schema import SignalConfig
from .prints.selection import print_accumulated


class SignalSelection:
    """Groups triggers into signal-window groups based on time overlap and correlation."""

    def __init__(self, signal: SignalConfig, replay_mode: str):
        """Cache the selection parameters once: signal.period_back, corr_threshold and signal_criterion, and the
        point-in-time rule, true for a realistic replay_mode: grouping then takes eligible signals by (signal_begin,
        group_id), known at each trigger's begin, never by signal_end."""
        self._period_back = signal.period_back
        self._corr_threshold = signal.corr_threshold
        self._signal_criterion = signal.signal_criterion
        self._point_in_time = replay_mode == REPLAY_REALISTIC
        # Map series_id -> DataFrame with value columns (var3, trigger_value)
        self._series: Dict[str, pd.DataFrame] = {}
        # Pre-extracted var3 arrays for fast slope/correlation computation
        self._series_arrays: Dict[str, np.ndarray] = {}
        self._triggers: List[pd.DataFrame] = []
        # Correlation per series pair, begin date and period, kept until the build ends. It grows faster than the
        # series count (about +0.9 GiB at 1k series). If memory becomes the limit, drop a cached correlation once its
        # begin date leaves the period: it is a cache, so the outputs stay the same.
        self._correlation_cache: Dict[Tuple[str, str, pd.Timestamp, int], float] = {}
        # Shared-index test per unordered series pair (series are final once correlations start)
        self._shared_index_cache: Dict[Tuple[str, str], bool] = {}
        # (signal, join) pairs whose correlation fell back to date alignment -> computations, in first-seen sequence:
        # one summary record per build instead of one per computation
        self._fallback_pairs: Dict[Tuple[str, str], int] = {}

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def add_series_values(self, series_id: str, signal_df: pd.DataFrame) -> None:
        """Store value columns (var3, trigger_value) indexed by date for correlation."""
        if signal_df is None or len(signal_df) == 0:
            return
        # Expect index to be DatetimeIndex; if not, convert it (a failed conversion raises)
        if not isinstance(signal_df.index, pd.DatetimeIndex):
            signal_df = signal_df.copy()
            signal_df.index = pd.to_datetime(signal_df.index)

        if VAR3 not in signal_df.columns:
            return

        cols_to_store = [VAR3]
        if TRIGGER_VALUE in signal_df.columns:
            cols_to_store.append(TRIGGER_VALUE)

        stored = signal_df[cols_to_store].dropna(subset=[VAR3])
        self._series[str(series_id)] = stored
        self._series_arrays[str(series_id)] = stored[VAR3].to_numpy(
            dtype=float, copy=False
        )

    def add_series_triggers(
        self, series_id: str, triggers_df: Optional[pd.DataFrame]
    ) -> None:
        """Accumulate per-series triggers as-is for later grouping.

        Expected columns in triggers_df:
        - series_id, trigger_begin_dates, trigger_end_dates,
          trigger_start_values, trigger_end_values, is_plus_trigger, is_true
        """
        if triggers_df is None or len(triggers_df) == 0:
            return

        base_cols = [
            SERIES_ID,
            TRIGGER_BEGIN_DATES,
            TRIGGER_END_DATES,
            TRIGGER_START_VALUES,
            TRIGGER_END_VALUES,
            IS_PLUS_TRIGGER,
            IS_TRUE,
        ]
        cols = [c for c in base_cols if c in triggers_df.columns]
        self._triggers.append(triggers_df[cols].copy())

    def build_signal_window_groups(
        self, printout_io: bool = False, full_output: bool = False
    ) -> pd.DataFrame:
        """Group triggers by signal-seeded fixed windows with time overlap and correlation-based membership.

        Returns a single dataframe (groups_df) with one row per original trigger:
          - group_id, is_signal
          - On signal rows only: num_series, series_ids (comma-separated)
        """
        if not self._triggers:
            return pd.DataFrame(columns=SIGNAL_TABLE_COLUMNS)

        all_triggers = self._prepare_all_triggers()
        if printout_io:
            print_accumulated(all_triggers, full_output=full_output)

        # Keeps, per series, the period_back rows before its row at the earliest begin through latest_end (counted in rows,
        # as _precompute_window_bounds counts them, so gaps in the dates never shorten a window)
        self._trim_series_to_trigger_range(all_triggers, self._period_back)

        all_triggers = self._assign_group_ids(all_triggers)
        if all_triggers.empty:
            return pd.DataFrame(columns=SIGNAL_TABLE_COLUMNS)

        members_df = self._attach_group_aggregates(all_triggers)

        members_df = (
            members_df[SIGNAL_TABLE_COLUMNS]
            .sort_values([GROUP_ID, TRIGGER_BEGIN_DATES, SERIES_ID])
            .reset_index(drop=True)
        )

        if self._fallback_pairs:
            logger.warning(
                "DatetimeIndex mismatch: %d correlations over %d series pairs fell back to date-aligned "
                "correlation; first pairs: %s",
                sum(self._fallback_pairs.values()),
                len(self._fallback_pairs),
                list(self._fallback_pairs)[:SELECTION_FALLBACK_PAIRS_LOGGED],
            )
        self._release_build_state()
        return members_df

    def _release_build_state(self) -> None:
        """Release internal state used only during build_signal_window_groups."""
        del self._triggers
        del self._correlation_cache
        del self._shared_index_cache
        del self._series_arrays
        del self._fallback_pairs

    # =========================================================================
    # GROUP BUILDING HELPERS
    # =========================================================================

    def _prepare_all_triggers(self) -> pd.DataFrame:
        """Concatenate all triggers, normalize date columns, and sort for deterministic processing."""
        all_triggers = pd.concat(self._triggers, ignore_index=True)
        self._triggers = []  # Free per-series list; data now lives in all_triggers

        all_triggers[TRIGGER_BEGIN_DATES] = ensure_datetime_series(
            all_triggers[TRIGGER_BEGIN_DATES]
        )
        all_triggers[TRIGGER_END_DATES] = ensure_datetime_series(
            all_triggers[TRIGGER_END_DATES]
        )
        # Normalize series_id to str once to avoid repeated str() casts downstream
        all_triggers[SERIES_ID] = all_triggers[SERIES_ID].astype(str)

        all_triggers.sort_values(
            [TRIGGER_BEGIN_DATES, SERIES_ID], inplace=True, kind="mergesort"
        )
        all_triggers.reset_index(drop=True, inplace=True)

        return all_triggers

    def _trim_series_to_trigger_range(
        self,
        all_triggers: pd.DataFrame,
        period_back: int,
    ) -> None:
        """Trim stored series to the rows required by downstream consumers.

        Slope/correlation read period_back rows before the row at or before each trigger_begin_dates
        (_precompute_window_bounds). Variation plotting needs rows from trigger_begin_dates to
        trigger_end_dates. Each series keeps the rows from period_back rows before its row at or
        before the earliest begin through latest_end; later begins sit on later rows, so their
        windows are kept too. The look-back counts rows, never calendar days.

        Modifies self._series and self._series_arrays in place.
        """

        earliest_begin = all_triggers[TRIGGER_BEGIN_DATES].min()
        latest_end = all_triggers[TRIGGER_END_DATES].max()

        for sid in list(self._series.keys()):
            df = self._series[sid]
            begin_row = int(df.index.searchsorted(earliest_begin, side="right")) - 1
            first_row = max(0, begin_row - period_back)
            trimmed = df.iloc[first_row:]
            trimmed = trimmed.loc[trimmed.index <= latest_end]
            if trimmed.empty:
                del self._series[sid]
                self._series_arrays.pop(sid, None)
                continue
            self._series[sid] = trimmed
            self._series_arrays[sid] = trimmed[VAR3].to_numpy(
                dtype=float, copy=False
            )

    def _assign_group_ids(self, all_triggers: pd.DataFrame) -> pd.DataFrame:
        """Assign group IDs based on time overlap, correlation threshold, and signal_criterion.

        For each trigger, computes absolute slope, checks eligible signals by time overlap,
        and assigns the trigger to the first group whose signal passes the correlation threshold.
        Dynamically updates signals based on signal_criterion to ensure correct correlation targets.
        Sets is_signal column directly to avoid a redundant post-hoc pass.

        Args:
            all_triggers: DataFrame with all triggers (must be sorted by trigger_begin_dates)

        Returns:
            DataFrame with group_id, slope, is_signal, and corr_to_signal columns added
        """
        corr_threshold = self._corr_threshold
        period_back = self._period_back
        point_in_time = self._point_in_time
        n = len(all_triggers)

        # Pre-compute all window bounds and slopes before the loop
        slopes, bounds_map = self._precompute_bounds_and_slopes(all_triggers, period_back)

        # Active signals deque sorted by signal_end for efficient expiration
        active_signals: Deque[Dict[str, Any]] = deque()
        group_ids: List[int] = [0] * n
        slope_vals: List[float] = [np.nan] * n
        corr_vals: List[float] = [np.nan] * n
        group_signal_idx: Dict[int, int] = {}
        # Track groups whose signal was updated (need correlation recomputation)
        signal_updated_groups: set = set()

        group_id_counter = 0
        for idx in range(n):
            sid = all_triggers.at[idx, SERIES_ID]
            begin = all_triggers.at[idx, TRIGGER_BEGIN_DATES]
            end = all_triggers.at[idx, TRIGGER_END_DATES]

            # Pre-computed slope: O(1) lookup instead of O(period_back) polyfit
            s = slopes[idx]
            slope_vals[idx] = s

            # Drop expired signals from left
            while active_signals and active_signals[0]["signal_end"] <= begin:
                active_signals.popleft()

            # Find eligible signals where begin is in [signal_begin, signal_end)
            eligible = [
                signal
                for signal in active_signals
                if signal["signal_begin"] <= begin < signal["signal_end"]
            ]
            if point_in_time:
                eligible.sort(key=lambda signal: (signal["signal_begin"], signal[GROUP_ID]))

            assigned_gid = 0
            assigned_corr = np.nan
            for signal in eligible:
                signal_sid = signal["signal_series_id"]
                r = self._lookup_or_compute_correlation(
                    signal_sid, sid, begin, period_back, bounds_map
                )

                # Assign only to the first eligible signal passing threshold: earliest (signal_begin, group_id)
                # when point_in_time, else earliest signal_end, newest first among equal ends
                if r is not None and abs(r) >= corr_threshold:
                    assigned_gid = signal[GROUP_ID]
                    assigned_corr = r
                    break

            if assigned_gid == 0:
                # No group found, create a new signal
                group_id_counter += 1
                assigned_gid = group_id_counter
                new_signal = {
                    GROUP_ID: assigned_gid,
                    "signal_begin": begin,
                    "signal_end": end,
                    "signal_series_id": sid,
                    "signal_slope": s if np.isfinite(s) else np.nan,
                }
                insert_pos = bisect.bisect_left(
                    [signal["signal_end"] for signal in active_signals], end
                )
                active_signals.insert(insert_pos, new_signal)
                group_signal_idx[assigned_gid] = idx
                # Signal correlates perfectly with itself
                corr_vals[idx] = 1.0
            else:
                # Group found, check if trigger qualifies for signal update
                old_signal_idx = group_signal_idx.get(assigned_gid)
                self._verify_signal_update(
                    active_signals, assigned_gid, sid, begin, end, s, group_signal_idx, idx
                )
                # If signal changed, mark group for correlation recomputation
                if group_signal_idx[assigned_gid] != old_signal_idx:
                    signal_updated_groups.add(assigned_gid)
                # Store correlation inline
                corr_vals[idx] = assigned_corr

            group_ids[idx] = assigned_gid

        all_triggers[GROUP_ID] = group_ids
        all_triggers[SLOPE] = slope_vals

        is_signal_vals = [False] * n
        for signal_idx in group_signal_idx.values():
            is_signal_vals[signal_idx] = True
        all_triggers[IS_SIGNAL] = is_signal_vals
        all_triggers[CORR_TO_SIGNAL] = corr_vals

        # Recompute correlations only for groups where signal was updated
        if signal_updated_groups:
            self._recompute_correlations_for_updated_signals(
                all_triggers, signal_updated_groups, period_back, bounds_map
            )

        return all_triggers

    def _verify_signal_update(
        self,
        active_signals: Deque[Dict[str, Any]],
        group_id: int,
        sid: str,
        begin: pd.Timestamp,
        end: pd.Timestamp,
        slope: Optional[float],
        group_signal_idx: Dict[int, int],
        idx: int,
    ) -> None:
        """Update signal of a group if this trigger qualifies under signal_criterion.

        Args:
            active_signals: Deque of active signal items (sorted by signal_end)
            group_id: Group ID the trigger was assigned to
            sid: Series ID of the joining trigger
            begin: Trigger begin date of the joining trigger
            end: Trigger end date of the joining trigger
            slope: Absolute slope of the joining trigger (may be None)
            group_signal_idx: Maps group_id to the DataFrame row index of its current signal
            idx: DataFrame row index of the joining trigger
        """
        signal_criterion = self._signal_criterion

        if signal_criterion == SIGNAL_EARLIEST_BEGIN:
            return
        elif signal_criterion == SIGNAL_EARLIEST_BEGIN_HIGHEST_SLOPE:
            if slope is None or np.isnan(slope):
                return

            for i, signal in enumerate(active_signals):
                if signal[GROUP_ID] != group_id:
                    continue

                # Only compete if same begin date
                if begin != signal["signal_begin"]:
                    return

                current_slope = signal.get("signal_slope", np.nan)
                needs_update = (
                    np.isnan(current_slope)
                    or slope > current_slope
                    or (slope == current_slope and sid < signal["signal_series_id"])
                )

                if not needs_update:
                    return

                old_end = signal["signal_end"]
                signal["signal_series_id"] = sid
                signal["signal_slope"] = slope
                signal["signal_end"] = end
                group_signal_idx[group_id] = idx

                # Reposition in deque if signal_end changed (maintains sort order)
                if end != old_end:
                    del active_signals[i]
                    insert_pos = bisect.bisect_left(
                        [s["signal_end"] for s in active_signals], end
                    )
                    active_signals.insert(insert_pos, signal)

                return

    def _recompute_correlations_for_updated_signals(
        self,
        all_triggers: pd.DataFrame,
        signal_updated_groups: set,
        period_back: int,
        bounds_map: Dict[Tuple[str, pd.Timestamp], Optional[Tuple[int, int]]],
    ) -> None:
        """Recompute correlations only for groups where the signal was updated.

        Called after the main _assign_group_ids loop when _verify_signal_update
        changed a signal. Only the affected groups need recomputation.

        Args:
            all_triggers: DataFrame with group assignments
            signal_updated_groups: Set of group IDs whose signal changed
            period_back: Lookback window size
            bounds_map: Pre-computed window bounds for O(1) lookup
        """
        for gid in signal_updated_groups:
            group_mask = all_triggers[GROUP_ID] == gid
            group = all_triggers.loc[group_mask]
            signal_rows = group[group[IS_SIGNAL]]
            if signal_rows.empty:
                continue

            signal_series_id = signal_rows.iloc[0][SERIES_ID]

            for idx in group.index:
                trigger_series_id = all_triggers.at[idx, SERIES_ID]
                trigger_begin = all_triggers.at[idx, TRIGGER_BEGIN_DATES]

                if trigger_series_id == signal_series_id:
                    all_triggers.at[idx, CORR_TO_SIGNAL] = 1.0
                else:
                    corr_value = self._lookup_or_compute_correlation(
                        signal_series_id, trigger_series_id,
                        trigger_begin, period_back, bounds_map,
                    )
                    all_triggers.at[idx, CORR_TO_SIGNAL] = corr_value

    def _attach_group_aggregates(self, all_triggers: pd.DataFrame) -> pd.DataFrame:
        """Attach num_series and series_ids columns to all rows within each group.

        Series IDs are listed by decreasing slope value within each group.

        Args:
            all_triggers: DataFrame with group_id, is_signal, slope columns

        Returns:
            DataFrame with NUM_SERIES and SERIES_IDS columns added
        """
        members_df = all_triggers.copy()
        members_df[NUM_SERIES] = pd.NA
        members_df[SERIES_IDS] = pd.NA

        for gid, g in members_df.groupby(GROUP_ID, sort=True):
            series_unique = list(pd.unique(g[SERIES_ID]))
            # Sort series_ids by decreasing slope value
            if SLOPE in g.columns:
                series_slopes = (
                    g.groupby(SERIES_ID)[SLOPE].max().sort_values(ascending=False)
                )
                series_by_slope = series_slopes.index.tolist()
                series_with_slope = [
                    s for s in series_by_slope if pd.notna(series_slopes[s])
                ]
                series_without_slope = [
                    s for s in series_unique if s not in series_with_slope
                ]
                series_unique = series_with_slope + series_without_slope
            series_ids_str = ",".join(map(str, series_unique))
            members_df.loc[g.index, NUM_SERIES] = int(len(series_unique))
            members_df.loc[g.index, SERIES_IDS] = series_ids_str

        return members_df

    # =========================================================================
    # COMPUTATION HELPERS
    # =========================================================================

    def _precompute_bounds_and_slopes(
        self,
        all_triggers: pd.DataFrame,
        period_back: int,
    ) -> Tuple[List[float], Dict[Tuple[str, pd.Timestamp], Optional[Tuple[int, int]]]]:
        """Pre-compute all window bounds and slopes before the assignment loop.

        Slopes depend only on (series_id, begin_date, period_back) -- all known
        before the loop. Window bounds are reused for lazy correlation lookups.

        Args:
            all_triggers: Sorted triggers DataFrame
            period_back: Lookback window size

        Returns:
            (slopes, bounds_map) where slopes[idx] is the absolute slope for trigger idx
            and bounds_map[(sid, begin)] is (start_idx, end_idx) or None
        """
        unique_sids = list(pd.unique(all_triggers[SERIES_ID].astype(str)))
        unique_begins = pd.unique(all_triggers[TRIGGER_BEGIN_DATES])

        bounds_map = self._precompute_window_bounds(
            unique_sids, unique_begins, period_back
        )
        slopes = self._precompute_slopes(all_triggers, bounds_map)

        return slopes, bounds_map

    def _precompute_window_bounds(
        self,
        unique_series_ids: List[str],
        unique_begins: np.ndarray,
        period_back: int,
    ) -> Dict[Tuple[str, pd.Timestamp], Optional[Tuple[int, int]]]:
        """Vectorized window bounds for all (series_id, begin_date) pairs.

        Uses np.searchsorted per series across all begin dates at once,
        replacing per-call pd.Index.get_indexer([val], method="ffill").

        Equivalence: searchsorted(side="right") - 1 finds the largest index
        value <= val on a sorted array, identical to get_indexer with ffill.

        Args:
            unique_series_ids: Unique series IDs from triggers
            unique_begins: Unique trigger_begin_dates from triggers
            period_back: Lookback window size

        Returns:
            Dict mapping (series_id, begin_timestamp) to (start_idx, end_idx) or None
        """
        bounds_map: Dict[Tuple[str, pd.Timestamp], Optional[Tuple[int, int]]] = {}

        begins_i8 = unique_begins.astype("int64")

        for sid in unique_series_ids:
            series_df = self._series.get(sid)
            if series_df is None or not isinstance(series_df.index, pd.DatetimeIndex):
                continue

            ts_i8 = series_df.index.values.astype("int64")

            # Vectorized: find largest index value <= each begin date
            end_indices = np.searchsorted(ts_i8, begins_i8, side="right") - 1
            start_indices = np.maximum(0, end_indices - period_back)
            valid = (end_indices >= 0) & (end_indices - start_indices + 1 >= 2)

            for k in range(len(unique_begins)):
                begin_ts = pd.Timestamp(unique_begins[k])
                if valid[k]:
                    bounds_map[(sid, begin_ts)] = (
                        int(start_indices[k]),
                        int(end_indices[k]),
                    )
                else:
                    bounds_map[(sid, begin_ts)] = None

        return bounds_map

    def _precompute_slopes(
        self,
        all_triggers: pd.DataFrame,
        bounds_map: Dict[Tuple[str, pd.Timestamp], Optional[Tuple[int, int]]],
    ) -> List[float]:
        """Batch slope computation for all triggers using pre-computed bounds.

        Args:
            all_triggers: Sorted triggers DataFrame
            bounds_map: Pre-computed window bounds

        Returns:
            List of absolute slopes (np.nan for non-computable)
        """
        n = len(all_triggers)
        slopes: List[float] = [np.nan] * n

        for idx in range(n):
            sid = all_triggers.at[idx, SERIES_ID]
            begin = all_triggers.at[idx, TRIGGER_BEGIN_DATES]

            arr = self._series_arrays.get(sid)
            if arr is None:
                continue

            bounds = bounds_map.get((sid, begin))
            if bounds is None:
                continue

            s_idx, e_idx = bounds
            y_raw = arr[s_idx : e_idx + 1]

            if not np.isfinite(y_raw[-1]):
                continue
            mask = np.isfinite(y_raw)
            if mask.sum() < 2:
                continue

            x = np.arange(len(y_raw), dtype=float)[mask]
            ref = float(y_raw[-1])
            # Normalize to join_begin so slopes are comparable across series
            y = (y_raw[mask] / ref) - 1.0
            slope = np.polyfit(x, y, 1)[0]
            if not np.isfinite(slope):
                continue
            slopes[idx] = float(abs(slope))

        return slopes

    def _lookup_or_compute_correlation(
        self,
        signal_sid: str,
        trigger_sid: str,
        begin: pd.Timestamp,
        period_back: int,
        bounds_map: Dict[Tuple[str, pd.Timestamp], Optional[Tuple[int, int]]],
    ) -> Optional[float]:
        """Look up cached correlation or compute and cache it.

        Args:
            signal_sid: Series ID of the signal trigger
            trigger_sid: Series ID of the joining trigger
            begin: Trigger begin date
            period_back: Lookback window size
            bounds_map: Pre-computed window bounds

        Returns:
            Pearson correlation coefficient or None if insufficient data
        """
        cache_key = self._pair_cache_key(signal_sid, trigger_sid, begin, period_back)
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]

        signal_bounds = bounds_map.get((signal_sid, begin))
        if signal_bounds is None:
            return None

        r = self._compute_corr_from_bounds(signal_sid, trigger_sid, signal_bounds)
        if r is not None:
            self._correlation_cache[cache_key] = r
        return r

    def _pair_cache_key(
        self, sid_a: str, sid_b: str, begin: pd.Timestamp, period_back: int
    ) -> Tuple[str, str, pd.Timestamp, int]:
        """Create the cache key for a pair of series.

        corr(A, B) and corr(B, A) share one item only when both series lie on one index (the fast
        path, where the orientations differ at most by rounding); otherwise each orientation has its
        own item, because the fallback correlation runs over the signal's own window.
        """
        if sid_a <= sid_b or not self._shares_index(sid_a, sid_b):
            return (sid_a, sid_b, begin, period_back)
        return (sid_b, sid_a, begin, period_back)

    def _shares_index(self, sid_a: str, sid_b: str) -> bool:
        """Both series are stored on one DatetimeIndex.

        Memoized per unordered pair (the test is symmetric), so a correlation-cache hit stays O(1).
        """
        pair = (sid_a, sid_b) if sid_a <= sid_b else (sid_b, sid_a)
        shared = self._shared_index_cache.get(pair)
        if shared is None:
            a, b = self._series.get(sid_a), self._series.get(sid_b)
            shared = a is not None and b is not None and len(a.index) == len(b.index) and bool((a.index == b.index).all())
            self._shared_index_cache[pair] = shared
        return shared

    def _compute_corr_from_bounds(
        self,
        signal_series_id: str,
        join_series_id: str,
        bounds: Tuple[int, int],
    ) -> Optional[float]:
        """Compute correlation between signal and join series using pre-looked-up bounds.

        Bounds come from the signal's DatetimeIndex. If both series share the
        same DatetimeIndex, fast positional slicing is used. Otherwise, falls
        back to date-aligned correlation via index intersection.

        Args:
            signal_series_id: Series ID of the signal trigger
            join_series_id: Series ID of the joining trigger
            bounds: Pre-computed (start_idx, end_idx) from the signal's index

        Returns:
            Pearson correlation coefficient or None if insufficient data
        """
        signal_arr = self._series_arrays.get(signal_series_id)
        join_arr = self._series_arrays.get(join_series_id)
        if signal_arr is None or join_arr is None:
            return None

        signal_df = self._series.get(signal_series_id)
        join_df = self._series.get(join_series_id)
        if signal_df is None or join_df is None:
            return None

        start_idx, end_idx = bounds

        # Fast path: identical indexes allow direct positional slicing
        signal_index = signal_df.index
        join_index = join_df.index
        if self._shares_index(signal_series_id, join_series_id):
            signal_vals = signal_arr[start_idx : end_idx + 1]
            join_vals = join_arr[start_idx : end_idx + 1]
        else:
            # Fallback: date-aligned correlation via index intersection, counted for the build's summary record
            pair = (signal_series_id, join_series_id)
            self._fallback_pairs[pair] = self._fallback_pairs.get(pair, 0) + 1
            window_dates = signal_index[start_idx : end_idx + 1]
            common_dates = window_dates.intersection(join_index)
            if len(common_dates) < 2:
                return None
            signal_vals = signal_df.loc[common_dates, VAR3].to_numpy(
                dtype=float
            )
            join_vals = join_df.loc[common_dates, VAR3].to_numpy(
                dtype=float
            )

        mask = np.isfinite(signal_vals) & np.isfinite(join_vals)
        if mask.sum() < 2:
            return None
        r = np.corrcoef(signal_vals[mask], join_vals[mask])[0, 1]
        return None if np.isnan(r) else float(r)
