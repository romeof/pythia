"""
Ideal mode analysis for trigger strategies.

This module provides the IdealModeMixin class that implements trigger detection
using full series visibility (future rows visible).

Structure:
1. IdealModeMixin class with starting point _analyze_series_ideal
2. Latest trigger begin lookup (_get_latest_trigger_begin_idx)
3. Duplicate cleaning methods
4. Active trigger flag methods
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common.constants import (
    BEGIN_COLS,
    END_IDEAL_MODE,
    END_REASON,
    ACTIVE_TRIGGER_MINUS_BEGIN,
    ACTIVE_TRIGGER_PLUS_BEGIN,
    TRIGGER_MINUS_BEGIN,
    TRIGGER_MINUS_END,
    TRIGGER_PLUS_BEGIN,
    TRIGGER_PLUS_END,
    DIRECTION_MINUS,
    DIRECTION_PLUS,
)

logger = logging.getLogger(__name__)

# Trigger pairs for deduplication: (begin_col, end_col, pair_type_label)
_TRIGGER_PAIRS = [
    (TRIGGER_PLUS_BEGIN, TRIGGER_PLUS_END, DIRECTION_PLUS),
    (TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END, DIRECTION_MINUS),
]

# Mapping from pair_type label to (begin_col, end_col) for final pair write-back
_PAIR_TYPE_TO_COLS = {
    DIRECTION_PLUS: (TRIGGER_PLUS_BEGIN, TRIGGER_PLUS_END),
    DIRECTION_MINUS: (TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END),
}


class IdealModeMixin:
    """Mixin providing ideal mode analysis methods.

    Ideal mode processes the full series with future rows visible:
    - Detects all triggers using full series visibility
    - Applies post-processing to clean duplicates
    - Marks active triggers for incomplete trigger pairs
    """

    def _analyze_series_ideal(self, series_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """Ideal mode analysis starting point (future rows visible)."""
        df = self._generate_trigger_begin(df)
        # The gate on the previous row: a begin on an extremum row whose previous row fails it is removed (a row
        # without a begin already holds 0), so the ends copied from the opposite begins below lose it too
        df.loc[~self._begin_allowed, BEGIN_COLS] = 0
        latest_plus_idx = self._get_latest_trigger_begin_idx(df, TRIGGER_PLUS_BEGIN)
        latest_minus_idx = self._get_latest_trigger_begin_idx(df, TRIGGER_MINUS_BEGIN)
        df = self._generate_trigger_end(df)
        df = self._clean_duplicates(df)
        df = self._add_active_trigger_flags(df, latest_plus_idx, latest_minus_idx)
        return df

    def _get_latest_trigger_begin_idx(
        self, df: pd.DataFrame, begin_col: str
    ) -> Optional[Any]:
        """Get the index of the latest trigger begin.

        Returns:
            The index value of the latest begin, or None if no begins exist.
        """
        if begin_col not in df.columns:
            return None

        begins_mask = df[begin_col].astype(bool)
        begin_rows = np.flatnonzero(begins_mask.to_numpy())

        if begin_rows.size == 0:
            return None

        last_begin_row = begin_rows[-1]
        return df.index[last_begin_row]

    def _clean_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean duplicate triggers using vectorized rising-edge pass.

        Processes each plus/minus pair separately, then removes overlapping intervals.
        """
        all_pairs: List[Tuple[int, int, str]] = []

        for begin_col, end_col, pair_type in _TRIGGER_PAIRS:
            if begin_col not in df.columns or end_col not in df.columns:
                continue

            pair_cols = [begin_col, end_col]
            triggers_mask = df[pair_cols].any(axis=1)

            if not triggers_mask.any():
                continue

            ev = df.loc[triggers_mask, pair_cols].astype(np.int8).copy()
            ev_index = ev.index

            # Rising edges (0->1 transitions) on triggers-only timeline
            s = ev.to_numpy(dtype=np.int8)
            rising = np.zeros_like(s, dtype=np.int8)
            rising[0] = s[0]
            if s.shape[0] > 1:
                rising[1:] = ((s[1:] - s[:-1]) == 1).astype(np.int8)
            ev.loc[:, pair_cols] = rising

            # Extract valid pairs in chronological order: each begin takes the first end after it;
            # an end with no unmatched begin before it and a final begin with no end after it stay unmatched
            begins_idx = np.flatnonzero(ev.iloc[:, 0] == 1)
            ends_idx = np.flatnonzero(ev.iloc[:, 1] == 1)

            i, j = 0, 0
            while i < len(begins_idx) and j < len(ends_idx):
                begin_pos = begins_idx[i]
                end_pos = ends_idx[j]
                if begin_pos < end_pos:
                    begin_idx = ev_index[begin_pos]
                    end_idx = ev_index[end_pos]
                    all_pairs.append((begin_idx, end_idx, pair_type))
                    i += 1
                    j += 1
                else:
                    j += 1

        # Sort all pairs by begin index
        all_pairs.sort(key=lambda x: x[0])

        # Remove overlapping pairs — O(N) greedy check against last accepted pair
        # (all_pairs is sorted by begin_idx, so only the most recent endpoint matters)
        final_pairs: List[Tuple[int, int, str]] = []
        for begin_idx, end_idx, pair_type in all_pairs:
            if not final_pairs or begin_idx >= final_pairs[-1][1]:
                final_pairs.append((begin_idx, end_idx, pair_type))

        # Update DataFrame: zero all trigger columns and clear every end reason, then write back final pairs,
        # so an end reason stays only on a final pair's end row
        trigger_columns = [col for bc, ec, _ in _TRIGGER_PAIRS for col in (bc, ec)]
        df.loc[:, trigger_columns] = 0
        df.loc[:, END_REASON] = ""

        for begin_idx, end_idx, pair_type in final_pairs:
            begin_col, end_col = _PAIR_TYPE_TO_COLS[pair_type]
            df.at[begin_idx, begin_col] = 1
            df.at[end_idx, end_col] = 1
            df.at[end_idx, END_REASON] = END_IDEAL_MODE

        return df

    def _add_active_trigger_flags(
        self,
        df: pd.DataFrame,
        latest_plus_idx: Optional[Any],
        latest_minus_idx: Optional[Any],
    ) -> pd.DataFrame:
        """Add columns marking active (unmatched) latest begin before cleaning.

        Rules:
        - active_trigger_*_begin = 1 at pre-cleaned latest begin index if that index
          does NOT appear as the latest begin after cleaning; else 0.
        - Validate mutual exclusivity (except in realistic mode).
        """
        n = len(df)
        df[ACTIVE_TRIGGER_PLUS_BEGIN] = np.zeros(n, dtype=np.uint8)
        df[ACTIVE_TRIGGER_MINUS_BEGIN] = np.zeros(n, dtype=np.uint8)

        latest_cleaned_plus_idx = self._get_latest_trigger_begin_idx(
            df, TRIGGER_PLUS_BEGIN
        )
        latest_cleaned_minus_idx = self._get_latest_trigger_begin_idx(
            df, TRIGGER_MINUS_BEGIN
        )

        for side, latest_idx, latest_cleaned_idx, flag_col in [
            (
                DIRECTION_PLUS,
                latest_plus_idx,
                latest_cleaned_plus_idx,
                ACTIVE_TRIGGER_PLUS_BEGIN,
            ),
            (
                DIRECTION_MINUS,
                latest_minus_idx,
                latest_cleaned_minus_idx,
                ACTIVE_TRIGGER_MINUS_BEGIN,
            ),
        ]:
            if latest_idx is None:
                continue

            pre_pos = df.index.get_loc(latest_idx)

            if latest_cleaned_idx is not None:
                clean_pos = df.index.get_loc(latest_cleaned_idx)
                if clean_pos > pre_pos:
                    raise ValueError(
                        f"Cleaned {side} latest begin row ({clean_pos}) > pre-cleaned row ({pre_pos})"
                    )
                if latest_cleaned_idx != latest_idx:
                    df.at[latest_idx, flag_col] = np.uint8(1)
            else:
                df.at[latest_idx, flag_col] = np.uint8(1)

        return df
