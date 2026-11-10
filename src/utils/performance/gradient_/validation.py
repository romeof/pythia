"""
Sweep-line weight-constraint helpers for the gradient computation.

This module contains the helpers that back ``validate_and_filter_signals``:
1. precompute_signal_bounds — derive (begin, end) per signal from its pct index
2. sweep_line_filter — chronological pass that removes signals violating the
   max concurrent weight constraint
3. log_signal_removal / log_filter_result — logging helpers for the sweep pass
"""

import logging
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Sort priorities of the sweep's events on one date
_END_EARLIER = 0  # end of a signal that began on an earlier date
_BEGIN = 1
_END_SAME_DAY = 2  # end of a zero-length signal (begin = end): after its day's begin events


def precompute_signal_bounds(
    signals_data: Dict[int, Dict[str, Any]],
) -> Dict[int, Tuple[pd.Timestamp, pd.Timestamp]]:
    """Precompute (begin, end) timestamps per signal from their pct index.

    Returns:
        Dict mapping group_id to (begin, end) timestamps. Signals with
        empty or missing pct series are excluded.
    """
    bounds: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for gid, d in signals_data.items():
        pct_series = d.get("pct")
        if pct_series is not None and len(pct_series) > 0:
            bounds[gid] = (pct_series.index.min(), pct_series.index.max())
    return bounds


def sweep_line_filter(
    signal_bounds: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]],
    signals_data: Dict[int, Dict[str, Any]],
    weight: float,
    concurrent_weight_limit: float,
) -> Tuple[Set[int], int]:
    """Sweep-line pass that removes signals violating the weight constraint.

    Builds begin/end events from precomputed bounds and sweeps chronologically.
    Begin events are placed at (begin, _BEGIN, rank, gid) and end events at
    (end, _END_EARLIER, 0, gid), or at (end, _END_SAME_DAY, 0, gid) for a
    zero-length signal. Sorting by (date, type) processes the end events of a date
    before its begin events, so a signal ending on date D no longer counts when new
    signals beginning on D are evaluated; a zero-length signal ends after the
    begin events of its date, so it counts on its day.

    When a violation is detected, the newest active signal (highest rank) is
    removed -- preserving the original removal policy.

    Args:
        signal_bounds: Precomputed (begin, end) per signal
        signals_data: Full signal data dict (for logging metadata)
        weight: Weight per signal
        concurrent_weight_limit: Maximum allowed concurrent weight

    Returns:
        Tuple of (removed_group_ids set, max_concurrent count)
    """
    # Build events: (date, type, rank, gid)
    # type: _END_EARLIER, _BEGIN, _END_SAME_DAY (the sort priority on one date)
    # rank: the signals_data iteration rank of begin events
    events: List[Tuple[pd.Timestamp, int, int, int]] = []
    for rank, gid in enumerate(signal_bounds):
        begin, end = signal_bounds[gid]
        events.append((begin, _BEGIN, rank, gid))
        events.append((end, _END_SAME_DAY if end == begin else _END_EARLIER, 0, gid))

    # Sort: by date first, then type (_END_EARLIER, _BEGIN, _END_SAME_DAY), then rank
    events.sort()

    removed_group_ids: Set[int] = set()
    # active_signals: gid → rank for O(1) insert and delete.
    # Finding the newest (max rank) is O(K) but avoids O(K) list rebuild per end event.
    active_signals: Dict[int, int] = {}
    max_concurrent = 0

    for _, event_type, rank, gid in events:
        if gid in removed_group_ids:
            continue

        if event_type != _BEGIN:
            # End event: O(1) removal instead of O(K) list rebuild
            active_signals.pop(gid, None)
        else:
            # Begin event: add to active set
            active_signals[gid] = rank
            concurrent_count = len(active_signals)

            if concurrent_count * weight > concurrent_weight_limit:
                newest_gid = max(active_signals, key=active_signals.__getitem__)
                del active_signals[newest_gid]
                removed_group_ids.add(newest_gid)

                log_signal_removal(
                    newest_gid, signal_bounds,
                    [(rank, g) for g, rank in active_signals.items()],
                    weight, concurrent_count, concurrent_weight_limit,
                )
            else:
                max_concurrent = max(max_concurrent, concurrent_count)

    return removed_group_ids, max_concurrent


def log_signal_removal(
    removed_gid: int,
    signal_bounds: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]],
    remaining_active: List[Tuple[int, int]],
    weight: float,
    concurrent_count: int,
    concurrent_weight_limit: float,
) -> None:
    """Log a single signal removal with details about the violation."""
    begin, end = signal_bounds[removed_gid]
    existing_str = "; ".join(
        f"group_id={g} (begin={signal_bounds[g][0].date()}, end={signal_bounds[g][1].date()})"
        for _, g in remaining_active
    )
    logger.warning(
        f"Removed signal group_id={removed_gid} (begin={begin.date()}, "
        f"end={end.date()}) - caused violation: "
        f"weight={weight:.4f} - concurrent={concurrent_count} = "
        f"{weight * concurrent_count:.4f} > {concurrent_weight_limit:.4f}\n"
        f"Existing signals: {existing_str}"
    )


def log_filter_result(
    removed_group_ids: set,
    filtered_signals_data: Dict[int, Dict[str, Any]],
    weight: float,
    max_concurrent: int,
    concurrent_weight_limit: float,
) -> None:
    """Log the final result of the weight constraint validation."""
    if removed_group_ids:
        logger.warning(
            f"Total signals removed: {len(removed_group_ids)}. "
            f"Remaining signals: {len(filtered_signals_data)}"
        )
    else:
        logger.info(
            f"Weight constraint satisfied: weight={weight:.4f}, "
            f"max_concurrent_signals={max_concurrent}, "
            f"concurrent_weight={weight * max_concurrent:.4f} <= {concurrent_weight_limit:.4f}"
        )
