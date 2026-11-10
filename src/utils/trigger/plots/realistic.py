"""
Realistic mode plot orchestration for trigger detection.

Provides functions to generate debug plots for suppressed, ended, and active triggers
during the realistic incremental loop.

Structure:
1. Debug directory setup (setup_debug_dir)
2. Plot filter (should_plot_suppressed)
3. Plot orchestration (plot_suppressed_trigger, plot_ended_trigger, plot_active_trigger)
4. Helper functions (find_marking_info, compute_suppressed_metrics)
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...common.constants import TRIGGER_VALUE
from ..variation import compute_trigger_variation, directional_variation_pct
from .detection import plot_window_with_triggers

logger = logging.getLogger(__name__)


# =============================================================================
# DEBUG DIRECTORY SETUP
# =============================================================================


def setup_debug_dir(
    series_id: str, output_file: Path, plot_debug: bool
) -> Optional[Path]:
    """Setup debug directory if plot debugging is enabled: plots/<series_id> beside the trigger output file."""
    if not plot_debug:
        return None

    debug_dir = Path(output_file).parent / "plots" / series_id
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


# =============================================================================
# PLOT FILTER
# =============================================================================


def should_plot_suppressed(candidate_date, active_trigger: Optional[Tuple]) -> bool:
    """Determine if a suppressed trigger should be plotted."""
    return active_trigger is None or candidate_date > active_trigger[1]


# =============================================================================
# PLOT ORCHESTRATION
# =============================================================================


def plot_suppressed_trigger(
    series_id: str,
    df: pd.DataFrame,
    window_df: pd.DataFrame,
    i: int,
    current_idx,
    current_value: float,
    candidate_date,
    trigger_type: str,
    suppression_reason: str,
    trigger_cols: List[str],
    state: Dict[str, Any],
    debug_dir: Optional[Path],
) -> None:
    """Generate debug plot for a suppressed trigger."""
    duration, _, _ = compute_suppressed_metrics(
        df, trigger_type, candidate_date, current_idx, current_value
    )

    trigger_begin_value, trigger_end_value, signed_pct = compute_trigger_variation(
        df, trigger_type, current_idx, current_idx
    )

    # For suppressed triggers, all indices are at the current row
    detection_begin_idx = df.index.get_loc(candidate_date)
    current_pos_idx = df.index.get_loc(current_idx)

    plot_window_with_triggers(
        window_df=window_df,
        trigger_cols=trigger_cols,
        iteration=i,
        current_date=current_idx,
        series_id=series_id,
        debug_dir=str(debug_dir),
        detection_date=candidate_date,
        detection_end_date=current_idx,
        trigger_end_date=current_idx,
        duration=duration,
        begin_value=trigger_begin_value,
        end_value=trigger_end_value,
        variation_pct=signed_pct,
        trigger_type=trigger_type,
        detection_begin_idx=detection_begin_idx,
        current_idx=i,
        detection_end_idx=current_pos_idx,
        trigger_end_idx=current_pos_idx,
        trigger_count=state["suppressed_counters"][trigger_type],
        is_suppressed=True,
        suppression_reason=suppression_reason,
    )


def plot_ended_trigger(
    series_id: str,
    df: pd.DataFrame,
    window_df: pd.DataFrame,
    i: int,
    current_idx,
    trigger_cols: List[str],
    state: Dict[str, Any],
    debug_dir: Optional[Path],
    duration: int,
    end_reason: str,
    detection_end_date=None,
) -> None:
    """Generate debug plot for an ended trigger."""
    trigger_type, begin_date, _, trigger_begin_idx, _, _ = state["active_trigger"]

    marking_date, marking_iter, evt_count = find_marking_info(
        state["all_marked_triggers"], trigger_type, begin_date
    )

    trigger_begin_date = marking_date if marking_date else current_idx
    trigger_begin_value, trigger_end_value, signed_pct = compute_trigger_variation(
        df, trigger_type, trigger_begin_date, current_idx
    )

    # Compute row indices for legend
    detection_begin_idx = df.index.get_loc(begin_date)
    actual_detection_end_date = (
        detection_end_date if detection_end_date is not None else current_idx
    )
    detection_end_idx = df.index.get_loc(actual_detection_end_date)
    trigger_end_idx = df.index.get_loc(current_idx)

    plot_window_with_triggers(
        window_df=window_df,
        trigger_cols=trigger_cols,
        iteration=i,
        current_date=trigger_begin_date,
        series_id=series_id,
        debug_dir=str(debug_dir),
        detection_date=begin_date,
        detection_end_date=actual_detection_end_date,
        trigger_end_date=current_idx,
        duration=duration,
        begin_value=trigger_begin_value,
        end_value=trigger_end_value,
        variation_pct=signed_pct,
        is_active=False,
        trigger_type=trigger_type,
        end_reason=end_reason,
        detection_begin_idx=detection_begin_idx,
        current_idx=marking_iter,
        detection_end_idx=detection_end_idx,
        trigger_end_idx=trigger_end_idx,
        trigger_count=evt_count,
    )


def plot_active_trigger(
    series_id: str,
    df: pd.DataFrame,
    window_df: pd.DataFrame,
    trigger_cols: List[str],
    state: Dict[str, Any],
    debug_dir: Optional[Path],
    n: int,
) -> None:
    """Generate debug plot for an active trigger.

    Args:
        window_df: Window DataFrame with triggers already generated by the caller.
    """
    trigger_type, begin_date, _, trigger_begin_idx, _, _ = state["active_trigger"]

    final_idx = df.index[-1]
    begin_pos = df.index.get_loc(begin_date)
    trigger_begin_pos = df.index.get_loc(trigger_begin_idx)
    final_pos = df.index.get_loc(final_idx)
    active_duration = final_pos - trigger_begin_pos

    marking_date, marking_iter, evt_count = find_marking_info(
        state["all_marked_triggers"], trigger_type, begin_date
    )

    trigger_begin_date = marking_date if marking_date else final_idx
    trigger_begin_value, trigger_end_value, signed_pct = compute_trigger_variation(
        df, trigger_type, trigger_begin_date, final_idx
    )

    plot_window_with_triggers(
        window_df=window_df,
        trigger_cols=trigger_cols,
        iteration=n - 1,
        current_date=trigger_begin_date,
        series_id=series_id,
        debug_dir=str(debug_dir),
        detection_date=begin_date,
        duration=active_duration,
        begin_value=trigger_begin_value,
        end_value=trigger_end_value,
        variation_pct=signed_pct,
        is_active=True,
        trigger_type=trigger_type,
        detection_begin_idx=begin_pos,
        current_idx=marking_iter,
        trigger_count=evt_count,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def find_marking_info(
    all_marked_triggers: List[Tuple], trigger_type: str, begin_date
) -> Tuple[Optional[Any], Optional[int], Optional[int]]:
    """Find marking info for a trigger from the tracked list.

    Returns:
        Tuple of (marking_date, marking_iter, trigger_count) or (None, None, None) if not found.
    """
    for trg_type, trg_date, mark_date, _, mark_iter, evt_cnt in all_marked_triggers:
        if trg_type == trigger_type and trg_date == begin_date:
            return mark_date, mark_iter, evt_cnt
    return None, None, None


def compute_suppressed_metrics(
    df: pd.DataFrame,
    trigger_type: str,
    candidate_date,
    current_idx,
    current_value: float,
) -> Tuple[int, float, float]:
    """Compute metrics for a suppressed trigger candidate.

    Returns:
        Tuple of (duration, signed_pct, candidate_value)
    """
    candidate_value = df.loc[candidate_date, TRIGGER_VALUE]
    candidate_pos = df.index.get_loc(candidate_date)
    current_pos = df.index.get_loc(current_idx)
    duration = current_pos - candidate_pos

    signed_pct = directional_variation_pct(trigger_type, candidate_value, current_value)

    return duration, signed_pct, candidate_value
