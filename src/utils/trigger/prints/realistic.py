"""
Realistic mode print utilities for trigger detection.

Provides print functions for trigger summary and finalization output
during realistic mode analysis.

Structure:
1. print_trigger_summary (the full trigger summary table), print_finalize_realistic_triggers (the
   trigger count summary)
2. Helper functions (_format_date, _print_active_trigger_row, _print_ended_trigger_row)
"""

from typing import List, Tuple

import pandas as pd

from ...common.constants import (
    END_REASON,
    ACTIVE_TRIGGER_MINUS_BEGIN,
    ACTIVE_TRIGGER_PLUS_BEGIN,
    TRIGGER_MINUS_BEGIN,
    TRIGGER_MINUS_END,
    TRIGGER_PLUS_BEGIN,
    TRIGGER_PLUS_END,
    TRIGGER_VALUE,
)
from ..variation import directional_variation_pct


def print_trigger_summary(
    series_id: str, df: pd.DataFrame, all_marked_triggers: List[Tuple]
) -> None:
    """Print summary of all triggers (TRIGGER SUMMARY section)."""
    print("")
    print("TRIGGER - DETECTION")
    print("=" * 80)
    print(f"SERIES ID: {series_id}")
    print("=" * 80)

    plus_ends = df[df[TRIGGER_PLUS_END] == 1].index.tolist()
    minus_ends = df[df[TRIGGER_MINUS_END] == 1].index.tolist()
    active_plus = df[df[ACTIVE_TRIGGER_PLUS_BEGIN] == 1].index.tolist()
    active_minus = df[df[ACTIVE_TRIGGER_MINUS_BEGIN] == 1].index.tolist()

    sorted_triggers = sorted(all_marked_triggers, key=lambda x: x[2])

    if not sorted_triggers:
        print("\nNo triggers marked.")
        print("=" * 80)
        print("")
        return

    # Column widths
    w = {
        "type": 20,
        "detection_begin": 18,
        "trigger_begin": 18,
        "detection_end": 18,
        "trigger_end": 18,
        "begin": 12,
        "end_val": 12,
        "var": 12,
        "duration": 10,
        "reason": 25,
    }

    # Header
    print(
        f"\n{'Type':<{w['type']}} {'Detection Begin Date':<{w['detection_begin']}} {'Trigger Begin Date':<{w['trigger_begin']}} "
        f"{'Detection End Date':<{w['detection_end']}} {'Trigger End Date':<{w['trigger_end']}} "
        f"{'Begin Value':<{w['begin']}} {'End Value':<{w['end_val']}} {'Variation':<{w['var']}} {'Duration':<{w['duration']}} {'End Reason':<{w['reason']}}"
    )
    print("-" * sum(w.values()))

    for (
        trigger_type,
        detection_date,
        current_idx,
        begin_value,
        marking_iter,
        evt_cnt,
    ) in sorted_triggers:
        begin_pos = df.index.get_loc(detection_date)
        detection_str = _format_date(detection_date)
        trigger_begin_str = _format_date(current_idx)

        is_active = detection_date in active_plus or detection_date in active_minus

        if is_active:
            _print_active_trigger_row(
                df, w, trigger_type, detection_str, trigger_begin_str, begin_value
            )
        else:
            matching_ends = [
                e
                for e in (
                    plus_ends if trigger_type == TRIGGER_PLUS_BEGIN else minus_ends
                )
                if e > detection_date
            ]
            _print_ended_trigger_row(
                df,
                w,
                trigger_type,
                detection_str,
                trigger_begin_str,
                begin_value,
                begin_pos,
                detection_date,
                matching_ends,
                sorted_triggers,
            )

    plus_count = sum(1 for s in sorted_triggers if s[0] == TRIGGER_PLUS_BEGIN)
    minus_count = sum(1 for s in sorted_triggers if s[0] == TRIGGER_MINUS_BEGIN)
    active_count = len(active_plus) + len(active_minus)

    print("\n" + "=" * 80)
    print(
        f"Total triggers: {len(sorted_triggers)} (Plus: {plus_count}, Minus: {minus_count}, Active: {active_count})"
    )
    print("=" * 80)
    print("")


def print_finalize_realistic_triggers(series_id: str, df: pd.DataFrame) -> None:
    """Print trigger count summary at end of realistic mode."""
    print(f"Series '{series_id}': Realistic mode trigger generation complete")
    print(f"  {TRIGGER_PLUS_BEGIN}: {df[TRIGGER_PLUS_BEGIN].sum()}")
    print(f"  {TRIGGER_MINUS_BEGIN}: {df[TRIGGER_MINUS_BEGIN].sum()}")
    print(f"  {TRIGGER_PLUS_END}: {df[TRIGGER_PLUS_END].sum()}")
    print(f"  {TRIGGER_MINUS_END}: {df[TRIGGER_MINUS_END].sum()}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _format_date(date_val) -> str:
    """Format a date value to string."""
    return (
        date_val.strftime("%Y-%m-%d")
        if hasattr(date_val, "strftime")
        else str(date_val)[:10]
    )


def _print_active_trigger_row(
    df, w, trigger_type, detection_str, trigger_begin_str, begin_value
):
    """Print a single active trigger row."""
    end_value = df.iloc[-1][TRIGGER_VALUE]
    variation_pct = directional_variation_pct(trigger_type, begin_value, end_value)
    var_str = f"{variation_pct:+.2f}%"
    print(
        f"{trigger_type:<{w['type']}} {detection_str:<{w['detection_begin']}} {trigger_begin_str:<{w['trigger_begin']}} "
        f"{'ACTIVE':<{w['detection_end']}} {'ACTIVE':<{w['trigger_end']}} "
        f"{begin_value:<{w['begin']}.2f} {end_value:<{w['end_val']}.2f} {var_str:<{w['var']}} {'-':<{w['duration']}} {'active':<{w['reason']}}"
    )


def _print_ended_trigger_row(
    df,
    w,
    trigger_type,
    detection_str,
    trigger_begin_str,
    begin_value,
    begin_pos,
    detection_date,
    matching_ends,
    sorted_triggers,
):
    """Print a single ended trigger row."""
    if not matching_ends:
        print(
            f"{trigger_type:<{w['type']}} {detection_str:<{w['detection_begin']}} {trigger_begin_str:<{w['trigger_begin']}} "
            f"{'-':<{w['detection_end']}} {'-':<{w['trigger_end']}} "
            f"{begin_value:<{w['begin']}.2f} {'-':<{w['end_val']}} {'-':<{w['var']}} {'-':<{w['duration']}} {'none, active':<{w['reason']}}"
        )
        return

    end_date = matching_ends[0]
    trigger_end_str = _format_date(end_date)
    # Find next trigger by Detection Begin Date: earliest trigger that starts after current trigger
    # and before/equal to Trigger End Date (indicates opposite_trigger swap)
    candidates = [
        other_det_date
        for _, other_det_date, _, _, _, _ in sorted_triggers
        if other_det_date > detection_date and other_det_date <= end_date
    ]
    if candidates:
        next_det_begin = min(candidates)
        detection_end_str = _format_date(next_det_begin)
    else:
        detection_end_str = trigger_end_str

    end_value = df.loc[end_date, TRIGGER_VALUE]
    end_reason = df.loc[end_date, END_REASON] if END_REASON in df.columns else ""
    variation_pct = directional_variation_pct(trigger_type, begin_value, end_value)
    var_str = f"{variation_pct:+.2f}%"
    # Duration should be between trigger begin and trigger end, not detection begin and trigger end
    trigger_begin_date = pd.to_datetime(trigger_begin_str)
    trigger_begin_pos = df.index.get_loc(trigger_begin_date)
    trigger_end_pos = df.index.get_loc(end_date)
    duration = trigger_end_pos - trigger_begin_pos
    print(
        f"{trigger_type:<{w['type']}} {detection_str:<{w['detection_begin']}} {trigger_begin_str:<{w['trigger_begin']}} "
        f"{detection_end_str:<{w['detection_end']}} {trigger_end_str:<{w['trigger_end']}} "
        f"{begin_value:<{w['begin']}.2f} {end_value:<{w['end_val']}.2f} {var_str:<{w['var']}} {duration:<{w['duration']}} {end_reason:<{w['reason']}}"
    )
