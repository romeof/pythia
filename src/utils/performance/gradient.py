"""
Computation module for computing the gradient from signals.

This module exposes the public starting points for the gradient pipeline. Internal
helpers live in the ``gradient_`` subpackage:
- ``gradient_.attenuation`` — yearly attenuation schedule + application
- ``gradient_.validation`` — sweep-line weight-constraint helpers
- ``gradient_.compounding`` — pct→per-period conversion, multiplicative engine,
  deduction, and begin-value policies
- ``gradient_.metrics`` — gradient metric calculations

Structure (public surface):
1. Global index (``build_global_index``)
2. Signal segments (``build_signal_segments_cumulative_pct_with_maxweight_constraint``) and the signal
   variation helpers (``compute_signal_variation``, ``validate_and_filter_signals``, ``compute_variation_pct``)
3. Total variation (``build_actual_variation_pair``)
4. Reference (``build_reference_variation_series_map``, ``build_reference_terminal_attenuations``)
"""

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..common.common import variation_pct
from ..common.constants import (
    GROUP_ID,
    IS_PLUS_TRIGGER,
    IS_SIGNAL,
    IS_TRUE,
    SERIES_ID,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
    TRIGGER_END_VALUES,
    TRIGGER_START_VALUES,
    TRIGGER_VALUE,
)
from .gradient_.compounding import (
    convert_cumulative_pct_to_signed_period_variations,
    cumulative_pct_to_period_variations_series,
    multiplicative_variation,
)
from .gradient_.validation import (
    log_filter_result,
    precompute_signal_bounds,
    sweep_line_filter,
)

logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL INDEX
# =============================================================================


def build_global_index(
    observed: Sequence[pd.DatetimeIndex],
    start: Optional[pd.Timestamp],
    end_date: str,
) -> pd.DatetimeIndex:
    """Build the global curve index for gradient computation.

    The index is the union of the observed dates of the processed series and
    the references, never a weekday calendar. It starts at the earliest
    effective start over the processed series (index[lookback - 1] of each
    preprocessed frame), the first date on which any series may emit a trigger.

    Args:
        observed: Date indexes to unite: each processed series' dates and each
            reference's observed dates.
        start: Earliest effective start, or None when no series was processed.
        end_date: ISO date string for the end of the window (inclusive).

    Returns:
        pd.DatetimeIndex of the sorted, duplicate-free observed dates from start
        to end_date; empty when start is None or no index is given.
    """
    if start is None or not observed:
        return pd.DatetimeIndex([])
    union = pd.DatetimeIndex(np.unique(np.concatenate([pd.DatetimeIndex(dates).to_numpy() for dates in observed])))
    return union[(union >= start.normalize()) & (union <= pd.to_datetime(end_date))]


# =============================================================================
# SIGNAL SEGMENTS
# =============================================================================


def build_signal_segments_cumulative_pct_with_maxweight_constraint(
    signals_df: pd.DataFrame,
    series: Dict[str, pd.DataFrame],
    global_idx: pd.DatetimeIndex,
    weight: float,
    concurrent_weight_limit: float,
) -> Tuple[Dict[int, Dict[str, Any]], pd.DataFrame]:
    """Collect per-signal variation segments for those signals passing max weight constraint.

    First pass: compute percent-change series for each signal row.
    Then filter out signals that violate the concurrent_weight_limit constraint.

    Args:
        signals_df: DataFrame with group assignments (must have is_signal column)
        series: Dict mapping series_id to DataFrame with trigger_value column
        global_idx: Global curve index (the observed dates) for alignment
        weight: Weight per signal (fractional allocation per trigger)
        concurrent_weight_limit: Upper bound on the summed weights of concurrently active signals

    Returns:
        Tuple of (signals_data dict, filtered signal sequence DataFrame)
    """
    signal_rows = signals_df[signals_df[IS_SIGNAL]].copy()
    if signal_rows.empty:
        return {}, signal_rows

    signal_sequence = signal_rows.sort_values(
        [TRIGGER_BEGIN_DATES, GROUP_ID]
    ).reset_index(drop=True)

    signals_data: Dict[int, Dict[str, Any]] = {}
    for i in range(len(signal_sequence)):
        row = signal_sequence.iloc[i]
        gid = int(row[GROUP_ID])
        result = compute_signal_variation(
            signal_row=row,
            series=series,
            global_idx=global_idx,
        )
        if result is None:
            continue
        pct, sign_mult, metadata = result
        signals_data[gid] = {
            GROUP_ID: gid,
            "pct": pct,
            "sign": sign_mult,
            SERIES_ID: metadata.get(SERIES_ID),
            "start_val": metadata.get("start_val"),
            "end_val": metadata.get("end_val"),
            IS_PLUS_TRIGGER: bool(row[IS_PLUS_TRIGGER]),
            IS_TRUE: bool(row[IS_TRUE]),
            TRIGGER_START_VALUES: float(row[TRIGGER_START_VALUES]),
            TRIGGER_END_VALUES: float(row[TRIGGER_END_VALUES]),
        }

    signals_data = validate_and_filter_signals(
        signals_data=signals_data,
        weight=weight,
        concurrent_weight_limit=concurrent_weight_limit,
    )

    filtered_gids = set(signals_data.keys())
    filtered_signal_sequence = signal_sequence[
        signal_sequence[GROUP_ID].isin(filtered_gids)
    ].reset_index(drop=True)

    return signals_data, filtered_signal_sequence


def compute_signal_variation(
    signal_row: pd.Series,
    series: Dict[str, pd.DataFrame],
    global_idx: pd.DatetimeIndex,
) -> Optional[Tuple[pd.Series, int, Dict[str, Any]]]:
    """
    Compute percent-change series for a single signal.

    Args:
        signal_row: Row from signals_df containing signal metadata
        series: Dict mapping series_id to DataFrame with trigger_value column
        global_idx: Global curve index (the observed dates) for alignment

    Returns:
        Tuple of (pct_series, sign_multiplier, metadata_dict) or None if computation fails
    """
    series_id = str(signal_row[SERIES_ID])
    begin = pd.to_datetime(signal_row[TRIGGER_BEGIN_DATES])
    end = pd.to_datetime(signal_row[TRIGGER_END_DATES])
    is_plus = bool(signal_row[IS_PLUS_TRIGGER])
    start_val = float(signal_row.get(TRIGGER_START_VALUES))
    if pd.isna(start_val) or start_val == 0.0:
        return None
    end_val = float(signal_row.get(TRIGGER_END_VALUES))

    cat_df = series.get(series_id)
    if cat_df is None or len(cat_df) == 0:
        return None
    if TRIGGER_VALUE not in cat_df.columns:
        return None

    if not isinstance(cat_df.index, pd.DatetimeIndex):
        cat_df = cat_df.copy()
        cat_df.index = pd.to_datetime(cat_df.index)

    seg = cat_df.loc[(cat_df.index >= begin) & (cat_df.index <= end)]
    if seg.empty:
        return None

    values = seg[TRIGGER_VALUE]

    # Compute percent-change from the trigger's begin value
    pct = variation_pct(values, start_val)

    # Reindex to the observed dates within [begin, end] to show full trigger timeline
    seg_global_idx = global_idx[(global_idx >= begin) & (global_idx <= end)]
    pct = pct.reindex(seg_global_idx).ffill().dropna()
    if pct.empty:
        return None

    # Force signal day to 0% (must be after reindex to survive alignment)
    pct.iloc[0] = 0.0

    sign_mult = +1 if is_plus else -1

    return (
        pct,
        sign_mult,
        {SERIES_ID: series_id, "start_val": start_val, "end_val": end_val},
    )


# Mode-agnostic: this pre-filter bounds concurrent allocation using the configured weight w.
# Exact for additive (each event allocates w); conservative lower-bound for multiplicative
# (each event allocates w * V(start_i) >= w when the total value has grown).
# Computing the exact multiplicative concurrent allocation here would require the full path-dependent
# compounding, creating a circular dependency -- so the configured-weight check is the right compromise.
def validate_and_filter_signals(
    signals_data: Dict[int, Dict[str, Any]],
    weight: float,
    concurrent_weight_limit: float,
) -> Dict[int, Dict[str, Any]]:
    """
    Validate that concurrent signal weights don't exceed concurrent_weight_limit,
    or remove the signal that violates this condition.

    Uses a sweep-line algorithm: O(S*log(S) + E) where S = number of signals and
    E = number of begin/end events, instead of the naive O(D*S) approach.

    Each signal produces two events: a "begin" event at begin and an "end" event at end.
    Sorting these events and sweeping left-to-right tracks the active count
    without rescanning all signals per date. On the same date, end events are
    processed before begin events, so a signal ending on a given date no longer
    counts for new signals beginning that day; a zero-length signal (begin = end)
    ends after the begin events of its date, so it counts on its day.

    Args:
        signals_data: Dictionary of signal data in begin_date sequence with keys: 'pct', 'sign', etc.
        weight: Weight per signal (fractional allocation per trigger)
        concurrent_weight_limit: Upper bound on the summed weights of concurrently active signals

    Returns:
        Filtered signals_data dictionary with only valid signals
    """
    # Precompute (begin, end) per signal once -- avoids repeated .index.min()/.index.max()
    signal_bounds = precompute_signal_bounds(signals_data)
    if not signal_bounds:
        return signals_data

    removed_group_ids, max_concurrent = sweep_line_filter(
        signal_bounds, signals_data, weight, concurrent_weight_limit
    )

    filtered_signals_data = {
        gid: data for gid, data in signals_data.items() if gid not in removed_group_ids
    }

    log_filter_result(
        removed_group_ids, filtered_signals_data, weight, max_concurrent,
        concurrent_weight_limit,
    )

    return filtered_signals_data


def compute_variation_pct(var0: float, end_value: float) -> float:
    """
    Compute variation percentage between start and end values.

    Args:
        var0: Starting value
        end_value: Ending value

    Returns:
        Variation percentage, or NaN for a zero or missing value (any other invalid input raises)
    """
    if var0 == 0 or pd.isna(var0) or pd.isna(end_value):
        return float("nan")
    return variation_pct(end_value, var0)


# =============================================================================
# TOTAL VARIATION
# =============================================================================


def build_actual_variation_pair(
    signals_data: Dict[int, Dict[str, Any]],
    weight: float,
    idx: pd.DatetimeIndex,
    variation_method: str,
    yearly_attenuation_fraction: float = 0.0,
    attenuation_month: int = 1,
    boundary_deduction_pct: float = 0.0,
) -> Tuple[Optional[pd.Series], Optional[pd.Series], float]:
    """Build both pure and attenuation-adjusted gradient with a single conversion pass.

    Converts signal variations once (O(S×N)), then runs the compounding loop
    without attenuation and (when yearly_attenuation_fraction > 0) with attenuation.
    When yearly_attenuation_fraction == 0, both returned series are the same object.

    The attenuation-adjusted series carries every calendar year's attenuation, the last
    year's at idx[-1]. That last-year amount is also returned as a scalar so
    the plotting layer can mark the attenuated point at idx[-1].

    Args:
        signals_data: Dictionary of signal data with 'pct' and 'sign' keys.
                     Each item's pct.index[0] is the event start date.
        weight: Weight per signal (fraction per trigger)
        idx: Global curve index (the observed dates)
        variation_method: Weight mechanism at the aggregate level
            (validated by schema Literal constraint):
            - "multiplicative_fixed"
            - "multiplicative_total_with_active_signals"
            - "multiplicative_total_before_active_signals"
        yearly_attenuation_fraction: Fraction of the yearly increase removed [0.0, 1.0]
        attenuation_month: Month (1-12) when the yearly attenuation is applied
        boundary_deduction_pct: Deduction as percentage at each event boundary (e.g., 0.1 = 0.1% at each boundary)

    Returns:
        Tuple of (pure_gradient, attenuation_adjusted_gradient, terminal_attenuation_amount).
        The first two are None if signals_data is empty; terminal is 0.0.
        The first two are the same object and terminal is 0.0 when
        yearly_attenuation_fraction == 0.
    """
    if not signals_data:
        return None, None, 0.0

    # Single conversion pass — shared by both compounding runs
    signal_variations = convert_cumulative_pct_to_signed_period_variations(
        signals_data, idx
    )

    pure, _ = multiplicative_variation(
        signal_variations, weight, idx, variation_method, boundary_deduction_pct=boundary_deduction_pct,
    )

    if yearly_attenuation_fraction > 0.0:
        attenuation_adjusted, terminal_attenuation_amount = multiplicative_variation(
            signal_variations, weight, idx, variation_method,
            yearly_attenuation_fraction, attenuation_month, boundary_deduction_pct=boundary_deduction_pct,
        )
    else:
        attenuation_adjusted = pure
        terminal_attenuation_amount = 0.0

    return pure, attenuation_adjusted, terminal_attenuation_amount


# =============================================================================
# REFERENCE
# =============================================================================


def build_reference_variation_series_map(
    reference_series: Dict[str, pd.Series],
    global_idx: pd.DatetimeIndex,
) -> Dict[str, pd.Series]:
    """
    Compute per-reference variation curves (no aggregation, single-series compounding).

    Rows before global_idx[0], which a reference keeps untrimmed when its preprocessed
    frame is shorter than the lookback, are dropped before the base is taken, so each
    curve is 0 on the reference's first row on the index; a reference without a row on
    a non-empty index gets no curve.

    References are passive baselines: they compound freely during the years
    and the returned series is always pre-attenuation (full compounded variation).
    The post-attenuation terminal value is computed separately by
    build_reference_terminal_attenuations() and rendered by the plotting layer
    as a marker point.

    Args:
        reference_series: Dictionary mapping reference IDs to time series
        global_idx: Global curve index (the observed dates) for alignment

    Returns:
        Dictionary mapping reference IDs to pre-attenuation cumulative variation series
    """
    out: Dict[str, pd.Series] = {}
    if not reference_series:
        return out
    idx = global_idx

    for series_id, ts in reference_series.items():
        if ts is None or len(ts) == 0:
            continue
        ts_float = ts.astype(float)
        if len(idx) > 0:
            ts_float = ts_float[ts_float.index >= idx[0]]
            if len(ts_float) == 0:
                continue
        base = float(ts_float.iloc[0])
        if base == 0:
            continue
        pct = variation_pct(ts_float, base)
        variation = cumulative_pct_to_period_variations_series(pct)
        variation = variation.reindex(idx).fillna(0.0)
        ref_cum = (1.0 + variation).cumprod() - 1.0
        out[series_id] = ref_cum

    return out


def build_reference_terminal_attenuations(
    reference_map: Dict[str, pd.Series],
    yearly_attenuation_fraction: float,
) -> Dict[str, float]:
    """
    Compute the terminal attenuation amount for each reference.

    References compound freely throughout: the attenuation is applied once at the
    end on the total accumulated increase (increase = ref_cum.iloc[-1], floored at 0).
    The returned value is the absolute amount the plot subtracts from
    ref_cum.iloc[-1] to place the post-attenuation marker.

    Args:
        reference_map: Pre-attenuation reference cumulative variation series
        yearly_attenuation_fraction: Fraction of the total increase removed [0.0, 1.0]

    Returns:
        Dict mapping reference IDs to non-negative attenuation amounts. Empty
        when yearly_attenuation_fraction <= 0 or reference_map is empty.
    """
    if yearly_attenuation_fraction <= 0.0 or not reference_map:
        return {}
    attenuations: Dict[str, float] = {}
    for ref_id, ref_cum in reference_map.items():
        if ref_cum is None or len(ref_cum) == 0:
            continue
        increase = max(0.0, float(ref_cum.iloc[-1]))
        attenuations[ref_id] = yearly_attenuation_fraction * increase
    return attenuations


