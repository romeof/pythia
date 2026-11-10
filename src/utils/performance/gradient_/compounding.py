"""
Variation aggregation and multiplicative compounding for the gradient.

This module contains the engine that turns per-signal cumulative-percent
series into the aggregate gradient curve:

1. cumulative_pct_to_period_variations_series — single-signal pct → per-period rates
2. convert_cumulative_pct_to_signed_period_variations — apply this to all signals,
   reindex to the global axis, and apply signal direction
3. multiplicative_variation — geometric per-event compounding plus cross-event
   aggregation under one of three begin-value policies (with optional attenuation and
   deduction)
4. apply_event_deduction — begin/end deduction applied to a single event's series
5. value_at_start_{fixed, multiplicative_total_with_active_signals, multiplicative_total_before_active_signals} —
   the three pluggable begin-value policies selected by ``mode``
"""

from typing import Any, Dict, List, Tuple

import pandas as pd

from ...common.constants import (
    VARIATION_MULTIPLICATIVE_FIXED,
    VARIATION_MULTIPLICATIVE_BEFORE_ACTIVE_SIGNALS,
    VARIATION_MULTIPLICATIVE_WITH_ACTIVE_SIGNALS,
    GROUP_ID,
    IS_PLUS_TRIGGER,
    IS_TRUE,
    SERIES_ID,
)
from .attenuation import apply_yearly_attenuation, build_attenuation_schedule


# --- Design note: discrete vs continuous compounding ---
# An alternative to discrete per-period rates is continuous (log-space) compounding:
#   log_r_t = log(1 + r_t)           instead of r_t
#   event_cum = exp(cumsum(log_r)) - 1  instead of cumprod(1 + r) - 1
#
# For a single series the two are algebraically identical:
#   exp(Σ log(1 + r_t)) ≡ Π(1 + r_t)
#
# However, cross-event aggregation uses a linear sum in level-space:
#   total(t) = Σ locked_weight_i × event_cum_i(t)
#
# The fundamental incompatibility is the summation across events: the exponential
# is not linear over addition, i.e. exp(a + b) ≠ exp(a) + exp(b). So the moment
# two or more log-variation terms are summed and then exponentiated, the result
# diverges from summing the discrete terms directly. This holds regardless of the
# weight value — even with w = 1 the mismatch appears as soon as multiple events
# are combined:
#
#   log(1 + R_total) ≠ Σ w_i × log(1 + r_i)       (for any w, when N > 1)
#
# (A secondary, smaller effect: with a single event, w × log(1+r) ≠ log(1 + w×r)
#  when w ≠ 1. But this is minor compared to the multi-event structural issue.)
#
# Numerical example — two events, w = 1.0, after one period:
#   Event A: +20% (r_A = 0.20),  Event B: -10% (r_B = -0.10)
#
#   Discrete (correct):
#     R = 1.0 × 0.20 + 1.0 × (-0.10) = +10.00%
#
#   Log-space (incorrect for multi-event aggregation):
#     log_A = log(1.20) = 0.18232,  log_B = log(0.90) = -0.10536
#     log_R = 1.0 × 0.18232 + 1.0 × (-0.10536) = 0.07696
#     R_log = exp(0.07696) - 1 = +8.00%   ← ~2 pp off
#
# The gap grows with variation magnitude and number of events. For the ±10–50%
# range typical of signal segments, the discrepancy is material enough to make the
# summary table disagree with the plot. Log-variations are standard for statistical
# analysis (dispersion models, hypothesis testing) because they are additive over time
# and approximately normal, but for summing event variations the discrete form
# is correct.
#
# Therefore this module uses discrete compounding throughout.
# -------------------------------------------------------
def convert_cumulative_pct_to_signed_period_variations(
    signals_data: Dict[int, Dict[str, Any]],
    idx: pd.DatetimeIndex,
) -> List[Dict[str, Any]]:
    """
    Convert each signal's percent-from-start series into a signed per-period variation series.

    For each signal:
    1. Convert cumulative pct into per-period rates:
       r_t = (1+pct_t/100)/(1+pct_{t-1}/100) - 1.
       Since pct_t = (v_t / v_i - 1) * 100.0 (cumulative from initial value v_i),
       v_i cancels out and:
       r_t = (1 + pct_t/100) / (1 + pct_{t-1}/100) - 1.0 = v_t / v_{t-1} - 1.0.
       If raw values were used directly at this stage, v_t / v_{t-1} - 1 would be
       equivalent (or the ratio v_t / v_{t-1} instead of 1 + r_t).
       Since pct is already computed for the gradient plot middle panel, this step
       derives per-period variation rates directly from pct_t.
    2. Reindex to the full global curve index (the observed dates) and set non-active dates to 0.
       This keeps valid signal-interval values and assigns neutral variation outside
       the signal interval, so compounding uses multiplicative identity (1 + 0 = 1).
    3. Apply signal direction: sign=+1 keeps as-is; sign=-1 negates.
       For inverse signals, negative underlying moves contribute positively and
       positive moves contribute negatively.
    4. Per-event cumulative variation is geometric:
       event_cum = (1.0 + sv["variation"]).cumprod() - 1.0.
    5. Weighting is applied after per-event compounding according to the selected
       variation method.

    Returns list sorted by event start date (ascending).

    Args:
        signals_data: Dictionary of signal data with 'pct' and 'sign' keys
        idx: Global curve index (the observed dates)

    Returns:
        List of dicts with:
        - 'variation': signed per-period Series
        - 'start_date': event start
        - 'end_date': event end
        sorted by start_date ascending
    """
    result = []
    for d in signals_data.values():
        pct_series = d["pct"]

        variation = cumulative_pct_to_period_variations_series(pct_series)
        variation = variation.reindex(idx).fillna(0.0)

        # Apply signal direction by simple negation for inverse signals:
        # inverse signals contribute positively on negative moves, and negatively on positive moves.
        # Note we avoid using -variation / (1.0 + variation), which is the inverse variation and for a variation of -50% would give 100% (not what we want)
        if d["sign"] == -1:
            variation = -variation

        # pct.index[0] is the event start date (signal day, forced to 0% in compute_signal_variation)
        result.append(
            {
                "variation": variation,
                "start_date": pct_series.index[0],
                "end_date": pct_series.index[-1],
                GROUP_ID: d.get(GROUP_ID, "unknown"),
                SERIES_ID: d.get(SERIES_ID, "unknown"),
                IS_PLUS_TRIGGER: d.get(IS_PLUS_TRIGGER, "unknown"),
                IS_TRUE: d.get(IS_TRUE, "unknown"),
                "pct": pct_series,
            }
        )

    # Enforce chronological order explicitly -- multiplicative_variation requires this
    result.sort(key=lambda sv: sv["start_date"])

    return result


def cumulative_pct_to_period_variations_series(pct: pd.Series) -> pd.Series:
    """Convert cumulative percent series to per-period decimal variation rates.

    Discrete per-period variation rate: r_t = (1 + pct_t/100) / (1 + pct_{t-1}/100) - 1.
    Algebraically equal to v_t / v_{t-1} - 1.
    See "Design note: discrete vs continuous compounding" above.

    Args:
        pct: Cumulative percent-from-start series.

    Returns:
        Per-period decimal variation rates. First value is the rate from 0% to pct[0].
    """
    pct_shifted = pct.shift(1).fillna(0.0)
    return (1.0 + pct / 100.0) / (1.0 + pct_shifted / 100.0) - 1.0


def multiplicative_variation(
    signal_variations: List[Dict[str, Any]],
    weight: float,
    idx: pd.DatetimeIndex,
    mode: str,
    yearly_attenuation_fraction: float = 0.0,
    attenuation_month: int = 1,
    boundary_deduction_pct: float = 0.0,
) -> Tuple[pd.Series, float]:
    """
    Compound using multiplicative weighting.

    Each event i receives a locked allocation:
        locked_weight_i = w * V(start_i)

    where V(start_i) is determined by the mode:

    - multiplicative_fixed:
      Always returns 1.0, giving a fixed allocation equal to w for every event.
      Equivalent to "fixed-initial-value" mode: each event is allocated a fixed
      fraction of the initial baseline regardless of prior events.

    - multiplicative_total_with_active_signals:
      V(start_i) includes contributions from all previously processed events
      as observed at start_i, including events still active at start_i:
          V(start_i) = 1 + total_cum(start_i)

    - multiplicative_total_before_active_signals:
      V(start_i) includes only prior events already ended before start_i
      (end_date < start_i). Events still active at start_i are excluded.

    The locked weight differs across modes because _total_with_active_signals incorporates the
    active variations of all series up to V(start_i), while _total_before_active_signals only
    counts variations from already-ended events; _fixed ignores prior state entirely.

    Note: cumsum(w * r_i) would be incorrect -- it adds per-period rates linearly,
    ignoring geometric compounding within each event. Example: +10% then -10%
    yields -1% geometrically (1.10 * 0.90 = 0.99), but cumsum gives 0%.

    In all modes, per-event compounding remains geometric:
        event_cum_i(t) = cumprod(1 + r_i)_t - 1
    and each event is scaled once by locked_weight_i (no re-weighting after lock).

    When yearly_attenuation_fraction > 0, periodic deductions are interleaved with event
    processing: before each event, any pending attenuations whose attenuation_date <= the
    event's start_date are applied. Remaining attenuations are applied after all events.
    This ensures subsequent events compound on a reduced base value.

    The schedule's last item is the last calendar year, attenuated at idx[-1]:
    it lowers only the last point of total_cum, and its amount is returned
    beside total_cum so the caller can mark that point.

    Requires signal_variations sorted by start_date ascending.

    Args:
        signal_variations: List of per-signal variation dicts from convert_cumulative_pct_to_signed_period_variations
        weight: Fractional weight per event (applied to value at event start)
        idx: Global curve index (the observed dates)
        mode: Variation method determining the begin-value policy
        yearly_attenuation_fraction: Fraction of the yearly increase removed [0.0, 1.0]
        attenuation_month: Month (1-12) when the yearly attenuation is applied
        boundary_deduction_pct: Deduction as percentage at each event boundary (e.g., 0.1 = 0.1% at each boundary)

    Returns:
        Tuple of (cumulative variation series, last-year attenuation amount); the
        amount is 0.0 without attenuation or when the last year has no increase
    """
    # Resolve the begin-value policy from the mode
    if mode == VARIATION_MULTIPLICATIVE_FIXED:
        value_at_start_fn = value_at_start_fixed
    elif mode == VARIATION_MULTIPLICATIVE_WITH_ACTIVE_SIGNALS:
        value_at_start_fn = value_at_start_multiplicative_total_with_active_signals
    elif mode == VARIATION_MULTIPLICATIVE_BEFORE_ACTIVE_SIGNALS:
        value_at_start_fn = value_at_start_multiplicative_total_before_active_signals
    else:
        raise ValueError(f"Unknown variation method: {mode}")

    # Build attenuation schedule (empty when yearly_attenuation_fraction == 0)
    attenuation_schedule = (
        build_attenuation_schedule(idx, attenuation_month) if yearly_attenuation_fraction > 0.0 else []
    )
    next_attenuation_idx = 0
    cumulative_attenuation = 0.0
    attenuation_amount = 0.0

    total_cum = pd.Series(0.0, index=idx)
    processed_events: List[Dict[str, Any]] = []

    for sv in signal_variations:
        start_date = sv["start_date"]

        # Apply pending attenuations whose attenuation_date <= this event's start_date.
        # Attenuation deducts yearly_attenuation_fraction * max(0, year_increase) from total_cum[attenuation_date:],
        # so subsequent value_at_start sees a reduced base value.
        while (
            next_attenuation_idx < len(attenuation_schedule)
            and attenuation_schedule[next_attenuation_idx]["attenuation_date"] <= start_date
        ):
            attenuation_amount = apply_yearly_attenuation(
                total_cum, attenuation_schedule[next_attenuation_idx], yearly_attenuation_fraction
            )
            cumulative_attenuation += attenuation_amount
            next_attenuation_idx += 1

        # Geometric compounding of discrete per-period rates.
        # Continuous alternative: np.expm1(sv["variation"].cumsum())
        # Not used — exp is not linear over addition, so summing log-variations across
        # events then exponentiating diverges from summing discrete terms directly
        # (see design note above convert_cumulative_pct_to_signed_period_variations).
        # On non-active days variation=0, so (1+0)=1 preserves the last active value via cumprod.
        event_cum = (1.0 + sv["variation"]).cumprod() - 1.0

        # Apply deduction at each event boundary, begin (scales down allocation from day 1) and end (scales down end-boundary variation)
        if boundary_deduction_pct > 0.0:
            event_cum = apply_event_deduction(event_cum, sv["end_date"], boundary_deduction_pct)

        # Total value at event start according to the selected mode
        event_base = value_at_start_fn(
            start_date=start_date,
            idx=idx,
            total_cum=total_cum,
            processed_events=processed_events,
            cumulative_attenuation=cumulative_attenuation,
        )

        # Event's locked allocation = w * V(start_date)
        locked_weight = weight * event_base

        # Accumulate this event's scaled cumulative variation into the total.
        # In-place mutation avoids allocating a new Series on each iteration.
        # Safe because event_cum shares the same idx (reindex guarantees alignment).
        total_cum.values[:] += locked_weight * event_cum.values

        processed_events.append(
            {
                "start_date": sv["start_date"],
                "end_date": sv["end_date"],
                "event_cum": event_cum,
                "locked_weight": locked_weight,
            }
        )

    # Apply any attenuations scheduled after the last event's start_date. The
    # schedule ends with the last calendar year, so attenuation_amount ends as its attenuation.
    while next_attenuation_idx < len(attenuation_schedule):
        attenuation_amount = apply_yearly_attenuation(
            total_cum, attenuation_schedule[next_attenuation_idx], yearly_attenuation_fraction
        )
        next_attenuation_idx += 1

    return total_cum, attenuation_amount


def apply_event_deduction(
    event_cum: pd.Series,
    end_date: pd.Timestamp,
    boundary_deduction_pct: float,
) -> pd.Series:
    """Apply begin and end deduction to an event's cumulative variation series.

    Begin deduction scales down the event allocation from event start.
    End deduction additionally scales down the end-boundary variation from end_date onward.

    Mathematically equivalent to:
    - Begin: scaling the allocation by (1 - c) from the first active day
    - End: applying a second (1 - c) scale factor from end_date onward
    where c = boundary_deduction_pct / 100.

    During event: adjusted_t = (1 + event_cum_t) × (1 - c) - 1
    After  end:   adjusted_t = (1 + event_cum_t) × (1 - c)² - 1

    Args:
        event_cum: Raw cumulative variation series for the event
        end_date: Date when the event ends
        boundary_deduction_pct: Deduction at each event boundary as percentage (e.g., 0.1 = 0.1%)

    Returns:
        Deduction-adjusted cumulative variation series
    """
    deduction_factor = 1.0 - boundary_deduction_pct / 100.0
    # Begin deduction: allocation scaled down from day 1
    adjusted = (1.0 + event_cum) * deduction_factor - 1.0
    # End deduction: end-boundary variation scaled down from end_date onward
    end_mask = adjusted.index >= end_date
    adjusted.loc[end_mask] = (1.0 + adjusted.loc[end_mask]) * deduction_factor - 1.0
    return adjusted


def value_at_start_fixed(
    start_date: pd.Timestamp,
    idx: pd.DatetimeIndex,
    total_cum: pd.Series,
    processed_events: List[Dict[str, Any]],
    cumulative_attenuation: float = 0.0,
) -> float:
    """Return constant base value 1.0, giving a fixed allocation equal to w for every event."""
    del start_date, idx, total_cum, processed_events, cumulative_attenuation
    return 1.0


def value_at_start_multiplicative_total_with_active_signals(
    start_date: pd.Timestamp,
    idx: pd.DatetimeIndex,
    total_cum: pd.Series,
    processed_events: List[Dict[str, Any]],
    cumulative_attenuation: float = 0.0,
) -> float:
    """Return base value at event start including active and ended prior events.

    total_cum already incorporates attenuation deductions, so cumulative_attenuation is unused.
    """
    del processed_events, cumulative_attenuation
    pos = idx.searchsorted(start_date, side="right")
    if pos > 0:
        return float(1.0 + total_cum.iloc[pos - 1])
    return 1.0


def value_at_start_multiplicative_total_before_active_signals(
    start_date: pd.Timestamp,
    idx: pd.DatetimeIndex,
    total_cum: pd.Series,
    processed_events: List[Dict[str, Any]],
    cumulative_attenuation: float = 0.0,
) -> float:
    """Return base value at event start including only prior ended events.

    This function reconstructs value from individual event contributions (which
    do not include attenuation deductions), so cumulative_attenuation is subtracted to
    reflect the reduced base value after attenuations.
    """
    del total_cum
    eval_pos = idx.searchsorted(start_date, side="right") - 1
    if eval_pos < 0:
        return 1.0

    value_at_start = 1.0
    for ev in processed_events:
        if ev["end_date"] < start_date:
            # Reading event_cum at eval_pos (new event's start row) is valid because
            # after end_date all per-period variation rates are 0 (reindex fillna(0)), so
            # cumprod factors are 1 and event_cum is flat from end_date onward. The value
            # at eval_pos therefore equals the value at the event's own final active day.
            value_at_start += float(ev["locked_weight"] * ev["event_cum"].iloc[eval_pos])
    return value_at_start - cumulative_attenuation
