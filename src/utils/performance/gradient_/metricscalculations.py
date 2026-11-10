"""
Scalar formula helpers for gradient-level metric calculations.

All metrics operate on periodic (per-bar) variations derived from the
cumulative percentage series. Annualization uses an empirical factor
m = (N-1) / T_years where N-1 is the count of variation intervals (N data points).

Structure:
1. Shared helpers (_cumulative_pct_to_periodic_variations, the shared conversion;
   _compute_annualization_factor, the shared annualization)
2. Variation rate metrics (_calculate_acr)
3. Fall metrics (_compute_gradient_and_running_max, _calculate_max_fall, _calculate_fall_dur,
   _calculate_below_max_dur)
"""

import numba
import numpy as np
import pandas as pd


# =============================================================================
# SHARED HELPERS
# =============================================================================


def _cumulative_pct_to_periodic_variations(cumulative_pct: np.ndarray) -> np.ndarray:
    """Convert cumulative percentage series to periodic decimal variations.

    Formula: R_t = (1 + C_t/100) / (1 + C_{t-1}/100) - 1

    Args:
        cumulative_pct: 1-D array of cumulative variation percentages.

    Returns:
        1-D array of periodic decimal variations (length = len(cumulative_pct) - 1).
    """
    gradient = 1.0 + cumulative_pct / 100.0
    return gradient[1:] / gradient[:-1] - 1.0


def _compute_annualization_factor(
    index: pd.DatetimeIndex,
) -> tuple[float, float]:
    """Compute empirical annualization factor and observation span in years.

    Args:
        index: DatetimeIndex of the series.

    Returns:
        Tuple of (m, n_years) where m = periods per year, n_years = total span.
    """
    span_days = (index[-1] - index[0]).days
    n_years = span_days / 365.25
    if n_years < 1e-6:
        return 1.0, 0.0
    n_intervals = len(index) - 1
    m = n_intervals / n_years
    return m, n_years


# =============================================================================
# VARIATION RATE METRICS
# =============================================================================


def _calculate_acr(total_variation_decimal: float, n_years: float) -> float:
    """Annualized Compound Rate.

    acr = (1 + total_variation)^(1/n_years) - 1

    Args:
        total_variation_decimal: Total cumulative variation as a decimal
            (e.g. 0.50 for 50%).
        n_years: Observation span in years.

    Returns:
        acr as a percentage.
    """
    if n_years < 1e-6:
        return 0.0
    base = 1.0 + total_variation_decimal
    if base <= 0.0:
        return -100.0
    acr = base ** (1.0 / n_years) - 1.0
    return acr * 100.0


# =============================================================================
# FALL METRICS
# =============================================================================


def _compute_gradient_and_running_max(
    cumulative_pct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cumulative percentage to gradient level and its running maximum.

    Pre-computes the shared arrays used by all fall metric functions.

    Args:
        cumulative_pct: 1-D array of cumulative variation percentages.

    Returns:
        Tuple of (gradient, running_max) arrays.
    """
    gradient = 1.0 + cumulative_pct / 100.0
    running_max = np.maximum.accumulate(gradient)
    return gradient, running_max


def _calculate_max_fall(gradient: np.ndarray, running_max: np.ndarray) -> float:
    """Maximum fall from the running maximum.

    fall_t = (gradient_t - running_max_t) / running_max_t
    max_fall = |min(fall_t)| * 100

    Args:
        gradient: 1-D array of (1 + cumulative_pct / 100).
        running_max: Running maximum of gradient.

    Returns:
        Maximum fall from the running maximum as a positive percentage.
    """
    fall = (gradient - running_max) / running_max
    return float(abs(np.min(fall)) * 100.0)


@numba.njit(cache=True)
def _calculate_fall_dur(gradient: np.ndarray, running_max: np.ndarray) -> int:
    """Duration of the maximum fall from the running maximum (deepest point).

    Finds the deepest point of the maximum decline, then walks backward to
    the preceding maximum and forward to recovery (or end of series).

    Args:
        gradient: 1-D array of (1 + cumulative_pct / 100).
        running_max: Running maximum of gradient.

    Returns:
        Duration in periods of the curve index of the deepest fall episode.

    Notes:
        Scope: this measures the *deepest* fall from the running maximum by
        magnitude, not the *longest* one. A series can have a shallower
        but longer below-maximum period that _calculate_below_max_dur captures
        instead.

        Ongoing declines: if the series ends before the deepest point
        fully recovers, the recovery walk stops at the last available
        observation. The returned duration is therefore truncated at the
        end of the data, which is standard convention for an unrecovered
        fall from the running maximum.

        Boundary subtlety: the recovery loop condition is
        ``recovery_idx < len(gradient) - 1``, so the last index is reached
        by incrementing past the second-to-last element, not by evaluating
        the last element directly. As a result the loop cannot distinguish
        "recovered exactly at the final point" from "still below the running maximum at
        the final point" — both end with recovery_idx at the last index.
        The returned duration is correct in either case; a separate
        recovered/unrecovered flag would require an explicit post-loop check.
    """
    fall = (gradient - running_max) / running_max

    deepest_idx = int(np.argmin(fall))

    prior_max_idx = deepest_idx
    while prior_max_idx > 0 and gradient[prior_max_idx] < running_max[prior_max_idx]:
        prior_max_idx -= 1

    recovery_idx = deepest_idx
    prior_max = running_max[deepest_idx]
    while recovery_idx < len(gradient) - 1 and gradient[recovery_idx] < prior_max:
        recovery_idx += 1

    return recovery_idx - prior_max_idx


@numba.njit(cache=True)
def _calculate_below_max_dur(gradient: np.ndarray, running_max: np.ndarray) -> int:
    """Longest below-maximum period across all falls from the running maximum.

    Finds the maximum gap between consecutive points at the running maximum (where decline = 0).
    May correspond to a different event than the deepest decline.

    Args:
        gradient: 1-D array of (1 + cumulative_pct / 100).
        running_max: Running maximum of gradient.

    Returns:
        Longest decline duration in periods of the curve index.
    """
    at_running_max = gradient >= running_max

    max_duration = 0
    current_duration = 0
    for is_running_max in at_running_max:
        if is_running_max:
            max_duration = max(max_duration, current_duration)
            current_duration = 0
        else:
            current_duration += 1
    max_duration = max(max_duration, current_duration)
    return max_duration

