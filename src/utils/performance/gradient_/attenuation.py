"""
Attenuation helper functions for the gradient computation.

This module contains pure helper functions for the yearly attenuation mechanism:
1. build_attenuation_schedule — pre-compute attenuation dates from the curve index (the observed dates)
2. apply_yearly_attenuation — apply a single yearly attenuation to total_cum in-place
"""

from typing import Dict, List

import pandas as pd


def build_attenuation_schedule(
    idx: pd.DatetimeIndex,
    attenuation_month: int,
) -> List[Dict[str, pd.Timestamp]]:
    """Build yearly attenuation schedule from the curve index (the observed dates).

    Every calendar year Y in the index gets an item:
        attenuation_date = first observed date on or after the 1st of attenuation_month in Y+1,
        or idx[-1] when the index ends before it (always so for the last year).

    Args:
        idx: Global curve index (the observed dates)
        attenuation_month: Month (1-12) when the yearly attenuation is applied

    Returns:
        List of dicts with 'attenuation_date', 'year_start', 'year_end' keys,
        sorted by attenuation_date ascending; the last item is the last calendar
        year. Empty only for an empty idx.
    """
    if len(idx) == 0:
        return []

    years = sorted(idx.year.unique())
    schedule: List[Dict[str, pd.Timestamp]] = []

    for year in years:
        year_dates = idx[idx.year == year]
        if len(year_dates) == 0:
            continue

        year_start = year_dates[0]
        year_end = year_dates[-1]

        # First observed date on or after the 1st of attenuation_month in the next year
        target = pd.Timestamp(year + 1, attenuation_month, 1)
        candidates = idx[idx >= target]
        attenuation_date = candidates[0] if len(candidates) > 0 else idx[-1]

        schedule.append({
            "attenuation_date": attenuation_date,
            "year_start": year_start,
            "year_end": year_end,
        })

    return schedule


def apply_yearly_attenuation(
    total_cum: pd.Series,
    attenuation_item: Dict[str, pd.Timestamp],
    yearly_attenuation_fraction: float,
) -> float:
    """Apply a single yearly attenuation to total_cum in-place.

    Computes the year's increase (clamped at zero for a year that decreases)
    and deducts yearly_attenuation_fraction * increase from all rows at attenuation_date onward.

    Args:
        total_cum: Cumulative variation series (modified in-place)
        attenuation_item: Dict with 'attenuation_date', 'year_start', 'year_end' timestamps
        yearly_attenuation_fraction: Fraction of the yearly increase to remove [0.0, 1.0]

    Returns:
        Attenuation amount applied (non-negative)
    """
    attenuation_amount = yearly_attenuation_fraction * _year_increase(
        total_cum, attenuation_item["year_start"], attenuation_item["year_end"]
    )
    total_cum.loc[attenuation_item["attenuation_date"]:] -= attenuation_amount
    return attenuation_amount


def _year_increase(
    total_cum: pd.Series,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
) -> float:
    """Compute the non-negative increase on total_cum between year_start and year_end."""
    v_start = 1.0 + float(total_cum.at[year_start])
    v_end = 1.0 + float(total_cum.at[year_end])
    return max(0.0, v_end - v_start)
