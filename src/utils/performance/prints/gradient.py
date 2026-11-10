"""
Print utilities for the gradient performance output.

Provides:
- print_total_variation: print grouping total variation series
- print_gradient_plot_values: print all top-panel curves (gradient, attenuation, references)

Structure:
1. print_total_variation (standalone)
2. print_gradient_plot_values (standalone)
"""

from typing import Dict, Optional

import pandas as pd

from ...common.printframe import print_dataframe_output


def print_total_variation(
    gradient_series: Optional[pd.Series],
) -> None:
    """Print per-date gradient delta series.

    Args:
        gradient_series: Cumulative gradient series (skipped if None)
    """
    if gradient_series is None:
        return
    # Per-date view: change versus the previous observed date.
    per_date_delta = gradient_series.diff().fillna(gradient_series.iloc[0])
    per_date_delta.name = getattr(gradient_series, "name", None)
    print_dataframe_output(
        task_name="PERFORMANCE",
        sub_task_name="GRADIENT PER DATE",
        series_id=None,
        df=per_date_delta,
        full_output=True,
    )


def print_gradient_plot_values(
    gradient_series: Optional[pd.Series],
    attenuation_adjusted_gradient: Optional[pd.Series] = None,
    reference_variation_map: Optional[Dict[str, pd.Series]] = None,
) -> None:
    """Print per-date values for all top-panel curves.

    Columns: total_variation (always), total_variation_after_attenuation (if attenuation
    active), reference_{id} (one per reference).

    Args:
        gradient_series: Pure cumulative gradient series
        attenuation_adjusted_gradient: Attenuation-adjusted gradient (None when no attenuation)
        reference_variation_map: Pre-computed reference cumulative variation series
    """
    if gradient_series is None:
        return
    cols: Dict[str, pd.Series] = {"total_variation": gradient_series}
    if attenuation_adjusted_gradient is not None:
        cols["total_variation_after_attenuation"] = attenuation_adjusted_gradient
    for ref_id in sorted((reference_variation_map or {}).keys()):
        ref_series = reference_variation_map[ref_id]
        if ref_series is not None and len(ref_series) > 0:
            cols[f"reference_{ref_id}"] = ref_series
    df = pd.DataFrame(cols)
    print_dataframe_output(
        task_name="PERFORMANCE",
        sub_task_name="GRADIENT PLOT",
        series_id=None,
        df=df,
        full_output=True,
    )
