"""
Signed trigger variation, shared by the realistic debug prints and plots.

Structure:
1. compute_trigger_variation (trigger values and their signed variation)
2. directional_variation_pct (variation signed by trigger type)
"""

from typing import Tuple

import pandas as pd

from ..common.common import variation_pct
from ..common.constants import TRIGGER_PLUS_BEGIN, TRIGGER_VALUE


def compute_trigger_variation(
    df: pd.DataFrame, trigger_type: str, begin_idx, end_idx
) -> Tuple[float, float, float]:
    """Lookup trigger values and compute variation percentage.

    Returns:
        Tuple of (begin_value, end_value, signed_pct)
    """
    begin_value = df.loc[begin_idx, TRIGGER_VALUE]
    end_value = df.loc[end_idx, TRIGGER_VALUE]
    signed_pct = directional_variation_pct(trigger_type, begin_value, end_value)
    return begin_value, end_value, signed_pct


def directional_variation_pct(
    trigger_type: str, begin_value: float, end_value: float
) -> float:
    """Compute variation percentage based on trigger type.

    For plus triggers: positive direction = value UP, so variation = variation_pct(end, begin)
    For minus triggers: positive direction = value DOWN, so variation = -variation_pct(end, begin)
    """
    pct = variation_pct(end_value, begin_value)
    if trigger_type == TRIGGER_PLUS_BEGIN:
        return pct
    return -pct
