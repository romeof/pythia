"""
The gate on the previous bar (trigger.gate): the applicable items, the pre-filter's predicate, the begin vector
both replay modes read with its columns and their check, and the one-item test.

Structure:
1. Applicable items (applicable_items)
2. Pre-filter predicate (blocking_item)
3. Begin vector (begin_allowed, gate_columns, check_gate_columns)
4. Item test (item_passes)
"""

import operator
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..common.constants import GATE_OP_GE, GATE_OP_LE
from ..common.schema import GateItem

# The comparison of each item op
GATE_OPERATIONS = {GATE_OP_GE: operator.ge, GATE_OP_LE: operator.le}

# =============================================================================
# APPLICABLE ITEMS
# =============================================================================


def applicable_items(series_id: str, items: Sequence[GateItem]) -> List[GateItem]:
    """The items whose exempt_series does not name the series."""
    return [item for item in items if series_id not in item.exempt_series]


# =============================================================================
# PRE-FILTER PREDICATE
# =============================================================================


def blocking_item(df: pd.DataFrame, items: Sequence[GateItem]) -> Optional[GateItem]:
    """The first item that passes on no row of the frame, None when each passes somewhere (the pre-filter)."""
    return next((item for item in items if not item_passes(df, item).any()), None)


# =============================================================================
# BEGIN VECTOR
# =============================================================================


def begin_allowed(series_id: str, df: pd.DataFrame, items: Sequence[GateItem]) -> np.ndarray:
    """allowed[i]: bar i-1 passes every item; allowed[0] is False (no previous bar). Checks the items' columns first."""
    check_gate_columns(series_id, df, gate_columns(items))
    passes = np.ones(len(df), dtype=bool)
    for item in items:
        passes &= item_passes(df, item)
    allowed = np.zeros(len(df), dtype=bool)
    allowed[1:] = passes[:-1]
    return allowed


def gate_columns(items: Sequence[GateItem]) -> List[str]:
    """The distinct columns the items name, sorted: the reader's extra columns and both column checks."""
    return sorted({item.column for item in items})


def check_gate_columns(series_id: str, df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise ValueError naming the series and the configured gate columns the frame lacks, sorted."""
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"series {series_id} lacks the configured gate columns {missing}")


# =============================================================================
# ITEM TEST
# =============================================================================


def item_passes(df: pd.DataFrame, item: GateItem) -> np.ndarray:
    """value op threshold per row, in the column's dtype (a Python float casts to it, NEP 50); a NaN compares False."""
    return GATE_OPERATIONS[item.op](df[item.column].to_numpy(copy=False), item.threshold)
