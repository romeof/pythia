"""Regression for the correlation cache of the signal selection (HYP-19, R-S3).

Two series on different DatetimeIndex values take the date-aligned fallback correlation, which
depends on the signal's own window, so the two orientations differ. Each orientation computed on
a fresh SignalSelection (series added with add_series_values, bounds from
_precompute_window_bounds) gives its own value (non-vacuity: they differ); the second orientation
computed on the instance that already holds the first must equal its fresh value. The unfixed
code returns the first orientation's cached value through the symmetric key.

Structure:
1. main
2. The case
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import numpy as np
import pandas as pd

from src.utils.common.constants import VAR3
from src.utils.signal.selection import SignalSelection

ROWS = 20
DROPPED = [15, 17]


def main() -> None:
    """The two orientations, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config = testkit.validated_config(testkit.config_copy(KIT, folder))
    reverse_orientation(config)
    KIT.finish()


# =============================================================================
# THE CASE
# =============================================================================


def reverse_orientation(config) -> None:
    """Fresh values per orientation differ; the reverse one on the same instance equals its fresh value."""
    dates = pd.bdate_range("2021-01-04", periods=ROWS)
    first = pd.DataFrame({VAR3: 100.0 + np.arange(ROWS) ** 1.5}, index=dates)
    second = pd.DataFrame({VAR3: 50.0 + np.sin(np.arange(ROWS - len(DROPPED)))}, index=dates.delete(DROPPED))
    begin = dates[-1]
    period_back = config.signal.period_back

    def fresh(signal: str, joining: str) -> tuple[SignalSelection, dict, float]:
        selection = SignalSelection(config.signal, config.trigger.replay.mode)
        selection.add_series_values("A", first)
        selection.add_series_values("B", second)
        bounds = selection._precompute_window_bounds(["A", "B"], np.array([begin.to_datetime64()], dtype="datetime64[ns]"),
                                                     period_back)
        return selection, bounds, selection._lookup_or_compute_correlation(signal, joining, begin, period_back, bounds)

    selection, bounds, r_ab = fresh("A", "B")
    _, _, r_ba = fresh("B", "A")
    runnable = KIT.verdict("non-vacuity: the fresh orientations are two different values", r_ab is not None and r_ba is not None
                           and not np.isclose(r_ab, r_ba), "two different values", {"A,B": r_ab, "B,A": r_ba})
    reverse = selection._lookup_or_compute_correlation("B", "A", begin, period_back, bounds) if runnable else None
    KIT.verdict("the reverse orientation on the same instance equals its fresh value", runnable and np.isclose(reverse, r_ba),
                r_ba, reverse)


if __name__ == "__main__":
    main()
