"""Regression for the NaN the compound kernel used to hide (F-21).

_multiplicative_compound on [10.0, NaN] as float32, the production dtype of the per-series
variations, returns NaN (before F-21: fastmath let the -100 % clamp take the NaN for a variation of
-100 % or below, so -100.0); on finite float32 inputs it equals the compound written out,
[(1 + r1/100) * ... * (1 + rn/100) - 1] * 100.

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import numpy as np

from src.utils.performance.perseries_.metricscalculations import _multiplicative_compound

NAN_INPUT = [10.0, float("nan")]
FINITE_INPUTS = ([10.0, -5.0, 3.0], [50.0, -50.0], [0.0], [-20.0, 25.0, 1.5])


def main() -> None:
    """The NaN input and the finite inputs, then the summary (return code 1 on any FAIL)."""
    found = _multiplicative_compound(np.array(NAN_INPUT, dtype=np.float32))
    KIT.verdict(f"{NAN_INPUT} as float32 compounds to NaN", bool(np.isnan(found)), "nan", found)
    for values in FINITE_INPUTS:
        variations = np.array(values, dtype=np.float32)
        expected = (np.prod(1.0 + variations.astype(np.float64) / 100.0) - 1.0) * 100.0
        found = _multiplicative_compound(variations)
        KIT.verdict(f"{values} as float32 compounds as written out", bool(np.isclose(found, expected)), expected, found)
    KIT.finish()


if __name__ == "__main__":
    main()
