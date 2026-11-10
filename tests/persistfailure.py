"""Regression for a failed Performance write (HYP-28).

testkit's planted-failure check on Performance._persist_gradient_plot_data under --task all, on
a one-series realistic copy: the failure must reach the pipeline's failure record and a non-zero
return code. The unfixed code logs "Error writing performance HDF5" and returns 0.

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

TARGET = "src.tasks.performance:Performance._persist_gradient_plot_data"
TASK = "all"


def main() -> None:
    """The planted write failure, then the summary (return code 1 on any FAIL)."""
    testkit.planted_failure(KIT, target=TARGET, task=TASK)
    KIT.finish()


if __name__ == "__main__":
    main()
