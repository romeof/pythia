"""Regression for a series that fails in the Trigger producer (HYP-29).

testkit's planted-failure check on Preprocess.process_series under --task trigger, on a
one-series realistic copy: the failure must reach the pipeline's failure record and a non-zero
return code. The unfixed code logs "Error preprocessing series 0", drops the series and returns 0.

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

TARGET = "src.utils.preprocess.preprocess:Preprocess.process_series"
TASK = "trigger"


def main() -> None:
    """The planted preprocessing failure, then the summary (return code 1 on any FAIL)."""
    testkit.planted_failure(KIT, target=TARGET, task=TASK)
    KIT.finish()


if __name__ == "__main__":
    main()
