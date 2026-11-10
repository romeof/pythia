"""Regression for a failed reference write in the Trigger task (HYP-30).

testkit's planted-failure check on HDF5Manager.put_dataframe_normalized under --task trigger, on
a one-series realistic copy, planted only for keys that begin with reference_series_ (every
other write delegates): the failure must reach the pipeline's failure record and a non-zero return code.
The unfixed code logs "Error saving reference raw data for series 0", publishes a trigger file
without /reference_series_0 and returns 0.

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

TARGET = "src.utils.common.hdf5manager:HDF5Manager.put_dataframe_normalized"
REFERENCE_PREFIX = "reference_series_"
TASK = "trigger"


def main() -> None:
    """The planted reference write failure, then the summary (return code 1 on any FAIL)."""
    testkit.planted_failure(KIT, target=TARGET, task=TASK, key_prefix=REFERENCE_PREFIX)
    KIT.finish()


if __name__ == "__main__":
    main()
