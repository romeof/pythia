"""Regression for a read error in Performance's reference loader (HYP-09, its own site).

testkit's planted-failure check on HDF5Manager.get_dataframe_fast under --task all on a
one-series realistic copy, planted only for reference_series_ keys read from the signal store
(the Performance reference loader; the Signal task reads the references from the trigger
store, which delegates): the failure must reach the pipeline's failure record and a non-zero
return code. The unfixed loader logs "Skipping malformed reference series" at WARNING and drops the
reference curve while the run returns 0.

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

TARGET = "src.utils.common.hdf5manager:HDF5Manager.get_dataframe_fast"
REFERENCE_PREFIX = "reference_series_"
SIGNAL_STORE = "signal"
TASK = "all"


def main() -> None:
    """The planted reference read failure, then the summary (return code 1 on any FAIL)."""
    testkit.planted_failure(KIT, target=TARGET, task=TASK, key_prefix=REFERENCE_PREFIX, store_folder=SIGNAL_STORE)
    KIT.finish()


if __name__ == "__main__":
    main()
