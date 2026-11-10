"""Regression for the failures the task consumers and the gradient plot used to swallow (G-9, F-14 group A).

testkit's planted-failure check under --task all on a one-series realistic copy, one case per
site: SignalSelection.add_series_values (the Signal consumer), calculate_metrics (the
Performance signal consumer), normalize_dataframe (process_perseries, reached first by the
Performance trigger consumer) and plot_gradient (the gradient plot, no consumer). Each: one
"Task failed" record names the planted error, then return != 0; for the three consumer sites
exactly one ERROR record names series 0, read from the Child that planted_failure returns. The
code before G-9 logs each failure in a handler of its own and the run returns 0.

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

TASK = "all"
# (target, reached through a queue consumer)
CASES = (("src.tasks.signal:SignalSelection.add_series_values", True),
         ("src.tasks.performance:calculate_metrics", True),
         ("src.utils.performance.perseries:normalize_dataframe", True),
         ("src.tasks.performance:plot_gradient", False))


def main() -> None:
    """One planted failure per site, the series record of the consumer sites, then the summary (return code 1 on any FAIL)."""
    for target, consumer in CASES:
        child = testkit.planted_failure(KIT, target=target, task=TASK)
        if consumer:
            records = [line for line in testkit.plain_lines(child.text)
                       if line.startswith(testkit.ERROR_PREFIX) and testkit.SERIES_RECORD in line]
            KIT.verdict(f"{target} planted under --task {TASK}: one ERROR record names series 0", len(records) == 1, 1,
                        records)
    KIT.finish()


if __name__ == "__main__":
    main()
