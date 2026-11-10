"""Regression for the plot failures that used to be logged and dropped (G-9, F-14 group H).

testkit's planted-failure check under --task all on a one-series realistic copy, dev flags as
shipped (the per-series plots, the gradient plot and its signal plots on), one case per target:
Figure.savefig (before G-9: "Failed to save plot", return code 0), the gradient plot's
generate_summary_table (before: "Failed to build summary table", return code 0) and
PlotBase.should_emit_signal_plot (before: silent, return code 0); each: one "Task failed" record names
the planted error, then return != 0. The debug plot case: a testkit.planted_run child, --task
trigger on a one-series realistic copy whose dev file sets trigger.strategy.plot_debug, with
plt.subplots of the trigger debug plots planted: one "Task failed" record names the planted
error, then return != 0 (before: "Error creating debug plot", return code 0).

Structure:
1. main
2. The debug plot case
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

TASK = "all"
TARGETS = ("matplotlib.figure:Figure.savefig",
           "src.utils.performance.plots.gradient:generate_summary_table",
           "src.utils.performance.plots.gradient:PlotBase.should_emit_signal_plot")
DEBUG_TARGET = "src.utils.trigger.plots.detection:plt.subplots"
DEBUG_TASK = "trigger"
DEBUG_PLOTS = {"trigger.strategy.plot_debug": True}


def main() -> None:
    """The three shipped-flag cases and the debug plot case, then the summary (return code 1 on any FAIL)."""
    for target in TARGETS:
        testkit.planted_failure(KIT, target=target, task=TASK)
    debug_plot_failure()
    KIT.finish()


# =============================================================================
# THE DEBUG PLOT CASE
# =============================================================================


def debug_plot_failure() -> None:
    """With the trigger debug plots on, a planted plt.subplots failure fails the run."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES), dev=dict(DEBUG_PLOTS))
        child = testkit.planted_run(KIT, folder, "debugplot", config_file, target=DEBUG_TARGET, task=DEBUG_TASK)
        records = [line for line in testkit.plain_lines(child.text)
                   if testkit.HANDLER_PREFIX in line and testkit.PLANTED_ERROR in line]
        KIT.verdict(f"{DEBUG_TARGET} planted under --task {DEBUG_TASK}, debug plots on: one '{testkit.HANDLER_PREFIX}' "
                    "record names the planted error", len(records) == 1, 1, records)
        KIT.verdict(f"{DEBUG_TARGET} planted under --task {DEBUG_TASK}, debug plots on: return != 0",
                    child.return_code not in (0, None), "return != 0", {"return_code": child.return_code, "timed_out": child.timed_out})


if __name__ == "__main__":
    main()
