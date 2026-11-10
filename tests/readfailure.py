"""Regression for read errors in the HDF5Manager read helpers (HYP-09: one function family).

One planted failure per merged site, each placed where at base only that site's handler catches
it: get_dataframe_fast (HDFStore.get raises for /series_0), iter_series (get_dataframe_fast
raises for /series_0), get_keys_set (HDFStore.keys raises on the trigger store), each through testkit's
planted-failure check under --task all on a one-series realistic copy (the failure record and a
non-zero return code expected); and get_dataframe through its remaining caller, the replot (HDFStore.get
raises for /signals while replot_gradient_from_hdf5 reads a performance file of a healthy run):
the planted error must end the replot with a non-zero return code. The unfixed helpers log the error
and return None or an empty set, so the series, the keys or the signals vanish while the run
returns 0.

Absent keys of the replot: a healthy run whose display lists are all empty writes no
/gradient_param_sequences (Performance writes it only for a non-empty list); the replot of its
store must return 0 with the figure written and no line naming that optional key, and the replot of
a copy without /gradient_params must return non-zero with the descriptive error that names the key.
Base logs the optional key's absence as a read error; with the helpers raising, a replot that reads
the key without consulting the key list fails on it.

Structure:
1. main
2. The replot cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import shutil
from pathlib import Path

import pandas as pd

TASK = "all"
SERIES_KEY = "series_0"
# (target, key prefix, store folder): HDFStore.keys raises only on the trigger store, read first by the Signal task
PIPELINE_CASES = (("pandas.io.pytables:HDFStore.get", SERIES_KEY, ""),
                  ("src.utils.common.hdf5manager:HDF5Manager.get_dataframe_fast", SERIES_KEY, ""),
                  ("pandas.io.pytables:HDFStore.keys", "", "trigger"))
SIGNALS_KEY = "/signals"
PARAMS_KEY, SEQUENCES_KEY = "/gradient_params", "/gradient_param_sequences"
ABSENT_PARAMS = f"{PARAMS_KEY} is absent from the HDF5 store"
# Every list the store keeps in /gradient_param_sequences, emptied (the whitelist lives in dev.yaml)
NO_DISPLAY_LISTS = {"performance.metrics.display_order_signals": [], "performance.metrics.display_order_gradient": []}
NO_WHITELIST = {"performance.gradient.plots_whitelist": []}
NO_PLANTED_KEY = ""
REPLOT_CODE = testkit.CHILD_BOOTSTRAP + (
    "\nfrom pathlib import Path\nimport pandas as pd\n"
    "from src.utils.performance.replot import replot_gradient_from_hdf5\n"
    "original = pd.HDFStore.get\n"
    "def planted(self, key):\n"
    "    if key == sys.argv[6]:\n"
    "        raise OSError(testkit.PLANTED_ERROR)\n"
    "    return original(self, key)\n"
    "pd.HDFStore.get = planted\n"
    "replot_gradient_from_hdf5(Path(sys.argv[4]), save_path=Path(sys.argv[5]))\n")


def main() -> None:
    """The three pipeline cases and the two replot cases, then the summary (return code 1 on any FAIL)."""
    for target, key_prefix, store_folder in PIPELINE_CASES:
        testkit.planted_failure(KIT, target=target, task=TASK, key_prefix=key_prefix, store_folder=store_folder)
    replot_read_failure()
    replot_absent_keys()
    KIT.finish()


# =============================================================================
# THE REPLOT CASES
# =============================================================================


def replot_read_failure() -> None:
    """get_dataframe's read error of /signals ends the replot with a non-zero return code."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        healthy = testkit.run_pipeline(KIT, folder, "healthy", config_file, [TASK])
        performance = testkit.output_file(config_file, "performance")
        KIT.verdict("replot case: non-vacuity: a healthy run wrote a performance store", healthy.return_code == 0
                    and performance.is_file(), "return code 0 and a store", {"return_code": healthy.return_code, "store": performance.is_file()})
        child = run_replot(folder, "replot", performance, folder / "replot.png", SIGNALS_KEY)
        reported = [line for line in testkit.plain_lines(child.text) if testkit.PLANTED_ERROR in line]
        KIT.verdict("replot case: the planted read error of /signals is reported", bool(reported), testkit.PLANTED_ERROR,
                    reported[:3])
        KIT.verdict("replot case: return != 0", child.return_code not in (0, None), "return != 0",
                    {"return_code": child.return_code, "timed_out": child.timed_out})


def replot_absent_keys() -> None:
    """Without the optional /gradient_param_sequences the replot draws the figure and no line names the key;
    without /gradient_params it ends non-zero with the error that names that key."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy={**testkit.ONE_SERIES, **NO_DISPLAY_LISTS},
                                          dev=dict(NO_WHITELIST))
        healthy = testkit.run_pipeline(KIT, folder, "healthy", config_file, [TASK])
        performance = testkit.output_file(config_file, "performance")
        keys = testkit.store_keys(performance)
        KIT.verdict(f"absent-key case: non-vacuity: a healthy run wrote {PARAMS_KEY} without {SEQUENCES_KEY}",
                    healthy.return_code == 0 and keys is not None and PARAMS_KEY in keys and SEQUENCES_KEY not in keys,
                    f"return code 0, {PARAMS_KEY} without {SEQUENCES_KEY}", {"return_code": healthy.return_code, "keys": keys})
        figure = folder / "nosequences.png"
        child = run_replot(folder, "nosequences", performance, figure, NO_PLANTED_KEY)
        KIT.verdict(f"absent-key case: the replot without {SEQUENCES_KEY} returns 0 with the figure",
                    child.return_code == 0 and figure.is_file(), "return code 0 and a figure",
                    {"return_code": child.return_code, "timed_out": child.timed_out, "figure": figure.is_file()})
        naming = [line for line in testkit.plain_lines(child.text) if SEQUENCES_KEY in line]
        KIT.verdict(f"absent-key case: no line names the optional {SEQUENCES_KEY}", not naming, "none", naming[:3])
        stripped = folder / "noparams.h5"
        shutil.copyfile(performance, stripped)
        with pd.HDFStore(str(stripped), mode="a") as store:
            store.remove(PARAMS_KEY)
        child = run_replot(folder, "noparams", stripped, folder / "noparams.png", NO_PLANTED_KEY)
        named = [line for line in testkit.plain_lines(child.text) if ABSENT_PARAMS in line]
        KIT.verdict(f"absent-key case: the replot without {PARAMS_KEY} ends non-zero with the error naming it",
                    child.return_code not in (0, None) and bool(named), f"return != 0 and '{ABSENT_PARAMS}'",
                    {"return_code": child.return_code, "timed_out": child.timed_out, "named": named[:2]})


def run_replot(folder: Path, role: str, store: Path, figure: Path, planted_key: str) -> testkit.Child:
    """replot_gradient_from_hdf5 on the store in a child that saves the figure, HDFStore.get raising for the
    planted key (for none when it is empty)."""
    return testkit.run_child(KIT, folder, role, ["-c", REPLOT_CODE, *testkit.child_args(KIT), str(store), str(figure),
                                                 planted_key])


if __name__ == "__main__":
    main()
